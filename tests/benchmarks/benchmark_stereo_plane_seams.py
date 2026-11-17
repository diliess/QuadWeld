"""Stereo plane-intersection seams on saved captures, with independent USD evaluation.

Prediction reads only left/right RGB, the stereo calibration, the saved plate
masks and the arm's camera poses.  USD seams, CAD parts and rendered depth are
opened afterwards, by the evaluation stage.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from tests.benchmarks.seam_metrics import digest, error_stats, runs, segment_distances, write_json
from weld_vision.stereo_io import describe, draw_overlay, load_pair
from weld_vision import stereo_seam as ss


def transform(matrix, points):
    return (np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T)[:, :3]


def left_camera_pose(view, baseline):
    """Arm forward kinematics pose of the left eye (the plan stores the optical midpoint)."""
    offset = np.eye(4)
    offset[0, 3] = -.5 * baseline
    return np.asarray(view['world_from_optical_midpoint'], float) @ offset


def load_case(run, masks_dir, case):
    with np.load(masks_dir / (case + '_left') / 'masks.npz') as stored:
        masks = [stored['raw_horizontal'], stored['raw_vertical']]
    return load_pair(run / 'captures' / case / 'input.json', masks)


# Deliberate input degradations: each returns (left, right, masks).
def _shift(mask, dx, dy):
    return cv2.warpAffine(mask.astype(np.uint8), np.float32([[1, 0, dx], [0, 1, dy]]), mask.shape[::-1]).astype(bool)


def _morph(mask, pixels):
    kernel = np.ones((2 * abs(pixels) + 1,) * 2, np.uint8)
    return (cv2.dilate if pixels > 0 else cv2.erode)(mask.astype(np.uint8), kernel).astype(bool)


def _exposure(image, gain, gamma):
    return np.clip(255 * (image / 255.) ** gamma * gain, 0, 255).astype(np.uint8)


def _noise(image, sigma, seed):
    return np.clip(image + np.random.default_rng(seed).normal(0, sigma, image.shape), 0, 255).astype(np.uint8)


STRESS = {
    '右目变暗（增益 0.7，gamma 1.3）': lambda l, r, m: (l, _exposure(r, .7, 1.3), m),
    '左右目加噪声（sigma 3/255）': lambda l, r, m: (_noise(l, 3, 1), _noise(r, 3, 2), m),
    '掩码腐蚀 5 px': lambda l, r, m: (l, r, [_morph(x, -5) for x in m]),
    '掩码膨胀 5 px': lambda l, r, m: (l, r, [_morph(x, 5) for x in m]),
    '掩码整体平移 (6, 4) px': lambda l, r, m: (l, r, [_shift(x, 6, 4) for x in m]),
    '左右目模糊（sigma 1.5 px）': lambda l, r, m: (cv2.GaussianBlur(l, (0, 0), 1.5), cv2.GaussianBlur(r, (0, 0), 1.5), m),
    '右目竖直错位 0.5 px（校正残差）': lambda l, r, m: (l, cv2.warpAffine(
        r, np.float32([[1, 0, 0], [0, 1, .5]]), r.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), m),
}


def predict(run, masks_dir, output, stress):
    plan = json.loads((run / 'station_plan.json').read_text())
    (output / 'predictions').mkdir(parents=True, exist_ok=True)
    records = []
    for view in plan['views']:
        case = view['id']
        left, right, masks, k, baseline, _ = load_case(run, masks_dir, case)
        start = time.time()
        result = ss.locate_seam(left, right, k, baseline, *masks)
        record = describe(result, runtime_s=time.time() - start)
        pose = left_camera_pose(view, baseline)
        record.update(case=case, station=view['station'], patch=view['patch'], intrinsics_index=k.tolist(),
                      baseline_m=baseline, world_from_left_camera=pose.tolist(),
                      pose_source='Planned arm forward kinematics; no scene truth')
        if result['status'] == 'ok':
            record['joint_ends_world'] = transform(pose, result['joint_ends_xyz']).tolist()
        variants = {}
        for name, change in (STRESS.items() if stress else ()):
            l, r, m = change(left, right, masks)
            variants[name] = ss.locate_seam(l, r, k, baseline, *m)
        record['variants'] = {name: describe(v) for name, v in variants.items()}
        write_json(output / 'predictions' / (case + '.json'), record)
        records.append(record)
        print('STEREO_SEAM_PREDICTED', case, record['status'], json.dumps(record['checks']), flush=True)
    for station in dict.fromkeys(r['station'] for r in records):
        good = [r for r in records if r['station'] == station and r['status'] == 'ok']
        fused = {'station': station, 'views': [r['case'] for r in good], 'status': 'ok' if len(good) >= 2 else 'too_few_views'}
        if len(good) >= 2:
            # Only gross failures are strays (a wrong match is off by pixels of disparity);
            # dropping a merely noisy view would cost its part of the joint.
            tolerance = 1e-3 * float(np.mean([r['checks']['depth_per_disparity_px_mm'] for r in good]))
            merged = ss.fuse_seams([np.array(r['joint_ends_world']) for r in good], tolerance=tolerance)
            fused['stray_tolerance_mm'] = 1000 * tolerance
            fused.update(joint_ends_world=merged['ends'].tolist(), direction=merged['direction'].tolist(),
                         length_m=merged['length_m'], view_deviation_mm=[1000 * v for v in merged['view_deviation_m']],
                         view_used=merged['view_used'])
        write_json(output / 'predictions' / (station + '_fused.json'), fused)


def line_metrics(ends, reference_parts):
    """Reference path against a predicted segment: distance to its line and coverage by the finite segment."""
    ends = np.asarray(ends)
    truth = np.concatenate(reference_parts)
    direction = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
    offset = truth - ends[0]
    to_line = np.linalg.norm(offset - np.outer(offset @ direction, direction), axis=1)
    to_segment = segment_distances(truth, [ends])[0]
    samples = ends[0] + np.linspace(0, 1, max(2, int(np.linalg.norm(ends[1] - ends[0]) / .002) + 1))[:, None] * (ends[1] - ends[0])
    return {'reference_to_line': error_stats(to_line), 'coverage_at_5mm': float(np.mean(to_segment <= .005)),
            'coverage_at_10mm': float(np.mean(to_segment <= .01)),
            'predicted_points_to_reference': error_stats(segment_distances(samples, reference_parts)[0])}


def end_errors(ends, joint_ends):
    ends, joint_ends = np.asarray(ends), np.asarray(joint_ends)
    if np.linalg.norm(ends[0] - joint_ends[0]) > np.linalg.norm(ends[0] - joint_ends[1]):
        ends = ends[::-1]
    return [float(1000 * np.linalg.norm(a - b)) for a, b in zip(ends, joint_ends)]


def evaluate(run, output):
    """Independent stage: the only code here that opens USD seams, CAD parts or the render pose."""
    from pxr import Usd
    from tests.benchmarks.seam_metrics import ground_truth
    from weld_scene.overview_planning import t_geometry
    from weld_vision.stereo_io import read_rgb
    (output / 'images').mkdir(exist_ok=True)
    views, stations = [], {}
    for path in sorted((output / 'predictions').glob('*.json')):
        record = json.loads(path.read_text())
        if 'case' not in record:
            continue
        case, k = record['case'], np.array(record['intrinsics_index'])
        capture_dir = run / 'captures' / case
        capture = json.loads((capture_dir / 'evaluation/capture.json').read_text())
        left, right = (read_rgb(capture_dir / name) for name in ('left.png', 'right.png'))
        curves, pose = ground_truth(run, {'case': case, 'view': 'left'}, k, left.shape[:2])
        parts = [c['xyz'][a:b] for c in curves for a, b in runs(c['in_frame'])]
        fixture, bounds, _, _ = t_geometry(Usd.Stage.Open(capture['usd']), record['station'])
        web, base = bounds['web'], bounds['base']
        joint_world = transform(fixture, [[web[0, 0], web[0, 1], base[1, 2]], [web[1, 0], web[0, 1], base[1, 2]]])
        stations.setdefault(record['station'], {'path_world': np.concatenate([c['world'] for c in curves]),
                                                'joint_world': joint_world})
        row = {'case': case, 'station': record['station'], 'status': record['status'], 'checks': record['checks'],
               'mean_depth_m': float(np.concatenate(parts)[:, 2].mean()), 'runtime_s': record['runtime_s'],
               'planned_vs_rendered_camera_mm': float(1000 * np.linalg.norm(
                   np.array(record['world_from_left_camera'])[:3, 3] - pose[:3, 3]))}
        result = {'status': record['status']}
        if record['status'] == 'ok':
            ends = np.array(record['joint_ends_xyz'])
            row.update(line_metrics(ends, parts), length_m=record['length_m'], opening_deg=record['opening_deg'],
                       end_is_open=record['end_is_open'])
            joint_camera = transform(np.linalg.inv(pose), joint_world)
            visible = [bool(0 <= x <= left.shape[1] - 1 and 0 <= y <= left.shape[0] - 1) for x, y in ss.project(joint_camera, k)]
            row['joint_end_error_mm'] = end_errors(ends, joint_camera) if all(visible) else None
            samples = ends[0] + np.linspace(0, 1, 200)[:, None] * (ends[1] - ends[0])
            result.update(xy=ss.project(samples, k), xy_right=ss.project(samples - [record['baseline_m'], 0, 0], k))
        row['variants'] = {name: ({'status': v['status'], **line_metrics(np.array(v['joint_ends_xyz']), parts)}
                                  if v['status'] == 'ok' else {'status': v['status']})
                           for name, v in record.get('variants', {}).items()}
        truth = np.concatenate(parts)
        draw_overlay(left, right, result, output / 'images' / (case + '.png'),
                     reference=(ss.project(truth, k), ss.project(truth - [record['baseline_m'], 0, 0], k)))
        views.append(row)
        print('STEREO_SEAM_EVALUATED', case, json.dumps({key: row.get(key) for key in (
            'status', 'reference_to_line', 'coverage_at_10mm', 'joint_end_error_mm')}), flush=True)
    fused = []
    for station, truth in stations.items():
        record = json.loads((output / 'predictions' / (station + '_fused.json')).read_text())
        row = {'station': station, 'status': record['status'], 'views': record['views']}
        if record['status'] == 'ok':
            row.update(line_metrics(record['joint_ends_world'], [truth['path_world']]), length_m=record['length_m'],
                       joint_end_error_mm=end_errors(record['joint_ends_world'], truth['joint_world']),
                       view_deviation_mm=record['view_deviation_mm'], view_used=record['view_used'])
        fused.append(row)
        print('STEREO_SEAM_FUSED', json.dumps(row), flush=True)
    metrics = {
        'method': 'Per-plate stereo disparity-plane fit + plane intersection; scale from the calibrated baseline only',
        'inputs': 'Left/right RGB, stereo calibration, saved SAM3 plate masks, planned arm camera poses',
        'reference_to_line': 'In-frame nominal USD path samples (1 mm) to the predicted infinite 3D line',
        'coverage': 'In-frame nominal path samples within 5/10 mm of the predicted finite segment',
        'predicted_points_to_reference': ('Predicted segment sampled every 2 mm to the nearest finite in-frame nominal '
                                          'path; the joint is 30 mm longer than the nominal path at each end'),
        'joint_end_error_mm': 'Predicted ends against the CAD ends of the web/base contact; null when an end is out of view',
        'truth_usage': 'Evaluation only, after all predictions were written; nothing was fitted to or selected by truth',
        'source_hashes': {name: digest(ROOT / name) for name in (
            'weld_vision/stereo_seam.py', 'weld_vision/stereo_io.py', 'tests/benchmarks/benchmark_stereo_plane_seams.py')},
        'views': views, 'fused': fused}
    write_json(output / 'metrics.json', metrics)
    return metrics


def report(run, output, metrics):
    views, fused = metrics['views'], metrics['fused']
    good = [v for v in views if v['status'] == 'ok']
    def mean(values):
        return float(np.mean(values)) if len(values) else float('nan')
    lines = ['# 双目平面交线焊缝定位', '',
             f'数据：`{run.relative_to(ROOT)}`，{len(views)} 组同步双目，{len(good)} 组输出焊缝。',
             '预测只读取左右目 RGB、双目标定、已保存的 SAM3 板件掩码和机械臂相机位姿；'
             'USD 焊缝、CAD 与渲染深度仅在评价阶段打开，未用于拟合或筛选。', '',
             '## 逐视图', '',
             '| 视图 | 距离 m | 线误差 均值 / 最大 mm | 覆盖@5mm | 覆盖@10mm | 端点误差 mm | 夹角 ° | 半分重复性 mm | 耗时 s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v in views:
        if v['status'] != 'ok':
            lines.append(f"| {v['case']} | {v['mean_depth_m']:.2f} | 未输出：{v['status']} | 0% | 0% | – | – | – | {v['runtime_s']:.1f} |")
            continue
        e = v['reference_to_line']
        ends = ' / '.join(f'{x:.1f}' for x in v['joint_end_error_mm']) if v['joint_end_error_mm'] else '端点在画外'
        lines.append(f"| {v['case']} | {v['mean_depth_m']:.2f} | {e['mean_mm']:.2f} / {e['max_mm']:.2f} | "
                     f"{v['coverage_at_5mm']:.0%} | {v['coverage_at_10mm']:.0%} | {ends} | {v['opening_deg']:.1f} | "
                     f"{v['checks']['split_half_spread_mm']:.2f} | {v['runtime_s']:.1f} |")
    lines += ['', '线误差：画内名义焊缝采样点到预测三维直线的距离。覆盖：这些采样点到预测有限线段的距离在阈值内的比例。',
              '端点误差：预测两端与 CAD 腹板/底板接触线两端的距离（名义焊接路径两端各内缩 30 mm，不作为端点真值）。',
              '半分重复性：把每块板的像素分成交错的两半分别重拟合后焊缝的最大偏移的一半，无需真值，可在实机上作为质量指示。', '',
              '## 各工位多视角融合', '',
              '每个机械臂位姿的结果用相机位姿变换到世界坐标后合并成一条线段；长度取各视图并集。', '',
              '| 工位 | 视图数 | 线误差 均值 / 最大 mm | 整条名义焊缝覆盖@5mm | @10mm | 端点误差 mm | 各视图到融合线最大距离 mm |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for f in fused:
        if f['status'] != 'ok':
            lines.append(f"| {f['station']} | {len(f['views'])} | 未融合 | – | – | – | – |")
            continue
        e = f['reference_to_line']
        lines.append(f"| {f['station']} | {len(f['views'])} | {e['mean_mm']:.2f} / {e['max_mm']:.2f} | "
                     f"{f['coverage_at_5mm']:.0%} | {f['coverage_at_10mm']:.0%} | "
                     f"{' / '.join(f'{x:.1f}' for x in f['joint_end_error_mm'])} | "
                     f"{' / '.join(f'{x:.1f}' for x in f['view_deviation_mm'])} |")
    names = list(dict.fromkeys(n for v in views for n in v['variants']))
    if names:
        lines += ['', '## 输入扰动', '', '同一批图像逐项劣化后重跑，其余不变。', '',
                  '| 条件 | 成功 / 总数 | 线误差均值 mm（各视图平均） | 最差视图 mm | 覆盖@10mm |', '|---|---:|---:|---:|---:|',
                  f"| 原始输入 | {len(good)} / {len(views)} | {mean([v['reference_to_line']['mean_mm'] for v in good]):.2f} | "
                  f"{max(v['reference_to_line']['mean_mm'] for v in good):.2f} | {mean([v['coverage_at_10mm'] for v in good]):.0%} |"]
        for name in names:
            rows = [v['variants'][name] for v in views if name in v['variants']]
            ok = [r for r in rows if r['status'] == 'ok']
            lines.append(f"| {name} | {len(ok)} / {len(rows)} | {mean([r['reference_to_line']['mean_mm'] for r in ok]):.2f} | "
                         f"{max([r['reference_to_line']['mean_mm'] for r in ok], default=float('nan')):.2f} | "
                         f"{mean([r['coverage_at_10mm'] for r in ok]):.0%} |")
    lines += ['', '## 图像', '', '绿色：预测焊缝（圆圈为端点）；品红虚线：名义焊缝，仅用于评价。', '']
    lines += [f"![{v['case']}](images/{v['case']}.png)" for v in views]
    lines += ['', '静态仿真、理想针孔与理想相机位姿；不代表实机精度。逐项数据见 `metrics.json` 与 `predictions/`。', '']
    (output / 'README.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True, help='Existing capture run with captures/ and station_plan.json')
    p.add_argument('--masks-dir', type=Path, help='Folder of <case>_left/masks.npz; default <run-dir>/plate_parts')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--stage', choices=('predict', 'evaluate', 'all'), default='all')
    p.add_argument('--no-stress', action='store_true', help='Skip the degraded-input repeats')
    a = p.parse_args()
    run, output = a.run_dir.resolve(), a.output_dir.resolve()
    if a.stage != 'evaluate':
        predict(run, (a.masks_dir or run / 'plate_parts').resolve(), output, not a.no_stress)
    if a.stage != 'predict':
        report(run, output, evaluate(run, output))
    print('STEREO_SEAM_BENCHMARK_COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
