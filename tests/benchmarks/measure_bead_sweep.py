"""Score the saved sweep against the simulation mesh."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from weld_scene.scene import STEREO_SCENE as SCENE, write_json
from tests.benchmarks.benchmark_bead_toes import stats, to_curve, truth_toes
from weld_vision import bead_seam
from weld_vision.sweep import SPACING, line_error, measure


def root_stats(ends, true_root):
    start, stop = np.asarray(ends)
    direction = (stop - start) / np.linalg.norm(stop - start)
    return {**stats(line_error(true_root, start, direction)),
            'endpoint_mm': (np.linalg.norm(np.array([start, stop]) - true_root[[0, -1]], axis=1) * 1000).tolist()}


def true_axis(true_root):
    start, stop = true_root[[0, -1]]
    length = np.linalg.norm(stop - start)
    return start, (stop - start) / length, float(length)


def toe_stats(segments, curve, true_root):
    start, axis, length = true_axis(true_root)
    count = int(np.ceil(length / SPACING))
    centres = (np.arange(count) + .5) * length / count
    coverage = np.zeros(count, bool)
    for segment in segments:
        along = (np.asarray(segment) - start) @ axis
        coverage |= (centres >= along.min() - SPACING / 2) & (centres <= along.max() + SPACING / 2)
    gaps = np.flatnonzero(np.diff(np.r_[False, ~coverage, False]))
    error = stats(to_curve(np.concatenate(segments), curve)) if segments else {}
    return {**error, 'coverage': float(coverage.mean()), 'covered_length_m': float(coverage.mean() * length),
            'max_gap_mm': float(np.max(gaps[1::2] - gaps[::2], initial=0) * length / count * 1000),
            'segments': len(segments)}, coverage


def actual_standoff(view, true_root):
    eye = np.asarray(view['world_from_optical_midpoint'])[:3, 3]
    start, axis, length = true_axis(true_root)
    nearest = start + np.clip((eye - start) @ axis, 0., length) * axis
    return float(np.linalg.norm(eye - nearest))


def evaluate(run, output):
    from pxr import Usd

    plan = json.loads((run / 'close_plan.json').read_text())
    predictions = json.loads((output / 'predictions.json').read_text())
    overview_views = {v['station']: v for v in json.loads((run / 'overview' / 'views.json').read_text())['views']}
    close_views = {v['id']: v for v in json.loads((run / 'close' / 'views.json').read_text())['views']}
    stage = Usd.Stage.Open(str(SCENE))
    metrics = {}
    for station, observation in plan['overview'].items():
        base_truth, web_truth, root_truth = truth_toes(stage, station, 'front')
        overview_view = overview_views[station]
        midpoint = np.asarray(overview_view['world_from_optical_midpoint'])[:3, 3]
        true_range = float(np.linalg.norm(root_truth[[0, -1]].mean(0) - midpoint))
        coarse = {'predicted_distance_m': observation['estimated_range_m'], 'truth_distance_m': true_range,
                  'distance_error_mm': (observation['estimated_range_m'] - true_range) * 1000,
                  'root': root_stats(observation['root_world_m'], root_truth)}
        fused = predictions['fused'][station]
        merged = {'status': fused['status']}
        if 'root_world_m' in fused:
            merged.update(root=root_stats(fused['root_world_m'], root_truth), views_used=len(fused['views_used']),
                          root_length_m=fused['root_length_m'])
            coverages = []
            for name, curve in (('base', base_truth), ('web', web_truth)):
                merged[name], coverage = toe_stats(fused['toes'][name]['segments_world_m'], curve, root_truth)
                coverages.append(coverage)
            merged['both_toes_coverage'] = float(np.mean(coverages[0] & coverages[1]))
        ordered = [v for v in predictions['views'] if v['station'] == station]
        scan = [v for v in ordered if v['view'] == 'sweep']
        central = scan[len(scan) // 2]['id']
        per_view = []
        for view in ordered:
            capture = close_views[view['id']]
            record = {key: view[key] for key in ('id', 'status', 'distance_m', 'seconds')}
            record['truth_distance_m'] = actual_standoff(capture, root_truth)
            record['range_comparison'] = view['id'] == central or view['view'] == 'range_check'
            if view['status'] == 'ok':
                record['root'] = root_stats(view['root'], root_truth)
                base_index = int(np.argmax(np.abs(np.asarray(view['normals'])[:, 2])))
                for name, index, curve in (('base', base_index, base_truth), ('web', 1 - base_index, web_truth)):
                    toe = view['toes'][index]
                    toe['xyz'] = np.asarray(toe['xyz'])
                    record[name], _ = toe_stats(bead_seam.polyline_segments(toe, min_points=1), curve, root_truth)
            per_view.append(record)
        metrics[station] = {'overview': coarse, 'fused': merged, 'views': per_view,
                            'transitions': [r for r in plan['transitions'] if r['from'].startswith(station + '_')]}
    write_json(output / 'metrics.json', metrics)
    write_report(output, metrics)


def write_report(output, metrics):
    lines = ['# 2 米粗定位与 0.6 米焊道扫查仿真', '',
             '左、右图由 Isaac Sim 的机械臂末端相机采集。距离与误差基于焊道网格，位置以米计，误差以毫米计。', '',
             '## 粗定位', '',
             '| 工位 | 双目估计 m | 真值 m | 距离偏差 mm | 根线平均 / P95 mm |',
             '|---|---:|---:|---:|---:|']
    for station, record in metrics.items():
        c = record['overview']
        lines.append(f"| {station} | {c['predicted_distance_m']:.4f} | {c['truth_distance_m']:.4f} | "
                     f"{c['distance_error_mm']:+.2f} | {c['root']['mean_mm']:.2f} / {c['root']['p95_mm']:.2f} |")
    lines += ['', '## 0.6 米扫查融合', '',
              '| 工位 | 融合视图 | 根线平均 / P95 mm | 底板焊趾平均 / P95 / 最大 mm | 底板覆盖 | '
              '立板焊趾平均 / P95 / 最大 mm | 立板覆盖 | 两侧同时覆盖 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']

    def errors(record, include_max=False):
        keys = ('mean_mm', 'p95_mm', 'max_mm') if include_max else ('mean_mm', 'p95_mm')
        return ' / '.join(f'{record[key]:.2f}' for key in keys) if record and 'mean_mm' in record else '—'

    for station, record in metrics.items():
        f = record['fused']
        if 'root' not in f:
            lines.append(f"| {station} | {f['status']} | | | | | | |")
            continue
        lines.append(f"| {station} | {f['views_used']} | {errors(f['root'])} | {errors(f['base'], True)} | "
                     f"{f['base']['coverage']:.1%} | {errors(f['web'], True)} | {f['web']['coverage']:.1%} | "
                     f"{f['both_toes_coverage']:.1%} |")
    lines += ['', '根线误差取网格根线点到拟合直线的距离。焊趾误差取有效三维点到网格焊趾折线的距离。'
              '覆盖率按真值根线方向的约 1 mm 单元统计，分母为整道焊缝。', '',
              '## 同位点工作距离比较', '',
              '| 工位 | 名义距离 m | 实际距离 m | 状态 | 底板平均 / P95 mm | 底板有效长度 mm | 立板平均 / P95 mm | 立板有效长度 mm |',
              '|---|---:|---:|---|---:|---:|---:|---:|']
    for station, record in metrics.items():
        variants = sorted((v for v in record['views'] if v['range_comparison']), key=lambda v: v['distance_m'])
        for v in variants:
            base, web = v.get('base', {}), v.get('web', {})
            base_length, web_length = (1000 * a.get('covered_length_m', 0.) for a in (base, web))
            lines.append(f"| {station} | {v['distance_m']:.2f} | {v['truth_distance_m']:.3f} | {v['status']} | "
                         f"{errors(base)} | {base_length:.0f} | {errors(web)} | {web_length:.0f} |")
    lines += ['', '各距离保持相同的瞄准点和相机朝向。有效长度反映该次拍摄覆盖和匹配情况。', '',
              '## 扫查位姿与文件', '']
    for station, record in metrics.items():
        failed = sum(row['failed_samples'] for row in record['transitions'])
        tested = sum(row['static_samples'] for row in record['transitions'])
        f = record['fused']
        lines.append(f"- {station} 的相邻位姿插值检查有 {tested} 个静态样点，其中 {failed} 个检查失败。")
        if 'root' in f:
            lines.append(f"  底板和立板最大未覆盖区分别为 {f['base']['max_gap_mm']:.0f} mm 和 {f['web']['max_gap_mm']:.0f} mm。"
                         f" [融合叠图]({station}/overview_overlay.png) · [三维折线]({station}/fused.json)")
    lines += ['', '采集使用实际末端相机外参和离散仿真位姿。碰撞检查使用 MuJoCo 场景及补充的工件包围盒。'
              '未执行行走与机械臂的动力学轨迹，也未测量实机深度精度。', '']
    (output / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--min-views', type=int, default=2)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    if args.min_views < 1:
        parser.error('--min-views must be positive')
    output = args.run_dir / 'results'
    cv2.setNumThreads(1)
    if not args.evaluate_only:
        measure(args.run_dir, output, args.min_views)
    evaluate(args.run_dir, output)
    print('BEAD_SWEEP_REPORT', output / 'README.md')


if __name__ == '__main__':
    main()
