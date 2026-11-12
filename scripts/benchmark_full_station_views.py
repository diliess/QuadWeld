"""Six-image joint depth, foreground seam inference, and independent far-view evaluation."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from benchmark_joint_station_views import infer_station, projected_segments
from benchmark_stereo_seams import digest, error_stats, ground_truth, predict, reuse_semantics, runs, segment_distances, write_json
from da3_multiview import group_station_views, install_multiview_alignment


def grouped(root):
    plan = json.loads((root / 'station_plan.json').read_text())
    stations = list(dict.fromkeys(v['station'] for v in plan['views']))
    return plan, {s: group_station_views(root, plan, s) for s in stations}


def depth_stage(root, model_dir, process_res):
    import torch
    from predict_depth import load_model
    plan, groups = grouped(root)
    model, info = load_model(model_dir, stream_blocks=True)
    install_multiview_alignment(model)
    write_json(root / 'model.json', {**info, 'checkpoint_sha256': digest(model_dir / 'model.safetensors'),
               'model_dir': str(model_dir), 'process_res': process_res, 'stream_blocks': True,
               'weight_dtype': str(next(model.parameters()).dtype), 'torch': torch.__version__})
    records = []
    for station, (recs, images, k, e, world) in groups.items():
        destination = root / 'predictions' / station
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / 'inputs.json', {'records': recs, 'intrinsics': k.tolist(),
                   'extrinsics_w2c_common_frame': e.tolist(), 'world_from_camera_optical': world.tolist(),
                   'pose_source': 'Arm FK and physical stereo baseline; no scene-depth truth'})
        if not (destination / 'report.json').is_file():
            infer_station(model, recs, images, k, e, destination, process_res)
        records.extend({**r, 'depth': str(destination / (r['id'] + '_depth.npz'))} for r in recs)
        indices = [i for i, r in enumerate(recs) if r['patch'] == 'center']
        pair_destination = root / 'pair_predictions' / station
        if not (pair_destination / 'report.json').is_file():
            pair_e = np.linalg.inv(world[indices]) @ world[indices[0]]
            infer_station(model, [recs[i] for i in indices], [images[i] for i in indices],
                          k[indices], pair_e.astype(np.float32), pair_destination, process_res)
        gc.collect()
        torch.cuda.empty_cache()
    seam_dir = root / 'seams'
    seam_dir.mkdir(exist_ok=True)
    write_json(seam_dir / 'plan.json', {'source_run': str(root), 'records': records,
               'truth_mode': 'usd_reference', 'scope': 'Predicted depth, RGB, calibration and SAM foreground masks only',
               'candidate_selection': 'Both foreground plate masks support contact; predicted range [1.2, 2.8] m'})


def seam_stage(root):
    from PIL import Image
    plan = json.loads((root / 'seams/plan.json').read_text())
    destination = root / 'seams'
    reuse_semantics(root / 'plate_parts/semantics.npz', plan['records'], destination / 'semantics.npz')
    predict(plan, destination)
    # Preserve every detector output. Select only contacts supported by both
    # foreground parts, without consulting nominal seam locations or errors.
    filtered = destination / 'foreground_predictions'
    filtered.mkdir(exist_ok=True)
    with np.load(destination / 'semantics.npz') as semantics:
        for i, record in enumerate(plan['records']):
            name = record['id']
            description = json.loads((destination / 'predictions' / (name + '.json')).read_text())
            with np.load(destination / 'predictions' / (name + '.npz')) as data:
                arrays = {k: data[k] for k in data.files}
            height, width = arrays['depth_m'].shape
            masks = [cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                     for m in semantics[f'{i}_plate_masks']]
            adjacent = [cv2.dilate(m, np.ones((15, 15), np.uint8)).astype(bool) for m in masks]
            band = adjacent[0] & adjacent[1]
            count_before = count_after = 0
            for j in range(len(description['candidates'])):
                xy, xyz, valid = [arrays[f'candidate_{j}_{key}'] for key in ('xy', 'xyz', 'valid')]
                pixels = np.rint(xy).astype(int)
                on_image = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
                supported = np.zeros(len(xy), bool)
                supported[on_image] = band[pixels[on_image, 1], pixels[on_image, 0]]
                distance = np.linalg.norm(xyz, axis=1)
                selected = valid & supported & (distance >= 1.2) & (distance <= 2.8)
                count_before += int(valid.sum())
                count_after += int(selected.sum())
                arrays[f'candidate_{j}_valid'] = selected
            np.savez_compressed(filtered / (name + '.npz'), **arrays)
            write_json(filtered / (name + '.json'), {**description, 'selection': {
                'rule': 'Within 7 processed pixels of each distinct foreground plate mask; predicted range 1.2-2.8 m',
                'original_valid_points': count_before, 'retained_valid_points': count_after, 'truth_used': False}})
            print('FOREGROUND_SEAM_SELECTED', name, count_after, '/', count_before, flush=True)


def depth_errors(root, record, native_k, shape):
    """Render-depth/CAD visibility is accessed only in this evaluation function."""
    from pxr import Usd, UsdGeom
    from hj90.overview_planning import t_geometry, transform_points
    capture_dir = root / 'captures' / record['case']
    capture = json.loads((capture_dir / 'evaluation/capture.json').read_text())
    pose = np.asarray(capture['world_from_camera_optical'][0 if record['view'] == 'left' else 1])
    stage = Usd.Stage.Open(capture['usd'])
    fixture, bounds, _, _ = t_geometry(stage, record['station'])
    with np.load(capture_dir / 'evaluation/truth.npz') as truth:
        raw_z = truth[record['view'] + '_z_m']
    depth = cv2.resize(raw_z, shape[::-1], interpolation=cv2.INTER_NEAREST)
    k = native_k.copy()
    k[:2, 2] -= .5
    yy, xx = np.indices(shape)
    rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ np.linalg.inv(k).T
    points = rays * np.where(np.isfinite(depth), depth, 0.)[..., None]
    local = transform_points(np.linalg.inv(fixture) @ pose, points.reshape(-1, 3)).reshape(*shape, 3)
    faces = []
    for part in ('base', 'web'):
        low, high = bounds[part]
        contained = ((local >= low - .005) & (local <= high + .005)).all(2)
        face_axis, face_value = (2, high[2]) if part == 'base' else (1, low[1])
        visible = contained & (np.abs(local[..., face_axis] - face_value) < .005) & np.isfinite(depth) & (depth > 0)
        faces.append(cv2.erode(visible.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool))
    return depth, faces


def report_stage(root):
    from PIL import Image, ImageDraw, ImageFont
    plan, groups = grouped(root)
    seam_plan = json.loads((root / 'seams/plan.json').read_text())
    image_dir = root / 'images'
    image_dir.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    view_metrics, stations, centre_tiles = [], [], []
    lines = ['# 整体距离下的机械臂多视角联合焊缝识别', '',
             '每个工位固定机器狗底座，只改变机械臂；三个臂姿的六张双目图像一次联合输入 Nested。绿色为前方横板与竖板共同支持的预测焊缝。', '']
    for station, group in groups.items():
        station_errors, joint_depth_errors, pair_depth_errors = [], [], []
        retained_points, raw_points = 0, 0
        for record in group[0]:
            name = record['id']
            folder = root / 'seams/foreground_predictions'
            desc = json.loads((folder / (name + '.json')).read_text())
            with np.load(folder / (name + '.npz')) as data:
                k, shape = data['intrinsics_index'], data['depth_m'].shape
                curves, pose = ground_truth(root, record, k, shape)
                refs = [c['xyz'][a:b] for c in curves for a, b in runs(c['in_frame'])]
                parts, projections = [], []
                for j in range(len(desc['candidates'])):
                    xyz, valid = data[f'candidate_{j}_xyz'], data[f'candidate_{j}_valid']
                    parts.extend(xyz[a:b] for a, b in runs(valid))
                    projections.extend(projected_segments(xyz, valid, record['intrinsics']))
                points = np.concatenate(parts) if parts else np.zeros((0, 3))
                errors = segment_distances(points, refs)[0]
                station_errors.extend(errors.tolist())
                raw_points += desc['selection']['original_valid_points']
                retained_points += len(points)
                targets = np.concatenate(refs)
                reverse = segment_distances(targets, parts)[0] if parts else np.full(len(targets), np.inf)
                metrics = {'id': name, 'station': station, 'position_error': error_stats(errors),
                           'coverage_at_10mm': float(np.mean(reverse <= .01)),
                           'coverage_at_20mm': float(np.mean(reverse <= .02)),
                           'retained_points': len(points), 'corner_recovery': desc['corner_recovery']}
                view_metrics.append(metrics)
                if record['patch'] == 'center':
                    truth, face_masks = depth_errors(root, record, data['intrinsics_native'], shape)
                    mask = np.logical_or.reduce(face_masks)
                    joint_depth_errors.extend(np.abs(data['depth_m'][mask] - truth[mask]).tolist())
                    with np.load(root / 'pair_predictions' / station / (name + '_depth.npz')) as pair:
                        assert pair['depth_m'].shape == shape
                        np.testing.assert_allclose(pair['intrinsics'], data['intrinsics_native'], atol=1e-5)
                        pair_depth_errors.extend(np.abs(pair['depth_m'][mask] - truth[mask]).tolist())
            original = Image.open(record['image']).convert('RGB')
            overlay = original.copy()
            drawing = ImageDraw.Draw(overlay)
            for projected in projections:
                coordinates = [tuple(p) for p in projected]
                drawing.line(coordinates, fill=(0, 30, 0), width=7)
                drawing.line(coordinates, fill=(0, 255, 60), width=4)
            tile = Image.new('RGB', (1280, 1144), 'white')
            tile.paste(original, (0, 44))
            tile.paste(overlay, (640, 44))
            draw = ImageDraw.Draw(tile)
            draw.text((12, 8), f"{station} {record['patch']} {record['view']} | RGB", fill='black', font=font)
            draw.text((652, 8), 'Predicted foreground seam', fill='black', font=font)
            tile.save(image_dir / (name + '.png'))
            if record['patch'] == 'center' and record['view'] == 'left':
                centre_tiles.append(tile)
        row = {'station': station, 'position_error': error_stats(station_errors),
               'depth_center_joint': error_stats(joint_depth_errors), 'depth_center_pair': error_stats(pair_depth_errors),
               'retained_points': retained_points, 'raw_points': raw_points,
               'angles_deg': [v['azimuth_delta_deg'] for v in plan['views'] if v['station'] == station]}
        stations.append(row)
        mean = f"{row['position_error']['mean_mm']:.2f} mm" if row['position_error'] else '未识别到有效焊缝'
        lines += [f'## {station}', '', f'焊缝平均三维误差：**{mean}**', '']
        for view in (v for v in plan['views'] if v['station'] == station):
            lines += [f"### 机械臂角度 {view['azimuth_delta_deg']:+d}°", '']
            for side in ('left', 'right'):
                name = view['id'] + '_' + side
                lines += [f'![{name}：原图与预测焊缝](images/{name}.png)', '']
        print('FULL_JOINT_STATION_RESULT', json.dumps(row), flush=True)
    metrics = {'stations': stations, 'views': view_metrics,
               'error_definition': 'Selected predicted 3D points to nearest finite in-frame nominal seam segment; no GT alignment or error-based rejection',
               'selection': 'Both predicted foreground plate masks within 7 processed pixels and predicted Euclidean range [1.2, 2.8] m; not manual classification',
               'depth_comparison': 'Identical center RGB pixels, visible CAD/RTX agreeing face interiors eroded 3 processed pixels; joint six-image vs center stereo two-image',
               'truth_usage': 'Independent evaluation after predictions and selection are saved'}
    write_json(root / 'metrics.json', metrics)
    lines += ['平均误差统计通过前方两块板共同支持筛选的全部预测点；未按真实误差删点，仍可能包含误检。部分视角只恢复局部焊缝，平均误差不代表整条焊缝已完整识别。', '',
              '相机约 1.9–2.0 m，超出 HJ-90 标称 300–800 mm 测量距离。双目基线和机械臂相机位姿用于尺度恢复；CAD 焊缝和渲染深度仅用于独立评价。', '']
    (root / 'README.md').write_text('\n'.join(lines))
    preview = Image.new('RGB', (1536, 458), 'white')
    for i, tile in enumerate(centre_tiles):
        preview.paste(tile.resize((512, 458), Image.Resampling.LANCZOS), (i * 512, 0))
    preview.save(image_dir / 'centres_preview.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--stage', choices=('depth', 'seams', 'report'), required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--process-res', type=int, default=504)
    p.add_argument('--model-dir', type=Path, default=ROOT / 'thirdparty/depth-anything-3/weights/DA3NESTED-GIANT-LARGE-1.1')
    a = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['HF_HUB_OFFLINE'] = '1'
    if a.stage == 'depth':
        depth_stage(a.run_dir.resolve(), a.model_dir.resolve(), a.process_res)
    elif a.stage == 'seams':
        seam_stage(a.run_dir.resolve())
    else:
        report_stage(a.run_dir.resolve())
