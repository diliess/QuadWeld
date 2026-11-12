"""Joint six-image Nested inference and a minimal RGB/seam-projection report."""

import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from benchmark_stereo_seams import digest, error_stats, predict, reuse_semantics, runs, segment_distances, write_json
from da3_multiview import group_station_views, install_multiview_alignment


def infer_station(model, records, images, k, e, output, process_res):
    import torch

    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    camera_inputs, joint_inputs = [], []

    def camera_audit(module, inputs):
        camera_inputs.append({'normalized_extrinsics_w2c': inputs[0].detach().cpu().numpy().tolist(),
                              'processed_intrinsics': inputs[1].detach().cpu().numpy().tolist(),
                              'processed_hw': list(inputs[2])})

    def joint_audit(module, inputs):
        joint_inputs.append(list(inputs[0].shape))

    hooks = [model.model.da3.cam_enc.register_forward_pre_hook(camera_audit),
             model.model.da3.backbone.register_forward_pre_hook(joint_audit)]
    started = time.monotonic()
    try:
        prediction = model.inference(images, intrinsics=k.copy(), extrinsics=e.copy(),
                                     align_to_input_ext_scale=True, process_res=process_res,
                                     process_res_method='upper_bound_resize')
    finally:
        for hook in hooks:
            hook.remove()
    torch.cuda.synchronize()
    if len(camera_inputs) != 1 or len(joint_inputs) != 1 or joint_inputs[0][1] != len(images):
        raise RuntimeError('Images/camera parameters were not jointly forwarded in one model call')
    if len(prediction.depth) != len(images) or not prediction.is_metric:
        raise RuntimeError('Incomplete joint metric output')
    np.savez_compressed(output / 'before_camera_alignment.npz', **model.multiview_raw)
    for i, record in enumerate(records):
        depth = np.asarray(prediction.depth[i], dtype=np.float32)
        rgb = np.asarray(prediction.processed_images[i], dtype=np.uint8)
        if not np.isfinite(depth).all() or not (depth > 0).all() or rgb.shape[:2] != depth.shape:
            raise RuntimeError('Invalid depth output: ' + record['id'])
        arrays = {'depth_m': depth, 'processed_rgb': rgb, 'intrinsics': prediction.intrinsics[i],
                  'extrinsics_w2c': prediction.extrinsics[i], 'original_hw': record['image_size_wh'][::-1],
                  'is_metric': np.array(True), 'image_sha256': np.array(record['image_sha256'])}
        if prediction.conf is not None:
            arrays['confidence'] = prediction.conf[i]
        np.savez_compressed(output / (record['id'] + '_depth.npz'), **arrays)
    report = {'images': images, 'image_sha256': [digest(image) for image in images],
              'image_ids': [r['id'] for r in records], 'input_intrinsics': k.tolist(),
              'input_extrinsics_w2c': e.tolist(), 'joint_forward_input_shapes': joint_inputs,
              'camera_encoder_inputs': camera_inputs, 'camera_parameters_used_in_forward': True,
              'alignment': model.multiview_alignment_record, 'seconds': time.monotonic() - started,
              'peak_gpu_memory_mib': torch.cuda.max_memory_allocated() / 2**20,
              'truth_usage': 'RGB and camera calibration/poses only; no rendered depths, CAD seams or error thresholds'}
    write_json(output / 'report.json', report)
    print('JOINT_INFERRED', records[0]['station'], len(images), 'images',
          round(report['seconds'], 2), 'seconds', round(report['peak_gpu_memory_mib']), 'MiB', flush=True)


def projected_segments(points, valid, raw_intrinsics):
    """Project retained 3D predictions onto the full, uncropped original RGB."""
    valid = valid & np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    k_index = np.asarray(raw_intrinsics, dtype=float).copy()
    k_index[:2, 2] -= .5  # Native camera intrinsics locate pixel centres at u+.5.
    segments = []
    for a, b in runs(valid):
        projected = points[a:b] @ k_index.T
        segments.append(projected[:, :2] / projected[:, 2:])
    return segments


def evaluate_and_report(source, output, plan, records):
    from PIL import Image, ImageDraw, ImageFont

    by_id = {record['id']: record for record in records}
    all_errors, retained_errors, counts = {}, {}, {}
    projections, rows = {}, []
    for record in records:
        name, station = record['id'], record['station']
        source_reference = source / 'seams/evaluation' / (name + '.npz')
        with np.load(source_reference) as ref:
            xyz, mask = ref['truth_0_xyz'], ref['truth_0_in_frame']
            refs = [xyz[a:b] for a, b in runs(mask)]
        with np.load(output / 'seams/predictions' / (name + '.npz')) as predicted:
            description = json.loads((output / 'seams/predictions' / (name + '.json')).read_text())
            # Reference geometry is camera-local; ensure its processed grid still
            # matches the source grid before reusing independently saved truth.
            with np.load(source / 'seams/predictions' / (name + '.npz')) as original:
                np.testing.assert_allclose(predicted['intrinsics_native'], original['intrinsics_native'], atol=1e-5)
                if predicted['depth_m'].shape != original['depth_m'].shape:
                    raise ValueError('Reference grid mismatch')
            parts, kept, lines = [], [], []
            for j in range(len(description['candidates'])):
                points = predicted[f'candidate_{j}_xyz']
                for a, b in runs(predicted[f'candidate_{j}_valid']):
                    part = points[a:b]
                    retained = (np.linalg.norm(part, axis=1) >= .3) & (np.linalg.norm(part, axis=1) <= .8)
                    parts.append(part)
                    kept.append(part[retained])
                    lines.extend(projected_segments(part, retained, record['intrinsics']))
            points = np.concatenate(parts) if parts else np.zeros((0, 3))
            selected = np.concatenate(kept) if kept else np.zeros((0, 3))
            errors = segment_distances(selected, refs)[0]
            raw_errors = segment_distances(points, refs)[0]
            retained_errors.setdefault(station, []).extend(errors.tolist())
            all_errors.setdefault(station, []).extend(raw_errors.tolist())
            count = counts.setdefault(station, {'original_points': 0, 'retained_points': 0, 'excluded_points': 0})
            count['original_points'] += len(points)
            count['retained_points'] += len(selected)
            count['excluded_points'] += len(points) - len(selected)
            rows.append({'id': name, 'station': station, 'retained_position_error': error_stats(errors),
                         'all_candidate_position_error': error_stats(raw_errors),
                         'retained_points': len(selected), 'excluded_points': len(points) - len(selected),
                         'reference_sha256': digest(source_reference)})
            projections[name] = lines

    stations = []
    for station, errors in retained_errors.items():
        stations.append({'station': station, **counts[station], 'position_error': error_stats(errors),
                         'all_candidate_position_error': error_stats(all_errors[station]),
                         'joint_input_images': sum(r['station'] == station for r in records)})
    metrics = {'selection_rule': 'Retain predicted camera-frame Euclidean distances in [0.3, 0.8] m, independent of truth/error; in-range false positives are not separately classified.',
               'error_definition': 'Predicted sampled 3D points to nearest finite in-frame nominal CAD seam segment; no scene/GT alignment.',
               'source_run': str(source), 'views': rows, 'stations': stations,
               'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in
                                 (Path(__file__), ROOT / 'da3_multiview.py', ROOT / 'vision/weld_detector.py')}}
    write_json(output / 'metrics.json', metrics)
    image_dir = output / 'images'
    image_dir.mkdir(exist_ok=True)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 25)
    markdown = ['# 多角度联合焊缝测试', '',
                '每个工件三个机械臂拍摄位置、共六张双目图像一次联合输入 Nested。平均误差按预测距离300–800 mm内的候选点统计，为三维点到真实焊缝线段的平均距离。绿色为预测焊缝的投影。', '']
    patch_labels = {'left': '左侧拍摄位置', 'center': '中间拍摄位置', 'right': '右侧拍摄位置'}
    for station_result in stations:
        station = station_result['station']
        stats = station_result['position_error']
        mean = f'{stats["mean_mm"]:.2f} mm' if stats else 'N/A（无保留候选）'
        markdown += [f'## {station}', '', f'平均误差：**{mean}**', '']
        for view in (v for v in plan['views'] if v['station'] == station):
            panels = []
            for side in ('left', 'right'):
                name = view['id'] + '_' + side
                original = Image.open(by_id[name]['image']).convert('RGB')
                overlay = original.copy()
                draw = ImageDraw.Draw(overlay)
                for line in projections[name]:
                    coordinates = [tuple(p) for p in line]
                    draw.line(coordinates, fill=(0, 40, 0), width=7)
                    draw.line(coordinates, fill=(0, 255, 60), width=4)
                width, height = original.size
                panel = Image.new('RGB', (width * 2, height + 48), 'white')
                panel.paste(original, (0, 48))
                panel.paste(overlay, (width, 48))
                labels = ImageDraw.Draw(panel)
                labels.text((14, 10), side.capitalize() + ' | Original RGB', fill='black', font=font)
                labels.text((width + 14, 10), side.capitalize() + ' | Predicted seams', fill='black', font=font)
                panels.append(panel)
            figure = Image.new('RGB', (panels[0].width, sum(p.height for p in panels)), 'white')
            y = 0
            for panel in panels:
                figure.paste(panel, (0, y))
                y += panel.height
            filename = view['id'] + '.png'
            figure.save(image_dir / filename)
            markdown += [f'### {patch_labels[view["patch"]]}', '',
                         f'![{station} {patch_labels[view["patch"]]}：原图与预测焊缝投影](images/{filename})', '']
    (output / 'README.md').write_text('\n'.join(markdown))
    print('JOINT_STATION_MEAN_ERRORS', json.dumps(stations), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=ROOT / 'outputs/benchmarks/t_station_close_20261110_01')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--process-res', type=int, default=504)
    p.add_argument('--model-dir', type=Path, default=ROOT / 'thirdparty/depth-anything-3/weights/DA3NESTED-GIANT-LARGE-1.1')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--report-only', action='store_true')
    p.add_argument('--semantic-cache', type=Path, help='Verified SAM cache, including foreground plate-part masks')
    a = p.parse_args()
    source = a.source_run.resolve()
    output = (a.output_dir or ROOT / 'outputs/benchmarks' / datetime.now(ZoneInfo('Asia/Taipei')).strftime('t_station_joint_%Y%m%d_%H%M%S')).resolve()
    if output.exists() and any(output.iterdir()) and not (a.resume or a.report_only):
        p.error('Use a new output directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('MPLCONFIGDIR', str(output / 'mpl_cache'))
    plan = json.loads((source / 'station_plan.json').read_text())
    stations = list(dict.fromkeys(v['station'] for v in plan['views']))
    groups = {station: group_station_views(source, plan, station) for station in stations}
    records = [r for station in stations for r in groups[station][0]]
    if not a.report_only:
        pending = [s for s in stations if not (output / 'predictions' / s / 'report.json').is_file()]
        if pending:
            import torch
            from predict_depth import load_model
            model, model_info = load_model(a.model_dir, stream_blocks=True)
            install_multiview_alignment(model)
            write_json(output / 'model.json', {**model_info, 'model_dir': str(a.model_dir.resolve()),
                       'checkpoint_sha256': digest(a.model_dir / 'model.safetensors'),
                       'config_sha256': digest(a.model_dir / 'config.json'), 'process_res': a.process_res,
                       'weight_dtype': str(next(model.parameters()).dtype), 'stream_blocks': True,
                       'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
                       'adapter_sha256': digest(ROOT / 'da3_multiview.py'), 'script_sha256': digest(Path(__file__))})
            for station in pending:
                station_records, images, k, e, world_poses = groups[station]
                destination = output / 'predictions' / station
                destination.mkdir(parents=True, exist_ok=True)
                write_json(destination / 'inputs.json', {'records': station_records, 'intrinsics': k.tolist(),
                           'extrinsics_w2c_common_frame': e.tolist(), 'world_from_camera_optical': world_poses.tolist(),
                           'pose_source': 'authored station_plan optical midpoint + physical stereo baseline'})
                infer_station(model, station_records, images, k, e, destination, a.process_res)
                gc.collect()
                torch.cuda.empty_cache()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        seam_output = output / 'seams'
        seam_output.mkdir(exist_ok=True)
        seam_records = [{**r, 'depth': str(output / 'predictions' / r['station'] / (r['id'] + '_depth.npz'))} for r in records]
        semantic_cache = (a.semantic_cache or source / 'seams/semantics.npz').resolve()
        seam_plan = {'records': seam_records, 'scope': 'Saved RGB semantics and jointly predicted Nested depth only; no seam truth',
                     'semantic_cache': str(semantic_cache), 'semantic_cache_sha256': digest(semantic_cache)}
        write_json(seam_output / 'plan.json', seam_plan)
        reuse_semantics(semantic_cache, records, seam_output / 'semantics.npz')
        predict(seam_plan, seam_output)
    evaluate_and_report(source, output, plan, records)
    print('JOINT_STATION_BENCHMARK_COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
