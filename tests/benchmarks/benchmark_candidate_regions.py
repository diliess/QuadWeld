"""Capture and evaluate the six preselected current-scene stereo regions.

Capture uses scene geometry to place a free stereo rig. Prediction sees RGB,
calibration, camera poses and SAM text candidates. CAD and renderer labels are
reserved for evaluation or an explicitly separate labels ablation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / 'assets/scenes/industrial-welding/industrial_welding_full.usda'
REGIONS = ('A01', 'A03', 'A23', 'A06', 'A04', 'B01')
DISTANCE = .65


def serial(value):
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def region_geometry(stage, station):
    """Capture/evaluation only: exact CAD contact and the two relevant meshes."""
    from pxr import UsdGeom
    fixture = stage.GetPrimAtPath('/World/Fixtures/' + station)
    cache = UsdGeom.XformCache()
    world = np.array(cache.GetLocalToWorldTransform(fixture)).T
    if station == 'A04':
        from tests.a04_reference import reference_from_stage
        references = reference_from_stage(stage)
        line = next(item['control_world_m'] for item in references if item['name'] == 'A04_vertical')
        return np.asarray(line), world, ['A04_back', 'A04_return']
    suffix = 'inclined_web' if station == 'A06' else 'parallel_0_west' if station == 'B01' else 'web'
    prim = stage.GetPrimAtPath(f'/World/Fixtures/{station}/{station}_{suffix}')
    raw = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
    selected = np.isclose(raw[:, 2], raw[:, 2].min(), atol=1e-7) & np.isclose(raw[:, 1], raw[:, 1].min(), atol=1e-7)
    edge = transform(raw[selected], np.array(cache.GetLocalToWorldTransform(prim)).T)
    local = transform(edge, np.linalg.inv(world))
    edge = edge[np.argsort(local[:, 0])][[0, -1]]
    if station == 'B01':
        local = transform(edge, np.linalg.inv(world))
        local[:, 0] = [-3., -2.]
        edge = transform(local, world)
    base = 'inclined_base' if station == 'A06' else 'deck' if station == 'B01' else 'base'
    return edge, world, [station + '_' + base, station + '_' + suffix]


def camera_pose(line, fixture, station, fraction):
    from weld_scene.overview_planning import look_at
    tangent = line[1] - line[0]
    tangent /= np.linalg.norm(tangent)
    target = line[0] + fraction * (line[1] - line[0])
    if station == 'A04':
        outward = fixture[:3, :3] @ np.array([1., -1., .18])
        outward /= np.linalg.norm(outward)
    else:
        front = fixture[:3, :3] @ np.array([0., -1., 0.])
        up = np.cross(front, tangent)
        up /= np.linalg.norm(up)
        outward = np.cos(np.radians(35)) * front + np.sin(np.radians(35)) * up
    pose = look_at(target + DISTANCE * outward, target)
    c, s = np.cos(np.radians(-35)), np.sin(np.radians(-35))
    pose[:3, :3] = pose[:3, :3] @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return pose


def capture(args):
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting', 'active_gpu': args.gpu,
                         'multi_gpu': False, 'open_usd': str(SCENE), 'width': 640, 'height': 1100,
                         'sync_loads': True,
                         'extra_args': ['--/exts/omni.kit.async_engine/keep_loop_running=false']})
    try:
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdSemantics
        from weld_scene.optics import load_calibration
        from weld_scene.usd import set_intrinsics
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        for _ in range(60):
            app.update()
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        stage.GetPrimAtPath('/World/Robot/e05l_welding_torch/hj90_active/laser').GetAttribute('inputs:intensity').Set(0.)
        for station in REGIONS:
            for prim in stage.GetPrimAtPath('/World/Fixtures/' + station).GetChildren():
                if prim.IsA(UsdGeom.Mesh):
                    UsdSemantics.LabelsAPI.Apply(prim, 'class').CreateLabelsAttr().Set([prim.GetName()])
        calibration = load_calibration()
        size = tuple(calibration['image_size_wh'])
        k, baseline = np.array(calibration['intrinsics']), calibration['baseline_m']
        paths = ['/World/Cameras/candidate_stereo_' + side for side in ('left', 'right')]
        cameras = [UsdGeom.Camera.Define(stage, path) for path in paths]
        for camera in cameras:
            set_intrinsics(camera, k, size)
        products, groups = [], []
        for path in paths:
            product = rep.create.render_product(path, size)
            group = [rep.AnnotatorRegistry.get_annotator('rgb'),
                     rep.AnnotatorRegistry.get_annotator('distance_to_image_plane'),
                     rep.AnnotatorRegistry.get_annotator('semantic_segmentation', init_params={'colorize': False})]
            for item in group:
                item.attach(product)
            products.append(product)
            groups.append(group)
        records, truth_records = [], {}
        try:
            for station in REGIONS:
                line, fixture, parts = region_geometry(stage, station)
                truth_records[station] = {'root_world_m': line, 'expected_parts': parts, 'has_deposited_bead': False,
                                          'reference_scope': 'selected 1 m window' if station == 'B01' else 'physical contact'}
                for index, fraction in enumerate((.25, .50, .75)):
                    ident = f'{station}_view{index}'
                    folder = args.run_dir / 'captures' / ident
                    (folder / 'evaluation').mkdir(parents=True, exist_ok=True)
                    midpoint = camera_pose(line, fixture, station, fraction)
                    poses = []
                    for camera, shift in zip(cameras, (-baseline / 2, baseline / 2)):
                        pose = midpoint.copy()
                        pose[:3, 3] += shift * midpoint[:3, 0]
                        camera.MakeMatrixXform().Set(Gf.Matrix4d((pose @ np.diag([1., -1., -1., 1.])).T.tolist()))
                        poses.append(pose)
                    for _ in range(8):
                        app.update()
                    omni.usd.get_context().reset_renderer_accumulation()
                    for _ in range(2):
                        rep.orchestrator.step(rt_subframes=32, pause_timeline=True, delta_time=0.)
                    views, truths, label_maps = [], {}, {}
                    for side, group, pose in zip(('left', 'right'), groups, poses):
                        rgb = np.asarray(group[0].get_data()).copy()[..., :3]
                        if rgb.shape != (size[1], size[0], 3) or rgb.std() < 1:
                            raise RuntimeError('Invalid RGB for ' + ident)
                        file = folder / (side + '.png')
                        Image.fromarray(rgb).save(file)
                        truths[side + '_z_m'] = np.asarray(group[1].get_data()).copy().squeeze().astype(np.float32)
                        semantics = group[2].get_data()
                        truths[side + '_part_ids'] = np.asarray(semantics['data']).copy().squeeze()
                        label_maps[side] = semantics['info']['idToLabels']
                        views.append({'id': side, 'image': file.name, 'image_size_wh': size, 'intrinsics': k,
                                      'image_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                                      'extrinsics_w2c': np.linalg.inv(pose) @ poses[0],
                                      'distortion_coefficients': [0] * 5})
                    write_json(folder / 'input.json', {'schema': 'quadweld-stereo-v1', 'synchronized': True,
                               'views': views, 'baseline_m': baseline, 'length_unit': 'm',
                               'world_from_left_optical': poses[0], 'calibration_status': 'simulation pinhole stereo',
                               'laser_enabled': False})
                    np.savez_compressed(folder / 'evaluation/truth.npz', **truths)
                    write_json(folder / 'evaluation/label_maps.json', label_maps)
                    records.append({'id': ident, 'station': station, 'view': index,
                                    'camera_pose_source': 'free static stereo rig', 'nominal_range_m': DISTANCE})
                    write_json(args.run_dir / 'views.json', {'views': records})
                    print('CAPTURED', ident, flush=True)
        finally:
            for product, group in zip(products, groups):
                for item in group:
                    item.detach(product)
                product.destroy()
        source = SCENE.with_name('industrial_welding.usdc')
        write_json(args.run_dir / 'evaluation/reference.json', {'regions': truth_records,
                   'source_usdc_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'scope': 'CAD contacts and rendered labels are evaluation-only; capture placement uses scene geometry'})
    finally:
        app.close()


def label_masks(run, view):
    reference = json.loads((run / 'evaluation/reference.json').read_text())['regions'][view['station']]
    folder = run / 'captures' / view['id'] / 'evaluation'
    maps = json.loads((folder / 'label_maps.json').read_text())['left']
    with np.load(folder / 'truth.npz') as data:
        ids = data['left_part_ids']
        masks = []
        for name in reference['expected_parts']:
            found = [int(key) for key, value in maps.items() if name.lower() in str(value).lower()]
            masks.append(np.isin(ids, found))
    return np.asarray(masks), np.ones(2)


def predict(args):
    import cv2
    from weld_vision import bead_seam, stereo_seam
    from weld_vision.stereo_io import load_pair
    cv2.setNumThreads(1)
    output = args.run_dir / ('labels_control' if args.mask_source == 'labels' else 'results')
    records = []
    for view in json.loads((args.run_dir / 'views.json').read_text())['views']:
        if args.stations and view['station'] not in args.stations:
            continue
        record = dict(view, mask_source=args.mask_source)
        if args.mask_source == 'labels':
            masks, scores = label_masks(args.run_dir, view)
        else:
            with np.load(args.run_dir / 'sam' / (view['id'] + '.npz')) as data:
                masks, scores = data['masks'], data['scores']
        start = time.monotonic()
        pair = ((0, 1) if args.mask_source == 'labels' else bead_seam.pick_plate_pair(masks, scores))
        record.update(candidate_count=len(masks), selected_pair=pair)
        if pair is None:
            record.update(root_status='no_plate_pair', toe_status='not_run', seconds=time.monotonic() - start)
        else:
            path = args.run_dir / 'captures' / view['id'] / 'input.json'
            left, right, selected, k, baseline, rotation = load_pair(path, [masks[i] for i in pair])
            world = np.array(json.loads(path.read_text())['world_from_left_optical'])
            world[:3, :3] = world[:3, :3] @ rotation
            result = bead_seam.locate_bead_toes(left, right, k, baseline, *selected, depth_range=(.20, 1.50))
            record['initial_status'] = result['status']
            record['nominal_range_prior_used'] = result['status'] == 'ambiguous_stereo_match'
            if record['nominal_range_prior_used']:
                result = bead_seam.locate_bead_toes(left, right, k, baseline, *selected,
                                                  depth_range=(.20, 1.50), expected_distance=DISTANCE)
            seam = result['seam']
            record.update(root_status=seam['status'], toe_status=result['status'], checks=seam['checks'],
                          seconds=time.monotonic() - start, toes=[])
            if seam['status'] == 'ok':
                record.update(root_world_m=transform(seam['joint_ends_xyz'], world),
                              root_camera_m=seam['joint_ends_xyz'], length_m=seam['length_m'])
                for toe in result['toes']:
                    segments = [transform(curve, world) for curve in bead_seam.polyline_segments(toe)]
                    length = sum(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum() for curve in segments)
                    record['toes'].append({'valid_points': int(toe['valid'].sum()), 'valid_length_m': float(length),
                                           'segments_world_m': segments})
            image = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
            for mask, color in zip(selected, ((0, 165, 255), (255, 220, 0))):
                image[mask] = (.70 * image[mask] + .30 * np.asarray(color)).astype(np.uint8)
            (output / 'masks').mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output / 'masks' / (view['id'] + '.jpg')), image)
        records.append(record)
        write_json(output / 'predictions.json', {'views': records})
        print('PREDICTED', view['id'], record['root_status'], 'toes=' + record['toe_status'],
              f"{record['seconds']:.1f}s", flush=True)
    fused = {}
    for station in dict.fromkeys(view['station'] for view in records):
        usable = [view for view in records if view['station'] == station and view['root_status'] == 'ok']
        fused[station] = {'status': 'insufficient_views', 'available_views': len(usable)}
        if len(usable) >= 2:
            result = stereo_seam.fuse_seams([view['root_world_m'] for view in usable], tolerance=.005)
            fused[station] = {'status': 'ok', 'root_world_m': result['ends'], 'length_m': result['length_m'],
                              'views_used': [view['id'] for view, used in zip(usable, result['view_used']) if used],
                              'view_deviation_m': result['view_deviation_m']}
            points = np.linspace(*result['ends'], max(2, int(np.ceil(result['length_m'] / .001)) + 1))
            (output / station).mkdir(parents=True, exist_ok=True)
            np.savetxt(output / station / 'root_world.csv', points, delimiter=',', header='x_m,y_m,z_m', comments='')
    write_json(output / 'predictions.json', {'views': records, 'fused': fused,
               'scope': 'automatic SAM candidate selection' if args.mask_source == 'sam' else 'evaluation-label masks ablation'})


def line_metrics(ends, reference):
    from tests.benchmarks.seam_metrics import error_stats, segment_distances
    ends, reference = np.asarray(ends, float), np.asarray(reference, float)
    samples = np.linspace(*reference, max(2, int(np.ceil(np.linalg.norm(reference[1] - reference[0]) / .001)) + 1))
    direction = ends[1] - ends[0]
    direction /= np.linalg.norm(direction)
    offset = samples - ends[0]
    errors = np.linalg.norm(offset - np.outer(offset @ direction, direction), axis=1)
    finite_errors = segment_distances(samples, [ends])[0]
    actual_direction = reference[1] - reference[0]
    actual_direction /= np.linalg.norm(actual_direction)
    return {'line_error': error_stats(errors), 'coverage_at_5mm': float(np.mean(finite_errors <= .005)),
            'coverage_at_10mm': float(np.mean(finite_errors <= .010)),
            'direction_error_deg': float(np.degrees(np.arccos(np.clip(abs(direction @ actual_direction), 0, 1))))}


def evaluate(args):
    import cv2
    from weld_vision.stereo_io import read_rgb
    from weld_vision.stereo_seam import project
    output = args.run_dir / ('labels_control' if args.mask_source == 'labels' else 'results')
    predictions = json.loads((output / 'predictions.json').read_text())
    refs = json.loads((args.run_dir / 'evaluation/reference.json').read_text())
    source = SCENE.with_name('industrial_welding.usdc')
    if refs['source_usdc_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('Scene changed since capture')
    (output / 'images').mkdir(parents=True, exist_ok=True)
    for record in predictions['views']:
        reference = refs['regions'][record['station']]['root_world_m']
        if record['root_status'] == 'ok':
            record['metrics'] = line_metrics(record['root_world_m'], reference)
        folder = args.run_dir / 'captures' / record['id']
        pair = json.loads((folder / 'input.json').read_text())
        pose = np.array(pair['world_from_left_optical'])
        panels = []
        for index, side in enumerate(('left', 'right')):
            image = cv2.cvtColor(read_rgb(folder / (side + '.png')), cv2.COLOR_RGB2BGR)
            k = np.array(pair['views'][index]['intrinsics'])
            k[:2, 2] -= .5
            camera = np.linalg.inv(pose)
            if index:
                camera = np.array(pair['views'][index]['extrinsics_w2c']) @ camera
            def draw(points, color, width):
                local = transform(points, camera)
                xy = np.rint(project(local, k)).astype(np.int32)
                cv2.polylines(image, [xy], False, color, width, cv2.LINE_AA)
            draw(reference, (0, 220, 255), 2)
            if record['root_status'] == 'ok':
                draw(record['root_world_m'], (40, 230, 40), 2)
            for toe in record.get('toes', []):
                for segment in toe['segments_world_m']:
                    draw(segment, (230, 0, 230), 2)
            panels.append(image)
        cv2.imwrite(str(output / 'images' / (record['id'] + '.jpg')), np.concatenate(panels, axis=1))
    summary = {}
    for station, fused in predictions['fused'].items():
        views = [view for view in predictions['views'] if view['station'] == station]
        row = {'root_outputs': sum(view['root_status'] == 'ok' for view in views), 'total_views': len(views),
               'toe_positive_views': sum(view['toe_status'] == 'ok' for view in views),
               'toe_positive_length_m': sum(toe['valid_length_m'] for view in views for toe in view.get('toes', [])),
               'fusion': fused}
        row['longest_false_toe_segment_m'] = max((
            float(np.linalg.norm(np.diff(np.asarray(segment), axis=0), axis=1).sum())
            for view in views for toe in view.get('toes', []) for segment in toe['segments_world_m']), default=0.)
        if fused['status'] == 'ok':
            reference = np.asarray(refs['regions'][station]['root_world_m'])
            row['metrics'] = line_metrics(fused['root_world_m'], reference)
            row['metrics']['reference_length_m'] = float(np.linalg.norm(reference[1] - reference[0]))
            if refs['regions'][station]['reference_scope'] == 'physical contact':
                ends = np.asarray(fused['root_world_m'])
                if np.linalg.norm(ends[0] - reference[1]) < np.linalg.norm(ends[0] - reference[0]):
                    ends = ends[::-1]
                row['metrics']['endpoint_error_mm'] = (np.linalg.norm(ends - reference, axis=1) * 1000).tolist()
        summary[station] = row
    write_json(output / 'metrics.json', {'regions': summary, 'views': predictions['views'],
               'error_scope': 'world-frame CAD contact to predicted line; coverage uses finite predicted segment',
               'truth_used_for_prediction': args.mask_source == 'labels'})
    report(args, output, summary, predictions['views'])
    visual_summary(args.run_dir, output, summary, predictions['views'])
    print('BENCHMARK_REPORT', output / 'README.md', flush=True)


def report(args, output, summary, views):
    lines = ['# 其他相似区域的双目识别测试', '',
             '当前 USD 场景；每个区域 3 个静态自由双目视角，工作距离 0.65 m，90 mm 基线，640×1100 像素，绕光轴 −35°。',
             '相机布置根据预选场景区域确定；本次不检验机械臂可达性、运动或动力学。',
             '自动测试只用 RGB、标定、相机外参和 SAM3.1 文本候选；CAD 接触线与渲染标签用于独立评价。',
             '周期纹理多解时可使用预先设定的 0.65 m 工作距离消歧，与当前主流程的距离先验方式一致。',
             '**本报告为渲染标签掩码对照，属于使用真值分割的诊断实验。**' if args.mask_source == 'labels'
             else '**本报告为自动 SAM 选板的原始结果，未用真值更改选板或筛选融合视图。**', '',
             '![测试总览](summary.jpg)', '',
             '| 区域 | 根线输出/视图 | 融合线误差均值 / 最大 mm | 两端点误差 mm | 覆盖@5mm / @10mm | 焊趾误输出视图 |',
             '|---|---:|---:|---:|---:|---:|']
    for station, row in summary.items():
        metrics = row.get('metrics')
        error = f"{metrics['line_error']['mean_mm']:.3f} / {metrics['line_error']['max_mm']:.3f}" if metrics else '未融合'
        endpoints = (' / '.join(f'{value:.1f}' for value in metrics['endpoint_error_mm'])
                     if metrics and 'endpoint_error_mm' in metrics else '局部区段 / 不评价')
        coverage = f"{metrics['coverage_at_5mm']:.1%} / {metrics['coverage_at_10mm']:.1%}" if metrics else '–'
        lines.append(f"| {station} | {row['root_outputs']}/{row['total_views']} | {error} | {endpoints} | {coverage} | {row['toe_positive_views']}/{row['total_views']} |")
    lines += ['', '根线输出表示算法状态为 ok，不等于选中了预选接缝；位置、方向与覆盖率需结合判断。',
              '线误差由整条选定 CAD 接触线到预测无限直线的距离统计；覆盖率按有限预测段计算。',
              '覆盖率不会惩罚预测段超出真实接头；轨迹范围需要另看端点误差。A04 的竖直接缝在本轮出现明显外延。',
              'B01 评价对象是预选的 1 m 局部区段，其端点不是整条纵肋的物理端点。',
              '所有区域都没有已沉积焊道。本轮为负例诊断而强制调用焊趾分支，状态为 ok 记为误输出；其有效长度见 metrics.json。',
              '当前焊趾函数假设调用方已确认存在焊道，不承担自动区分有无焊道的任务。', '',
              '## 逐视图', '', '| 视图 | 根线状态 | 焊趾状态 | 线误差均值 mm | 距离先验 | 耗时 s |',
              '|---|---|---|---:|---|---:|']
    for view in views:
        error = f"{view['metrics']['line_error']['mean_mm']:.3f}" if 'metrics' in view else '–'
        lines.append(f"| {view['id']} | {view['root_status']} | {view['toe_status']} | {error} | {view.get('nominal_range_prior_used', False)} | {view['seconds']:.1f} |")
    lines += ['', '## 图像', '', '黄色：CAD 真值；绿色：预测根线；品红：算法输出的焊趾（这些无焊道区域中属于误报）。',
              '每张图均为完整左、右目，未裁剪；分割候选选板叠加图位于 masks/。', '']
    for station in summary:
        lines += [f'### {station}', '', f'![{station}](images/{station}_view1.jpg)', '']
    lines += ['## 复跑', '', '```bash',
              f'.venv/bin/python -m tests.benchmarks.benchmark_candidate_regions capture --run-dir <new_run_dir>',
              '.venv-sam/bin/python -m model_runtime.sam_adapter <new_run_dir>/captures/*/input.json --output-dir <new_run_dir>/sam --gpu 0',
              f'.venv/bin/python -m tests.benchmarks.benchmark_candidate_regions predict --run-dir {args.run_dir}',
              f'.venv/bin/python -m tests.benchmarks.benchmark_candidate_regions evaluate --run-dir {args.run_dir}',
              '```', '', '结果均位于世界坐标系，单位米；CSV 为融合根线。理想仿真标定与外参，不构成实机精度验证。', '']
    (output / 'README.md').write_text('\n'.join(lines))


def visual_summary(run, output, summary, views):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/quadweld-candidate-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from PIL import Image
    from weld_vision.stereo_seam import project

    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    bold = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    references = json.loads((run / 'evaluation/reference.json').read_text())['regions']
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 11.7), dpi=145, facecolor='#f5f7fa')
    for ax, (station, row) in zip(axes.flat, summary.items()):
        record = next(view for view in views if view['station'] == station and view['view'] == 1)
        folder = run / 'captures' / record['id']
        pair = json.loads((folder / 'input.json').read_text())
        k = np.asarray(pair['views'][0]['intrinsics'], float)
        k[:2, 2] -= .5
        inverse = np.linalg.inv(np.asarray(pair['world_from_left_optical']))
        ax.imshow(Image.open(folder / 'left.png'))
        def draw(points, **style):
            xy = project(transform(points, inverse), k)
            ax.plot(xy[:, 0], xy[:, 1], **style)
        draw(references[station]['root_world_m'], color='#ffcc24', lw=4, linestyle='--')
        if row['fusion']['status'] == 'ok':
            draw(row['fusion']['root_world_m'], color='#19db53', lw=1.9)
        for toe in record.get('toes', []):
            for segment in toe['segments_world_m']:
                draw(segment, color='#ff39d1', lw=2.1)
        ax.set_xlim(0, 639)
        ax.set_ylim(850, 250)
        ax.set_axis_off()
        error = row.get('metrics', {}).get('line_error', {}).get('mean_mm')
        label = f'{station}  根线均值 {error:.3f} mm' if error is not None else f'{station}  未融合'
        ax.set_title(label, fontproperties=bold, fontsize=16, color='#14304a', loc='left', pad=11)
        coverage = row.get('metrics', {}).get('coverage_at_5mm', 0)
        ax.text(0, -.055, f"覆盖@5mm {coverage:.0%}   焊趾误输出 {row['toe_positive_views']}/{row['total_views']} 视图",
                transform=ax.transAxes, fontproperties=font, fontsize=11, color='#37485d')
        endpoints = row.get('metrics', {}).get('endpoint_error_mm')
        note = ('物理端点最大误差 ' + f'{max(endpoints):.1f} mm' if endpoints is not None
                else '仅评价预选的 1 m 区段，不评价物理端点')
        ax.text(0, -.11, note, transform=ax.transAxes, fontproperties=font, fontsize=11,
                color='#b33728' if endpoints and max(endpoints) > 20 else '#37485d')
    fig.suptitle('其他相似区域 · 双目识别测试结果', fontproperties=bold, fontsize=24,
                 color='#14304a', x=.03, ha='left', y=.99)
    fig.text(.03, .945, '黄色虚线：CAD 真值   绿色：融合根线   品红：无焊道条件下的焊趾误输出（中央视角）',
             fontproperties=font, fontsize=13, color='#37485d')
    root_count = sum(row['root_outputs'] for row in summary.values())
    total = sum(row['total_views'] for row in summary.values())
    fig.text(.03, .017, f'{root_count}/{total} 组输出根线。覆盖率不惩罚轨迹外延，需另看端点误差。自由相机、理想仿真标定。',
             fontproperties=font, fontsize=12, color='#37485d')
    fig.subplots_adjust(left=.03, right=.985, top=.89, bottom=.09, hspace=.33, wspace=.045)
    fig.savefig(output / 'summary.jpg', dpi=145, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('capture', 'predict', 'evaluate'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mask-source', choices=('sam', 'labels'), default='sam')
    parser.add_argument('--stations', nargs='+', choices=REGIONS)
    args = parser.parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    {'capture': capture, 'predict': predict, 'evaluate': evaluate}[args.stage](args)


if __name__ == '__main__':
    main()
