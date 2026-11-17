"""Fresh overview -> measured coarse geometry -> three close stereo observations.

CAD is accessed only by capture (initial aiming, independent reference export)
and report/evaluation. Planning and prediction never read CAD or render labels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from tests.benchmarks.benchmark_candidate_regions import ROOT, SCENE, serial, transform, write_json

REGIONS = ('A01', 'A03', 'A23', 'A06', 'A04', 'B01', 'A25_back', 'A26_back')


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference(stage, region):
    from tests.benchmarks.benchmark_candidate_regions import region_geometry
    if not region.endswith('_back'):
        line, fixture, parts = region_geometry(stage, region)
        return {'root_world_m': line, 'fixture': fixture, 'expected_parts': parts,
                'mode': 'root', 'reference_scope': 'selected 1 m window' if region == 'B01' else 'physical contact'}
    from tests.benchmarks.benchmark_bead_toes import truth_toes
    from pxr import UsdGeom
    station = region.split('_')[0]
    toes = truth_toes(stage, station, 'back')
    fixture = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath('/World/Fixtures/' + station))).T
    return {'root_world_m': toes[2][[0, -1]], 'fixture': fixture,
            'toes_world_m': toes[:2], 'expected_parts': [station + '_base', station + '_web'],
            'mode': 'bead', 'reference_scope': 'physical contact'}


def roll_pose(pose, degrees=-35.):
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    pose[:3, :3] = pose[:3, :3] @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return pose


def initial_pose(ref, region):
    from weld_scene.overview_planning import look_at
    line, fixture = np.asarray(ref['root_world_m']), np.asarray(ref['fixture'])
    tangent = line[1] - line[0]
    tangent /= np.linalg.norm(tangent)
    if region == 'A04':
        outward = fixture[:3, :3] @ np.array([1., -1., .18])
        outward /= np.linalg.norm(outward)
    else:
        front = fixture[:3, :3] @ np.array([0., 1. if region.endswith('_back') else -1., 0.])
        up = np.cross(front, tangent)
        if up[2] < 0:
            up = -up
        up /= np.linalg.norm(up)
        outward = np.cos(np.radians(35)) * front + np.sin(np.radians(35)) * up
    return roll_pose(look_at(line.mean(0) + 2. * outward, line.mean(0)))


def capture(args, phase):
    """Render fresh synchronized pairs; sensor truth is exported separately."""
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting', 'active_gpu': args.gpu,
                         'multi_gpu': False, 'open_usd': str(SCENE), 'width': 640, 'height': 1100,
                         'sync_loads': True,
                         'extra_args': ['--/exts/omni.kit.async_engine/keep_loop_running=false']})
    exit_code = 0
    try:
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdSemantics
        from weld_scene.optics import load_calibration
        from weld_scene.usd import set_intrinsics
        omni.timeline.get_timeline_interface().stop()
        for _ in range(60):
            app.update()
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        stage.GetPrimAtPath('/World/Robot/e05l_welding_torch/hj90_active/laser').GetAttribute('inputs:intensity').Set(0.)
        for station in dict.fromkeys(r.split('_')[0] for r in REGIONS):
            for prim in stage.GetPrimAtPath('/World/Fixtures/' + station).GetChildren():
                if prim.IsA(UsdGeom.Mesh):
                    UsdSemantics.LabelsAPI.Apply(prim, 'class').CreateLabelsAttr().Set([prim.GetName()])
        if phase == 'overview':
            refs = {region: reference(stage, region) for region in REGIONS}
            plan = [{'id': region + '_overview', 'station': region, 'mode': refs[region]['mode'],
                     'world_from_optical_midpoint': initial_pose(refs[region], region), 'nominal_range_m': 2.,
                     'pose_source': 'known scene region for initial aiming only'} for region in REGIONS]
            write_json(args.run_dir / 'evaluation/reference.json', {'regions': refs,
                       'source_usdc_sha256': digest(SCENE.with_name('industrial_welding.usdc')),
                       'scope': 'CAD reference and render labels are evaluation only; initial aiming uses scene region'})
        elif phase == 'coarse-retry':
            initial = read_json(args.run_dir / 'overview/results/predictions.json')['views']
            plan = []
            for item in initial:
                if item['root_status'] == 'ok':
                    continue
                # Forward along the known initial axis/stand-off, without CAD.
                pose = np.asarray(item['world_from_optical_midpoint']).copy()
                pose[:3, 3] += pose[:3, 2] * 1.
                plan.append({**{k: item[k] for k in ('station', 'mode')},
                             'id': item['station'] + '_coarse_retry', 'nominal_range_m': 1.,
                             'world_from_optical_midpoint': pose,
                             'pose_source': 'move 1 m forward along failed initial optical axis',
                             'retry_reason': item['root_status'], 'previous_id': item['id']})
            write_json(args.run_dir / 'coarse_retry_plan.json', {'views': plan, 'uses_cad': False})
        else:
            plan = read_json(args.run_dir / 'close_plan.json')['views']
        calibration = load_calibration()
        size = tuple(calibration['image_size_wh'])
        k, baseline = np.array(calibration['intrinsics']), calibration['baseline_m']
        paths = ['/World/Cameras/staged_stereo_' + side for side in ('left', 'right')]
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
        views_path = args.run_dir / phase / 'views.json'
        records = read_json(views_path)['views'] if views_path.exists() else []
        try:
            for item in plan:
                start = time.monotonic()
                folder = args.run_dir / phase / 'captures' / item['id']
                if (folder / 'input.json').exists():
                    saved = read_json(folder / 'input.json')
                    old = np.asarray(saved['world_from_left_optical'])
                    expected = np.asarray(item['world_from_optical_midpoint']).copy()
                    expected[:3, 3] -= baseline/2 * expected[:3, 0]
                    if not np.allclose(old, expected, atol=1e-9):
                        raise FileExistsError('Existing capture pose differs from plan: ' + str(folder))
                    validate_input(folder / 'input.json')
                    print('RESUMED_VALID_CAPTURE', item['id'], flush=True)
                    continue
                if folder.exists():
                    failed = args.run_dir / phase / 'failed_captures' / (item['id'] + '_attempt1')
                    failed.parent.mkdir(parents=True, exist_ok=True)
                    folder.rename(failed)
                (folder / 'evaluation').mkdir(parents=True, exist_ok=True)
                midpoint = np.asarray(item['world_from_optical_midpoint'])
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
                views, truths, labels = [], {}, {}
                for side, group, pose in zip(('left', 'right'), groups, poses):
                    rgb = np.asarray(group[0].get_data()).copy()[..., :3]
                    if rgb.shape != (size[1], size[0], 3) or rgb.std() < 1:
                        raise RuntimeError('Invalid RGB: ' + item['id'])
                    file = folder / (side + '.png')
                    Image.fromarray(rgb).save(file)
                    truths[side + '_z_m'] = np.asarray(group[1].get_data()).copy().squeeze().astype(np.float32)
                    semantics = group[2].get_data()
                    truths[side + '_part_ids'] = np.asarray(semantics['data']).copy().squeeze()
                    labels[side] = semantics['info']['idToLabels']
                    views.append({'id': side, 'image': file.name, 'image_size_wh': size, 'intrinsics': k,
                                  'image_sha256': digest(file), 'extrinsics_w2c': np.linalg.inv(pose) @ poses[0],
                                  'distortion_coefficients': [0] * 5})
                write_json(folder / 'input.json', {'schema': 'quadweld-stereo-v1', 'synchronized': True,
                           'views': views, 'baseline_m': baseline, 'length_unit': 'm',
                           'world_from_left_optical': poses[0], 'laser_enabled': False,
                           'calibration_status': 'simulation pinhole, already rectified',
                           'capture_frame': 'same static scene state, timeline stopped'})
                np.savez_compressed(folder / 'evaluation/truth.npz', **truths)
                write_json(folder / 'evaluation/label_maps.json', labels)
                records.append({**item, 'capture_seconds': time.monotonic() - start})
                write_json(args.run_dir / phase / 'views.json', {'views': records})
                print('CAPTURED', phase, item['id'], flush=True)
        finally:
            for product, group in zip(products, groups):
                for annotator in group:
                    annotator.detach(product)
                product.destroy()
    except Exception as error:
        import traceback
        exit_code = 1
        traceback.print_exc()
        write_json(args.run_dir / phase / 'capture_failure.json',
                   {'error':repr(error),'traceback':traceback.format_exc(),'view':item.get('id') if 'item' in locals() else None})
    finally:
        app.close(exit_code=exit_code)


def audit_samples(left, right, masks, seam, k, baseline, exclude=None):
    """Recover the solver's actual deterministic pixels, without fitting again."""
    import cv2
    from weld_vision import stereo_seam as ss
    images = ss.StereoImages(left, right)
    cleaned = [ss.clean_mask(m) for m in masks]
    reach = [cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5) for m in cleaned]
    arrays, reports = {}, []
    for i, fit in enumerate(seam['fits']):
        region = cleaned[i] & (reach[1-i] <= seam['neighbourhood_px'])
        if exclude is not None:
            region &= ~exclude
        margin = max(2, round(left.shape[1] / 320))
        domain = cv2.dilate(region.astype(np.uint8), np.ones((2*margin+1, 2*margin+1), np.uint8)).astype(bool)
        sampled = ss._Region(domain, 30000)
        assert len(sampled.x) == fit['pixels'], (i, len(sampled.x), fit['pixels'])
        xy = np.c_[sampled.x, sampled.y]
        disparity = np.c_[xy, np.ones(len(xy))] @ fit['params']
        right_xy = xy - np.c_[disparity, np.zeros(len(xy))]
        residual = (ss._sample(images.levels[-1][1], right_xy[:, 0], right_xy[:, 1])
                    - ss._sample(images.levels[-1][0], xy[:, 0], xy[:, 1]))
        finite = np.isfinite(residual)
        inlier = finite & (abs(residual) < .5)
        xyz = ss.backproject(xy, disparity, k, baseline)
        n, h = seam['planes'][i]
        np.testing.assert_allclose(xyz @ n, h, atol=1e-8)
        arrays.update({f'plate_{i}_mask': masks[i], f'plate_{i}_fit_domain': domain,
                       f'plate_{i}_sample_xy': xy, f'plate_{i}_right_xy': right_xy,
                       f'plate_{i}_disparity_px': disparity, f'plate_{i}_xyz_from_fitted_plane_m': xyz,
                       f'plate_{i}_photometric_residual': residual, f'plate_{i}_inlier': inlier})
        reports.append({'plate': i, 'sample_count': len(xy), 'finite_fraction': finite.mean(),
                        'final_inlier_fraction_all_samples': inlier.mean(),
                        'solver_inlier_fraction_finite_samples': fit['inlier_fraction'],
                        'normal_camera': n, 'offset_camera_m': h, 'disparity_plane_abc': fit['params'],
                        'residual_scale': fit['residual_scale'], 'hypotheses': fit['hypotheses']})
    if exclude is not None:
        arrays['bead_exclusion_mask'] = exclude
    return arrays, reports


def validate_input(path):
    from weld_vision.stereo_io import read_rgb
    spec = read_json(path)
    assert spec['synchronized'] and spec['length_unit'] == 'm'
    assert [v['id'] for v in spec['views']] == ['left', 'right']
    for v in spec['views']:
        image = path.parent / v['image']
        assert digest(image) == v['image_sha256']
        assert tuple(read_rgb(image).shape[1::-1]) == tuple(v['image_size_wh'])
    relative = np.asarray(spec['views'][1]['extrinsics_w2c']) @ np.linalg.inv(spec['views'][0]['extrinsics_w2c'])
    assert np.allclose(relative[:3, :3], np.eye(3), atol=1e-7)
    assert np.allclose(relative[:3, 3], [-spec['baseline_m'], 0, 0], atol=1e-7)
    pose = np.asarray(spec['world_from_left_optical'])
    assert np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(pose[:3, :3]), 1.)
    return {'status': 'ok', 'synchronized_static': True, 'rgb_hashes_verified': True,
            'calibration_and_pose_verified': True, 'already_rectified': True,
            'baseline_m': spec['baseline_m'], 'input_sha256': digest(path)}


def plan_close(run, observations):
    """Only measured overview endpoints/normals enter this planner."""
    from weld_scene.overview_planning import look_at
    plans, skipped = [], []
    for observation in observations:
        if observation['root_status'] != 'ok':
            skipped.append({'station': observation['station'], 'status': observation['root_status']})
            continue
        line = np.asarray(observation['root_world_m'])
        normals = np.asarray(observation['face_normals_world'])
        direction = line[1] - line[0]
        direction /= np.linalg.norm(direction)
        outward = normals.sum(0)
        outward -= direction * (outward @ direction)
        outward /= np.linalg.norm(outward)
        for index, (fraction, angle) in enumerate(zip((.25, .50, .75), (-10., 0., 10.))):
            target = line[0] + fraction * (line[1] - line[0])
            # For a vertical joint, a downward eye offset can put the rig below
            # the lower plate. Use an upward offset, based on world gravity and
            # measured direction only, without clipping endpoints to CAD.
            guarded = abs(direction[2]) > .7 and angle * direction[2] < 0
            if guarded:
                angle = -angle
            ray = np.cos(np.radians(angle)) * outward + np.sin(np.radians(angle)) * direction
            pose = roll_pose(look_at(target + .65 * ray, target))
            plans.append({'id': f"{observation['station']}_view{index}", 'station': observation['station'],
                          'mode': observation['mode'], 'view_index': index, 'fraction': fraction,
                          'along_axis_tilt_deg': angle, 'nominal_range_m': .65, 'target_world_m': target,
                          'vertical_low_camera_guard': guarded,
                          'world_from_optical_midpoint': pose,
                          'pose_source': 'measured overview root + measured visible face normals',
                          'coarse_id': observation['id']})
    write_json(run / 'close_plan.json', {'views': plans, 'overview': {v['station']: v for v in observations},
               'skipped': skipped, 'coarse_predictions_sha256': digest(run / 'coarse_selection.json'),
               'uses_cad': False, 'distance_m': .65, 'roll_deg': -35.})


def measure(args, phase):
    import cv2
    from weld_vision import bead_seam, stereo_seam as ss
    from weld_vision.stereo_io import describe, load_pair
    from weld_vision.sweep import pick_scan_pair
    cv2.setNumThreads(1)
    directory = args.run_dir / phase
    output = directory / 'results'
    records = []
    coarse = read_json(args.run_dir / 'close_plan.json')['overview'] if phase == 'close' else {}
    for view in read_json(directory / 'views.json')['views']:
        start = time.monotonic()
        folder = output / view['id']
        folder.mkdir(parents=True, exist_ok=True)
        path = directory / 'captures' / view['id'] / 'input.json'
        audit = validate_input(path)
        with np.load(directory / 'sam' / (view['id'] + '.npz')) as data:
            raw_masks, scores = data['masks'], data['scores']
            prompt_ids, prompt_texts = data['prompts'], data['prompt_texts']
        left, right, masks, k, baseline, rotation = load_pair(path, raw_masks)
        world = np.asarray(read_json(path)['world_from_left_optical'])
        world[:3, :3] = world[:3, :3] @ rotation
        masks = np.asarray(masks)
        pair = None
        selection = 'automatic adjacent SAM pair'
        if phase == 'close':
            pair = pick_scan_pair(masks, scores, k, world, coarse[view['station']])
            if pair is not None:
                selection = 'SAM pair matched to measured coarse planes'
        if pair is None:
            pair = bead_seam.pick_plate_pair(masks, scores)
        record = {**view, 'phase': phase, 'input_audit': audit, 'candidate_count': len(masks), 'selected_pair': pair,
                  'selection_method': selection, 'world_from_rectified_left': world,
                  'candidates': [{'index': i, 'score': float(s), 'prompt': str(prompt_texts[p]),
                                  'area_pixels': int(m.sum())} for i, (m, s, p) in enumerate(zip(masks, scores, prompt_ids))]}
        if pair is None:
            record.update(status='no_plate_pair', root_status='no_plate_pair', seconds=time.monotonic()-start)
        else:
            selected = [masks[i] for i in pair]
            np.savez_compressed(folder / 'selected_masks.npz', masks=selected, indices=pair, scores=scores[list(pair)])
            limits = ((1.4, 2.7) if phase == 'overview' else (.3, 1.7)) if phase != 'close' else (.20, 1.50)
            options = {'depth_range': limits}
            bead_mode = phase == 'close' and view['mode'] == 'bead'
            def locate(extra):
                if bead_mode:
                    return bead_seam.locate_bead_toes(left, right, k, baseline, *selected, **options, **extra)
                seam = ss.locate_seam(left, right, k, baseline, *selected, **options, **extra)
                return {'status': seam['status'], 'seam': seam, 'toes': []}
            result = locate({})
            record['initial_status'] = result['status']
            record['range_prior_attempted'] = result['status'] == 'ambiguous_stereo_match'
            if record['range_prior_attempted']:
                result = locate({'expected_distance': view['nominal_range_m']})
            seam = result['seam']
            record.update(status=result['status'], root_status=seam['status'], checks=seam['checks'],
                          seam=describe(seam), bead_refit_used='bead_mask' in result, toes=[])
            if 'fits' in seam:
                arrays, planes = audit_samples(left, right, selected, seam, k, baseline, result.get('bead_mask'))
                np.savez_compressed(folder / 'fit_samples.npz', **arrays)
                record['plane_audit'] = planes
                for plane in planes:
                    normal = world[:3, :3] @ plane['normal_camera']
                    plane['normal_world'] = normal
                    plane['offset_world_m'] = plane['offset_camera_m'] + normal @ world[:3, 3]
            if seam['status'] == 'ok':
                root = transform(seam['joint_ends_xyz'], world)
                record.update(root_world_m=root, face_normals_world=np.asarray(seam['face_normals']) @ world[:3, :3].T)
                np.savetxt(folder / 'root_world.csv', transform(seam['xyz'], world), delimiter=',',
                           header='x_m,y_m,z_m', comments='')
                if phase == 'close':
                    observed = np.asarray(coarse[view['station']]['root_world_m'])
                    direction = observed[1]-observed[0]
                    direction /= np.linalg.norm(direction)
                    offset = root-observed[0]
                    deviation = np.linalg.norm(offset-np.outer(offset @ direction, direction), axis=1).max()
                    angle = np.degrees(np.arccos(np.clip(abs((root[1]-root[0]) @ direction)/np.linalg.norm(root[1]-root[0]),0,1)))
                    record['coarse_consistency'] = {'max_line_deviation_m': deviation, 'angle_deg': angle,
                                                    'accepted': bool(deviation <= .08 and angle <= 10.)}
                for index, toe in enumerate(result['toes']):
                    xyz = transform(toe['xyz'], world)
                    np.savez_compressed(folder / f'toe_plate{index}.npz', **toe, xyz_world_m=xyz)
                    record['toes'].append({'plate': index, 'xyz': xyz, 'valid': toe['valid'],
                                           'segments_world_m': [transform(p, world) for p in bead_seam.polyline_segments(toe)],
                                           'coverage': toe['coverage']})
            record['seconds'] = time.monotonic() - start
        records.append(record)
        write_json(folder / 'record.json', record)
        write_json(output / 'predictions.json', {'views': records, 'phase': phase})
        print('MEASURED', phase, view['id'], record['root_status'], record['status'],
              f"{record['seconds']:.1f}s", flush=True)
    if phase != 'close':
        observations = {v['station']: v for v in read_json(args.run_dir / 'overview/results/predictions.json')['views']}
        if phase == 'coarse-retry':
            observations.update({v['station']: v for v in records})
        selected = list(observations.values())
        write_json(args.run_dir / 'coarse_selection.json', {'views': selected})
        plan_close(args.run_dir, selected)
    else:
        fuse(output, records)


def fuse(output, records):
    from weld_vision import bead_seam, stereo_seam as ss
    fused = {}
    for region in REGIONS:
        usable = [v for v in records if v['station'] == region and v['root_status'] == 'ok'
                  and v.get('coarse_consistency', {}).get('accepted', False)]
        record = {'status': 'insufficient_views', 'available_views': len(usable)}
        if len(usable) >= 2:
            axis = ss.fuse_seams([v['root_world_m'] for v in usable], tolerance=.005)
            record = {'status': 'ok', 'root_world_m': axis['ends'], 'length_m': axis['length_m'],
                      'views_used': [v['id'] for v, used in zip(usable, axis['view_used']) if used],
                      'view_deviation_m': axis['view_deviation_m'], 'toes': {}}
            folder = output / region
            folder.mkdir(exist_ok=True)
            np.savetxt(folder / 'root_world.csv', np.linspace(*axis['ends'], max(2,int(np.ceil(axis['length_m']/.001))+1)),
                       delimiter=',', header='x_m,y_m,z_m', comments='')
            toe_views = [{'root': v['root_world_m'], 'normals': v['face_normals_world'],
                          'toes': [{'xyz': t['xyz'], 'valid': t['valid']} for t in v['toes']]}
                         for v in usable if v['status'] == 'ok' and len(v['toes']) == 2]
            if len(toe_views) >= 2:
                result = bead_seam.fuse_toes(toe_views, min_views=2)
                base = int(np.argmax(abs(np.asarray(toe_views[0]['normals'])[:, 2])))
                for name, index in (('base', base), ('web', 1-base)):
                    toe = result['toes'][index]
                    record['toes'][name] = {'segments_world_m': bead_seam.polyline_segments(toe, min_points=2),
                                            'valid_samples': int(toe['valid'].sum()),
                                            'total_samples': len(toe['valid']), 'minimum_views': 2}
                    np.savez_compressed(folder / (name + '_toe.npz'), **toe)
                    xyz = np.where(toe['valid'][:, None], toe['xyz'], np.nan)
                    np.savetxt(folder / (name + '_toe.csv'),
                               np.column_stack((toe['s_m'], xyz, toe['valid'], toe['views'], toe['spread_m'])),
                               delimiter=',', header='s_m,x_m,y_m,z_m,valid,views,max_deviation_m', comments='')
                record['toe_status'] = 'ok' if all(t['valid_samples']>=2 for t in record['toes'].values()) else 'partial'
                record['toe_root_world_m'] = result['root']['ends']
        fused[region] = record
    write_json(output / 'fusion.json', fused)


def plan_bead_retry(run):
    """Repeat the two bead regions with the existing main-pipeline view law."""
    from weld_scene.sweep_planning import inspection_pose
    child = run / 'bead-retry'
    parent = read_json(run / 'close_plan.json')
    observations = {k:v for k,v in parent['overview'].items() if v['mode']=='bead' and v['root_status']=='ok'}
    views = []
    for region, observation in observations.items():
        ends = np.asarray(observation['root_world_m'])
        for index, fraction in enumerate((.25,.5,.75)):
            target = ends[0]+fraction*(ends[1]-ends[0])
            views.append({'id':f'{region}_retry_view{index}','station':region,'mode':'bead',
                          'view_index':index,'fraction':fraction,'along_axis_tilt_deg':0.,'nominal_range_m':.65,
                          'target_world_m':target,'world_from_optical_midpoint':inspection_pose(
                              target,observation['face_normals_world'],.65,elevation=35.,roll=-35.),
                          'pose_source':'existing inspection_pose: measured normals, 35 deg elevation, -35 deg roll',
                          'elevation_deg':35.,'roll_deg':-35.,'coarse_id':observation['id']})
    write_json(child / 'coarse_selection.json', {'views':list(observations.values())})
    write_json(child / 'close_plan.json', {'views':views,'overview':observations,'skipped':[],
               'coarse_predictions_sha256':digest(child / 'coarse_selection.json'),'uses_cad':False,
               'reason':'initial angle-bisector views had low base-toe coverage; repeat the existing 35 degree inspection law',
               'distance_m':.65,'roll_deg':-35.})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('capture-overview', 'measure-overview', 'capture-coarse-retry',
                                         'measure-coarse-retry', 'capture-close', 'measure-close', 'plan-bead-retry', 'report'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage.startswith('capture-'):
        capture(args, args.stage.split('-', 1)[1])
    elif args.stage.startswith('measure-'):
        measure(args, args.stage.split('-', 1)[1])
    elif args.stage == 'plan-bead-retry':
        plan_bead_retry(args.run_dir)
    else:
        from tests.benchmarks.report_staged_regions import report
        report(args.run_dir)


if __name__ == '__main__':
    main()
