"""Stereo pairs of the first-layer fillet beads (A25/A26), with separate evaluation labels.

By default the rig is placed from the fixture frame. With --pose-plan both
eyes are rigidly mounted on the arm and the base and joint poses come from
the scan planner.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# name: (target x along the joint, distance, elevation deg, azimuth deg, roll deg, joint side)
# Roll turns the rig about its optical axis.  With roll 0 the baseline is parallel to the joint:
# ripples, machining marks and the plate edges then give no unique stereo match (kept as a negative case).
VIEWS = {
    'r90_centre':  (0.00, .60, 35,   0,  90, 'front'),
    'r90_left':    (-.25, .60, 35,   0,  90, 'front'),
    'r90_right':   (0.25, .60, 35,   0,  90, 'front'),
    'r90_near':    (0.10, .45, 45,   0,  90, 'front'),
    'r90_high':    (-.10, .60, 55,   0,  90, 'front'),
    'r90_low':     (0.00, .60, 22,   0,  90, 'front'),
    'r90_back':    (0.00, .60, 35,   0,  90, 'back'),
    'r35_centre':  (0.00, .60, 35,   0,  35, 'front'),
    'r35_left':    (-.30, .60, 35,   0,  35, 'front'),
    'r-35_right':  (0.30, .60, 35,   0, -35, 'front'),
    'r35_near':    (0.10, .45, 45,   0,  35, 'front'),
    'r35_back':    (0.00, .60, 35,   0,  35, 'back'),
    'r60_centre':  (0.00, .60, 40,   0,  60, 'front'),
    'oblique+':    (-.15, .60, 35,  30,   0, 'front'),
    'oblique-':    (0.15, .60, 35, -30,   0, 'front'),
    'oblique+r35': (-.15, .60, 35,  30,  35, 'front'),
    'r0_centre':   (0.00, .60, 35,   0,   0, 'front'),
    'r0_far':      (0.00, .75, 35,   0,   0, 'front'),
}
LABELS = {'weld_bead': 1, 'base_plate': 2, 'web_plate': 3, 'spatter': 4}


def rig_pose(fixture, joint, target_x, distance, elevation, azimuth, roll, side):
    """World-from-optical pose of the rig midpoint (x right, y down, z forward) looking at the joint."""
    sign = -1. if side == 'front' else 1.
    target = np.array([target_x, sign * joint['half_web'], joint['base_top']])
    el, az = np.radians(elevation), np.radians(azimuth)
    offset = distance * np.array([np.cos(el) * np.sin(az), sign * np.cos(el) * np.cos(az), np.sin(el)])
    position = target + offset
    forward = -offset / distance
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    right, down = c * right + s * down, c * down - s * right
    local = np.eye(4)
    local[:3, :3] = np.stack((right, down, forward), axis=1)
    local[:3, 3] = position
    return fixture @ local


def joint_geometry(stage, station):
    from pxr import UsdGeom
    cache = UsdGeom.XformCache()
    prim = stage.GetPrimAtPath('/World/Fixtures/' + station)
    fixture = np.array(cache.GetLocalToWorldTransform(prim)).T
    def local_bounds(name):
        mesh = stage.GetPrimAtPath(f'/World/Fixtures/{station}/{station}_{name}')
        matrix = np.linalg.inv(fixture) @ np.array(cache.GetLocalToWorldTransform(mesh)).T
        points = np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
        points = np.c_[points, np.ones(len(points))] @ matrix.T
        return points[:, :3].min(0), points[:, :3].max(0)
    web = local_bounds('web')
    base = local_bounds('base')
    return fixture, {'half_web': float(web[1][1]), 'base_top': float(base[1][2]),
                     'x_range': [float(web[0][0]), float(web[1][0])]}


def inspection_views(stage, stations, names):
    for station in stations:
        fixture, joint = joint_geometry(stage, station)
        for name in names:
            x, distance, elevation, azimuth, roll, side = VIEWS[name]
            ident = f'{station}_{name}'.replace('+', 'p').replace('-', 'm')
            pose = rig_pose(fixture, joint, x, distance, elevation, azimuth, roll, side)
            yield {'id': ident, 'station': station, 'view': name, 'joint_side': side,
                   'target_x_m': x, 'distance_m': distance, 'elevation_deg': elevation,
                   'azimuth_deg': azimuth, 'roll_deg': roll, 'world_from_optical_midpoint': pose.tolist()}


def mounted_cameras(stage, view, k, size):
    from pxr import Gf, UsdGeom
    from weld_scene.arm_kinematics import ArmKinematics
    from weld_scene.overview_planning import world_matrix
    from weld_scene.usd import RIG_PATH, set_intrinsics

    cache = UsdGeom.XformCache()
    root = stage.GetPrimAtPath('/World/Robot')
    base = world_matrix(cache, stage.GetPrimAtPath('/World/Robot/base_link'))
    world = np.asarray(view['base_world']) @ np.linalg.inv(base) @ world_matrix(cache, root)
    local = np.linalg.inv(world_matrix(cache, root.GetParent())) @ world
    UsdGeom.Xformable(root).MakeMatrixXform().Set(Gf.Matrix4d(local.T.tolist()))
    arm = ArmKinematics(stage)
    arm.apply(view['arm_q_rad'])
    np.testing.assert_allclose(arm.camera(view['arm_q_rad']), view['world_from_optical_midpoint'], atol=2e-5)
    cache = UsdGeom.XformCache()
    poses, paths = [], []
    for eye in ('left', 'right'):
        path = RIG_PATH + '/' + eye
        camera = UsdGeom.Camera.Get(stage, path)
        set_intrinsics(camera, k, size)
        poses.append(world_matrix(cache, camera.GetPrim()) @ np.diag([1., -1., -1., 1.]))
        paths.append(path)
    return poses, paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--stations', nargs='+', default=['A25', 'A26'])
    p.add_argument('--views', nargs='+', default=list(VIEWS), choices=list(VIEWS))
    p.add_argument('--scale', type=int, default=1, help='Integer factor on the simulated sensor resolution')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--pose-plan', type=Path, help='JSON of mounted camera waypoints from scan_bead_seam.py')
    p.add_argument('--rgb-only', action='store_true', help='Capture images and camera poses without evaluation channels')
    a = p.parse_args()
    if a.scale < 1:
        p.error('--scale must be positive')
    output = a.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        p.error('Use a new empty output directory')
    (output / 'captures').mkdir(parents=True)
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    from weld_scene.optics import load_calibration
    calibration = load_calibration()
    width, height = (a.scale * v for v in calibration['image_size_wh'])
    k = np.array(calibration['intrinsics'], float)
    k[:2] *= a.scale
    baseline = calibration['baseline_m']
    bundle = ROOT / 'assets/scenes/industrial-welding'
    scene_hash = hashlib.sha256((bundle / 'industrial_welding.usdc').read_bytes()).hexdigest()
    plan = json.loads(a.pose_plan.read_text()) if a.pose_plan else None
    if plan is not None and plan['source_usdc_sha256'] != scene_hash:
        p.error('Pose plan was made for a different scene')
    stations = {v['station'] for v in plan['views']} if plan is not None else a.stations
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting', 'active_gpu': a.gpu, 'multi_gpu': False,
                         'open_usd': str(bundle / 'industrial_welding_full.usda'), 'width': width, 'height': height,
                         'sync_loads': True, 'extra_args': ['--/exts/omni.kit.async_engine/keep_loop_running=false']})
    try:
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdSemantics
        from weld_scene.usd import set_intrinsics
        stage = omni.usd.get_context().get_stage()
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        for _ in range(60):
            app.update()
        stage.SetEditTarget(stage.GetSessionLayer())
        stage.GetPrimAtPath('/World/Robot/e05l_welding_torch/hj90_active/laser').GetAttribute('inputs:intensity').Set(0.)
        if plan is not None:
            from weld_scene.mount_offset import apply_usd_offset
            apply_usd_offset(stage, plan['mount_offset_m'])
        for prim in (() if a.rgb_only else stage.Traverse()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            role = prim.GetAttribute('welding:role')
            role = role.Get() if role else None
            name = prim.GetName()
            label = ('weld_bead' if role == 'weld_metal' else 'spatter' if role == 'weld_spatter' else
                     'base_plate' if name.endswith(('_base', '_base_surface')) else
                     'web_plate' if name.endswith(('_web', '_wall_surface', '_wall_surface_back')) else None)
            station = name.split('_')[0]
            if label and station in stations:      # per station: the neighbouring fixture is background
                UsdSemantics.LabelsAPI.Apply(prim, 'class').CreateLabelsAttr().Set([f'{station.lower()}_{label}'])
        records = []
        planned_views = plan['views'] if plan is not None else inspection_views(stage, a.stations, a.views)
        for view in planned_views:
            ident, station = view['id'], view['station']
            midpoint = np.asarray(view['world_from_optical_midpoint'])
            poses, paths = [], []
            if plan is not None:
                poses, paths = mounted_cameras(stage, view, k, (width, height))
                eye_delta = poses[1][:3, 3] - poses[0][:3, 3]
                np.testing.assert_allclose(eye_delta, baseline * midpoint[:3, 0], atol=2e-6)
            else:
                for eye, shift in (('left', -baseline / 2), ('right', baseline / 2)):
                    pose = midpoint.copy()
                    pose[:3, 3] += shift * midpoint[:3, 0]
                    camera = UsdGeom.Camera.Define(stage, f'/World/Cameras/bead_rig_{ident}_{eye}')
                    set_intrinsics(camera, k, (width, height))
                    camera.MakeMatrixXform().Set(Gf.Matrix4d((pose @ np.diag([1., -1., -1., 1.])).T.tolist()))
                    poses.append(pose)
                    paths.append(str(camera.GetPath()))
            products, annotators = [], []
            folder = output / 'captures' / ident
            folder.mkdir(parents=True)
            try:
                for path in paths:
                    product = rep.create.render_product(path, (width, height))
                    group = [rep.AnnotatorRegistry.get_annotator('rgb')]
                    if not a.rgb_only:
                        group.extend((rep.AnnotatorRegistry.get_annotator('distance_to_image_plane'),
                                      rep.AnnotatorRegistry.get_annotator(
                                          'semantic_segmentation', init_params={'colorize': False})))
                    for annotator in group:
                        annotator.attach(product)
                    products.append(product)
                    annotators.append(group)
                for _ in range(28):
                    app.update()
                omni.usd.get_context().reset_renderer_accumulation()
                for _ in range(2):
                    rep.orchestrator.step(rt_subframes=32, pause_timeline=True, delta_time=0.)
                truth, views = {}, []
                for eye, pose, group in zip(('left', 'right'), poses, annotators):
                    image = np.asarray(group[0].get_data()).copy()[..., :3]
                    if image.shape != (height, width, 3) or image.std() < 1:
                        raise RuntimeError('Invalid render: ' + ident)
                    if not a.rgb_only:
                        depth, semantic = group[1:]
                        z = np.asarray(depth.get_data()).copy().squeeze().astype(np.float32)
                        result = semantic.get_data()
                        ids = np.asarray(result['data']).squeeze()
                        labels = np.zeros(ids.shape, np.uint8)
                        for key, value in result['info']['idToLabels'].items():
                            for name, code in LABELS.items():
                                if f'{station.lower()}_{name}' in str(value).lower():
                                    labels[ids == int(key)] = code
                        if z.shape != (height, width) or not (labels == LABELS['weld_bead']).any():
                            raise RuntimeError('Invalid evaluation channels for ' + ident)
                        truth[eye + '_z_m'] = z
                        truth[eye + '_labels'] = labels
                    file = folder / (eye + '.png')
                    Image.fromarray(image).save(file)
                    views.append({'id': eye, 'image': file.name, 'image_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                                  'image_size_wh': [width, height], 'intrinsics': k.tolist(),
                                  'extrinsics_w2c': (np.linalg.inv(pose) @ poses[0]).tolist(),
                                  'distortion_coefficients': [0, 0, 0, 0, 0]})
                if timeline.is_playing():
                    raise RuntimeError('Timeline advanced during capture')
                (folder / 'input.json').write_text(json.dumps({
                    'schema': 'quadweld-stereo-v1', 'synchronized': True, 'views': views, 'baseline_m': baseline,
                    'length_unit': 'm', 'laser_enabled': False,
                    'coordinate_convention': 'OpenCV world-to-camera; world is left optical frame',
                    'world_from_left_optical': poses[0].tolist(),
                    'calibration_status': 'Simulated HJ-90 pinhole stereo'}, indent=2) + '\n')
                record = {**view, 'world_from_left_optical': poses[0].tolist(), 'camera_paths': paths}
                if not a.rgb_only:
                    (folder / 'evaluation').mkdir()
                    np.savez_compressed(folder / 'evaluation/truth.npz', **truth)
                    (folder / 'evaluation/capture.json').write_text(json.dumps({
                        **record, 'labels': LABELS, 'source_usdc_sha256': scene_hash, 'physics_executed': False,
                        'truth_usage': 'Independent evaluation only; not a predictor input'}, indent=2) + '\n')
                records.append(record)
                print('BEAD_STEREO_CAPTURED', ident, flush=True)
            finally:
                for product, group in zip(products, annotators):
                    for annotator in group:
                        annotator.detach(product)
                    product.destroy()
        if hashlib.sha256((bundle / 'industrial_welding.usdc').read_bytes()).hexdigest() != scene_hash:
            raise RuntimeError('Source scene changed')
        (output / 'views.json').write_text(json.dumps({
            'views': records, 'resolution_wh': [width, height], 'intrinsics': k.tolist(), 'baseline_m': baseline,
            'scale': a.scale, 'source_usdc_sha256': scene_hash,
            'pose_source': 'Arm forward kinematics' if plan is not None else 'Free rig placed from fixture frame',
            'physics_executed': False}, indent=2) + '\n')
        print('BEAD_STEREO_COMPLETE', len(records), flush=True)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        app.close(exit_code=1)
        raise
    else:
        app.close()


if __name__ == '__main__':
    main()
