"""Capture synchronized stereo plus a swept laser; reconstruct from RGB correspondences."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hj90.optics import load_calibration


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    scene = json.loads((ROOT/'assets/scenes/industrial-welding/manifest.json').read_text())['full_scene']
    p.add_argument('--pose', choices=tuple(scene['poses']), default=scene['default_pose'])
    p.add_argument('--fixture', choices=('plane','step'), help='Synthetic optical validation target, not a robot placement')
    p.add_argument('--calibration', type=Path, default=ROOT/'calibration/hj90_sim.json')
    p.add_argument('--mode', choices=('static','dynamic'), default='static')
    p.add_argument('--lines', type=int, help='Scan samples (default from calibration)')
    p.add_argument('--duration', type=float, help='Simulated acquisition duration, not wall-clock render time')
    p.add_argument('--angle-deg', type=float, default=0., help='Fixed laser angle in dynamic mode')
    p.add_argument('--target-speed', type=float, default=0., help='Synthetic fixture X speed in m/s, dynamic validation only')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--evaluate', action='store_true', help='Save separate renderer truth and compare; never used for reconstruction')
    p.add_argument('--subframes', type=int, default=16)
    p.add_argument('--isaac-python', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.calibration = a.calibration.expanduser().resolve()
    c = load_calibration(a.calibration)
    a.lines = a.lines if a.lines is not None else c['scan_lines']
    a.duration = a.duration if a.duration is not None else c['full_scan_duration_s']
    if a.lines < 1 or a.duration <= 0 or a.gpu < 0 or a.subframes < 1:
        p.error('Use positive lines/duration/subframes and a nonnegative GPU index')
    if a.lines/a.duration > c['max_line_rate_hz']:
        p.error('Requested line rate exceeds the datasheet maximum')
    if a.mode == 'static' and a.duration < max(.2, 1/c['max_full_frame_fps']):
        p.error('Static full scans require at least 0.2 s simulated acquisition')
    if abs(a.angle_deg) > c['scan_half_angle_deg']:
        p.error('Dynamic angle is outside the configured scan range')
    if a.target_speed and (not a.fixture or a.mode != 'dynamic'):
        p.error('--target-speed is only for a synthetic dynamic fixture')
    stamp = datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y%m%d_%H%M%S_%f')
    a.output_dir = (a.output_dir or ROOT/'outputs/hj90'/stamp).expanduser().resolve()
    return a, c


def validation_fixture(stage, kind):
    """Known geometry is used by rendering/evaluation only, never triangulation."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade
    from hj90.usd import SOURCE_CAMERA
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    source = UsdGeom.Camera.Define(stage, SOURCE_CAMERA)
    source.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.))
    target = UsdGeom.Xform.Define(stage, '/World/Target')
    target.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    mat = UsdShade.Material.Define(stage, '/World/Material')
    shader = UsdShade.Shader.Define(stage, '/World/Material/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.3, .3, .3))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.65)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    specs = [('Plane', (0, 0, .39), (.9, 1.3, .02))]
    if kind == 'step':
        specs += [('Step', (.22, 0, .45), (.4, 1.3, .12)), ('Occluder', (-.09, 0, .61), (.035, .26, .10))]
    for name, xyz, size in specs:
        cube = UsdGeom.Cube.Define(stage, '/World/Target/'+name)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    light = UsdLux.DomeLight.Define(stage, '/World/Light')
    light.CreateIntensityAttr(300.)
    return target


def worker(a, c):
    import time
    import numpy as np
    from isaacsim import SimulationApp
    manifest = json.loads((ROOT/'assets/scenes/industrial-welding/manifest.json').read_text())
    usd = ROOT/'assets/scenes/industrial-welding'/manifest['full_scene']['poses'][a.pose]['asset']
    base_hash = hashlib.sha256(usd.read_bytes()).hexdigest()
    config = {'headless': a.headless, 'active_gpu': a.gpu, 'multi_gpu': False,
              'renderer': 'RaytracedLighting', 'width': 640, 'height': 1100, 'sync_loads': True,
              'extra_args': ['--/exts/omni.kit.async_engine/keep_loop_running=false']}
    if not a.fixture:
        config['open_usd'] = str(usd)
    report = {'passed': False, 'mode': a.mode, 'fixture': a.fixture, 'pose': a.pose,
              'calibration': c, 'depth_source': 'laser stripes detected in two rendered RGB images; calibrated stereo triangulation',
              'truth_usage': 'evaluation only' if a.evaluate else 'no renderer depth annotator attached',
              'physics_executed': False, 'real_hardware_fidelity': 'uncalibrated functional approximation'}
    write_json(a.output_dir/'report.json', report)
    app = SimulationApp(config)
    resources = []
    try:
        import carb
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from PIL import Image
        from pxr import Gf, UsdGeom
        from hj90.usd import attach, laser_texture, set_laser
        from hj90.optics import reconstruct, rasterize, write_ply, write_pcd
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        for _ in range(60):
            app.update()
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        target = validation_fixture(stage, a.fixture) if a.fixture else None
        texture = laser_texture(a.output_dir/'laser_profile.png', c)
        cameras, laser, rig = attach(stage, c, texture)
        # No temporal denoising/exposure adaptation may carry a previous stripe into a later frame.
        settings = carb.settings.get_settings()
        settings.set('/rtx/post/histogram/enabled', False)
        settings.set('/rtx/post/dlss/execMode', 2)
        settings.set('/rtx/post/aa/op', 0)
        for side, path in cameras.items():
            product = rep.create.render_product(path, tuple(c['image_size_wh']))
            rgb = rep.AnnotatorRegistry.get_annotator('rgb')
            rgb.attach(product)
            truth = None
            if a.evaluate:
                truth = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
                truth.attach(product)
            resources.append((product, rgb, truth))
        for _ in range(40):
            app.update()
        def render(enabled, angle):
            set_laser(laser, angle, enabled, c)
            omni.usd.get_context().reset_renderer_accumulation()
            rep.orchestrator.step(rt_subframes=a.subframes, pause_timeline=True, delta_time=0.)
            # Both products are rendered by the same orchestrator tick at one pose/time.
            images = [np.asarray(rgb.get_data()).copy()[..., :3] for _, rgb, _ in resources]
            if any(im.shape != (c['image_size_wh'][1], c['image_size_wh'][0], 3) for im in images):
                raise RuntimeError('Invalid stereo image shape')
            return images
        angles = (np.linspace(-c['scan_half_angle_deg'], c['scan_half_angle_deg'], a.lines)
                  if a.mode == 'static' and a.lines > 1 else np.full(a.lines, a.angle_deg))
        points, world_points, uv_all, confidence_all, ids, errors = [], [], [], [], [], []
        frames = []
        started = time.monotonic()
        calibration = {'schema': 'hj90-active-stereo-v1', 'optics': c, 'cameras': cameras,
                       'point_frame': 'stereo midpoint OpenCV (+X right, +Y down, +Z forward)'}
        write_json(a.output_dir/'camera.json', calibration)
        static_off = render(False, 0.) if a.mode == 'static' else None
        if static_off is not None:
            for side, image in zip(('left','right'), static_off):
                Image.fromarray(image).save(a.output_dir/(side+'_off.png'))
                Image.fromarray(image).convert('L').save(a.output_dir/(side+'_gray.png'))
        for i, degrees in enumerate(angles):
            stamp = i*a.duration/a.lines
            if target and a.mode == 'dynamic':
                target.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(a.target_speed*stamp, 0, 0))
            angle = float(np.radians(degrees))
            off = static_off if static_off is not None else render(False, angle)
            on = render(True, angle)
            xyz, uv, confidence = reconstruct(on[0], off[0], on[1], off[1], c, angle)
            cv_from_usd = np.diag([1., -1., -1., 1.])
            world_from_mid = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(rig.GetPrim())).T @ cv_from_usd
            points.append(xyz)
            world_points.append(xyz @ world_from_mid[:3,:3].T + world_from_mid[:3,3])
            uv_all.append(uv)
            confidence_all.append(confidence)
            ids.append(np.full(len(xyz), i, np.int32))
            name = f'{i:04d}'
            arrays = {'left_on': on[0], 'right_on': on[1]}
            if static_off is None:
                arrays.update(left_off=off[0], right_off=off[1])
            np.savez_compressed(a.output_dir/'frames'/(name+'.npz'), **arrays)
            if i in (0, len(angles)//2, len(angles)-1):
                Image.fromarray(on[0]).save(a.output_dir/f'left_on_{name}.png')
                Image.fromarray(on[1]).save(a.output_dir/f'right_on_{name}.png')
            evaluation = None
            if a.evaluate:
                truth = np.asarray(resources[0][2].get_data()).copy().squeeze()
                np.savez_compressed(a.output_dir/'evaluation'/(name+'.npz'), left_truth_z_m=truth)
                if len(xyz):
                    xy = np.floor(uv).astype(int)
                    gt = truth[xy[:,1], xy[:,0]]
                    good = np.isfinite(gt) & (gt > 0)
                    delta = np.abs(xyz[good,2]-gt[good])
                    errors.extend(delta.tolist())
                    evaluation = {'samples': len(delta), 'mae_m': float(delta.mean()) if len(delta) else None}
            frames.append({'index': i, 'timestamp_s': stamp, 'laser_angle_deg': float(degrees),
                           'world_from_mid_optical': world_from_mid.tolist(), 'points': len(xyz),
                           'evaluation': evaluation})
            if i % 16 == 0 or i == len(angles)-1:
                print('HJ90_SCAN', i+1, '/', len(angles), 'points', sum(map(len, points)), flush=True)
        xyz = np.concatenate(points)
        world_xyz = np.concatenate(world_points)
        depth = rasterize(xyz, c) if a.mode == 'static' else None
        payload = {'points_mid_optical_m': xyz.astype(np.float32), 'points_world_m': world_xyz.astype(np.float32),
                   'left_uv': np.concatenate(uv_all).astype(np.float32),
                   'confidence': np.concatenate(confidence_all).astype(np.float32), 'frame_index': np.concatenate(ids)}
        if depth is not None:
            payload.update(depth_m=depth, valid_depth_mask=np.isfinite(depth))
        np.savez_compressed(a.output_dir/'scan.npz', **payload)
        write_ply(a.output_dir/'cloud.ply', world_xyz)
        write_pcd(a.output_dir/'cloud.pcd', world_xyz)
        np.savetxt(a.output_dir/'cloud.txt', world_xyz, fmt='%.7f')
        write_json(a.output_dir/'frames.json', frames)
        if depth is not None:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.imsave(a.output_dir/'depth.png', depth, cmap='turbo', vmin=.3, vmax=.8)
        report.update(passed=len(xyz)>0, points=len(xyz), duration_simulated_s=a.duration,
                      duration_wall_s=time.monotonic()-started, line_rate_hz=a.lines/a.duration,
                      frames=a.lines, timeline_playing=timeline.is_playing(), source_layer_unchanged=hashlib.sha256(usd.read_bytes()).hexdigest()==base_hash,
                      raw_frames='frames/*.npz', output='scan.npz', point_cloud='cloud.ply',
                      depth_coverage_fraction=float(np.isfinite(depth).mean()) if depth is not None else None,
                      evaluation={'points': len(errors), 'mae_m': float(np.mean(errors)) if errors else None,
                                  'p95_m': float(np.percentile(errors,95)) if errors else None})
        write_json(a.output_dir/'report.json', report)
        print('HJ90_SCAN_COMPLETE', json.dumps({k:v for k,v in report.items() if k!='calibration'}), flush=True)
        if not report['passed']:
            raise RuntimeError('No stereo laser correspondences reconstructed; inspect on/off images')
    except BaseException as error:
        report.update(passed=False, error=str(error))
        write_json(a.output_dir/'report.json', report)
        raise
    finally:
        for product, rgb, truth in resources:
            rgb.detach(product)
            if truth is not None:
                truth.detach(product)
            product.destroy()
        app.close()


def main(argv=None):
    a, c = parse_args(argv)
    if a._worker:
        worker(a, c)
        return
    from main import find_python, child_environment
    executable, error = find_python('isaac', a.isaac_python)
    if not executable:
        raise RuntimeError(error)
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise ValueError('Use a new empty output directory')
    for child in ('frames', 'evaluation'):
        (a.output_dir/child).mkdir(parents=True, exist_ok=True)
    command = [executable, str(Path(__file__).resolve()), '--_worker', '--pose', a.pose,
               '--calibration', str(a.calibration), '--mode', a.mode, '--lines', str(a.lines),
               '--duration', str(a.duration), '--angle-deg', str(a.angle_deg), '--target-speed', str(a.target_speed),
               '--gpu', str(a.gpu), '--subframes', str(a.subframes), '--output-dir', str(a.output_dir)]
    if a.fixture:
        command += ['--fixture', a.fixture]
    for flag in ('headless','evaluate'):
        if getattr(a, flag):
            command.append('--'+flag)
    environment = child_environment()
    environment['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    with (a.output_dir/'runtime.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('HJ-90 scan:', a.output_dir)


if __name__ == '__main__':
    main()
