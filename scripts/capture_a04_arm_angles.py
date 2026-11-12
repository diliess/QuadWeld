"""Capture A04 stereo at different actual arm joint poses, with a fixed base."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture_hj90_stereo import write_json
from benchmark_stereo_seams import digest


def plan_poses(stage, collision_file):
    import mujoco as mj
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Usd, UsdGeom
    from hj90.arm_kinematics import ArmKinematics

    arm = ArmKinematics(stage)
    cache = UsdGeom.XformCache()
    model = mj.MjModel.from_xml_path(str(collision_file))
    data = mj.MjData(model)
    base = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot/base_link'))).T
    quat = Rotation.from_matrix(base[:3, :3]).as_quat()
    data.qpos[:3], data.qpos[3:7] = base[:3, 3], np.r_[quat[3], quat[:3]]
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot/Joints')):
        attr = prim.GetAttribute('state:angular:physics:position')
        if attr:
            index = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, prim.GetName())
            if index >= 0:
                data.qpos[model.jnt_qposadr[index]] = np.radians(attr.Get())
    addresses = [model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f'e05l_joint{i}')] for i in range(1, 7)]
    mj.mj_forward(model, data)
    for path, expected in arm.forward(arm.initial).items():
        index = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, path.split('/')[-1])
        if index >= 0 and (not np.allclose(data.xpos[index], expected[:3, 3], atol=2e-5, rtol=0) or
                           not np.allclose(data.xmat[index].reshape(3, 3), expected[:3, :3], atol=2e-5, rtol=0)):
            raise ValueError('MuJoCo collision model does not agree with captured USD arm')
    requested = [('A04_initial_arm', arm.initial.copy(), 'initial')]
    for joint, degrees in ((0, 8), (1, 6), (3, 10), (4, 6)):
        for sign in (-1, 1):
            q = arm.initial.copy()
            q[joint] += np.radians(sign*degrees)
            name = f'A04_arm_j{joint+1}_{"m" if sign < 0 else "p"}{degrees:02d}'
            requested.append((name, q, f'joint {joint+1}: {sign*degrees:+d} deg from initial'))
    records = []
    initial_camera = arm.camera(arm.initial)
    for name, q, description in requested:
        arm.forward(q)  # Enforce USD joint limits before collision checks.
        samples = max(2, int(np.ceil(np.max(np.abs(q-arm.initial))/.04))+1)
        worst = 0.
        for fraction in np.linspace(0, 1, samples):
            data.qpos[addresses] = arm.initial+(q-arm.initial)*fraction
            mj.mj_forward(model, data)
            bad = [contact for contact in data.contact if contact.dist < -.0005]
            if bad:
                raise ValueError(f'{name}: collision detected along sampled joint interpolation')
            worst = min(worst, min((float(c.dist) for c in data.contact), default=0.))
        camera = arm.camera(q)
        records.append({'id': name, 'description': description, 'arm_q_rad': q.tolist(),
                        'arm_q_deg': np.degrees(q).tolist(), 'interpolation_collision_samples': samples,
                        'collision_penetration_limit_m': .0005, 'worst_contact_distance_m': worst,
                        'optical_orientation_change_deg': float(np.degrees(Rotation.from_matrix(
                            camera[:3, :3] @ initial_camera[:3, :3].T).magnitude())),
                        'optical_translation_change_m': float(np.linalg.norm(camera[:3, 3]-initial_camera[:3, 3]))})
    return arm, records


def capture(args):
    import numpy as np
    from isaacsim import SimulationApp
    from main import scene_setup
    setup = scene_setup()
    source = Path(setup['usd'])
    source_usdc = ROOT/'assets/scenes/industrial-welding/industrial_welding.usdc'
    source_hash = digest(source_usdc)
    app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting', 'active_gpu': args.gpu,
                         'multi_gpu': False, 'width': 640, 'height': 1100, 'sync_loads': True,
                         'open_usd': str(source),
                         'extra_args': ['--/exts/omni.kit.async_engine/keep_loop_running=false']})
    try:
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from PIL import Image
        from pxr import Sdf, UsdGeom
        from hj90.optics import load_calibration
        from hj90.usd import attach, set_laser

        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        for _ in range(60):
            app.update()
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        calibration = load_calibration()
        cameras, laser, _ = attach(stage, calibration, ROOT/'assets/scenes/industrial-welding/textures/hj90_laser_profile.png')
        set_laser(laser, 0., False, calibration)
        arm, poses = plan_poses(stage, args.collision_model)
        write_json(args.output_dir/'arm_plan.json', {'views': poses, 'station': 'A04', 'source_usd': str(source),
                   'capture_script_sha256': digest(Path(__file__)),
                   'kinematics_sha256': digest(ROOT/'hj90/arm_kinematics.py'),
                   'base_fixed': True, 'camera_attached_to_torch': True, 'baseline_m': .09,
                   'collision_model': str(args.collision_model), 'collision_model_sha256': digest(args.collision_model),
                   'physics_executed': False, 'scope': 'Joint FK snapshots and sampled static collision checks; not dynamic tracking'})
        for pose in poses:
            arm.apply(pose['arm_q_rad'])
            for _ in range(20):
                app.update()
            products, rgbs, depths = [], [], []
            output = args.output_dir/'captures'/pose['id']
            (output/'evaluation').mkdir(parents=True)
            try:
                for side in ('left', 'right'):
                    product = rep.create.render_product(cameras[side], (640, 1100))
                    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
                    depth = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
                    rgb.attach(product); depth.attach(product)
                    products.append(product); rgbs.append(rgb); depths.append(depth)
                context = rep.create.render_product('/World/Cameras/A04_initial_arm_photo_context', (960, 640))
                context_rgb = rep.AnnotatorRegistry.get_annotator('rgb')
                context_rgb.attach(context)
                for _ in range(25):
                    app.update()
                omni.usd.get_context().reset_renderer_accumulation()
                for _ in range(2):
                    rep.orchestrator.step(rt_subframes=32, pause_timeline=True, delta_time=0.)
                cache = UsdGeom.XformCache()
                world = [np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(cameras[side]))).T @
                         np.diag([1., -1., -1., 1.]) for side in ('left', 'right')]
                torch_world = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot/e05l_welding_torch'))).T
                # Midpoint optical pose must result from arm FK, not camera motion.
                midpoint = world[0].copy(); midpoint[:3, 3] = (world[0][:3, 3]+world[1][:3, 3])/2
                np.testing.assert_allclose(midpoint, torch_world @ arm.torch_to_camera, atol=2e-6)
                np.testing.assert_allclose(midpoint, arm.camera(pose['arm_q_rad']), atol=2e-6)
                baseline = float(np.linalg.norm(world[1][:3, 3]-world[0][:3, 3]))
                if not np.isclose(baseline, .09, atol=1e-8):
                    raise ValueError('Mounted stereo baseline changed')
                truths, views = {}, []
                for i, side in enumerate(('left', 'right')):
                    rgb = np.asarray(rgbs[i].get_data()).copy()[..., :3]
                    depth = np.asarray(depths[i].get_data()).copy().squeeze().astype(np.float32)
                    if rgb.shape != (1100, 640, 3) or depth.shape != (1100, 640) or rgb.std() < 1:
                        raise ValueError('Invalid arm-mounted RGB/depth capture')
                    image = output/(side+'.png'); Image.fromarray(rgb).save(image)
                    camera = UsdGeom.Camera(stage.GetPrimAtPath(cameras[side]))
                    f = camera.GetFocalLengthAttr().Get()
                    k = [[640*f/camera.GetHorizontalApertureAttr().Get(), 0, 320],
                         [0, 1100*f/camera.GetVerticalApertureAttr().Get(), 550], [0, 0, 1]]
                    views.append({'id': side, 'image': image.name, 'image_sha256': digest(image),
                                  'image_size_wh': [640, 1100], 'intrinsics': k,
                                  'extrinsics_w2c': (np.linalg.inv(world[i]) @ world[0]).tolist(),
                                  'distortion_coefficients': [0, 0, 0, 0, 0]})
                    truths[side+'_z_m'] = depth
                Image.fromarray(np.asarray(context_rgb.get_data()).copy()[..., :3]).save(output/'robot_context.png')
                np.savez_compressed(output/'evaluation/truth.npz', **truths)
                write_json(output/'input.json', {'schema': 'quadweld-stereo-v1', 'synchronized': True, 'views': views,
                           'baseline_m': baseline, 'length_unit': 'm', 'laser_enabled': False,
                           'coordinate_convention': 'OpenCV world-to-camera; world is left optical frame',
                           'calibration_status': 'Simulation pinhole and rigid arm-mounted stereo; no real calibration'})
                write_json(output/'evaluation/capture.json', {'pose': pose['id'], 'usd': str(source),
                           'source_usdc_sha256': source_hash, 'station': 'A04', 'camera_paths': cameras,
                           'world_from_camera_optical': [x.tolist() for x in world], 'physics_executed': False,
                           'actual_arm_q_rad': pose['arm_q_rad'], 'robot_base_fixed': True,
                           'camera_rigid_on_torch': True, 'truth_usage': 'Independent evaluation only'})
                stage.GetSessionLayer().Export(str(output/'arm_pose.usda'))
                layer = Sdf.Layer.FindOrOpen(str(output/'arm_pose.usda'))
                layer.subLayerPaths = [str(source)]; layer.Save()
                if timeline.is_playing():
                    raise RuntimeError('Timeline advanced during static arm capture')
                print('ARM_STEREO_CAPTURED', pose['id'], 'q_deg', pose['arm_q_deg'], flush=True)
            finally:
                for product, rgb, depth in zip(products, rgbs, depths):
                    rgb.detach(product); depth.detach(product); product.destroy()
                context_rgb.detach(context); context.destroy()
        if digest(source_usdc) != source_hash:
            raise ValueError('Source scene changed')
        write_json(args.output_dir/'capture_status.json', {'completed': True, 'arm_poses': len(poses), 'stereo_images': 2*len(poses)})
    finally:
        app.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--collision-model', type=Path, default=ROOT.parent/'robosimstudio/examples/live/twins/industrial_welding/assets/assembled/validation_world.xml')
    a = p.parse_args()
    a.output_dir = a.output_dir.expanduser().resolve(); a.collision_model = a.collision_model.resolve()
    if a.output_dir.exists() and any(path.name != 'capture.log' for path in a.output_dir.iterdir()):
        p.error('Use a new empty output directory')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    capture(a)
