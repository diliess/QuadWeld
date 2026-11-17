"""Save synchronized HJ-90 stereo RGB/calibration and separate evaluation truth."""

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def capture(args):
    import numpy as np
    from isaacsim import SimulationApp
    from weld_scene.optics import load_calibration
    c = load_calibration()
    bundle = ROOT/'assets/scenes/industrial-welding'
    manifest = json.loads((bundle/'manifest.json').read_text())
    first = bundle/manifest['full_scene']['poses'][args.poses[0]]['asset']
    source_hash = hashlib.sha256((bundle/'industrial_welding.usdc').read_bytes()).hexdigest()
    app = SimulationApp({'headless':True, 'renderer':'RaytracedLighting', 'active_gpu':args.gpu,
                         'multi_gpu':False, 'width':640, 'height':1100, 'sync_loads':True,
                         'open_usd':str(first),
                         'extra_args':['--/exts/omni.kit.async_engine/keep_loop_running=false']})
    try:
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from PIL import Image
        from pxr import UsdGeom, UsdLux
        from weld_scene.usd import attach, laser_texture, set_laser
        context = omni.usd.get_context()
        for case_index, pose in enumerate(args.poses):
            asset = bundle/manifest['full_scene']['poses'][pose]['asset']
            if case_index and not context.open_stage(str(asset)):
                raise RuntimeError('Could not switch to '+pose)
            timeline = omni.timeline.get_timeline_interface()
            timeline.stop()
            for _ in range(60):
                app.update()
            stage = context.get_stage()
            stage.SetEditTarget(stage.GetSessionLayer())
            output = args.output_dir/pose
            (output/'evaluation').mkdir(parents=True, exist_ok=True)
            camera_paths, laser, _ = attach(stage,c,laser_texture(output/'laser_profile.png',c))
            set_laser(laser,0.,False,c)
            products, rgbs, depths = [],[],[]
            try:
                for side in ('left','right'):
                    product = rep.create.render_product(camera_paths[side],tuple(c['image_size_wh']))
                    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
                    depth = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
                    rgb.attach(product)
                    depth.attach(product)
                    products.append(product)
                    rgbs.append(rgb)
                    depths.append(depth)
                for _ in range(40):
                    app.update()
                context.reset_renderer_accumulation()
                for _ in range(2):
                    rep.orchestrator.step(rt_subframes=32,pause_timeline=True,delta_time=0.)
                cache = UsdGeom.XformCache()
                world_poses = []
                intrinsics = []
                for side in ('left','right'):
                    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_paths[side]))
                    matrix = np.array(cache.GetLocalToWorldTransform(camera.GetPrim())).T @ np.diag([1.,-1.,-1.,1.])
                    world_poses.append(matrix)
                    focal = camera.GetFocalLengthAttr().Get()
                    w,h = c['image_size_wh']
                    intrinsics.append([[w*focal/camera.GetHorizontalApertureAttr().Get(),0,w/2],
                                       [0,h*focal/camera.GetVerticalApertureAttr().Get(),h/2],[0,0,1]])
                relative = [np.linalg.inv(pose_world) @ world_poses[0] for pose_world in world_poses]
                baseline = float(np.linalg.norm(world_poses[1][:3,3]-world_poses[0][:3,3]))
                if not np.isclose(baseline,.09,atol=1e-8):
                    raise ValueError('Unexpected HJ-90 baseline')
                views = []
                truths = {}
                for i,side in enumerate(('left','right')):
                    image = np.asarray(rgbs[i].get_data()).copy()[...,:3]
                    z = np.asarray(depths[i].get_data()).copy().squeeze().astype(np.float32)
                    if image.shape != (1100,640,3) or z.shape != (1100,640) or image.std()<1:
                        raise RuntimeError('Invalid stereo RGB/truth')
                    file = output/(side+'.png')
                    Image.fromarray(image).save(file)
                    truths[side+'_z_m'] = z
                    views.append({'id':side,'image':file.name,'image_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                                  'image_size_wh':[640,1100],'intrinsics':intrinsics[i],
                                  'extrinsics_w2c':relative[i].tolist(),'distortion_coefficients':[0,0,0,0,0]})
                if timeline.is_playing():
                    raise RuntimeError('Timeline advanced during stereo capture')
                write_json(output/'input.json',{'schema':'quadweld-stereo-v1','synchronized':True,
                           'views':views,'baseline_m':baseline,'length_unit':'m',
                           'coordinate_convention':'OpenCV world-to-camera; world is left optical frame',
                           'laser_enabled':False,'calibration_status':'simulation pinhole and pose readback; no real hardware calibration'})
                np.savez_compressed(output/'evaluation/truth.npz',**truths)
                write_json(output/'evaluation/capture.json',{'pose':pose,'usd':str(asset),
                           'source_usdc_sha256':source_hash,'camera_paths':camera_paths,
                           'world_from_camera_optical':[p.tolist() for p in world_poses],
                           'station':manifest['full_scene']['poses'][pose].get('station'),'physics_executed':False,
                           'actual_arm_q_rad':[float(np.radians(stage.GetPrimAtPath('/World/Robot/Joints/e05l_joint'+str(j)).GetAttribute(
                               'state:angular:physics:position').Get())) for j in range(1,7)],
                           'truth_usage':'independent evaluation only; not a predictor input'})
                print('STEREO_CAPTURED',pose,'baseline',baseline,flush=True)
            finally:
                for product,rgb,depth in zip(products,rgbs,depths):
                    rgb.detach(product)
                    depth.detach(product)
                    product.destroy()
        if hashlib.sha256((bundle/'industrial_welding.usdc').read_bytes()).hexdigest()!=source_hash:
            raise RuntimeError('Source scene changed')
    finally:
        app.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    scene=json.loads((ROOT/'assets/scenes/industrial-welding/manifest.json').read_text())['full_scene']
    p.add_argument('--poses',nargs='+',choices=tuple(scene['poses']),default=[scene['default_pose']])
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    a.output_dir=a.output_dir.expanduser().resolve()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Use a new empty output directory')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
    capture(a)


if __name__ == "__main__":
    main()
