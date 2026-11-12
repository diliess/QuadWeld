"""Frame A15/A03/A21 using actual robot/base FK, then capture mounted stereo."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_stereo_seams import digest, write_json


def prepare(args):
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom
    from hj90.arm_kinematics import ArmKinematics
    from hj90.optics import load_calibration
    from hj90.overview_planning import OverviewPlanner, look_at, transform_points, world_matrix
    from hj90.usd import attach, set_laser

    from main import scene_setup
    source = Path(scene_setup()['usd'])
    stage = Usd.Stage.Open(str(source))
    calibration = load_calibration()
    planner = OverviewPlanner(stage, args.collision_model, calibration)
    records = []
    for station in args.stations:
        best, candidates = planner.choose(station)
        write_json(args.output_dir/(station+'_view_candidates.json'), candidates)
        folder = args.output_dir/'captures'/best['id']; folder.mkdir(parents=True, exist_ok=True)
        layer = Sdf.Layer.CreateNew(str(folder/'arm_pose.usda'))
        layer.subLayerPaths = [str(source)]
        placed = Usd.Stage.Open(layer)
        delta = np.array(best['base_world']) @ np.linalg.inv(planner.source_base)
        UsdGeom.Xformable(placed.GetPrimAtPath('/World/Robot')).MakeMatrixXform().Set(Gf.Matrix4d(delta.T.tolist()))
        arm = ArmKinematics(placed)
        arm.apply(best['arm_q_rad'])
        np.testing.assert_allclose(arm.camera(best['arm_q_rad']), best['world_from_optical_midpoint'], atol=2e-5)
        cameras, laser, _ = attach(placed, calibration, ROOT/'assets/scenes/industrial-welding/textures/hj90_laser_profile.png')
        set_laser(laser, 0., False, calibration)
        cache = UsdGeom.XformCache()
        fixture = world_matrix(cache, placed.GetPrimAtPath('/World/Fixtures/'+station))
        centre = transform_points(fixture, np.array([[0., 0., best['world_from_optical_midpoint'][2][3]-.5]]))[0]
        centre[2] = np.clip(centre[2], .5, 1.2)
        context_pose = look_at(transform_points(fixture, np.array([[2.8, -3.3, 2.4]]))[0], centre)
        context = UsdGeom.Camera.Define(placed, '/World/Cameras/'+station+'_overview_context')
        context.MakeMatrixXform().Set(Gf.Matrix4d((context_pose @ np.diag([1., -1., -1., 1.])).T.tolist()))
        context.CreateFocalLengthAttr(22.)
        context.CreateHorizontalApertureAttr(36.)
        context.CreateVerticalApertureAttr(24.)
        context.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
        placed.SetDefaultPrim(placed.GetPrimAtPath('/World'))
        UsdGeom.SetStageMetersPerUnit(placed, 1.); UsdGeom.SetStageUpAxis(placed, 'Z')
        layer.Save()
        best.update(usd=str(folder/'arm_pose.usda'), camera_paths=cameras, context_camera=str(context.GetPath()))
        records.append(best)
        print('STATION_PLANNED', station, 'q_deg', best['arm_q_deg'], 'visibility',
              best['front_surface_ray_visibility_both_eyes'], flush=True)
    plan = {'views': records, 'source_usd': str(source),
            'source_usdc_sha256': digest(ROOT/'assets/scenes/industrial-welding/industrial_welding.usdc'),
            'collision_model': str(args.collision_model), 'collision_model_sha256': digest(args.collision_model),
            'source_hashes': {name: digest(ROOT/name) for name in ('scripts/capture_station_overviews.py',
                              'hj90/overview_planning.py','hj90/arm_kinematics.py','hj90/usd.py')},
            'view_selection': 'Fixed CAD framing/front-face ray visibility, supported static IK; no model-error feedback',
            'framing_margin_fraction': .04, 'baseline_m': .09, 'resolution_wh': calibration['image_size_wh'],
            'physics_executed': False, 'scope': 'Independent static initial placements; no walk/arm trajectory executed'}
    write_json(args.output_dir/'station_plan.json', plan)
    return plan


def capture(args, plan):
    import numpy as np
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting', 'active_gpu': args.gpu,
                         'multi_gpu': False, 'width': 640, 'height': 1100, 'sync_loads': True,
                         'extra_args': ['--/exts/omni.kit.async_engine/keep_loop_running=false']})
    try:
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from PIL import Image
        from pxr import Gf, UsdGeom
        from hj90.arm_kinematics import ArmKinematics
        from hj90.overview_planning import world_matrix, t_geometry, transform_points
        context = omni.usd.get_context()
        timeline = omni.timeline.get_timeline_interface()
        for view in plan['views']:
            output = args.output_dir/'captures'/view['id']
            if args.resume and (output/'input.json').is_file():
                from da3_stereo import load_pair
                load_pair(output/'input.json')
                continue
            context.open_stage(view['usd'])
            timeline.stop()
            for _ in range(60): app.update()
            stage = context.get_stage()
            if UsdGeom.GetStageMetersPerUnit(stage)!=1. or UsdGeom.GetStageUpAxis(stage)!='Z':
                raise ValueError('Capture requires an explicitly metre/Z-up root USD layer')
            arm = ArmKinematics(stage)
            np.testing.assert_allclose(arm.camera(arm.initial), view['world_from_optical_midpoint'], atol=2e-5)
            products, annotators = [], []
            try:
                rgbs, depths, semantics = [], [], []
                for side in ('left', 'right'):
                    product = rep.create.render_product(view['camera_paths'][side], tuple(plan['resolution_wh']))
                    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
                    depth = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
                    rgb.attach(product); depth.attach(product)
                    products.append(product); annotators.extend(((rgb, product), (depth, product)))
                    rgbs.append(rgb); depths.append(depth)
                    if plan.get('require_no_torch_pixels'):
                        semantic=rep.AnnotatorRegistry.get_annotator('semantic_segmentation',init_params={'colorize':False})
                        semantic.attach(product);annotators.append((semantic,product));semantics.append(semantic)
                control_rgb=control_semantic=None
                if plan.get('require_no_torch_pixels') and view is plan['views'][0]:
                    control=UsdGeom.Camera.Define(stage,'/World/Robot/e05l_welding_torch/hj90_visibility_control')
                    reference=UsdGeom.Camera(stage.GetPrimAtPath(view['camera_paths']['left']))
                    for attribute in ('FocalLength','HorizontalAperture','VerticalAperture','ClippingRange'):
                        getattr(control,'Create'+attribute+'Attr')(getattr(reference,'Get'+attribute+'Attr')().Get())
                    optical=np.array(plan['mount_adjustment']['old_torch_from_optical'])
                    optical[:3,3] -= optical[:3,0]*.045
                    control.MakeMatrixXform().Set(Gf.Matrix4d((optical@np.diag([1.,-1.,-1.,1.])).T.tolist()))
                    control_product=rep.create.render_product(str(control.GetPath()),tuple(plan['resolution_wh']))
                    control_rgb=rep.AnnotatorRegistry.get_annotator('rgb')
                    control_semantic=rep.AnnotatorRegistry.get_annotator('semantic_segmentation',init_params={'colorize':False})
                    for annotator in (control_rgb,control_semantic):
                        annotator.attach(control_product);annotators.append((annotator,control_product))
                    products.append(control_product)
                product = rep.create.render_product(view['context_camera'], (960, 640))
                rgb_context = rep.AnnotatorRegistry.get_annotator('rgb'); rgb_context.attach(product)
                products.append(product); annotators.append((rgb_context, product))
                for _ in range(25): app.update()
                context.reset_renderer_accumulation()
                for _ in range(2): rep.orchestrator.step(rt_subframes=32, pause_timeline=True, delta_time=0.)
                cache = UsdGeom.XformCache()
                world = [world_matrix(cache, stage.GetPrimAtPath(view['camera_paths'][side])) @ np.diag([1., -1., -1., 1.])
                         for side in ('left', 'right')]
                midpoint = world[0].copy(); midpoint[:3, 3] = (world[0][:3, 3]+world[1][:3, 3])/2
                np.testing.assert_allclose(midpoint, arm.camera(arm.initial), atol=2e-6)
                baseline = float(np.linalg.norm(world[0][:3, 3]-world[1][:3, 3]))
                np.testing.assert_allclose(baseline, .09, atol=1e-8)
                truths, views, torch_counts, depth_scale_checks = {}, [], {}, {}
                def torch_mask(annotator):
                    result=annotator.get_data()
                    labels=result['info']['idToLabels']
                    ids=[int(key) for key,label in labels.items() if 'welding_torch' in str(label)]
                    mask=np.isin(np.asarray(result['data']).squeeze(),ids)
                    if mask.shape != (1100,640):raise ValueError('Invalid torch segmentation shape')
                    return mask
                (output/'evaluation').mkdir(exist_ok=True)
                if control_semantic is not None:
                    mask=torch_mask(control_semantic)
                    if not mask.any():raise ValueError('Torch segmentation positive control failed')
                    Image.fromarray(np.asarray(control_rgb.get_data()).copy()[...,:3]).save(output/'evaluation/control_with_torch.png')
                    Image.fromarray(mask.astype(np.uint8)*255).save(output/'evaluation/control_torch_mask.png')
                    write_json(output/'evaluation/torch_segmentation_control.json',{'torch_pixels':int(mask.sum()),
                               'description':'Original left-eye mounting relative to the same visible torch; evaluation only'})
                for i, side in enumerate(('left', 'right')):
                    rgb = np.asarray(rgbs[i].get_data()).copy()[..., :3]
                    depth = np.asarray(depths[i].get_data()).copy().squeeze().astype(np.float32)
                    if rgb.shape != (1100, 640, 3) or depth.shape != (1100, 640) or rgb.std() < 1:
                        raise ValueError('Invalid station RGB/depth: '+view['id'])
                    image = output/(side+'.png'); Image.fromarray(rgb).save(image)
                    camera = UsdGeom.Camera(stage.GetPrimAtPath(view['camera_paths'][side]))
                    focal = camera.GetFocalLengthAttr().Get()
                    k = [[640*focal/camera.GetHorizontalApertureAttr().Get(), 0, 320],
                         [0, 1100*focal/camera.GetVerticalApertureAttr().Get(), 550], [0, 0, 1]]
                    if plan.get('require_no_torch_pixels'):
                        fixture,_,_,surface=t_geometry(stage,view['station'])
                        points=transform_points(np.linalg.inv(world[i]),transform_points(fixture,surface))
                        projected=points@np.array(k).T
                        pixels=np.floor(projected[:,:2]/projected[:,2:]).astype(int)
                        valid=(points[:,2]>0)&(pixels[:,0]>=0)&(pixels[:,0]<640)&(pixels[:,1]>=0)&(pixels[:,1]<1100)
                        ratio=float(np.median(depth[pixels[valid,1],pixels[valid,0]]/points[valid,2]))
                        if not .95<ratio<1.05:raise ValueError('Rendered depth scale disagrees with CAD geometry: '+str(ratio))
                        depth_scale_checks[side]=ratio
                    views.append({'id': side, 'image': image.name, 'image_sha256': digest(image),
                                  'image_size_wh': [640, 1100], 'intrinsics': k,
                                  'extrinsics_w2c': (np.linalg.inv(world[i]) @ world[0]).tolist(),
                                  'distortion_coefficients': [0, 0, 0, 0, 0]})
                    truths[side+'_z_m'] = depth
                    if semantics:
                        mask=torch_mask(semantics[i]);torch_counts[side]=int(mask.sum())
                        Image.fromarray(mask.astype(np.uint8)*255).save(output/'evaluation'/(side+'_torch_mask.png'))
                        if mask.any():raise ValueError('Torch entered '+view['id']+'/'+side+' view')
                Image.fromarray(np.asarray(rgb_context.get_data()).copy()[..., :3]).save(output/'robot_context.png')
                (output/'evaluation').mkdir(exist_ok=True)
                np.savez_compressed(output/'evaluation/truth.npz', **truths)
                write_json(output/'evaluation/capture.json', {'station': view['station'], 'pose': view['id'],
                           'usd': view['usd'], 'source_usdc_sha256': plan['source_usdc_sha256'],
                           'camera_paths': view['camera_paths'], 'world_from_camera_optical': [w.tolist() for w in world],
                           'actual_arm_q_rad': arm.initial.tolist(), 'base_world': view['base_world'],
                           'camera_rigid_on_torch': True, 'physics_executed': False,
                           'torch_pixels':torch_counts,
                           'cad_depth_scale_ratio':depth_scale_checks,
                           'truth_usage': 'Independent evaluation only'})
                write_json(output/'input.json', {'schema': 'quadweld-stereo-v1', 'synchronized': True, 'views': views,
                           'baseline_m': baseline, 'length_unit': 'm', 'laser_enabled': False,
                           'coordinate_convention': 'OpenCV world-to-camera; world is left optical frame',
                           'calibration_status': 'Simulation pinhole and rigid arm-mounted stereo; no real calibration'})
                if timeline.is_playing(): raise RuntimeError('Timeline advanced during capture')
                print('STATION_STEREO_CAPTURED', view['id'], flush=True)
            finally:
                for annotator, product in annotators: annotator.detach(product)
                for product in products: product.destroy()
        if digest(ROOT/'assets/scenes/industrial-welding/industrial_welding.usdc') != plan['source_usdc_sha256']:
            raise ValueError('Source scene changed')
        write_json(args.output_dir/'capture_status.json', {'completed': True, 'stereo_pairs': len(plan['views'])})
    finally:
        app.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--stations', nargs='+', choices=('A15','A03','A21'), default=['A15','A03','A21'])
    p.add_argument('--collision-model', type=Path, default=ROOT.parent/'robosimstudio/examples/live/twins/industrial_welding/assets/assembled/validation_world.xml')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    args.output_dir = args.output_dir.resolve(); args.collision_model = args.collision_model.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        p.error('Use an empty output directory or --resume')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    plan = json.loads((args.output_dir/'station_plan.json').read_text()) if args.resume and (args.output_dir/'station_plan.json').is_file() else prepare(args)
    if not args.plan_only: capture(args, plan)


if __name__ == '__main__':
    main()
