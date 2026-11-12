"""Place the dog and mounted stereo cameras so both eyes frame whole T workpieces."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_stereo_seams import digest, write_json
from prepare_close_station_views import solve_reach, static_check


def prepare(args):
    import mujoco as mj
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom
    from hj90.arm_kinematics import ArmKinematics
    from hj90.full_framing import fit_stereo_distance
    from hj90.mount_offset import apply_collision_offset, apply_usd_offset, torch_frustum_audit
    from hj90.optics import load_calibration
    from hj90.overview_planning import OverviewPlanner, look_at, t_geometry, transform_points, world_matrix

    previous = json.loads((args.baseline_run / 'station_plan.json').read_text())
    stage = Usd.Stage.Open(previous['source_usd'])
    stage.SetEditTarget(stage.GetSessionLayer())
    mount = apply_usd_offset(stage, previous['mount_adjustment']['lateral_offset_m'])
    calibration = load_calibration()
    planner = OverviewPlanner(stage, previous['collision_model'], calibration)
    apply_collision_offset(planner.model, mount)
    records = []
    for old in previous['views']:
        station = old['station']
        fixture, bounds, vertices, surfaces = t_geometry(stage, station)
        target = np.array([0., 0., bounds['base'][1, 2] + .5 * np.ptp(bounds['web'][:, 2])])
        world_surfaces = transform_points(fixture, surfaces)
        chosen, attempts = None, []
        for azimuth in (0, -15, 15):
            if chosen:
                break
            for elevation in (35, 30, 25, 20, 15, 10, 5):
                if chosen:
                    break
                distance, pose, boxes = fit_stereo_distance(fixture, target, vertices,
                    calibration['intrinsics'], calibration['image_size_wh'], elevation, azimuth,
                    calibration['baseline_m'], args.margin)
                print('FULL_FRAME_TRY', station, 'distance', distance, 'elevation', elevation, flush=True)
                for yaw_delta in (0, -20, 20):
                    if chosen:
                        break
                    yaw = np.arctan2(pose[1, 2], pose[0, 2]) + np.radians(yaw_delta)
                    for reach in (None, .45, .6):
                        solution = planner.solve(pose, yaw) if reach is None else solve_reach(planner, pose, yaw, reach)
                        attempt = {'elevation_deg': elevation, 'azimuth_deg': azimuth,
                                   'camera_to_target_m': distance, 'yaw_delta_deg': yaw_delta,
                                   'requested_arm_reach_m': reach, 'ik_passed': solution is not None}
                        attempts.append(attempt)
                        if solution is None:
                            continue
                        base, q = solution
                        passed, check = static_check(planner, base, q)
                        attempt['static_check_passed'] = passed
                        if not passed:
                            continue
                        visibility = []
                        for sign in (-1, 1):
                            eye = pose[:3, 3] + sign * calibration['baseline_m'] / 2 * pose[:3, 0]
                            count = 0
                            for point in world_surfaces:
                                ray = point - eye
                                ray_distance = np.linalg.norm(ray)
                                hit = mj.mj_ray(planner.model, planner.data, eye, ray / ray_distance,
                                                None, 1, -1, np.array([-1], np.int32))
                                count += hit >= 0 and abs(hit - ray_distance) < .003
                            visibility.append(count / len(world_surfaces))
                        attempt['front_surface_visibility_both_eyes'] = visibility
                        if min(visibility) < .9:
                            continue
                        local_base = transform_points(np.linalg.inv(fixture), base[:3, 3][None])[0]
                        closest_xy = np.clip(local_base[:2], bounds['base'][0, :2], bounds['base'][1, :2])
                        gap = float(np.linalg.norm(local_base[:2] - closest_xy))
                        chosen = {'id': station + '_full_stereo', 'station': station,
                                  'azimuth_deg': azimuth, 'elevation_deg': elevation,
                                  'camera_to_workpiece_center_m': distance,
                                  'robot_base_center_to_workpiece_xy_m': gap,
                                  'world_from_optical_midpoint': pose.tolist(), 'base_world': base.tolist(),
                                  'arm_q_rad': q.tolist(), 'arm_q_deg': np.degrees(q).tolist(),
                                  'mesh_vertices_in_frame_both_eyes': 1.,
                                  'mesh_normalized_bounds_both_eyes': boxes,
                                  'front_surface_ray_visibility_both_eyes': visibility,
                                  'complete_workpiece_margin_fraction': args.margin, **check}
                        break
        write_json(args.output_dir / (station + '_pose_attempts.json'), attempts)
        if chosen is None:
            raise ValueError('No supported full-frame pose for ' + station)
        folder = args.output_dir / 'captures' / chosen['id']
        folder.mkdir(parents=True)
        layer = Sdf.Layer.CreateNew(str(folder / 'arm_pose.usda'))
        layer.subLayerPaths = [old['usd']]
        placed = Usd.Stage.Open(layer)
        UsdGeom.SetStageMetersPerUnit(placed, 1.)
        UsdGeom.SetStageUpAxis(placed, 'Z')
        placed.SetDefaultPrim(placed.GetPrimAtPath('/World'))
        cache = UsdGeom.XformCache()
        current_base = world_matrix(cache, placed.GetPrimAtPath('/World/Robot/base_link'))
        current_robot = world_matrix(cache, placed.GetPrimAtPath('/World/Robot'))
        delta = np.asarray(chosen['base_world']) @ np.linalg.inv(current_base)
        UsdGeom.Xformable(placed.GetPrimAtPath('/World/Robot')).MakeMatrixXform().Set(Gf.Matrix4d((delta @ current_robot).T.tolist()))
        arm = ArmKinematics(placed)
        arm.apply(chosen['arm_q_rad'])
        np.testing.assert_allclose(arm.camera(chosen['arm_q_rad']), chosen['world_from_optical_midpoint'], atol=2e-5)
        target_world = transform_points(fixture, target[None])[0]
        context_target = (np.asarray(chosen['base_world'])[:3, 3] + target_world) / 2
        context_eye = context_target + fixture[:3, :3] @ np.array([3.8, -5., 3.])
        context_pose = look_at(context_eye, context_target) @ np.diag([1., -1., -1., 1.])
        context = UsdGeom.Camera(placed.GetPrimAtPath(old['context_camera']))
        context.MakeMatrixXform().Set(Gf.Matrix4d(context_pose.T.tolist()))
        chosen.update(usd=str(folder / 'arm_pose.usda'), camera_paths=old['camera_paths'],
                      context_camera=old['context_camera'])
        layer.Save()
        records.append(chosen)
        print('FULL_FRAME_PLANNED', station, 'camera_distance', chosen['camera_to_workpiece_center_m'],
              'robot_center_gap', chosen['robot_base_center_to_workpiece_xy_m'], flush=True)
    audit = torch_frustum_audit(stage, calibration)
    if any(row['triangles_potentially_intersecting_frustum'] for row in audit):
        raise ValueError('Torch intersects a stereo view')
    plan = {**previous, 'views': records, 'baseline_run': str(args.baseline_run),
            'capture_scope': 'complete_workpiece_in_each_eye', 'framing_margin_fraction': args.margin,
            'mount_adjustment': mount, 'torch_frustum_audit': audit, 'require_no_torch_pixels': True,
            'view_selection': 'Closest complete stereo framing at each angle, first feasible supported robot pose; fixed intrinsics and physical baseline',
            'source_hashes': {name: digest(ROOT / name) for name in ('scripts/prepare_full_station_views.py',
                'hj90/full_framing.py', 'hj90/overview_planning.py', 'hj90/arm_kinematics.py', 'hj90/mount_offset.py')},
            'physics_executed': False, 'scope': 'Static simulated robot placements; no walking or arm trajectory executed'}
    write_json(args.output_dir / 'station_plan.json', plan)
    print('FULL_FRAME_PLAN_COMPLETE', args.output_dir, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run', type=Path, default=ROOT / 'outputs/benchmarks/t_station_unobstructed_20261110_01')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--margin', type=float, default=.08)
    a = p.parse_args()
    a.baseline_run, a.output_dir = a.baseline_run.resolve(), a.output_dir.resolve()
    if not 0 < a.margin < .25:
        p.error('Margin must be between 0 and 0.25')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Use an empty output directory')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    prepare(a)
