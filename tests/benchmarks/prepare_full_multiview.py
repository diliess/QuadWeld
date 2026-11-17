"""Fixed dog bases, actual arm IK and overlapping whole-workpiece stereo views."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from tests.benchmarks.seam_metrics import digest, write_json
from tests.benchmarks.prepare_close_station_views import static_check
from weld_scene.arm_kinematics import ArmKinematics
from weld_scene.mount_offset import apply_collision_offset
from weld_scene.optics import load_calibration
from weld_scene.overview_planning import OverviewPlanner, framing, look_at, t_geometry, transform_points


def solve_fixed(arm, pose, initial):
    def residual(q):
        current = arm.camera(q)
        return np.r_[current[:3, 3] - pose[:3, 3],
                     .35 * Rotation.from_matrix(pose[:3, :3] @ current[:3, :3].T).as_rotvec()]
    fit = least_squares(residual, initial, bounds=arm.bounds.T, max_nfev=300,
                        ftol=1e-10, xtol=1e-10, gtol=1e-10)
    return fit.x if np.linalg.norm(residual(fit.x)) < 2e-6 else None


def prepare(source, output):
    previous = json.loads((source / 'station_plan.json').read_text())
    calibration = load_calibration()
    views, attempts = [], []
    for old in previous['views']:
        stage = Usd.Stage.Open(old['usd'])
        planner = OverviewPlanner(stage, previous['collision_model'], calibration)
        apply_collision_offset(planner.model, previous['mount_adjustment'])
        arm = planner.arm
        fixture, bounds, vertices, _ = t_geometry(stage, old['station'])
        world_vertices = transform_points(fixture, vertices)
        target_local = np.array([0., 0., bounds['base'][1, 2] + .5 * np.ptp(bounds['web'][:, 2])])
        target = transform_points(fixture, target_local[None])[0]
        centre = np.asarray(old['world_from_optical_midpoint'])
        local_offset = fixture[:3, :3].T @ (centre[:3, 3] - target)
        selected = None
        # Search angular spread, never move the dog or alter focal length.
        for degrees in (8, 6, 5, 4, 3, 2):
            batch = []
            for label, angle in (('left', -degrees), ('center', 0), ('right', degrees)):
                eye = target + fixture[:3, :3] @ Rotation.from_euler('z', angle, degrees=True).apply(local_offset)
                pose = look_at(eye, target)
                q = arm.initial.copy() if angle == 0 else solve_fixed(arm, pose, arm.initial)
                attempt = {'station': old['station'], 'angle_deg': angle, 'ik_passed': q is not None}
                attempts.append(attempt)
                if q is None:
                    break
                actual_pose = arm.camera(q)
                fraction, boxes = framing(actual_pose, world_vertices, calibration['intrinsics'],
                                          calibration['image_size_wh'], calibration['baseline_m'], .04)
                passed, checks = static_check(planner, np.asarray(old['base_world']), q)
                attempt.update(static_check_passed=passed, full_workpiece_fraction=fraction)
                if not passed or fraction < 1:
                    break
                batch.append({**old, **checks, 'id': old['station'] + '_full_' + label,
                              'patch': label, 'azimuth_delta_deg': angle,
                              'world_from_optical_midpoint': actual_pose.tolist(),
                              'arm_q_rad': q.tolist(), 'arm_q_deg': np.degrees(q).tolist(),
                              'mesh_normalized_bounds_both_eyes': boxes,
                              'complete_workpiece_margin_fraction': .04})
            if len(batch) == 3:
                selected = batch
                break
        if selected is None:
            raise RuntimeError('No fixed-base multi-angle pose set for ' + old['station'])
        for view in selected:
            folder = output / 'captures' / view['id']
            folder.mkdir(parents=True)
            layer = Sdf.Layer.CreateNew(str(folder / 'arm_pose.usda'))
            layer.subLayerPaths = [old['usd']]
            placed = Usd.Stage.Open(layer)
            UsdGeom.SetStageMetersPerUnit(placed, 1.)
            UsdGeom.SetStageUpAxis(placed, 'Z')
            placed.SetDefaultPrim(placed.GetPrimAtPath('/World'))
            placed_arm = ArmKinematics(placed)
            placed_arm.apply(view['arm_q_rad'])
            np.testing.assert_allclose(placed_arm.camera(view['arm_q_rad']), view['world_from_optical_midpoint'], atol=2e-6)
            layer.Save()
            view['usd'] = str(folder / 'arm_pose.usda')
            views.append(view)
        print('FIXED_BASE_MULTIVIEW', old['station'], 'angles', [v['azimuth_delta_deg'] for v in selected], flush=True)
    plan = {**previous, 'views': views, 'baseline_run': str(source),
            'framing_margin_fraction': .04, 'capture_scope': 'fixed_base_full_workpiece_multiview',
            'view_selection': 'Fixed prior dog bases and camera distance; arm-only IK with overlapping complete stereo framing',
            'source_hashes': {**previous['source_hashes'], str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}}
    write_json(output / 'station_plan.json', plan)
    write_json(output / 'pose_attempts.json', attempts)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=ROOT / 'outputs/benchmarks/20261110_01__stereo__multi3__framing')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Use an empty output directory')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    prepare(a.source_run.resolve(), a.output_dir.resolve())
