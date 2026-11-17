"""Mounted-camera joint sampling and independently defined three-seam references."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf, Usd, UsdGeom

from tests.a04_reference import contact_segments, reference_from_stage
from weld_scene.arm_kinematics import ArmKinematics
from tests.benchmarks.seam_metrics import per_seam_metrics, project


class ThreeContactTests(unittest.TestCase):
    def test_three_finite_contacts_share_the_real_inner_junction(self):
        bounds = {'base': [[-.55, -.3, .338], [.55, .3, .348]],
                  'back': [[-.29, .145, .348], [.29, .155, .748]],
                  'return': [[-.29, -.205, .348], [-.28, .145, .748]]}
        curves = contact_segments(bounds)
        self.assertEqual(len(curves), 3)
        for curve in curves.values():
            np.testing.assert_allclose(curve[0], [-.28, .145, .348])
        np.testing.assert_allclose([np.linalg.norm(v[1]-v[0]) for v in curves.values()], [.4, .57, .35])
        bounds['return'][1][1] = .13
        with self.assertRaises(ValueError):
            contact_segments(bounds)

    def test_reference_tracks_fixture_transform(self):
        stage = Usd.Stage.Open('assets/scenes/industrial-welding/industrial_welding.usdc')
        old = reference_from_stage(stage)
        fixture = stage.GetPrimAtPath('/World/Fixtures/A04')
        previous = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(fixture)).T
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('z', .7).as_matrix()
        transform[:3, 3] = [10, -3, 1]
        stage.SetEditTarget(stage.GetSessionLayer())
        UsdGeom.Xformable(fixture).MakeMatrixXform().Set(Gf.Matrix4d(transform.T.tolist()))
        new = reference_from_stage(stage)
        delta = transform @ np.linalg.inv(previous)
        for a, b in zip(old, new):
            expected = (np.c_[a['control_world_m'], np.ones(2)] @ delta.T)[:, :3]
            np.testing.assert_allclose(b['control_world_m'], expected, atol=1e-7)

    def test_missing_one_of_three_welds_reduces_its_coverage(self):
        origin = np.array([0., 0., 1.])
        k = np.array([[10., 0, 10], [0, 10., 10], [0, 0, 1.]])
        curves = []
        for index, end in enumerate(([.3, 0, 1], [0, .3, 1], [0, 0, 1.3])):
            xyz = np.linspace(origin, end, 51)
            curves.append({'name': str(index), 'xyz': xyz, 'xy': project(xyz, k),
                           'in_frame': np.ones(51, bool), 'visible': np.ones(51, bool), 'arc': np.linspace(0, .3, 51)})
        metrics = per_seam_metrics(curves, [c['xyz'] for c in curves], np.ones((32, 32)), k)
        self.assertTrue(all(m['coverage_at_10mm']==1. for m in metrics))
        missing = per_seam_metrics(curves, [c['xyz'] for c in curves[:2]], np.ones((32, 32)), k)
        self.assertLess(missing[2]['coverage_at_10mm'], .1)
        self.assertIsNone(missing[2]['position_error'])


class MountedArmTests(unittest.TestCase):
    def test_fk_matches_authored_arm_poses(self):
        import json
        from pathlib import Path
        scene = json.loads(Path('assets/scenes/industrial-welding/manifest.json').read_text())['full_scene']
        for file in {pose['asset'] for pose in scene['poses'].values()}:
            with self.subTest(file=file):
                stage = Usd.Stage.Open('assets/scenes/industrial-welding/'+file)
                arm = ArmKinematics(stage)
                actual = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(
                    '/World/Robot/e05l_welding_torch/hj90'))).T @ np.diag([1., -1, -1, 1])
                np.testing.assert_allclose(arm.camera(arm.initial), actual, atol=2e-6)

    def test_joint_pose_changes_arm_and_camera_while_base_and_mount_stay_fixed(self):
        stage = Usd.Stage.Open('assets/scenes/industrial-welding/industrial_welding.usdc')
        stage.SetEditTarget(stage.GetSessionLayer())
        arm = ArmKinematics(stage)
        base_before = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot/base_link')))
        q = arm.initial.copy(); q[0] += np.radians(8)
        arm.apply(q)
        cache = UsdGeom.XformCache()
        base_after = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot/base_link')))
        np.testing.assert_array_equal(base_before, base_after)
        torch = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot/e05l_welding_torch'))).T
        camera = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Robot/e05l_welding_torch/hj90'))).T @ np.diag([1., -1, -1, 1])
        np.testing.assert_allclose(np.linalg.inv(torch) @ camera, arm.torch_to_camera, atol=1e-9)
        angle = Rotation.from_matrix(camera[:3, :3] @ arm.camera(arm.initial)[:3, :3].T).magnitude()
        self.assertAlmostEqual(np.degrees(angle), 8., places=5)
        q[1] = 10
        with self.assertRaises(ValueError):
            arm.apply(q)


if __name__ == '__main__':
    unittest.main()
