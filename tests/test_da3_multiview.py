"""Camera-only scale and full-resolution projection for joint view inference."""

from pathlib import Path
import sys
import unittest

import numpy as np

from da3_multiview import camera_centres, camera_distance_scale

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_joint_station_views import projected_segments


def poses(centres):
    result = np.repeat(np.eye(4)[None], len(centres), axis=0)
    result[:, :3, 3] = -np.asarray(centres)
    return result


class JointViewGeometryTests(unittest.TestCase):
    def test_collinear_camera_scale_is_rigid_transform_invariant(self):
        centres = np.array([[x, 0, 0] for x in (0, .09, .3, .39, .6, .69)])
        calibrated = poses(centres)
        angle = .8
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        transformed = (centres / .4) @ rotation.T + [4, -2, 3]
        factor, record = camera_distance_scale(poses(transformed), calibrated)
        self.assertAlmostEqual(factor, .4)
        self.assertLess(record['camera_distance_residual_rmse_m'], 1e-12)
        self.assertEqual(len(record['view_pairs']), 15)

    def test_collapsed_trajectory_and_improper_rotations_fail(self):
        with self.assertRaisesRegex(ValueError, 'Collapsed'):
            camera_distance_scale(poses([[0, 0, 0], [0, 0, 0]]), poses([[0, 0, 0], [.09, 0, 0]]))
        invalid = poses([[0, 0, 0], [.09, 0, 0]])
        invalid[0, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'orthonormal'):
            camera_centres(invalid)

    def test_native_pixel_centres_and_excluded_gaps(self):
        k = np.array([[100, 0, 20.5], [0, 100, 30.5], [0, 0, 1.]])
        points = np.array([[0, 0, 1], [.1, 0, 1], [.2, 0, 1], [.3, 0, 1], [.4, 0, 1]])
        projected = projected_segments(points, np.array([True, True, False, True, True]), k)
        self.assertEqual(len(projected), 2)
        np.testing.assert_allclose(projected[0], [[20, 30], [30, 30]])
        np.testing.assert_allclose(projected[1], [[50, 30], [60, 30]])


if __name__ == '__main__':
    unittest.main()
