"""Analytic checks for the offline DA3 geometry experiment (no model/GPU)."""

import unittest

import numpy as np

from scripts.analyze_weld_geometry import (
    backproject, fit_plane, project_intersection, scale_pixels, surface_normals, trace_path,
)


class WeldGeometryTests(unittest.TestCase):
    def test_plane_intersection_projects_known_3d_line(self):
        intrinsics = np.array([[420., 0, 252], [0, 420, 189], [0, 0, 1]])
        # Planes y=.1 and z=.7 meet along a horizontal line.
        plane_a = np.array([0., 1, 0, -.1])
        plane_b = np.array([0., 0, 1, -.7])
        line, point, direction = project_intersection(plane_a, plane_b, intrinsics)
        xyz = np.array([[-.2, .1, .7], [0, .1, .7], [.3, .1, .7]])
        uv = xyz @ intrinsics.T
        uv /= uv[:, 2:]
        np.testing.assert_allclose(uv @ line, 0, atol=1e-10)
        np.testing.assert_allclose(np.stack((plane_a[:3], plane_b[:3])) @ point,
                                   [.1, .7], atol=1e-12)
        self.assertAlmostEqual(abs(direction[0]), 1.)
        with self.assertRaises(ValueError):
            project_intersection(plane_a, plane_a + [0, 0, 0, .1], intrinsics)

    def test_ransac_recovers_tilted_plate_with_outliers(self):
        rng = np.random.default_rng(42)
        xy = rng.uniform(-.2, .2, (900, 2))
        z = .7 + .25 * xy[:, 0] - .4 * xy[:, 1] + rng.normal(0, .0003, len(xy))
        points = np.column_stack((xy, z))
        points = np.vstack((points, rng.uniform([-.2, -.2, .8], [.2, .2, 1.2], (200, 3))))
        plane, report = fit_plane(points)
        expected = np.array([.25, -.4, -1.])
        expected /= np.linalg.norm(expected)
        self.assertGreater(abs(plane[:3] @ expected), .9999)
        self.assertLess(abs(plane[:3] @ [0, 0, .7] + plane[3]), .0001)
        self.assertGreater(report["inlier_fraction"], .80)
        self.assertLess(report["inlier_fraction"], .85)

    def test_backprojection_and_normals_of_front_plane(self):
        depth = np.full((40, 60), .8, dtype=np.float32)
        intrinsics = np.array([[100., 0, 30], [0, 100, 20], [0, 0, 1]])
        xyz = backproject(depth, intrinsics)
        np.testing.assert_allclose(xyz[20, 30], [0, 0, .8], atol=1e-7)
        normals = surface_normals(depth, intrinsics)
        np.testing.assert_allclose(normals[..., 2], -1, atol=1e-6)
        np.testing.assert_allclose(normals[..., :2], 0, atol=1e-6)

    def test_continuous_path_ignores_isolated_strong_distractor(self):
        scores = np.zeros((24, 40))
        scores[10, :] = 1
        scores[2, 20] = 5
        path = trace_path(scores, np.ones_like(scores, dtype=bool))
        np.testing.assert_array_equal(path, 10)
        invalid = np.ones_like(scores, dtype=bool)
        invalid[:, 20] = False
        with self.assertRaises(ValueError):
            trace_path(scores, invalid)

    def test_original_processed_coordinates_round_trip(self):
        points = np.array([[0., 0], [860, 1156], [1799, 2399]])
        scale = np.array([378 / 1800, 504 / 2400])
        np.testing.assert_allclose(scale_pixels(scale_pixels(points, scale), 1 / scale), points,
                                   atol=1e-12)


if __name__ == "__main__":
    unittest.main()
