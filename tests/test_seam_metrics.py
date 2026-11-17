"""Spatial seam metrics use finite paths and preserve pixel-centre conventions."""

import unittest

import numpy as np

from tests.benchmarks.seam_metrics import backproject, error_stats, project, runs, sample_depth, segment_distances


class SeamMetricTests(unittest.TestCase):
    def test_finite_segments_do_not_extend_endpoints_or_bridge_gaps(self):
        curves = [np.array([[0., 0, 0], [1, 0, 0]]), np.array([[3., 0, 0], [4, 0, 0]])]
        points = np.array([[.5, .003, .004], [-2, 0, 0], [2, 0, 0], [5, 0, 0]])
        errors, nearest = segment_distances(points, curves)
        np.testing.assert_allclose(errors, [.005, 2, 1, 1])
        np.testing.assert_allclose(nearest, [[.5, 0, 0], [0, 0, 0], [1, 0, 0], [4, 0, 0]])
        self.assertEqual(runs(np.array([True, True, False, True, True, False, True])), [(0, 2), (3, 5)])
        self.assertTrue(np.isinf(segment_distances(points, [])[0]).all())

    def test_pixel_centre_projection_and_subpixel_depth(self):
        k_native = np.array([[350., 0, 147], [0, 352, 252], [0, 0, 1]])
        k_index = k_native.copy()
        k_index[:2, 2] -= .5
        xyz = np.array([[.03, -.05, .6], [-.02, .07, .9]])
        xy = project(xyz, k_index)
        np.testing.assert_allclose(xy+.5, project(xyz, k_native))
        np.testing.assert_allclose(backproject(xy, xyz[:, 2], k_index), xyz, atol=1e-15)
        yy, xx = np.indices((5, 6))
        depth = .5+xx*.01+yy*.02
        samples = sample_depth(depth, np.array([[1.25, 2.5], [4.5, 3.25]]))
        np.testing.assert_allclose(samples, [.5625, .61])

    def test_statistics_distinguish_missing_outputs_from_zero_error(self):
        self.assertIsNone(error_stats([]))
        result = error_stats(np.array([.003, .004]))
        self.assertAlmostEqual(result['mean_mm'], 3.5)
        self.assertAlmostEqual(result['rmse_mm'], np.sqrt(12.5))
        with self.assertRaises(ValueError):
            error_stats([np.inf])


if __name__ == '__main__':
    unittest.main()
