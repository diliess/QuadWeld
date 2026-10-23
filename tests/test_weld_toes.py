"""Analytic toe geometry and rejection tests, independent of pretrained models."""

import unittest

import numpy as np

from vision.weld_detector import normals_from_depth
from vision.weld_toes import extract_weld_toes
from vision.weld_surfaces import continuous_support, local_host_surface


class WeldToeTests(unittest.TestCase):
    def setUp(self):
        self.k = np.array([[150., 0, 80], [0, 150, 60], [0, 0, 1]])
        self.mask = np.zeros((120, 160), dtype=bool)
        self.mask[50:71, 15:145] = True
        self.curve = np.column_stack((np.arange(20, 140), np.full(120, 60)))
        self.rgb = np.full((120, 160, 3), 90, dtype=np.uint8)
        self.rgb[self.mask] = 190

    def run_toes(self, depth, mask=None):
        return extract_weld_toes(self.mask if mask is None else mask, self.curve, depth, self.k,
                                 self.rgb, normals_from_depth(depth, self.k))

    def test_two_open_sides_follow_known_bead_toes(self):
        yy, _ = np.indices(self.mask.shape)
        profile = np.maximum(0, 1-((yy-60)/10.)**2)
        depth = (1-.015*profile*self.mask).astype(np.float32)
        result = self.run_toes(depth)
        self.assertEqual(result["status"], "paired_candidates")
        self.assertEqual(len(result["segments"]), 1)
        pair = result["segments"][0]
        self.assertEqual(len(pair["side_a"]["xy"]), len(pair["side_b"]["xy"]))
        self.assertLess(np.abs(pair["side_a"]["xy"][:, 1]-49.5).mean(), 2.5)
        self.assertLess(np.abs(pair["side_b"]["xy"][:, 1]-70.5).mean(), 2.5)
        self.assertTrue((pair["width_processed_px"] > 10).all())
        for key in ("side_a", "side_b"):
            side = pair[key]
            self.assertGreater(np.linalg.norm(side["xy"][-1]-side["xy"][0]), 100)
            self.assertGreater(side["depth_supported"].mean(), .5)
            self.assertGreater(side["geometry_supported"].mean(), .5)
            np.testing.assert_array_equal(side["geometry_supported"],
                                          side["depth_supported"] & side["surface_valid"] & side["depth_quality_valid"] & side["surface_continuity_valid"])
            self.assertTrue(np.isfinite(side["xyz_camera_estimated_m"]).all())
            projected = side["xyz_camera_estimated_m"] @ self.k.T
            projected = projected[:, :2] / projected[:, 2:]
            np.testing.assert_allclose(projected, side["xy"], atol=1e-5)

    def test_flat_depth_does_not_claim_toe_geometry_support(self):
        result = self.run_toes(np.ones(self.mask.shape, dtype=np.float32))
        self.assertEqual(len(result["segments"]), 1)
        for key in ("side_a", "side_b"):
            self.assertFalse(result["segments"][0][key]["depth_supported"].any())
            self.assertFalse(result["segments"][0][key]["geometry_supported"].any())

    def test_missing_bead_section_splits_both_sides(self):
        mask = self.mask.copy()
        mask[:, 75:85] = False
        result = self.run_toes(np.ones(mask.shape, dtype=np.float32), mask)
        self.assertEqual(len(result["segments"]), 2)
        first, second = result["segments"]
        for key in ("side_a", "side_b"):
            self.assertLess(first[key]["xy"][:, 0].max(), 75)
            self.assertGreater(second[key]["xy"][:, 0].min(), 84)

    def test_narrow_mask_has_no_fabricated_two_side_path(self):
        mask = np.zeros_like(self.mask)
        mask[60, 15:145] = True
        result = self.run_toes(np.ones(mask.shape, dtype=np.float32), mask)
        self.assertEqual(result["segments"], [])

    def test_invalid_depth_rejected(self):
        depth = np.ones(self.mask.shape, dtype=np.float32)
        depth[60, 60] = np.nan
        with self.assertRaises(ValueError):
            self.run_toes(depth)

    def test_curved_host_uses_validated_quadratic_instead_of_forcing_plane(self):
        yy, xx = np.indices(self.mask.shape)
        inverse_z = 1 + .0005*(yy-60.)**2 + .00001*(xx-80.)**2
        depth = (1/inverse_z).astype(np.float32)
        centers = self.curve.astype(float)
        across = np.tile([0., 1.], (len(centers), 1))
        curve = centers + [0., -10.]
        result = local_host_surface(curve, centers, across, np.full(len(curve), -10.), -1,
                                    depth, self.k, normals_from_depth(depth, self.k), np.ones(self.mask.shape, bool))
        self.assertGreater((result["kind"] == 3).mean(), .8)
        expected_z = 1/(1+.0005*(curve[:,1]-60)**2 + .00001*(curve[:,0]-80)**2)
        np.testing.assert_allclose(result["xyz"][:,2], expected_z, atol=1e-4)

    def test_low_quality_depth_retains_candidates_without_geometry_claim(self):
        depth = np.ones(self.mask.shape, dtype=np.float32)
        result = extract_weld_toes(self.mask, self.curve, depth, self.k, self.rgb,
                                   normals_from_depth(depth, self.k), quality_mask=np.zeros_like(self.mask))
        self.assertTrue(result["segments"])
        for name in ("side_a", "side_b"):
            side = result["segments"][0][name]
            self.assertFalse(side["surface_valid"].any())
            self.assertFalse(side["geometry_supported"].any())
            self.assertTrue(np.isfinite(side["surface_xyz_camera_estimated_m"]).all())

    def test_abrupt_surface_switch_is_not_a_continuous_3d_path(self):
        points = np.column_stack((np.arange(30)*.001, np.zeros(30), np.ones(30)))
        points[15:, 2] += .1
        supported = continuous_support(points, np.ones(30, bool))
        self.assertFalse(supported[14])
        self.assertFalse(supported[15])
        self.assertTrue(supported[:14].all())
        self.assertTrue(supported[16:].all())


if __name__ == "__main__":
    unittest.main()
