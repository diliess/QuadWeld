"""Observation-only checks using independently constructed synthetic RGB-D scenes."""

import unittest
from dataclasses import replace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from vision.contour_detector import detect_path
from vision.rgbd import RGBDCalibration, generate_rgbd_trajectory


class RGBDTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.calibration = RGBDCalibration((320, 240), np.array([[300., 0, 160], [0, 300, 120], [0, 0, 1]]),
                                           np.zeros(5), 1., np.eye(4))
        self.rgb = np.full((240, 320, 3), 220, np.uint8)
        cv2.rectangle(self.rgb, (80, 65), (240, 175), (10, 10, 10), 4)

    def depth(self, distance=1., normal=(0, 0, -1)):
        v, u = np.indices((240, 320))
        rays = np.stack(((u - 160) / 300, (v - 120) / 300, np.ones_like(u)), axis=-1)
        return distance * normal[2] / (rays @ np.asarray(normal))

    def test_depth_changes_height_without_any_workpiece_pose_input(self):
        for distance in (.7, 1.2):
            result = generate_rgbd_trajectory(self.rgb, self.depth(distance), self.calibration)
            np.testing.assert_allclose(result.contour_base[:, 2], distance, atol=1e-8)
            np.testing.assert_allclose(result.path.surface_points[0], result.path.surface_points[-1])
            trace = result.path.poses[np.array(result.path.phases) == "trace", :3, 3]
            np.testing.assert_allclose(trace[:, 2], distance - .01, atol=1e-8)
            self.assertLessEqual(np.linalg.norm(np.diff(result.path.surface_points, axis=0), axis=1).max(), .010001)

    def test_tilted_plane_survives_noise_holes_and_outliers(self):
        normal = np.array([.15, -.3, -1.])
        normal /= np.linalg.norm(normal)
        depth = self.depth(.9, normal)
        rng = np.random.default_rng(12)
        depth += rng.normal(0, .0007, depth.shape)
        depth[rng.random(depth.shape) < .15] = 0
        depth[rng.random(depth.shape) < .08] = 2.
        result = generate_rgbd_trajectory(self.rgb, depth, self.calibration)
        np.testing.assert_allclose(result.surface.normal, normal, atol=.002)
        residual = (result.contour_base - [0, 0, .9]) @ normal
        self.assertLess(np.abs(residual).max(), .0004)
        np.testing.assert_allclose(result.path.poses[0, :3, 2], -normal, atol=.002)

    def test_calibration_maps_camera_path_to_base(self):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [.3, -.2, .4]).as_matrix()
        transform[:3, 3] = [.2, -.1, .6]
        result = generate_rgbd_trajectory(self.rgb, self.depth(), replace(self.calibration, base_from_camera=transform))
        np.testing.assert_allclose(result.contour_base,
                                   result.contour_camera @ transform[:3, :3].T + transform[:3, 3])
        np.testing.assert_allclose(result.surface.normal, transform[:3, :3] @ [0, 0, -1], atol=1e-8)

    def test_depth_units_are_explicit(self):
        result = generate_rgbd_trajectory(self.rgb, np.full((240, 320), 850, np.uint16),
                                          replace(self.calibration, depth_scale_to_m=.001))
        np.testing.assert_allclose(result.contour_base[:, 2], .85, atol=1e-8)

    def test_missing_or_unregistered_depth_never_falls_back(self):
        for value in (0., np.nan, np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "no scene-coordinate fallback"):
                generate_rgbd_trajectory(self.rgb, np.full((240, 320), value), self.calibration)
        with self.assertRaisesRegex(ValueError, "match"):
            generate_rgbd_trajectory(self.rgb, np.ones((120, 160)), self.calibration)

    def test_nonplanar_depth_is_rejected(self):
        depth = np.random.default_rng(3).uniform(.5, 1.5, (240, 320))
        with self.assertRaisesRegex(RuntimeError, "dominant planar"):
            generate_rgbd_trajectory(self.rgb, depth, self.calibration)

    def test_image_translation_changes_path(self):
        before = generate_rgbd_trajectory(self.rgb, self.depth(), self.calibration)
        shifted = cv2.warpAffine(self.rgb, np.array([[1., 0, 25], [0, 1, 0]]), (320, 240), borderValue=(220, 220, 220))
        after = generate_rgbd_trajectory(shifted, self.depth(), self.calibration)
        np.testing.assert_allclose(after.contour_base.mean(axis=0) - before.contour_base.mean(axis=0),
                                   [25 / 300, 0, 0], atol=1e-8)

    def test_observed_surface_distinguishes_clearance_from_penetration(self):
        result = generate_rgbd_trajectory(self.rgb, self.depth(), self.calibration)
        self.assertFalse(result.surface.intersects(np.array([[0., 0, .98]]), .006))
        self.assertTrue(result.surface.intersects(np.array([[0., 0, 1.01]]), .006))
        self.assertFalse(result.surface.intersects(np.array([[2., 0, 1.01]]), .006))

    def test_open_stroke_traces_endpoint_to_endpoint_without_closing(self):
        image = np.full_like(self.rgb, 220)
        cv2.polylines(image, [np.array([(60, 50), (60, 170), (240, 170), (240, 50)])],
                      False, (10, 10, 10), 5)
        result = generate_rgbd_trajectory(image, self.depth(), self.calibration)
        self.assertFalse(result.closed)
        self.assertEqual(result.diagnostics["path_type"], "open")
        np.testing.assert_allclose(result.contour_pixel[[0, -1]], [[60, 50], [240, 50]], atol=2)
        np.testing.assert_allclose(result.path.surface_points[[0, -1]], result.contour_base[[0, -1]])
        self.assertGreater(np.linalg.norm(result.path.surface_points[0] - result.path.surface_points[-1]), .5)
        self.assertLessEqual(np.linalg.norm(np.diff(result.path.surface_points, axis=0), axis=1).max(), .010001)
        normal = result.surface.normal
        np.testing.assert_allclose(result.path.poses[-1, :3, 3],
                                   result.path.surface_points[-1] + .1 * normal)
        trace = result.path.poses[np.array(result.path.phases) == "trace", :3, 3]
        np.testing.assert_allclose(trace[-1], result.path.surface_points[-1] + .01 * normal)

    def test_branching_open_stroke_is_rejected(self):
        image = np.full_like(self.rgb, 220)
        cv2.line(image, (60, 80), (240, 80), (10, 10, 10), 5)
        cv2.line(image, (150, 80), (150, 180), (10, 10, 10), 5)
        with self.assertRaisesRegex(RuntimeError, "No unbranched open"):
            detect_path(image, mode="open")

    def test_closed_path_still_returns_to_its_start(self):
        result = generate_rgbd_trajectory(self.rgb, self.depth(), self.calibration)
        self.assertTrue(result.closed)
        self.assertEqual(result.diagnostics["path_type"], "closed")
        np.testing.assert_allclose(result.path.surface_points[0], result.path.surface_points[-1])


if __name__ == "__main__":
    unittest.main()
