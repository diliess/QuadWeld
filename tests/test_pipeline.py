import unittest
from dataclasses import replace

import cv2
import numpy as np

from config import SCENE
from planning.waypoint_generator import generate_waypoints
from robot.kinematics import SerialKinematics, pose_error
from robot.robot_controller import distance_to_polyline
from vision.contour_detector import detect_contour
from vision.rgbd import RGBDCalibration, generate_rgbd_trajectory


class VisionPlanningTests(unittest.TestCase):
    def test_image_changes_produce_different_paths(self):
        camera = RGBDCalibration((640, 480), np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]]),
                                  np.zeros(5), 1., np.diag([1., -1., -1., 1.]))
        centers = []
        for offset, angle in ((0, 0), (80, 23)):
            image = np.full((480, 640, 3), 200, np.uint8)
            corners = cv2.boxPoints(((270 + offset, 210), (180, 110), angle)).astype(np.int32)
            cv2.polylines(image, [corners], True, (10, 10, 10), 4)
            result = generate_rgbd_trajectory(image, np.ones((480, 640)), camera)
            contour_base, path = result.contour_base, result.path
            np.testing.assert_allclose(path.surface_points[0], path.surface_points[-1])
            self.assertLessEqual(np.linalg.norm(np.diff(path.surface_points, axis=0), axis=1).max(), 0.010001)
            np.testing.assert_allclose(path.poses[0, :3, 3], path.poses[-1, :3, 3])
            self.assertAlmostEqual(path.poses[0, 2, 3], -0.9)
            self.assertAlmostEqual(path.poses[path.phases.index('trace'), 2, 3], -0.99)
            np.testing.assert_allclose(path.poses[0, :3, 2], [0, 0, -1])
            # Base XY is counterclockwise because this installation inverts image v.
            area2 = np.sum(contour_base[:, 0] * np.roll(contour_base[:, 1], -1)
                           - contour_base[:, 1] * np.roll(contour_base[:, 0], -1))
            self.assertLess(area2, 0)
            centers.append(contour_base.mean(axis=0))
        self.assertGreater(centers[1][0] - centers[0][0], 0.1)

    def test_empty_and_open_outline_are_rejected(self):
        image = np.full((480, 640, 3), 200, np.uint8)
        with self.assertRaises(RuntimeError):
            detect_contour(image)
        cv2.line(image, (100, 100), (300, 100), (0, 0, 0), 5)
        with self.assertRaises(RuntimeError):
            detect_contour(image)

    def test_calibrated_rays_undo_image_distortion(self):
        camera = RGBDCalibration((640, 480), np.array([[600., 0, 320], [0, 590, 240], [0, 0, 1]]),
                                  np.array([.03, -.01, .002, -.003, 0]), 1., np.eye(4))
        points = np.array([[-.2, -.1, .7], [0, 0, 1.], [.3, .15, 1.2]])
        pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), camera.camera_matrix,
                                     camera.distortion_coefficients)
        np.testing.assert_allclose(camera.rays(pixels), points / points[:, 2, None], atol=1e-8)

    def test_invalid_clearance_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_waypoints([[0, 0, 0], [1, 0, 0], [0, 1, 0]], replace(SCENE, approach_height_m=0.005))


class KinematicsTests(unittest.TestCase):
    def test_distance_to_closed_path_includes_closing_edge(self):
        path = np.array([[0., 0., 0.], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
        distances = distance_to_polyline([[0, 0.5, 0], [1.1, 0.5, 0], [0.5, 0, 0.01]], path)
        np.testing.assert_allclose(distances, [0, 0.1, 0.01])

    def test_fk_ik_known_chain_and_unreachable_target(self):
        frames = np.repeat(np.eye(4)[None], 2, axis=0)
        frames[1, 0, 3] = 0.4
        tcp = np.eye(4)
        tcp[0, 3] = 0.3
        model = SerialKinematics(['a', 'b'], frames, np.repeat(np.eye(4)[None], 2, axis=0),
                                 np.array([[0, 0, 1.], [0, 0, 1.]]),
                                 np.array([[-3.14, 3.14], [-3.14, 3.14]]), np.eye(4), tcp)
        np.testing.assert_allclose(model.forward([0, 0])[:3, 3], [0.7, 0, 0])
        target = model.forward([0.8, -0.5])
        solution = model.inverse(target, [0.5, -0.2])
        self.assertLess(np.linalg.norm(pose_error(model.forward(solution), target)), 1e-5)
        target[0, 3] = 10
        with self.assertRaises(RuntimeError):
            model.inverse(target, solution)


if __name__ == '__main__':
    unittest.main()
