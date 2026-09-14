import unittest
from dataclasses import replace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from config import SCENE
from planning.waypoint_generator import generate_waypoints
from robot.kinematics import SerialKinematics, pose_error
from robot.robot_controller import distance_to_polyline
from vision.contour_detector import detect_contour
from vision.geometry import PlaneCamera


class VisionPlanningTests(unittest.TestCase):
    def test_image_changes_produce_different_paths(self):
        camera = PlaneCamera(640, 480, 600, 600, np.eye(4), np.array([0, 0, -1]), np.array([0, 0, 1]))
        centers = []
        for offset, angle in ((0, 0), (80, 23)):
            image = np.full((480, 640, 3), 200, np.uint8)
            corners = cv2.boxPoints(((270 + offset, 210), (180, 110), angle)).astype(np.int32)
            cv2.polylines(image, [corners], True, (10, 10, 10), 4)
            contour, _ = detect_contour(image)
            world = camera.pixel_to_world(contour)
            path = generate_waypoints(world, SCENE)
            np.testing.assert_allclose(path.surface_points[0], path.surface_points[-1])
            self.assertLessEqual(np.linalg.norm(np.diff(path.surface_points, axis=0), axis=1).max(), 0.010001)
            np.testing.assert_allclose(path.poses[0, :3, 3], path.poses[-1, :3, 3])
            self.assertAlmostEqual(path.poses[0, 2, 3], -0.9)
            self.assertAlmostEqual(path.poses[path.phases.index('trace'), 2, 3], -0.99)
            np.testing.assert_allclose(path.poses[0, :3, 2], [0, 0, -1])
            # World XY is counterclockwise because image v is inverted.
            area2 = np.sum(world[:, 0] * np.roll(world[:, 1], -1) - world[:, 1] * np.roll(world[:, 0], -1))
            self.assertLess(area2, 0)
            centers.append(world.mean(axis=0))
        self.assertGreater(centers[1][0] - centers[0][0], 0.1)

    def test_empty_and_open_outline_are_rejected(self):
        image = np.full((480, 640, 3), 200, np.uint8)
        with self.assertRaises(RuntimeError):
            detect_contour(image)
        cv2.line(image, (100, 100), (300, 100), (0, 0, 0), 5)
        with self.assertRaises(RuntimeError):
            detect_contour(image)

    def test_tilted_camera_round_trip_and_parallel_ray(self):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('xyz', [0.2, -0.1, 0.3]).as_matrix()
        transform[:3, 3] = [0.4, -0.1, 1.7]
        camera = PlaneCamera(640, 480, 600, 590, transform, np.array([0, 0, 0.5]), np.array([0., 0., 1.]))
        pixels = np.array([[200., 200.], [320., 240.], [440., 300.]])
        world = camera.pixel_to_world(pixels)
        np.testing.assert_allclose(world[:, 2], 0.5)
        np.testing.assert_allclose(camera.world_to_pixel(world), pixels, atol=1e-9)
        camera.world_from_camera = np.eye(4)
        camera.plane_normal = np.array([1., 0., 0.])
        with self.assertRaises(ValueError):
            camera.pixel_to_world([[320, 240]])

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
