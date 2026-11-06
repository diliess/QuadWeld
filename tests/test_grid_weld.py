"""Auto-weld grid pipeline on CPU: fillet poses, depth rescaling, seam selection and CLI modes."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from main import parse_args
from planning.fillet import FaceObstacles, clip_seam, fillet_path
from vision.auto_weld_seams import load_seam, support_plane, support_scale
from vision.rgbd import RGBDSettings

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "assets/scenes/ship-grid/ship_grid.usda"
WEB, FLOOR = np.array([-1., 0., 0.]), np.array([0., 0., 1.])


def auto_weld_module():
    spec = importlib.util.spec_from_file_location("auto_weld", ROOT / "scripts/auto_weld.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FilletPathTests(unittest.TestCase):
    def test_torch_bisects_the_corner_and_stands_off(self):
        seam = np.c_[np.ones(21), np.linspace(-.1, .1, 21), np.zeros(21)]
        settings = RGBDSettings(waypoint_spacing_m=.01, tcp_offset_m=.01, approach_height_m=.1)
        path = fillet_path(seam, WEB, FLOOR, settings)
        np.testing.assert_allclose(path.poses[:, :3, 2], np.tile([.5 ** .5, 0, -.5 ** .5], (len(path.poses), 1)), atol=1e-9)
        np.testing.assert_allclose(path.poses[:, :3, 0], np.tile([0, 1, 0], (len(path.poses), 1)), atol=1e-9)
        trace = path.poses[[i for i, p in enumerate(path.phases) if p == "trace"], :3, 3]
        gaps = FaceObstacles(np.zeros((2, 3)) + [1, 0, 0], np.vstack((WEB, FLOOR))).clearance(trace)
        np.testing.assert_allclose(gaps, .01 * .5 ** .5, atol=1e-9)
        self.assertAlmostEqual(np.linalg.norm(path.poses[0, :3, 3] - seam[0]), .1)
        self.assertEqual(path.phases[0], "approach")
        self.assertEqual(path.phases[-1], "retreat")
        rolled = fillet_path(seam, WEB, FLOOR, settings, roll=np.pi / 2)
        np.testing.assert_allclose(rolled.poses[:, :3, 2], path.poses[:, :3, 2], atol=1e-9)

    def test_faces_are_solid_behind_their_planes(self):
        faces = FaceObstacles(np.array([[1., 0, 0], [1., 0, 0]]), np.vstack((WEB, FLOOR)))
        self.assertFalse(faces.intersects(np.array([[.9, 0, .1]]), .05))
        self.assertTrue(faces.intersects(np.array([[1.05, 0, .1]]), .0))
        self.assertTrue(faces.intersects(np.array([[.9, 0, -.01]]), .0))

    def test_clip_keeps_length_around_the_nearest_sample(self):
        seam = np.c_[np.ones(101), np.linspace(-.5, .5, 101), np.zeros(101)]
        segment = clip_seam(seam, np.array([0., .2, 0.]), .3)
        self.assertAlmostEqual(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum(), .3, places=9)
        np.testing.assert_allclose(segment[[0, -1], 1], [.05, .35], atol=1e-9)
        end = clip_seam(seam, np.array([0., .5, 0.]), .3)
        np.testing.assert_allclose(end[[0, -1], 1], [.2, .5], atol=1e-9)


class AutoWeldSeamTests(unittest.TestCase):
    def write_report(self, folder, scale=0.8):
        """A camera looking 45 deg down at a web/floor corner, with DA3 depth shrunk by `scale`."""
        base_from_camera = np.eye(4)
        base_from_camera[:3, :3] = np.column_stack(((0, -1, 0), (-.5 ** .5, 0, -.5 ** .5), (.5 ** .5, 0, -.5 ** .5)))
        base_from_camera[:3, 3] = (0, 0, .5)
        rotation, translation = base_from_camera[:3, :3], base_from_camera[:3, 3]

        def plane(normal_base, point_base):
            normal = rotation.T @ normal_base
            point = rotation.T @ (point_base - translation) * scale
            return [*normal, float(-normal @ point)]

        corner = np.c_[np.full(30, .6), np.linspace(-.15, .15, 30), np.zeros(30)]
        camera = (corner - translation) @ rotation  # rows: R^T (p - t)
        planes = [{"id": 0, "equation_camera": plane(FLOOR, np.zeros(3)), "support_pixels": 5000},
                  {"id": 1, "equation_camera": plane(WEB, np.array([.6, 0, 0])), "support_pixels": 3000},
                  {"id": 2, "equation_camera": plane(np.array([0, -1., 0]), np.array([0, .3, 0])), "support_pixels": 900}]
        # A 0.22 m web (plane 1) seen through a pinhole, with DA3 depth shrunk like the planes.
        intrinsics = np.array([[200., 0, 100], [0, 200., 100], [0, 0, 1]])
        v, u = np.indices((200, 200))
        rays = np.stack((u, v, np.ones_like(u)), -1) @ np.linalg.inv(intrinsics).T
        normal = rotation.T @ WEB
        depth = (normal @ (rotation.T @ (np.array([.6, 0, 0]) - translation)) * scale) / (rays @ normal)
        heights = (rays * depth[..., None] / scale) @ rotation.T + translation
        web = (depth > 0) & (heights[..., 2] >= 0) & (heights[..., 2] <= .22) & (np.abs(heights[..., 1]) < .3)
        labels = np.where(web, 1, -1).astype(np.int16)
        result = {"planes": planes, "array_prefix": "image_000",
                  "joints": [{"planes": [0, 1]}, {"planes": [1, 2]}],
                  "paths": [{"id": "joint_0_segment_0", "kind": "plate_intersection", "joint_index": 0,
                             "state": "no_weld_detected", "xyz_key": "a",
                             "geometry_runs": [{"start_sample": 0, "stop_sample": 30}]},
                            {"id": "joint_1_segment_0", "kind": "plate_intersection", "joint_index": 1,
                             "state": "no_weld_detected", "xyz_key": "b",
                             "geometry_runs": [{"start_sample": 0, "stop_sample": 30}]}]}
        data = Path(folder) / "data"
        data.mkdir()
        (data / "auto_report.json").write_text(json.dumps({"results": [result]}))
        np.savez(data / "auto_results.npz", a=camera * scale, b=camera * scale, image_000_plane_labels=labels,
                 image_000_depth_m=np.where(depth > 0, depth, 1.), image_000_intrinsics=intrinsics)
        return base_from_camera, corner

    def test_support_plane_restores_metric_scale_in_base_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            base_from_camera, corner = self.write_report(folder, scale=.8)
            point, normal = support_plane(np.array([[0, .1, .032], [0, -.1, .032], [.2, 0, .032]]), .032, FLOOR)
            seam = load_seam(folder, base_from_camera, point, normal)
        self.assertAlmostEqual(seam.scale, 1 / .8, places=9)
        np.testing.assert_allclose(seam.points_base, corner, atol=1e-9)
        np.testing.assert_allclose(seam.normals_base, [WEB, FLOOR], atol=1e-9)
        self.assertEqual(seam.diagnostics["path_id"], "joint_0_segment_0")
        self.assertEqual(seam.diagnostics["rejected"], {"not_upright_on_lower_face": 1})
        self.assertAlmostEqual(seam.upright_top_m, .22, delta=.01)

    def test_leaning_monocular_web_keeps_the_procedure_faces(self):
        with tempfile.TemporaryDirectory() as folder:
            base_from_camera, _ = self.write_report(folder)
            report = Path(folder) / "data/auto_report.json"
            data = json.loads(report.read_text())
            rotation = base_from_camera[:3, :3]
            lean = np.array([-np.cos(np.radians(25)), 0, np.sin(np.radians(25))])  # Web tilted back 25 deg.
            normal = rotation.T @ lean
            point = rotation.T @ (np.array([.6, 0, 0]) - base_from_camera[:3, 3]) * .8
            data["results"][0]["planes"][1]["equation_camera"] = [*normal, float(-normal @ point)]
            report.write_text(json.dumps(data))
            seam = load_seam(folder, base_from_camera, np.zeros(3), FLOOR)
        np.testing.assert_allclose(seam.normals_base, [WEB, FLOOR], atol=1e-9)
        self.assertAlmostEqual(seam.diagnostics["monocular_upright_tilt_from_vertical_deg"], 25, places=6)

    def test_no_matching_support_plane_is_an_error(self):
        planes = [{"id": 0, "equation_camera": [1., 0, 0, .5], "support_pixels": 10}]
        with self.assertRaisesRegex(RuntimeError, "support plane"):
            support_scale(planes, np.array([0., 0, 1]), 1.)

    def test_calibrated_intrinsics_follow_the_processed_resolution(self):
        module = auto_weld_module()
        path = ROOT / "calibration/b2_e10l_front_rgbd.json"
        matrix = module.calibrated_intrinsics(path, [480, 640], (378, 504))
        self.assertAlmostEqual(matrix[0, 0], 400 * 504 / 640)
        self.assertAlmostEqual(matrix[1, 1], 400 * 378 / 480)
        self.assertAlmostEqual(matrix[0, 2], (320 + .5) * 504 / 640 - .5)
        self.assertAlmostEqual(matrix[1, 2], (240 + .5) * 378 / 480 - .5)
        with self.assertRaisesRegex(ValueError, "image size"):
            module.calibrated_intrinsics(path, [600, 800], (378, 504))


class GridModeTests(unittest.TestCase):
    def test_auto_weld_plans_in_the_environment(self):
        args = parse_args(["--robot", "b2_e10l", "--environment-usd", str(GRID), "--seam-source", "auto-weld",
                           "--robot-pose", "0", "0", "0", "--plan-only"])
        self.assertFalse(args.scene_only)
        self.assertEqual(args.robot_pose, [0., 0., 0.])
        self.assertTrue(parse_args(["--environment-usd", str(GRID)]).scene_only)

    def test_auto_weld_needs_corners_and_a_length(self):
        for extra in ([], ["--environment-usd", str(GRID), "--weld-length", "0"],
                      ["--environment-usd", str(GRID), "--demo-open"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--seam-source", "auto-weld", *extra])


if __name__ == "__main__":
    unittest.main()
