"""Full-image geometry checks using analytic depth, no model or annotations."""

import unittest
from pathlib import Path
import tempfile

import cv2
import numpy as np

from vision.weld_detector import detect, extract_plates, intersection_geometry, mask_centerline
from scripts.auto_weld import save_paths, serialize_result


class AutomaticWeldTests(unittest.TestCase):
    def setUp(self):
        self.shape = (120, 160)
        self.k = np.array([[150., 0, 80], [0, 150, 60], [0, 0, 1]])
        self.rgb = np.full((*self.shape, 3), 100, dtype=np.uint8)

    def corner(self, concave=True):
        y, _ = np.indices(self.shape)
        ray_y = np.abs((y - 60) / 150)
        return (1 / (1 + ray_y if concave else 1 - ray_y)).astype(np.float32)

    def test_flat_image_has_no_junction(self):
        result = detect(np.ones(self.shape, np.float32), self.k, self.rgb)
        self.assertEqual(len(result["plates"]), 1)
        self.assertEqual(result["joints"], [])

    def test_discovers_concave_contact_without_regions(self):
        result = detect(self.corner(), self.k, self.rgb)
        self.assertEqual(len(result["joints"]), 1)
        line = result["joints"][0]["xy"]
        self.assertGreater(np.ptp(line[:, 0]), 100)
        np.testing.assert_allclose(line[:, 1], 60, atol=1)
        joint = result["joints"][0]
        xyz = joint["xyz_camera_estimated_m"]
        for index in joint["planes"]:
            equation = result["plates"][index].equation
            np.testing.assert_allclose(xyz @ equation[:3]+equation[3], 0, atol=1e-6)
        np.testing.assert_allclose(np.cross(xyz-joint["origin_camera_estimated_m"], joint["direction_camera"]), 0, atol=1e-6)
        self.assertEqual(joint["segments"][0]["state"], "no_weld_detected")
        self.assertEqual(joint["segments"][0]["weld_state"], "unknown")

    def test_sam_interior_sampling_splits_faces_and_excludes_bead(self):
        mask = np.zeros(self.shape, dtype=bool)
        mask[10:110, 20:140] = True
        bead = np.zeros_like(mask)
        bead[55:65, 40:120] = True
        plates, labels, _ = extract_plates(self.corner(), self.k, plate_instances=[mask], exclude_mask=bead)
        self.assertGreaterEqual(len(plates), 2)
        self.assertFalse((labels[~mask] >= 0).any())
        self.assertFalse((labels[bead] >= 0).any())
        for plate in plates:
            self.assertEqual(plate.sampling_source, "sam_interior")
            self.assertEqual(plate.semantic_instance, 0)
        normals = np.array([plate.equation[:3] for plate in plates])
        self.assertLess(np.abs(normals @ normals.T).min(), .3)

    def test_partial_bead_preserves_fallback_intervals(self):
        bead = np.zeros(self.shape, dtype=bool)
        bead[55:66, 25:80] = True
        result = detect(self.corner(), self.k, self.rgb, weld_masks=[bead], weld_scores=[.8])
        self.assertEqual(len(result["welds"]), 1)
        states = [segment["state"] for joint in result["joints"] for segment in joint["segments"]]
        self.assertIn("weld_detected", states)
        self.assertIn("no_weld_detected", states)
        self.assertNotIn("unwelded", states)
        data = {"depth_m": self.corner(), "processed_rgb": self.rgb,
                "original_hw": np.array(self.shape), "intrinsics": self.k}
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory)/"mixed.png"
            cv2.imwrite(str(image), self.rgb)
            _, report, _ = serialize_result(0, image, data, result, 5)
        self.assertTrue(any(path["kind"] == "plate_intersection" for path in report["paths"]))
        self.assertEqual(sum(path["kind"] == "weld_toe" for path in report["paths"]), 2)
        self.assertFalse(any(path["kind"] == "plate_intersection" and path["state"] == "weld_detected"
                             for path in report["paths"]))

    def test_small_tack_evidence_is_not_called_unwelded(self):
        tack = np.zeros(self.shape, dtype=bool)
        tack[56:65, 70:79] = True
        result = detect(self.corner(), self.k, self.rgb, weld_masks=[tack], weld_scores=[.8])
        self.assertFalse(result["welds"])
        states = [segment["state"] for joint in result["joints"] for segment in joint["segments"]]
        self.assertIn("uncertain_weld_evidence", states)
        self.assertIn("no_weld_detected", states)

    def test_no_finite_depth_confidence_has_no_geometry_fallback(self):
        result = detect(self.corner(), self.k, self.rgb, confidence=np.full(self.shape, np.nan))
        self.assertFalse(result["plates"])
        self.assertFalse(result["joints"])

    def test_parallel_planes_have_no_unique_intersection(self):
        with self.assertRaises(ValueError):
            intersection_geometry(np.array([0., 0, 1., -1.]), np.array([0., 0, 1., -2.]))

    def test_convex_plate_outline_is_not_a_contact(self):
        result = detect(self.corner(concave=False), self.k, self.rgb)
        self.assertEqual(result["joints"], [])

    def test_two_faces_of_same_semantic_object_are_rejected(self):
        mask = np.ones(self.shape, dtype=bool)
        result = detect(self.corner(), self.k, self.rgb, plate_masks=[mask, mask.copy()])
        self.assertEqual(result["joints"], [])

    def test_weak_text_match_on_flat_surface_needs_contact_evidence(self):
        mask = np.zeros(self.shape, dtype=bool)
        mask[55:65, 25:135] = True
        result = detect(np.ones(self.shape, np.float32), self.k, self.rgb,
                        weld_masks=[mask], weld_scores=[.45])
        self.assertEqual(result["welds"], [])

    def test_mask_centerline_preserves_a_bend(self):
        mask = np.zeros((80, 100), dtype=np.uint8)
        cv2.polylines(mask, [np.array([[10, 15], [65, 15], [65, 65]])], False, 1, 7)
        curve = mask_centerline(mask)
        self.assertGreater(np.ptp(curve[:, 0]), 45)
        self.assertGreater(np.ptp(curve[:, 1]), 40)
        self.assertLess(np.linalg.norm(curve - [65, 15], axis=1).min(), 5)
        self.assertTrue((np.minimum(np.abs(curve[:, 1] - 15), np.abs(curve[:, 0] - 65)) < 4).all())

    def test_serialized_curve_and_class_map_use_original_resolution(self):
        depth = self.corner()
        data = {"depth_m": depth, "processed_rgb": self.rgb,
                "original_hw": np.array([240, 320]), "intrinsics": self.k}
        result = detect(depth, self.k, self.rgb)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "input.png"
            cv2.imwrite(str(image), np.zeros((240, 320, 3), dtype=np.uint8))
            arrays, report, _ = serialize_result(0, image, data, result, 5)
            save_paths(arrays, [report], Path(directory))
            import csv
            with (Path(directory)/"seam_paths.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["state"], "no_weld_detected")
            self.assertEqual(rows[0]["surface_kind"], "analytic_intersection")
            self.assertEqual(rows[0]["geometry_run"], "0")
        xy = arrays["image_000_joint_0_centerline_xy"]
        np.testing.assert_allclose(xy[:, 1], 120.5, atol=.1)
        xyz = arrays["image_000_joint_0_xyz_camera_estimated_m"]
        projected = xyz @ self.k.T
        np.testing.assert_allclose(projected[:,:2]/projected[:,2:], (xy+.5)/2-.5, atol=1e-5)
        mask = arrays["image_000_class_map_original"]
        self.assertEqual(mask.shape, (240, 320))
        self.assertEqual(mask[120, 160], 1)
        self.assertEqual(report["joints"][0]["weld_state"], "unknown")


if __name__ == "__main__":
    unittest.main()
