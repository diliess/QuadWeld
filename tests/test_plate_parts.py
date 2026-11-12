"""Prompt placement and exclusive part masks without scene annotations."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from vision.plate_parts import make_plate_prompts, resolve_plate_masks, sample_interior


class PlatePartTests(unittest.TestCase):
    def test_overview_uses_plate_at_vertical_foot_instead_of_larger_floor(self):
        depth = np.full((120, 100), 2.)
        k = np.array([[100., 0, 50.5], [0, 100, 40.5], [0, 0, 1]])
        vertical, base, floor, bevel = [np.zeros(depth.shape, bool) for _ in range(4)]
        vertical[20:50, 20:80] = True
        base[50:60, 15:85] = True
        floor[80:120, :] = True
        bevel[47:50, 20:80] = True
        planes = [SimpleNamespace(mask=floor, equation=np.array([0., -1, 0, -.9])),
                  SimpleNamespace(mask=base, equation=np.array([0., -1, 0, -.3])),
                  SimpleNamespace(mask=vertical, equation=np.array([0., 0, -1, 2.])),
                  SimpleNamespace(mask=bevel, equation=np.array([0., -.88, np.sqrt(1-.88**2), 0.]))]
        pose = np.eye(4)
        pose[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        with patch('vision.plate_parts.extract_plates', return_value=(planes, None, None)):
            prompts, foreground, _, _ = make_plate_prompts(depth, k, None, pose,
                distance_range=(1.2, 2.8), select_contact_pair=True)
        self.assertTrue(foreground[base].all())  # Overview depths survive the configurable range.
        self.assertTrue(np.array_equal(prompts['horizontal']['support_mask'], base))
        self.assertTrue(np.array_equal(prompts['vertical']['support_mask'], vertical))

    def test_positive_points_stay_inside_face_interior(self):
        mask = np.zeros((80, 100), bool)
        mask[10:65, 15:90] = True
        xy = sample_interior(mask)
        self.assertEqual(len(xy), 5)
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        self.assertTrue((distance[xy[:, 1].astype(int), xy[:, 0].astype(int)] >= 6).all())
        self.assertEqual(len(np.unique(xy, axis=0)), 5)

    def test_overlap_uses_predicted_plane_distance_and_excludes_far_pixels(self):
        points = np.zeros((8, 10, 3))
        points[:4, :, 1] = 1  # Vertical plane z=1.
        points[:4, :, 2] = 1
        points[4:, :, 2] = 2  # Horizontal plane y=0.
        foreground = np.ones((8, 10), bool)
        foreground[:, -1] = False
        raw = {role: np.ones((8, 10), bool) for role in ('horizontal', 'vertical')}
        prompts = {'horizontal': {'equation': np.array([0, 1, 0, 0]), 'support_mask': np.indices((8, 10))[0] >= 4},
                   'vertical': {'equation': np.array([0, 0, 1, -1]), 'support_mask': np.indices((8, 10))[0] < 4}}
        masks = resolve_plate_masks(raw, prompts, foreground, points)
        self.assertFalse((masks['horizontal'] & masks['vertical']).any())
        self.assertTrue(masks['vertical'][:4, :-1].all())
        self.assertTrue(masks['horizontal'][4:, :-1].all())
        self.assertFalse(masks['horizontal'][:, -1].any())
        self.assertFalse(masks['vertical'][:, -1].any())


if __name__ == '__main__':
    unittest.main()
