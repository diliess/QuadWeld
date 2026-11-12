"""Whole-object framing must hold for each physical stereo eye."""

import itertools
import unittest

import numpy as np

from hj90.full_framing import fit_stereo_distance
from hj90.overview_planning import framing


class FullFramingTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array(list(itertools.product((-.55, .55), (-.3, .3), (.69, .9))))
        self.target = np.array([0., 0., .79])
        self.k = np.array([[767.6, 0, 320], [0, 767.9, 550], [0, 0, 1.]])

    def test_both_eyes_keep_whole_object_and_margin(self):
        distance, pose, boxes = fit_stereo_distance(np.eye(4), self.target, self.vertices, self.k, [640, 1100], 15)
        for low, high in boxes:
            self.assertTrue((np.asarray(low) >= .08).all())
            self.assertTrue((np.asarray(high) <= .92).all())
        # Moving one centimetre closer violates at least one eye's margin.
        closer = pose.copy()
        closer[:3, 3] += closer[:3, 2] * .01
        self.assertLess(framing(closer, self.vertices, self.k, [640, 1100], margin=.08)[0], 1.)
        self.assertGreater(distance, .8)

    def test_wider_baseline_and_narrower_field_require_more_distance(self):
        args = (np.eye(4), self.target, self.vertices, self.k, [640, 1100], 15)
        ordinary = fit_stereo_distance(*args)[0]
        self.assertGreater(fit_stereo_distance(*args, baseline=.25)[0], ordinary)
        narrow = self.k.copy()
        narrow[0, 0] *= 1.2
        self.assertGreater(fit_stereo_distance(np.eye(4), self.target, self.vertices, narrow, [640, 1100], 15)[0], ordinary)

    def test_fixture_translation_and_yaw_preserve_framing(self):
        fixture = np.eye(4)
        angle = .7
        fixture[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        fixture[:3, 3] = [5, -3, .2]
        original = fit_stereo_distance(np.eye(4), self.target, self.vertices, self.k, [640, 1100], 15)
        moved = fit_stereo_distance(fixture, self.target, self.vertices, self.k, [640, 1100], 15)
        self.assertEqual(original[0], moved[0])
        np.testing.assert_allclose(original[2], moved[2], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
