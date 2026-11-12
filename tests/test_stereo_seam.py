"""Stereo plane intersection on a ray-traced T-joint with known geometry."""

import unittest

import cv2
import numpy as np

from vision import stereo_seam as ss


SEAM_ENDS = np.array([[-.25, 0., 0.], [.25, 0., 0.]])   # web foot on the base plate, world frame


def _texture(u, v, seed):
    """Band-limited random pattern over metric surface coordinates (periods 12-60 mm)."""
    rng = np.random.default_rng(seed)
    value = np.zeros_like(u)
    for _ in range(40):
        angle, period, phase = rng.uniform(0, np.pi), rng.uniform(.012, .06), rng.uniform(0, 2 * np.pi)
        value += np.sin(2 * np.pi * (u * np.cos(angle) + v * np.sin(angle)) / period + phase)
    return value / 40 ** .5


def _camera(position, target):
    """World-from-camera rotation (x right, y down, z forward) of a camera at ``position``."""
    forward = np.asarray(target, float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    return np.stack((right, np.cross(forward, right), forward), axis=1)


def _render(k, size, rotation, origin):
    """Image and surface labels (0 table, 1 base plate, 2 web) of the T-joint from one pinhole camera."""
    width, height = size
    xs, ys = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    rays = np.stack(((xs - k[0, 2]) / k[0, 0], (ys - k[1, 2]) / k[1, 1], np.ones_like(xs)), axis=-1) @ rotation.T
    depth = np.full(xs.shape, np.inf)
    image, label = np.zeros(xs.shape), np.zeros(xs.shape, int)
    # (plane axis, plane level, in-plane bounds, brightness, texture seed, label)
    surfaces = [(2, -.012, ((-9, 9), (-9, 9)), .35, 1, 0),
                (2, 0., ((-.30, .30), (-.22, .22)), .55, 2, 1),
                (1, 0., ((-.25, .25), (0., .16)), .45, 3, 2)]
    for axis, level, bounds, brightness, seed, name in surfaces:
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (level - origin[axis]) / rays[..., axis]
        hit = origin + rays * t[..., None]
        u, v = [hit[..., i] for i in range(3) if i != axis]
        inside = (t > 0) & (t < depth) & (u >= bounds[0][0]) & (u <= bounds[0][1]) \
            & (v >= bounds[1][0]) & (v <= bounds[1][1])
        image[inside] = (brightness + .12 * _texture(u, v, seed))[inside]
        label[inside], depth[inside] = name, t[inside]
    return (np.clip(image, 0, 1) * 255).astype(np.uint8), label


def _scene(size=(320, 240), focal=300., baseline=.06, position=(.06, -.62, .42)):
    k = np.array([[focal, 0, (size[0] - 1) / 2], [0, focal, (size[1] - 1) / 2], [0, 0, 1]])
    position = np.asarray(position, float)
    rotation = _camera(position, (0, -.03, .03))
    left, label = _render(k, size, rotation, position)
    right, _ = _render(k, size, rotation, position + baseline * rotation[:, 0])
    seam = (SEAM_ENDS - position) @ rotation          # left camera frame
    return left, right, k, baseline, label == 1, label == 2, seam, (rotation, position)


def _line_distance(points, ends):
    direction = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
    offset = points - ends[0]
    return np.linalg.norm(offset - np.outer(offset @ direction, direction), axis=1)


class GeometryTests(unittest.TestCase):
    K = np.array([[760., 0, 319.5], [0, 770., 549.5], [0, 0, 1]])

    def test_plane_and_disparity_parameters_round_trip(self):
        normal = np.array([.2, -.7, .6]) / np.linalg.norm([.2, -.7, .6])
        params = ss.disparity_from_plane(normal, 1.4, self.K, .09)
        recovered, offset = ss.plane_from_disparity(params, self.K, .09)
        np.testing.assert_allclose(recovered, normal, atol=1e-12)
        self.assertAlmostEqual(offset, 1.4)
        # The affine disparity reproduces the plane's points.
        xy = np.array([[100., 200.], [500., 900.], [320., 40.]])
        points = ss.backproject(xy, np.c_[xy, np.ones(3)] @ params, self.K, .09)
        np.testing.assert_allclose(points @ normal, 1.4, atol=1e-12)
        np.testing.assert_allclose(ss.project(points, self.K), xy, atol=1e-9)

    def test_intersection_lies_on_both_planes_and_rejects_parallel_plates(self):
        a = (np.array([0., -1, 0]), -.3)
        b = (np.array([0., 0, 1.]), 1.5)
        point, direction, angle = ss.intersect_planes(a, b)
        np.testing.assert_allclose(np.abs(direction), [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose([point @ a[0], point @ b[0]], [a[1], b[1]], atol=1e-12)
        self.assertAlmostEqual(angle, 90.)
        tilted = np.array([0., np.sin(np.radians(5)), np.cos(np.radians(5))])
        with self.assertRaises(ValueError):
            ss.intersect_planes(b, (tilted, 1.5))

    def test_disparity_plane_follows_image_reduction(self):
        params = np.array([.01, -.03, 25.])
        small = ss._rescale_params(params, .5)
        # Pixel i of the reduced image covers full-resolution pixels 2i and 2i + 1.
        for u, v in ((10, 20), (150, 7)):
            full = params @ [2 * u + .5, 2 * v + .5, 1]
            self.assertAlmostEqual(small @ [u, v, 1], full / 2)
        np.testing.assert_allclose(ss._rescale_params(small, 2), params, atol=1e-12)

    def test_depth_map_gives_starting_plane_despite_outliers(self):
        normal, offset = np.array([0., -.6, .8]), 1.2
        ys, xs = np.mgrid[:200, :300].astype(float)
        rays = np.stack(((xs - 150) / 400, (ys - 100) / 400, np.ones_like(xs)), axis=-1)
        depth = offset / (rays @ normal)
        depth[::7, ::5] *= 1.6
        mask = np.ones(depth.shape, bool)
        k = np.array([[400., 0, 150], [0, 400, 100], [0, 0, 1]])
        params = ss.disparity_plane_from_depth(depth, mask, 400., .1)
        np.testing.assert_allclose(params, ss.disparity_from_plane(normal, offset, k, .1), atol=1e-3)

    def test_fusion_takes_union_of_views_and_drops_a_stray_one(self):
        segments = [np.array([[0., 0, 0], [.5, 0, 0]]), np.array([[.3, .001, 0], [.9, .001, 0]]),
                    np.array([[.2, -.001, 0], [.7, -.001, 0]]), np.array([[.1, .05, .02], [.6, .05, .02]])]
        fused = ss.fuse_seams(segments)
        self.assertEqual(fused['view_used'], [True, True, True, False])
        self.assertAlmostEqual(fused['length_m'], .9, places=3)
        np.testing.assert_allclose(fused['ends'][:, 1:], 0, atol=1e-3)
        self.assertGreater(fused['direction'][0], .999)


class SyntheticJointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = _scene()
        left, right, k, baseline, base, web, _, _ = cls.scene
        cls.result = ss.locate_seam(left, right, k, baseline, base, web)

    def test_seam_is_recovered_in_metric_units_without_any_depth_input(self):
        seam, result = self.scene[6], self.result
        self.assertEqual(result['status'], 'ok')
        # One disparity pixel is ~30 mm of depth here; the fit must be far below that.
        self.assertGreater(result['checks']['depth_per_disparity_px_mm'], 25)
        self.assertLess(_line_distance(seam, result['joint_ends_xyz']).max(), .003)
        self.assertLess(_line_distance(result['xyz'], seam).max(), .003)
        self.assertLess(np.linalg.norm(result['joint_ends_xyz'] - seam, axis=1).max(), .012)
        self.assertAlmostEqual(result['length_m'], .5, delta=.015)
        self.assertEqual(result['end_is_open'], [False, False])
        self.assertLess(result['checks']['split_half_spread_mm'], 3)

    def test_faces_and_torch_axis(self):
        rotation = self.scene[7][0]
        result = self.result
        self.assertAlmostEqual(result['opening_deg'], 90, delta=2)
        # Visible faces: base plate top (+z) and the web side towards the camera (-y); torch splits them.
        expected = np.array([0., -1, 1]) / 2 ** .5 @ rotation
        self.assertGreater(result['torch_axis'] @ expected, np.cos(np.radians(3)))
        self.assertAlmostEqual(result['torch_axis'] @ result['direction'], 0, places=9)

    def test_projection_into_both_images_matches_the_rendered_joint(self):
        left, right, k, baseline, base, web, seam, _ = self.scene
        for xy, shift in ((self.result['xy'], 0.), (self.result['xy_right'], baseline)):
            truth = ss.project(seam - [shift, 0, 0], k)
            self.assertLess(_line_distance(np.c_[xy, np.zeros(len(xy))], np.c_[truth, np.zeros(2)]).max(), 1.)

    def test_rough_masks_and_exposure_difference_are_tolerated(self):
        left, right, k, baseline, base, web, seam, _ = self.scene
        kernel = np.ones((9, 9), np.uint8)
        cases = {
            'eroded': [cv2.erode(m.astype(np.uint8), kernel).astype(bool) for m in (base, web)],
            'dilated': [cv2.dilate(m.astype(np.uint8), kernel).astype(bool) for m in (base, web)],
            'shifted': [np.roll(m, (3, 4), axis=(0, 1)) for m in (base, web)],
        }
        for name, masks in cases.items():
            with self.subTest(name):
                result = ss.locate_seam(left, right, k, baseline, *masks)
                self.assertEqual(result['status'], 'ok')
                self.assertLess(_line_distance(seam, result['joint_ends_xyz']).max(), .006)
        dark = np.clip(255 * (right / 255.) ** 1.3 * .7, 0, 255).astype(np.uint8)
        result = ss.locate_seam(left, dark, k, baseline, base, web)
        self.assertLess(_line_distance(seam, result['joint_ends_xyz']).max(), .004)

    def test_wrong_depth_guess_does_not_hurt(self):
        left, right, k, baseline, base, web, seam, _ = self.scene
        wrong = np.full(left.shape, 1.6)   # twice the true distance, no structure
        result = ss.locate_seam(left, right, k, baseline, base, web, init_depth=wrong)
        self.assertEqual(result['status'], 'ok')
        self.assertLess(_line_distance(seam, result['joint_ends_xyz']).max(), .003)

    def test_unusable_inputs_are_reported_not_guessed(self):
        left, right, k, baseline, base, web, _, _ = self.scene
        empty = np.zeros_like(base)
        self.assertEqual(ss.locate_seam(left, right, k, baseline, base, empty)['status'], 'plate_mask_too_small')
        far = np.zeros_like(base)
        far[:12, :40] = True
        near = np.zeros_like(base)
        near[-12:, -40:] = True
        self.assertEqual(ss.locate_seam(left, right, k, baseline, far, near)['status'], 'plates_are_not_adjacent')
        # Two halves of the same plate are one plane, not a joint.
        columns = np.arange(base.shape[1]) < base.shape[1] // 2
        status = ss.locate_seam(left, right, k, baseline, base & columns, base & ~columns)['status']
        self.assertEqual(status, 'plates_nearly_parallel')


class LargeImageTests(unittest.TestCase):
    def test_high_resolution_pair_uses_reduced_search_and_stays_accurate(self):
        left, right, k, baseline, base, web, seam, _ = _scene(size=(1680, 1260), focal=1575.)
        result = ss.locate_seam(left, right, k, baseline, base, web, max_pixels=12000)
        self.assertEqual(result['status'], 'ok')
        self.assertLess(_line_distance(seam, result['joint_ends_xyz']).max(), .0015)
        self.assertLess(np.linalg.norm(result['joint_ends_xyz'] - seam, axis=1).max(), .006)


class RectificationTests(unittest.TestCase):
    def test_toed_in_pair_is_rectified_and_seam_returned_in_original_left_frame(self):
        size, focal, baseline = (320, 240), 300., .06
        k = np.array([[focal, 0, (size[0] - 1) / 2], [0, focal, (size[1] - 1) / 2], [0, 0, 1]])
        position = np.array([.06, -.62, .42])
        world_from_left = _camera(position, (0, -.03, .03))
        # Right camera: displaced along the left x axis and rotated 2 degrees about its y axis and 1 about z.
        toe = cv2.Rodrigues(np.radians([0., -2., 1.]))[0]
        world_from_right = world_from_left @ toe
        left, label = _render(k, size, world_from_left, position)
        right, _ = _render(k, size, world_from_right, position + baseline * world_from_left[:, 0])
        rotation = toe.T                                       # left camera points -> right camera
        translation = rotation @ np.array([-baseline, 0, 0])
        left_r, right_r, masks, k_r, b_r, back = ss.rectify_pair(
            left, right, k, np.zeros(5), k, np.zeros(5), rotation, translation, masks=(label == 1, label == 2))
        self.assertAlmostEqual(b_r, baseline, places=6)
        result = ss.locate_seam(left_r, right_r, k_r, b_r, *masks)
        self.assertEqual(result['status'], 'ok')
        seam = (SEAM_ENDS - position) @ world_from_left
        self.assertLess(_line_distance(seam, result['joint_ends_xyz'] @ back.T).max(), .004)


if __name__ == '__main__':
    unittest.main()
