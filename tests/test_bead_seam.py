"""Weld toes on a ray-traced T-joint that carries a fillet bead with wavy toes."""

import unittest

import numpy as np

from weld_vision import bead_seam as bs
from weld_vision import stereo_seam as ss

from tests.test_stereo_seam import _camera, _texture


def _legs(x):
    """Bead legs on the base plate and on the web (m) along the joint: not constant, so the toes are curves."""
    return .008 + .0012 * np.sin(2 * np.pi * x / .045), .0075 + .001 * np.sin(2 * np.pi * x / .06 + 1.)


def _render(k, size, rotation, origin, bead=True, stripes=False):
    """Image and labels (0 table, 1 base plate, 2 web, 3 bead).  Base plate z = 0, web y = 0, joint along x."""
    width, height = size
    xs, ys = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    rays = np.stack(((xs - k[0, 2]) / k[0, 0], (ys - k[1, 2]) / k[1, 1], np.ones_like(xs)), axis=-1) @ rotation.T
    depth = np.full(xs.shape, np.inf)
    image, label = np.zeros(xs.shape), np.zeros(xs.shape, int)
    surfaces = [(2, -.012, ((-9, 9), (-9, 9)), .30, 1, 0),
                (2, 0., ((-.30, .30), (-.22, .22)), .42, 2, 1),
                (1, 0., ((-.25, .25), (0., .16)), .34, 3, 2)]
    for axis, level, bounds, brightness, seed, name in surfaces:
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (level - origin[axis]) / rays[..., axis]
        hit = origin + rays * t[..., None]
        u, v = [hit[..., i] for i in range(3) if i != axis]
        inside = (t > 0) & (t < depth) & (u >= bounds[0][0]) & (u <= bounds[0][1]) \
            & (v >= bounds[1][0]) & (v <= bounds[1][1])
        # Stripes: a pattern that only varies along the joint and repeats, like machining marks.
        pattern = np.sin(2 * np.pi * u / .012) if stripes else _texture(u, v, seed)
        image[inside] = (brightness + .1 * pattern)[inside]
        label[inside], depth[inside] = name, t[inside]
    if bead:
        # The bead fills -y / leg_base(x) + z / leg_web(x) <= 1 in the corner.  March each ray up to
        # the plate it would hit and take the first crossing of that surface.
        steps = np.linspace(.0, 1., 400)
        finite = np.where(np.isfinite(depth), depth, 0.)
        start = np.maximum(finite - .06, 0.)
        previous, found = None, np.zeros(xs.shape, bool)
        crossing = np.zeros(xs.shape)
        for fraction in steps:
            t = start + fraction * (finite - start)
            point = origin + rays * t[..., None]
            base_leg, web_leg = _legs(point[..., 0])
            value = -point[..., 1] / base_leg + point[..., 2] / web_leg - 1
            value = np.where((np.abs(point[..., 0]) <= .25) & (point[..., 1] <= 1e-9) & (point[..., 2] >= -1e-9), value, 1.)
            if previous is not None:
                new = ~found & (previous[0] > 0) & (value <= 0)
                weight = previous[0] / np.maximum(previous[0] - value, 1e-12)
                crossing = np.where(new, previous[1] + weight * (t - previous[1]), crossing)
                found |= new
            previous = (value, t)
        point = origin + rays * crossing[..., None]
        ripple = .06 * np.sin(2 * np.pi * point[..., 0] / .0035)
        image[found] = (.78 + ripple)[found]
        label[found], depth[found] = 3, crossing[found]
    return (np.clip(image, 0, 1) * 255).astype(np.uint8), label


def _scene(bead=True, stripes=False, roll=0., size=(520, 400), focal=760., baseline=.06, position=(.02, -.26, .22)):
    k = np.array([[focal, 0, (size[0] - 1) / 2], [0, focal, (size[1] - 1) / 2], [0, 0, 1]])
    position = np.asarray(position, float)
    rotation = _camera(position, (0, -.004, .004))
    c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    rotation = rotation @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    left, label = _render(k, size, rotation, position, bead, stripes)
    right, _ = _render(k, size, rotation, position + baseline * rotation[:, 0], bead, stripes)
    return left, right, k, baseline, label, (rotation, position)


def _true_toes(pose, count=600):
    """Toe on the base plate and toe on the web, in the left camera frame."""
    rotation, position = pose
    x = np.linspace(-.25, .25, count)
    base_leg, web_leg = _legs(x)
    base = np.stack((x, -base_leg, np.zeros(count)), axis=1)
    web = np.stack((x, np.zeros(count), web_leg), axis=1)
    return [(curve - position) @ rotation for curve in (base, web)]


def _to_curve(points, curve):
    return np.array([np.linalg.norm(curve - p, axis=1).min() for p in points])


class BeadToeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = _scene()
        left, right, k, baseline, label, _ = cls.scene
        cls.result = bs.locate_bead_toes(left, right, k, baseline, label == 1, label == 2)

    def test_toes_are_curves_on_the_plates_within_a_millimetre(self):
        left, right, k, baseline, label, pose = self.scene
        result = self.result
        self.assertEqual(result['status'], 'ok')
        rotation, position = pose
        root = (np.array([[-.25, 0, 0], [.25, 0, 0]]) - position) @ rotation
        ends = result['seam']['joint_ends_xyz']
        direction = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
        offset = root - ends[0]
        self.assertLess(np.linalg.norm(offset - np.outer(offset @ direction, direction), axis=1).max(), .001)
        truth = _true_toes(pose)
        up = [abs((rotation @ n)[2]) for n, _ in result['seam']['planes']]
        for toe, curve in zip((result['toes'][int(np.argmax(up))], result['toes'][int(np.argmin(up))]), truth):
            valid = toe['valid']
            self.assertGreater(valid.mean(), .6)
            self.assertLess(_to_curve(toe['xyz'][valid], curve).mean(), .0007)
            self.assertLess(_to_curve(toe['xyz'][valid], curve).max(), .002)
            # Points lie on the fitted plate plane and project onto the bead edge in both images.
            normal, height = result['seam']['planes'][toe['plate']]
            np.testing.assert_allclose(toe['xyz'] @ normal, height, atol=1e-9)
            self.assertEqual(toe['xy'].shape, (len(toe['xyz']), 2))
            # The toe is not a straight line and the measurement follows it.
            self.assertGreater(np.ptp(toe['leg_m'][valid]), .0012)
            segments = bs.polyline_segments(toe)
            self.assertGreater(sum(len(s) for s in segments), .6 * len(valid))

    def test_bare_joint_reports_no_toe(self):
        left, right, k, baseline, label, _ = _scene(bead=False)
        result = bs.locate_bead_toes(left, right, k, baseline, label == 1, label == 2)
        self.assertEqual(result['status'], 'no_bead_toe_found')
        self.assertEqual(result['seam']['status'], 'ok')

    def test_masks_that_swallow_the_bead_are_tolerated(self):
        left, right, k, baseline, label, pose = self.scene
        truth = _true_toes(pose)
        rotation = pose[0]
        # A segmentation that gives the bead to the web plate.
        result = bs.locate_bead_toes(left, right, k, baseline, label == 1, (label == 2) | (label == 3))
        self.assertEqual(result['status'], 'ok')
        up = [abs((rotation @ n)[2]) for n, _ in result['seam']['planes']]
        for toe, curve in zip((result['toes'][int(np.argmax(up))], result['toes'][int(np.argmin(up))]), truth):
            self.assertGreater(toe['valid'].mean(), .5)
            self.assertLess(_to_curve(toe['xyz'][toe['valid']], curve).mean(), .001)


class AmbiguityTests(unittest.TestCase):
    def test_repeating_texture_along_the_baseline_is_refused_or_resolved_by_a_rough_distance(self):
        left, right, k, baseline, label, pose = _scene(bead=False, stripes=True)
        rotation, position = pose
        root = (np.array([[-.25, 0, 0], [.25, 0, 0]]) - position) @ rotation

        def error(result):
            ends = result['joint_ends_xyz']
            direction = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
            offset = root - ends[0]
            return np.linalg.norm(offset - np.outer(offset @ direction, direction), axis=1).max()

        free = ss.locate_seam(left, right, k, baseline, label == 1, label == 2)
        # Without help the answer is either refused or right, never a confident wrong depth.
        if free['status'] == 'ok':
            self.assertLess(error(free), .002)
        else:
            self.assertEqual(free['status'], 'ambiguous_stereo_match')
        self.assertGreater(free['checks']['depth_ambiguity'], .5)
        helped = ss.locate_seam(left, right, k, baseline, label == 1, label == 2,
                                expected_distance=float(np.linalg.norm(root.mean(0))) * 1.03)
        self.assertEqual(helped['status'], 'ok')
        self.assertLess(error(helped), .002)


class SelectionAndFusionTests(unittest.TestCase):
    def test_plate_pair_ignores_thin_strips_and_far_objects(self):
        shape = (300, 240)
        masks = np.zeros((5,) + shape, bool)
        masks[0, 40:140, 20:220] = True          # web
        masks[1, 150:280, 10:230] = True         # base plate
        masks[2, 140:150, 20:220] = True         # the bead, segmented as a "panel" of its own
        masks[3, 0:30, 0:60] = True              # something in the background
        masks[4, 282:292, 10:230] = True         # table edge below the plate
        pair = bs.pick_plate_pair(masks, [.6, .55, .7, .5, .4])
        self.assertEqual(set(pair), {0, 1})
        self.assertEqual(pair[0], 1)             # larger first
        self.assertIsNone(bs.pick_plate_pair(masks[2:], [.7, .5, .4]))

    def test_fusion_takes_the_median_of_views_and_needs_two(self):
        s = np.arange(0, .2001, .001)
        wave = .008 + .001 * np.sin(2 * np.pi * s / .05)

        def view(offset, start, stop):
            valid = (s >= start) & (s <= stop)
            base = np.stack((s, -wave + offset, np.zeros(len(s))), axis=1)
            web = np.stack((s, np.zeros(len(s)), wave + offset), axis=1)
            return {'root': np.array([[start, 0, 0.], [stop, 0, 0.]]), 'normals': [np.array([0, 0, 1.]), np.array([0, -1., 0])],
                    'toes': [{'xyz': base, 'valid': valid}, {'xyz': web, 'valid': valid}]}

        views = [view(0., 0., .15), view(.0002, .05, .2), view(-.0002, .0, .2)]
        swapped = view(.004, 0., .2)             # an outlier that also lists its plates the other way round
        swapped['normals'], swapped['toes'] = swapped['normals'][::-1], swapped['toes'][::-1]
        fused = bs.fuse_toes(views + [swapped])
        base = fused['toes'][0]
        self.assertTrue(base['valid'].sum() > 190)
        inner = base['valid'] & (base['s_m'] > .06) & (base['s_m'] < .14)
        expected = -(.008 + .001 * np.sin(2 * np.pi * base['s_m'] / .05))
        self.assertLess(np.abs(base['xyz'][inner, 1] - expected[inner]).max(), .0004)
        np.testing.assert_allclose(base['xyz'][inner, 2], 0, atol=3e-4)
        self.assertGreaterEqual(base['views'][inner].min(), 3)
        alone = bs.fuse_toes([views[0], view(0., .18, .2)])
        self.assertFalse(alone['toes'][0]['valid'][10:140].any())     # seen by one view only


if __name__ == '__main__':
    unittest.main()
