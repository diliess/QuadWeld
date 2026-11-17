"""Independent stage APIs: metric world frames, fusion support and refusals."""
import subprocess
import sys
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tests.test_bead_seam import _scene as bead_scene
from tests.test_stereo_seam import _scene as root_scene, _line_distance
from weld_vision import coarse_localization as coarse
from weld_vision import multiview_localization as close


class LocalizationPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A translated/rotated shared frame makes camera/world confusion visible.
        angle = np.radians(31.)
        cls.rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                                 [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        cls.translation = np.array([1.2, -.7, .3])
        left, right, k, baseline, a, b, _, pose = root_scene(
            size=(520, 400), focal=600., position=(.06, -1.6, 1.1))
        cls.observation = coarse.locate(left, right, k, baseline, a, b, cls.world_pose(pose),
                                        ident='overview', require_t_joint=True)
        cls.views = []
        for index, position in enumerate(((-.04, -.34, .28), (.04, -.34, .28))):
            left, right, k, baseline, labels, pose = bead_scene(position=position, roll=-35.)
            cls.views.append(close.locate_view(left, right, k, baseline, labels==1, labels==2,
                                               cls.world_pose(pose), cls.observation, ident=f'near_{index}'))

    @classmethod
    def world_pose(cls, pose):
        rotation, origin = pose
        world = np.eye(4)
        world[:3, :3] = cls.rotation @ rotation
        world[:3, 3] = cls.rotation @ origin + cls.translation
        return world

    def test_two_stages_measure_in_the_shared_world_frame(self):
        self.assertEqual(self.observation['status'], 'ok')
        truth = np.array([[-.25, 0, 0], [.25, 0, 0]]) @ self.rotation.T + self.translation
        self.assertLess(_line_distance(truth, self.observation['root_world_m']).max(), .005)
        self.assertTrue(1.7 < self.observation['estimated_range_m'] < 2.2)
        for view in self.views:
            self.assertEqual(view['status'], 'ok')
            self.assertLess(_line_distance(truth, view['root']).max(), .002)
        fused = close.fuse_views(self.views, minimum=2)
        self.assertEqual(fused['status'], 'ok')
        self.assertLess(_line_distance(truth, fused['root']['ends']).max(), .002)
        for toe in fused['toes']:
            self.assertGreater(toe['valid'].sum(), 20)
            self.assertTrue(np.all(toe['views'][toe['valid']] >= 2))
            local = (toe['xyz'][toe['valid']] - self.translation) @ self.rotation
            self.assertLess(np.max(np.abs(local[:, 1:])), .025)

    def test_missing_and_duplicate_views_cannot_supply_two_view_support(self):
        self.assertEqual(close.fuse_views(self.views[:1])['status'], 'insufficient_views')
        with self.assertRaisesRegex(ValueError, 'Duplicate view'):
            close.fuse_views([self.views[0], self.views[0]])

    def test_root_mode_is_explicit_and_does_not_require_toes(self):
        views = [{'id':str(i), 'mode':'root', 'status':'ok', 'root':v['root']} for i,v in enumerate(self.views)]
        fused = close.fuse_views(views, mode='root')
        self.assertEqual(fused['status'], 'ok')
        self.assertEqual(fused['toes'], [])
        with self.assertRaisesRegex(ValueError, 'mix root and bead'):
            close.fuse_views(views, mode='bead')

    def test_export_ignores_failed_views_and_preserves_failure_status(self):
        views = [{'id': str(i), 'mode': 'root', 'status': 'ok', 'root': v['root']}
                 for i, v in enumerate(self.views)]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            record = close.export_fusion(output, 'joint', [{'id': 'failed', 'status': 'no_plate_pair'}, *views],
                                         minimum=2, mode='root')
            self.assertEqual(record['status'], 'ok')
            self.assertEqual(record['views_used'], ['0', '1'])
            self.assertEqual(record['toes'], {})
            self.assertTrue((output / 'joint' / 'root.csv').is_file())
            record = close.export_fusion(output, 'insufficient', views[:1], minimum=2, mode='root')
            self.assertEqual(record['status'], 'insufficient_views')
            self.assertEqual(json.loads((output / 'insufficient' / 'fused.json').read_text()), record)


class InputContractTests(unittest.TestCase):
    def test_failure_status_crosses_the_stage_boundary_without_using_geometry(self):
        image = np.zeros((32, 32), np.uint8)
        mask = np.zeros_like(image, bool)
        k = np.array([[100.,0,15.5], [0,100.,15.5], [0,0,1.]])
        observed = coarse.locate(image, image, k, .09, mask, mask, np.eye(4))
        self.assertEqual(observed['status'], 'plate_mask_too_small')
        result = close.locate_view(image, image, k, .09, mask, mask, np.eye(4), observed)
        self.assertEqual(result['status'], 'coarse_not_ready')
        bad_pose = np.eye(4); bad_pose[0, 0] = 2.
        with self.assertRaisesRegex(ValueError, 'rigid'):
            coarse.locate(image, image, k, .09, mask, mask, bad_pose)

    def test_packages_import_without_simulator_or_model_runtime(self):
        code = ('import sys; from weld_vision import coarse_localization, multiview_localization; '
                'assert not set(("isaacsim", "omni", "mujoco", "pxr", "torch", "sam3")) & set(sys.modules)')
        subprocess.run([sys.executable, '-c', code], check=True)


if __name__ == '__main__':
    unittest.main()
