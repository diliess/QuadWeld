"""Scene refreshes must keep current geometry, sensor frames and valid presets together."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pxr import Usd

from main import scene_setup
from scripts.import_industrial_welding import import_scene

BUNDLE = Path(__file__).resolve().parents[1] / 'assets/scenes/industrial-welding'


class SceneImportTests(unittest.TestCase):
    def test_updated_hall_and_deposited_specimens_are_present(self):
        manifest = json.loads((BUNDLE/'manifest.json').read_text())
        config = json.loads((BUNDLE/'source/config.json').read_text())
        self.assertEqual(manifest['layout'], {'hall_size_m': [32, 40], 'A_stations': 28})
        self.assertEqual(len(config['stations']), 28)
        self.assertEqual(hashlib.sha256((BUNDLE/'industrial_welding.usdc').read_bytes()).hexdigest(),
                         manifest['source']['stage_sha256'])
        stage = Usd.Stage.Open(str(BUNDLE/'industrial_welding_full.usda'))
        for station in ('A25', 'A26', 'A27', 'A28'):
            prim = stage.GetPrimAtPath('/World/Fixtures/'+station)
            self.assertEqual(prim.GetAttribute('welding:depositionState').Get(), 'first_layer_deposited')
            self.assertEqual(prim.GetAttribute('welding:completedLayers').Get(), 1)

    def test_default_capture_uses_current_source_pose_and_rejects_retired_a04(self):
        setup = scene_setup()
        self.assertEqual(setup['pose'], 'original')
        self.assertIsNone(setup['station'])
        np.testing.assert_allclose(setup['base_xy_yaw'][:2], [-14, 0], atol=1e-8)
        np.testing.assert_allclose(setup['arm_q_rad'], [0, 1.2, 2.3, 0, 1.3, 0], atol=1e-8)
        with self.assertRaisesRegex(ValueError, 'replan the workstation'):
            scene_setup(pose_name='A04_initial_arm')
        self.assertFalse(list(BUNDLE.glob('industrial_welding_*_photo.usda')))

    def test_failed_rebuild_does_not_replace_existing_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'; source.mkdir()
            destination = root/'bundle'; destination.mkdir()
            user = destination/'saved.usda'
            user.write_text('original user layer')

            def fail_after_writes(_source, staging):
                (staging/'saved.usda').write_text('incomplete replacement')
                (staging/'partial.usdc').write_bytes(b'partial')
                raise ValueError('invalid new scene')

            with patch('scripts.import_industrial_welding.build_bundle', side_effect=fail_after_writes):
                with self.assertRaisesRegex(ValueError, 'invalid new scene'):
                    import_scene(source, destination)
            self.assertEqual(user.read_text(), 'original user layer')
            self.assertFalse((destination/'partial.usdc').exists())
            self.assertFalse(list(root.glob('.welding-import-*')))


if __name__ == '__main__':
    unittest.main()
