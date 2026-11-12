"""Missing semantic support can recover a measured corner without annotations."""

import unittest

import numpy as np

from vision.weld_detector import detect


def corner_scene():
    height, width = 96, 128
    k = np.array([[100., 0, 63.5], [0, 100., 47.5], [0, 0, 1]])
    _, xx = np.indices((height, width))
    # Two perpendicular planes meeting at optical Z=1 m. A separate flat
    # foreground patch models an irrelevant semantic object.
    depth = (1/(1+np.abs((xx-k[0, 2])/k[0, 0]))).astype(np.float32)
    wrong_mask = np.zeros_like(depth, dtype=bool)
    wrong_mask[68:92, 90:123] = True
    depth[wrong_mask] = .4
    rgb = np.full((*depth.shape, 3), 120, dtype=np.uint8)
    return depth, k, rgb, wrong_mask


class CornerRecoveryTests(unittest.TestCase):
    def test_wrong_flat_mask_cannot_hide_an_observed_corner(self):
        depth, k, rgb, wrong_mask = corner_scene()
        restricted = detect(depth, k, rgb, plate_masks=[wrong_mask])
        self.assertEqual(len(restricted['joints']), 0)
        recovered = detect(depth, k, rgb, plate_masks=[wrong_mask], recover_missing_corners=True)
        self.assertTrue(recovered['corner_recovery']['attempted'])
        self.assertEqual(recovered['plate_sampling_mode'], 'geometry_recovery_after_sam')
        self.assertGreater(len(recovered['joints']), 0)
        closest = min(recovered['joints'], key=lambda j: np.median(np.abs(j['xy'][:, 0]-63.5)))
        self.assertLess(np.median(np.abs(closest['xy'][:, 0]-63.5)), 1.)
        self.assertLess(np.median(np.abs(closest['xyz_camera_estimated_m'][:, 2]-1.)), .003)
        # The rejected semantic seed remains available for diagnosis.
        np.testing.assert_array_equal(recovered['semantic_plate_mask'], wrong_mask)

    def test_valid_semantic_corner_is_retained(self):
        depth, k, rgb, _ = corner_scene()
        masks = np.zeros((2, *depth.shape), dtype=bool)
        masks[0, :, :63] = True
        masks[1, :, 65:] = True
        result = detect(depth, k, rgb, plate_masks=masks, recover_missing_corners=True)
        self.assertGreater(len(result['joints']), 0)
        self.assertFalse(result['corner_recovery']['attempted'])
        self.assertEqual(result['plate_sampling_mode'], 'sam_interiors')

    def test_no_invented_corner_on_a_flat_surface(self):
        depth, k, rgb, wrong_mask = corner_scene()
        depth[:] = .7
        result = detect(depth, k, rgb, plate_masks=[wrong_mask], recover_missing_corners=True)
        self.assertTrue(result['corner_recovery']['attempted'])
        self.assertEqual(len(result['joints']), 0)


if __name__ == '__main__':
    unittest.main()
