"""RGB calibration must retain SDK units, frame direction and off-centre projection."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pxr import Gf, Usd, UsdGeom

from weld_scene.arm_kinematics import ArmKinematics
from weld_scene.usd import RGB_CAMERA, SOURCE_CAMERA, camera_intrinsics, set_intrinsics
from weld_scene.vendor_calibration import (DEFAULT_RGB_CALIBRATION, load_rgb_calibration,
                                     resize_rgb_calibration, rgb_pose_in_midpoint)
from weld_scene.scene import inspect_stage, scene_setup


class VendorRGBTests(unittest.TestCase):
    def test_vendor_values_and_translation_units_are_preserved(self):
        rgb = load_rgb_calibration()
        self.assertEqual(rgb['image_size_wh'], [3120, 4096])
        self.assertEqual(rgb['device_serial'], 'Jh26B196')
        k = np.asarray(rgb['intrinsics_opencv'])
        np.testing.assert_allclose(k.diagonal(), [3142.534912109375, 3142.534912109375, 1])
        np.testing.assert_allclose(k[:2, 2], [1556.43310546875, 2117.60546875])
        self.assertEqual(rgb['source_sha256'], hashlib.sha256(DEFAULT_RGB_CALIBRATION.read_bytes()).hexdigest())
        np.testing.assert_allclose(np.asarray(rgb['T_rgb_from_left_m'])[:3, 3],
                                   [-.05816891860961914, -.0107757568359375, -.013990567207336426])
        self.assertFalse(rgb['rectification_map_supplied'])

    def test_left_to_rgb_transform_is_not_mistaken_for_a_camera_pose(self):
        rgb = load_rgb_calibration()
        left_points = np.array([[.02, -.01, .5, 1], [-.1, .04, .7, 1]])
        midpoint_points = left_points.copy()
        midpoint_points[:, 0] -= .045
        rgb_from_midpoint = np.linalg.inv(rgb_pose_in_midpoint(rgb, .09))
        actual = midpoint_points @ rgb_from_midpoint.T
        raw = json.loads(DEFAULT_RGB_CALIBRATION.read_bytes())
        transform = np.asarray(raw['T_optical_camera'])
        expected = left_points[:, :3] @ transform[:3, :3].T + transform[:3, 3]/1000
        np.testing.assert_allclose(actual[:, :3], expected, atol=1e-12)
        self.assertLess(np.linalg.norm(rgb_pose_in_midpoint(rgb, .09)[:3, 3]), .03)

    def test_resized_principal_point_obeys_pixel_centre_conversion(self):
        rgb = load_rgb_calibration()
        small = resize_rgb_calibration(rgb, [780, 1024])
        np.testing.assert_allclose(np.asarray(small['intrinsics'])[:2], np.asarray(rgb['intrinsics'])[:2]/4)
        expected = (np.asarray(rgb['intrinsics_opencv'])[:2, 2]+.5)/4-.5
        np.testing.assert_allclose(np.asarray(small['intrinsics_opencv'])[:2, 2], expected)
        self.assertEqual(rgb['image_size_wh'], [3120, 4096])
        for invalid in ([0, 1024], [780.5, 1024], [780]):
            with self.assertRaises(ValueError):
                resize_rgb_calibration(rgb, invalid)

    def test_invalid_units_frames_or_raw_distortion_are_rejected(self):
        raw = json.loads(DEFAULT_RGB_CALIBRATION.read_bytes())
        for key, value in [('length_unit', 'm'), ('point_frame', 'right_eye'),
                           ('distortion', [.1, 0, 0, 0, 0])]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                record = dict(raw, **{key: value})
                path = Path(directory)/'rgb.json'
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    load_rgb_calibration(path)

    def test_usd_projection_matches_off_centre_vendor_raster(self):
        rgb = load_rgb_calibration()
        for size in ([3120, 4096], [780, 1024]):
            profile = resize_rgb_calibration(rgb, size)
            stage = Usd.Stage.CreateInMemory()
            camera = UsdGeom.Camera.Define(stage, '/Camera')
            k = np.asarray(profile['intrinsics'])
            set_intrinsics(camera, k, size)
            np.testing.assert_allclose(camera_intrinsics(camera, size), k, atol=.001, rtol=0)
            projection = camera.GetCamera(0).frustum.ComputeProjectionMatrix()
            for x, y, z in ((0, 0, .5), (.1, -.12, .6), (-.08, .15, .8)):
                ndc = projection.Transform(Gf.Vec3d(x, -y, -z))
                pixel = np.array([size[0]*(ndc[0]+1)/2, size[1]*(1-ndc[1])/2])
                expected = (k @ [x/z, y/z, 1])[:2]
                np.testing.assert_allclose(pixel, expected, atol=.001, rtol=0)

    def test_portable_rgb_extrinsics_follow_the_left_eye_during_arm_motion(self):
        stage = Usd.Stage.Open('assets/scenes/industrial-welding/industrial_welding_full.usda')
        stage.SetEditTarget(stage.GetSessionLayer())
        arm = ArmKinematics(stage)
        transform = np.asarray(load_rgb_calibration()['T_rgb_from_left_m'])
        for offset in (0, np.radians(8)):
            q = arm.initial.copy()
            q[0] += offset
            arm.apply(q)
            cache = UsdGeom.XformCache()
            optical = np.diag([1., -1, -1, 1])
            world = lambda p: np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p))).T @ optical
            left = world(RGB_CAMERA.rsplit('/', 1)[0]+'/left')
            actual = np.linalg.inv(world(RGB_CAMERA)) @ left
            np.testing.assert_allclose(actual, transform, atol=2e-7, rtol=0)

    def test_scene_setup_supports_vendor_and_legacy_calibration(self):
        for resolution in (None, [780, 1024]):
            setup = scene_setup('vendor', resolution)
            self.assertEqual(setup['camera'], RGB_CAMERA)
            stage = Usd.Stage.Open(setup['usd'])
            calibration, _ = inspect_stage(stage, setup)
            np.testing.assert_allclose(calibration['camera_matrix'], setup['rgb_calibration']['intrinsics'],
                                       atol=.001, rtol=0)
        legacy = scene_setup('legacy')
        self.assertEqual(legacy['camera'], SOURCE_CAMERA)
        self.assertEqual(legacy['resolution_wh'], [640, 1100])
        calibration, _ = inspect_stage(Usd.Stage.Open(legacy['usd']), legacy)
        np.testing.assert_allclose(np.asarray(calibration['camera_matrix'])[:2, 2], [320, 550])


if __name__ == '__main__':
    unittest.main()
