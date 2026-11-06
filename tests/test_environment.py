"""Imported environments retain scale, open cells, contact and portable assets."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from main import parse_args

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    Usd = None


BUNDLE = Path(__file__).resolve().parents[1] / "assets/scenes/ship-grid"
ASSET = BUNDLE / "ship_grid.usda"


class EnvironmentModeTests(unittest.TestCase):
    def test_environment_uses_preview_and_preserves_robot_selection(self):
        self.assertFalse(parse_args([]).scene_only)
        for robot in ("ur5e", "rm65"):
            args = parse_args(["--environment-usd", str(ASSET), "--robot", robot])
            self.assertTrue(args.scene_only)
            self.assertEqual(args.robot, robot)
            self.assertEqual(args.environment_usd, ASSET.resolve())

    def test_unsupported_modes_fail_before_simulator_launch(self):
        for extra in (["--plan-only"], ["--demo-open"], ["--workpiece", "open-box"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--environment-usd", str(ASSET)] + extra)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--environment-usd", str(BUNDLE / "missing.usda")])


@unittest.skipIf(Usd is None, "USD Python bindings are not installed")
class EnvironmentGeometryTests(unittest.TestCase):
    @staticmethod
    def bounds(stage, path):
        return UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
                                False, True).ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()

    def test_asset_has_exact_static_box_colliders_and_no_robot(self):
        stage = Usd.Stage.Open(str(ASSET))
        self.assertEqual(str(stage.GetDefaultPrim().GetPath()), "/ShipGrid")
        self.assertEqual(UsdGeom.GetStageUpAxis(stage), UsdGeom.Tokens.z)
        self.assertEqual(UsdGeom.GetStageMetersPerUnit(stage), 1.)
        colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        self.assertEqual(len(colliders), 33)
        for prim in colliders:
            self.assertTrue(prim.IsA(UsdGeom.Cube))
            self.assertTrue(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
        for prim in stage.Traverse():
            self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertFalse(prim.IsA(UsdPhysics.Joint))
            self.assertFalse(prim.IsA(UsdGeom.Camera))
        deck = self.bounds(stage, "/ShipGrid/Parts/ship_grid_deck/Collision")
        np.testing.assert_allclose(deck.GetSize(), [6.45, 6.45, .01], atol=1e-6)
        self.assertAlmostEqual(deck.GetMax()[2], 0.)

    def test_all_nine_cells_stay_open_and_two_metres_clear(self):
        stage = Usd.Stage.Open(str(ASSET))
        for row in range(3):
            for column in range(3):
                base = "/ShipGrid/Parts/ship_grid_"
                west = self.bounds(stage, f"{base}column_{column}_segment_{row}_web/Collision")
                east = self.bounds(stage, f"{base}column_{column + 1}_segment_{row}_web/Collision")
                south = self.bounds(stage, f"{base}row_{row}_web/Collision")
                north = self.bounds(stage, f"{base}row_{row + 1}_web/Collision")
                xmin, xmax = west.GetMax()[0], east.GetMin()[0]
                ymin, ymax = south.GetMax()[1], north.GetMin()[1]
                np.testing.assert_allclose([xmax - xmin, ymax - ymin], [2., 2.], atol=1e-6)
                np.testing.assert_allclose([west.GetMin()[1], west.GetMax()[1]], [ymin, ymax], atol=1e-6)
                self.assertLessEqual(south.GetMin()[0], xmin)
                self.assertGreaterEqual(south.GetMax()[0], xmax)
                centre = Gf.Vec3d((xmin+xmax)/2, (ymin+ymax)/2, .1)
                for prim in stage.Traverse():
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        self.assertFalse(self.bounds(stage, str(prim.GetPath())).Contains(centre))
        web = self.bounds(stage, "/ShipGrid/Parts/ship_grid_row_0_web/Collision")
        np.testing.assert_allclose([web.GetMin()[2], web.GetMax()[2]], [.01, .23], atol=1e-6)

    def test_bundle_relocates_with_textures_contact_and_parent_placement(self):
        from scene.environment import add_environment, environment_camera_view

        with tempfile.TemporaryDirectory() as folder:
            copied = Path(folder) / "bundle"
            shutil.copytree(BUNDLE, copied)
            source = Usd.Stage.Open(str(copied / ASSET.name))
            textures = [attr for prim in source.Traverse() for attr in prim.GetAttributes()
                        if attr.GetName() == "inputs:file"]
            self.assertEqual(len(textures), 3)
            for attr in textures:
                path = attr.Get()
                self.assertFalse(Path(path.path).is_absolute())
                self.assertTrue(Path(path.resolvedPath).is_file())
                self.assertTrue(Path(path.resolvedPath).is_relative_to(copied))
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.)
            root = add_environment(stage, copied / ASSET.name, at=(10, 20, 3))
            prefix = str(root.GetPath())
            deck_path = prefix + "/Parts/ship_grid_deck/Collision"
            deck = self.bounds(stage, deck_path)
            np.testing.assert_allclose(deck.GetMidpoint(), [10, 20, 2.995], atol=1e-6)
            prim = stage.GetPrimAtPath(deck_path)
            contact, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            self.assertTrue(contact)
            self.assertAlmostEqual(UsdPhysics.MaterialAPI(contact.GetPrim()).GetStaticFrictionAttr().Get(), .65, places=6)
            self.assertAlmostEqual(UsdPhysics.MaterialAPI(contact.GetPrim()).GetDynamicFrictionAttr().Get(), .65, places=6)
            eye, target = environment_camera_view(stage)
            np.testing.assert_allclose(target[:2], [10, 20], atol=1e-6)
            self.assertGreater(eye[2], target[2])
            with self.assertRaisesRegex(ValueError, "already exists"):
                add_environment(stage, copied / ASSET.name)

    def test_import_rejects_scale_axis_and_missing_default_prim(self):
        from scene.environment import add_environment

        for axis, units, default in (("Y", 1., True), ("Z", .01, True), ("Z", 1., False)):
            with tempfile.TemporaryDirectory() as folder:
                asset = Path(folder) / "invalid.usda"
                source = Usd.Stage.CreateNew(str(asset))
                root = UsdGeom.Xform.Define(source, "/Root")
                if default:
                    source.SetDefaultPrim(root.GetPrim())
                UsdGeom.SetStageUpAxis(source, axis)
                UsdGeom.SetStageMetersPerUnit(source, units)
                source.GetRootLayer().Save()
                stage = Usd.Stage.CreateInMemory()
                UsdGeom.SetStageUpAxis(stage, "Z")
                UsdGeom.SetStageMetersPerUnit(stage, 1.)
                with self.assertRaises(ValueError):
                    add_environment(stage, asset)
                self.assertFalse(stage.GetPrimAtPath("/World/Environment"))


if __name__ == "__main__":
    unittest.main()
