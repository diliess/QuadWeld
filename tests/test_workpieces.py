"""Check physical gaps, the open cavity, and reusable USD composition on CPU."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from config import WorkpieceConfig
from main import parse_args

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    Usd = None


class WorkpieceModeTests(unittest.TestCase):
    def test_preview_does_not_enter_unsupported_planner(self):
        self.assertFalse(parse_args([]).scene_only)
        for kind in ("plate-grid", "open-box"):
            self.assertTrue(parse_args(["--workpiece", kind]).scene_only)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--workpiece", kind, "--plan-only"])

    def test_invalid_dimensions_fail_before_simulator_starts(self):
        for option, value in (("--seam-gap", "0"), ("--tile-size", "nan"),
                              ("--grid-count", "1"), ("--wall-height", "-0.1")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--workpiece", "plate-grid", option, value])


@unittest.skipIf(Usd is None, "USD Python bindings are not installed")
class WorkpieceGeometryTests(unittest.TestCase):
    def export_and_open(self, folder, config):
        from scene.workpieces import export_workpiece_asset

        path = export_workpiece_asset(Path(folder) / "fixture.usda", config)
        stage = Usd.Stage.Open(str(path))
        self.assertEqual(str(stage.GetDefaultPrim().GetPath()), "/Workpiece")
        self.assertEqual(UsdGeom.GetStageUpAxis(stage), UsdGeom.Tokens.z)
        self.assertEqual(UsdGeom.GetStageMetersPerUnit(stage), 1.0)
        boxes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Cube)]
        for prim in boxes:
            self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
            self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
            bound_material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            self.assertTrue(bound_material)
        return stage, boxes, path

    @staticmethod
    def bounds(stage, prim):
        return UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()

    def test_grid_has_empty_gaps_and_correct_outer_extent(self):
        config = WorkpieceConfig(kind="plate-grid", grid_count=3, tile_size_m=0.14, gap_m=0.005)
        with tempfile.TemporaryDirectory() as folder:
            stage, boxes, _ = self.export_and_open(folder, config)
            self.assertEqual(len(boxes), 9)
            bounds = [self.bounds(stage, prim) for prim in boxes]
            left = self.bounds(stage, stage.GetPrimAtPath("/Workpiece/Parts/Plate_0_0"))
            right = self.bounds(stage, stage.GetPrimAtPath("/Workpiece/Parts/Plate_0_1"))
            self.assertAlmostEqual(right.GetMin()[0] - left.GetMax()[0], 0.005, places=7)
            gap_point = Gf.Vec3d((left.GetMax()[0] + right.GetMin()[0]) / 2,
                                left.GetMidpoint()[1], -config.thickness_m / 2)
            self.assertFalse(any(box.Contains(gap_point) for box in bounds))
            outer = self.bounds(stage, stage.GetDefaultPrim())
            np.testing.assert_allclose(outer.GetSize(), [0.43, 0.43, 0.006], atol=1e-7)
            self.assertAlmostEqual(outer.GetMax()[2], 0.0, places=7)

    def test_box_cavity_is_empty_and_walls_meet_floor(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, boxes, _ = self.export_and_open(folder, WorkpieceConfig(kind="open-box"))
            self.assertEqual(len(boxes), 5)
            bounds = [self.bounds(stage, prim) for prim in boxes]
            self.assertFalse(any(box.Contains(Gf.Vec3d(0, 0, 0.06)) for box in bounds))
            bottom = self.bounds(stage, stage.GetPrimAtPath("/Workpiece/Parts/Bottom"))
            for prim in boxes:
                if prim.GetName() != "Bottom":
                    self.assertAlmostEqual(self.bounds(stage, prim).GetMin()[2], bottom.GetMax()[2], places=7)
            outer = self.bounds(stage, stage.GetDefaultPrim())
            np.testing.assert_allclose(outer.GetSize(), [0.4, 0.4, 0.126], atol=1e-7)

    def test_reference_keeps_materials_and_respects_parent_placement(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, path = self.export_and_open(folder, WorkpieceConfig(kind="open-box"))
            scene = Usd.Stage.CreateInMemory()
            mount = UsdGeom.Xform.Define(scene, "/Placed")
            mount.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
            scene.DefinePrim("/Placed/Model", "Xform").GetReferences().AddReference(str(path))
            prim = scene.GetPrimAtPath("/Placed/Model/Parts/Bottom")
            self.assertTrue(prim)
            self.assertAlmostEqual(self.bounds(scene, prim).GetMax()[2], 3.0, places=7)
            bound_material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            self.assertEqual(str(bound_material.GetPath()), "/Placed/Model/Looks/Steel")


if __name__ == "__main__":
    unittest.main()
