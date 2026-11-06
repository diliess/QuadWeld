"""Every bundle under assets/scenes/ follows one portable-asset contract."""

from dataclasses import replace
import filecmp
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    from pxr import Usd, UsdGeom, UsdLux, UsdPhysics
except ImportError:
    Usd = None


SCENES = Path(__file__).resolve().parents[1] / "assets/scenes"
SHARED_KEYS = ("asset", "default_prim", "units", "up_axis", "origin", "robots_included", "source")


@unittest.skipIf(Usd is None, "USD Python bindings are not installed")
class SceneBundleTests(unittest.TestCase):
    @staticmethod
    def bounds(stage, path):
        return UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
                                False, True).ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()

    def test_bundles_share_layout_units_and_manifest_keys(self):
        bundles = sorted(path for path in SCENES.iterdir() if path.is_dir())
        self.assertEqual([path.name for path in bundles], ["open-box", "plate-grid", "ship-grid"])
        for bundle in bundles:
            with self.subTest(bundle=bundle.name):
                manifest = json.loads((bundle / "manifest.json").read_text())
                self.assertTrue(set(SHARED_KEYS) <= manifest.keys())
                self.assertEqual(manifest["asset"], bundle.name.replace("-", "_") + ".usda")
                self.assertEqual((manifest["units"], manifest["up_axis"]), ("metres", "Z"))
                self.assertFalse(manifest["robots_included"])
                stage = Usd.Stage.Open(str(bundle / manifest["asset"]))
                root = stage.GetDefaultPrim()
                self.assertEqual(str(root.GetPath()), manifest["default_prim"])
                self.assertEqual(UsdGeom.GetStageUpAxis(stage), UsdGeom.Tokens.z)
                self.assertEqual(UsdGeom.GetStageMetersPerUnit(stage), 1.)
                self.assertEqual([child.GetName() for child in root.GetChildren()], ["Looks", "Parts"])
                prims = list(stage.Traverse())
                self.assertTrue(any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in prims))
                for prim in prims:
                    self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
                    self.assertFalse(prim.HasAPI(UsdPhysics.ArticulationRootAPI))
                    self.assertFalse(prim.IsA(UsdGeom.Camera))
                    self.assertFalse(prim.HasAPI(UsdLux.LightAPI))

    def test_generated_bundles_rebuild_from_their_manifest(self):
        from config import WorkpieceConfig
        from scene.workpieces import export_workpiece_bundle

        for kind in ("plate-grid", "open-box"):
            committed = SCENES / kind
            config = WorkpieceConfig(**json.loads((committed / "manifest.json").read_text())["workpiece"])
            with tempfile.TemporaryDirectory() as folder:
                asset = export_workpiece_bundle(folder, config)
                for name in (asset.name, "manifest.json"):
                    self.assertTrue(filecmp.cmp(Path(folder) / name, committed / name, shallow=False),
                                    f"{kind}/{name} is stale; rebuild it with export_workpiece_bundle")

    def test_workpiece_reference_lands_on_plate_frame_and_checks_units(self):
        from config import SCENE, WorkpieceConfig
        from scene.workpieces import add_workpiece, material

        config = replace(SCENE, workpiece=WorkpieceConfig(kind="open-box"))
        asset = SCENES / "open-box/open_box.usda"
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        add_workpiece(stage, config, asset, material(stage, "/World/Looks/Stand", (0.1, 0.1, 0.1)))
        floor = self.bounds(stage, "/World/Workpiece/Assembly/Model/Parts/Bottom")
        x, y, _ = config.plate_center_world
        np.testing.assert_allclose([*floor.GetMidpoint()[:2], floor.GetMax()[2]],
                                   [x, y, config.plate_top_z_world], atol=1e-7)

        centimetres = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(centimetres, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(centimetres, .01)
        with self.assertRaisesRegex(ValueError, "metres"):
            add_workpiece(centimetres, config, asset, material(centimetres, "/World/Looks/Stand", (0.1, 0.1, 0.1)))


if __name__ == "__main__":
    unittest.main()
