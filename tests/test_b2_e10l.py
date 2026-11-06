"""B2 with the E10L-Pro welding arm: profile, floating articulation and arm kinematics on CPU."""

import json
import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from config import SCENE, scene_config_for_robot
from main import parse_args

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets/robots/b2_e10l/manifest.json"
URDF = ROOT / "assets/vendor/b2_e10l/B2-E10L.urdf"


def origin(node):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    result[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
    return result


class B2E10LProfileTests(unittest.TestCase):
    def test_profile_holds_legs_and_leaves_default_scene(self):
        config = scene_config_for_robot("b2_e10l")
        self.assertEqual(scene_config_for_robot("ur5e"), SCENE)
        self.assertEqual(config.joint_names, tuple(f"e10l_joint{i}" for i in range(1, 7)))
        legs = [name for name, _ in config.hold_joint_positions]
        self.assertEqual(len(legs), 12)
        self.assertTrue(all(name.endswith(("_hip_joint", "_thigh_joint", "_calf_joint")) for name in legs))
        self.assertIsNone(config.robot_stand_size_xy)
        self.assertTrue(config.camera_on_base_link)
        self.assertEqual(parse_args(["--robot", "b2_e10l"]).robot, "b2_e10l")
        # The rear feet stay on the floor.
        floor_y = config.floor_center_world[1] - config.floor_size[1] / 2
        self.assertLess(floor_y, config.robot_base_position_world[1] - 0.3285 - 0.1)

    def test_tool_frame_matches_built_manifest(self):
        config = scene_config_for_robot("b2_e10l")
        manifest = json.loads(MANIFEST.read_text())
        expected = np.array(manifest["link6_from_tcp"])
        actual = np.eye(4)
        actual[:3, :3] = Rotation.from_quat(config.flange_quaternion_wxyz, scalar_first=True).as_matrix()
        actual[:3, 3] = config.flange_translation
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        self.assertEqual(list(config.joint_names), manifest["arm_joints"])
        self.assertEqual(sorted(name for name, _ in config.hold_joint_positions), sorted(manifest["leg_joints"]))


@unittest.skipIf(Usd is None, "USD Python bindings are not installed")
class B2E10LAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = scene_config_for_robot("b2_e10l")
        if not Path(cls.config.robot_local_usd).is_file():
            raise unittest.SkipTest("Build the B2-E10L USD with scripts/build_b2_e10l.py")

    def compose(self):
        from robot.asset import attach_tool, author_joint_state

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        mount = UsdGeom.Xform.Define(stage, self.config.robot_mount_prim_path)
        mount.AddTranslateOp().Set(Gf.Vec3d(*self.config.robot_base_position_world))
        mount.AddOrientOp().Set(Gf.Quatf(*self.config.robot_base_quaternion_wxyz))
        UsdGeom.Xform.Define(stage, self.config.robot_prim_path).GetPrim().GetReferences().AddReference(
            self.config.robot_local_usd)
        attach_tool(stage, self.config)
        author_joint_state(stage, self.config)
        return stage

    def test_one_floating_articulation_with_legs_and_arm(self):
        from robot.asset import articulation_path

        stage = self.compose()
        root = articulation_path(stage, self.config.robot_prim_path)
        self.assertTrue(root.endswith("/base_link"))
        joints = [p for p in Usd.PrimRange(stage.GetPrimAtPath(self.config.robot_prim_path)) if p.IsA(UsdPhysics.Joint)]
        self.assertFalse([p for p in joints if not UsdPhysics.Joint(p).GetBody0Rel().GetTargets()])
        revolute = {p.GetName() for p in joints if p.IsA(UsdPhysics.RevoluteJoint)}
        self.assertEqual(revolute, set(self.config.joint_names) | {n for n, _ in self.config.hold_joint_positions})
        masses = [UsdPhysics.MassAPI(p).GetMassAttr().Get() for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        self.assertTrue(all(m > 1e-3 for m in masses), "massless links destabilise the articulation")
        manifest = json.loads(MANIFEST.read_text())
        expected = 74.6 + manifest["assumptions"]["arm_mass_kg"] + manifest["assumptions"]["torch_mass_kg"]
        self.assertAlmostEqual(sum(masses), expected + manifest["mount_plate"]["mass_kg"], delta=0.2)

    def test_initial_state_and_drive_units(self):
        stage = self.compose()
        joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
        for name, angle in self.config.hold_joint_positions:
            prim = joints[name]
            self.assertAlmostEqual(prim.GetAttribute("state:angular:physics:position").Get(), math.degrees(angle), places=4)
            drive = UsdPhysics.DriveAPI(prim, "angular")
            self.assertAlmostEqual(drive.GetTargetPositionAttr().Get(), math.degrees(angle), places=4)
            self.assertAlmostEqual(drive.GetStiffnessAttr().Get(), self.config.hold_stiffness * math.pi / 180, places=3)
        home = dict(zip(self.config.joint_names, self.config.home_joint_positions))
        for name, angle in home.items():
            self.assertAlmostEqual(joints[name].GetAttribute("state:angular:physics:position").Get(),
                                   math.degrees(angle), places=4)

    def test_arm_fk_matches_composed_urdf(self):
        from robot.kinematics import SerialKinematics

        if not URDF.is_file():
            self.skipTest("Compose the URDF with scripts/build_b2_e10l.py --urdf-only")
        stage = self.compose()
        model = SerialKinematics.from_stage(stage, self.config)
        self.assertIn("/base_link/e10l_mount_plate/e10l_base_link/", model.tcp_path)
        joints = {j.get("name"): j for j in ET.parse(URDF).getroot().findall("joint")}
        manifest = json.loads(MANIFEST.read_text())
        trunk = np.eye(4)
        trunk[:3, :3] = Rotation.from_quat(self.config.robot_base_quaternion_wxyz, scalar_first=True).as_matrix()
        trunk[:3, 3] = self.config.robot_base_position_world
        np.testing.assert_allclose(model.world_from_base, trunk @ np.array(manifest["trunk_from_arm_base"]), atol=1e-6)
        model.world_from_base = np.eye(4)
        home = np.asarray(self.config.home_joint_positions)
        self.assertTrue(np.all(home > model.limits[:, 0]) and np.all(home < model.limits[:, 1]))
        for q in (np.zeros(6), home, np.array([.3, -.2, .5, -.4, .6, -.1])):
            pose = np.eye(4)
            for name, angle in zip(self.config.joint_names, q):
                moving = np.eye(4)
                moving[:3, :3] = Rotation.from_rotvec(
                    np.fromstring(joints[name].find("axis").get("xyz"), sep=" ") * angle).as_matrix()
                pose = pose @ origin(joints[name].find("origin")) @ moving
            np.testing.assert_allclose(model.forward(q), pose @ np.array(manifest["link6_from_tcp"]), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
