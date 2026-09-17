"""Robot selection and imported RM65 kinematics against the manufacturer's URDF."""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from config import SCENE, scene_config_for_robot

try:
    from pxr import Usd, UsdGeom
except ImportError:
    Usd = None


class RobotProfileTests(unittest.TestCase):
    def test_profiles_do_not_mutate_default_scene(self):
        rm65 = scene_config_for_robot("rm65")
        self.assertEqual(scene_config_for_robot("ur5e"), SCENE)
        self.assertEqual(SCENE.robot_name, "ur5e")
        self.assertEqual(rm65.joint_names, tuple(f"joint_{i}" for i in range(1, 7)))
        self.assertLess(rm65.robot_base_position_world[1] + rm65.robot_stand_size_xy[1] / 2,
                        rm65.plate_center_world[1] - rm65.plate_size[1] / 2)
        with self.assertRaisesRegex(ValueError, "Unknown robot"):
            scene_config_for_robot("missing")

    @unittest.skipIf(Usd is None, "USD Python bindings are not installed")
    def test_rm65_nested_asset_fk_matches_source_urdf(self):
        from robot.asset import attach_tool, articulation_path
        from robot.kinematics import SerialKinematics

        config = scene_config_for_robot("rm65")
        urdf = Path(__file__).resolve().parents[1] / "assets/vendor/rm_models/RM65/urdf/RM65-6F/urdf/RM65-6F.urdf"
        if not Path(config.robot_local_usd).is_file() or not urdf.is_file():
            self.skipTest("Download and convert the RM65 assets to run this integration check")
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, config.robot_prim_path).GetPrim().GetReferences().AddReference(config.robot_local_usd)
        attach_tool(stage, config)
        model = SerialKinematics.from_stage(stage, config)
        self.assertTrue(articulation_path(stage, config.robot_prim_path).startswith(config.robot_prim_path))
        self.assertIn("/Geometry/base_link/link_1/link_2/", model.tcp_path)
        home = np.asarray(config.home_joint_positions)
        self.assertTrue(np.all(home > model.limits[:, 0]) and np.all(home < model.limits[:, 1]))
        joints = {joint.attrib["name"]: joint for joint in ET.parse(urdf).getroot().findall("joint")}
        for q in (np.zeros(6), home, np.array([.3, -.2, .5, -.4, .6, -.1])):
            pose = np.eye(4)
            for name, angle in zip(config.joint_names, q):
                joint = joints[name]
                origin = joint.find("origin")
                fixed = np.eye(4)
                fixed[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.attrib["rpy"], sep=" ")).as_matrix()
                fixed[:3, 3] = np.fromstring(origin.attrib["xyz"], sep=" ")
                moving = np.eye(4)
                moving[:3, :3] = Rotation.from_rotvec(np.fromstring(joint.find("axis").attrib["xyz"], sep=" ") * angle).as_matrix()
                pose = pose @ fixed @ moving
            tip = np.eye(4)
            tip[2, 3] = config.tool_length_m
            np.testing.assert_allclose(model.forward(q), pose @ tip, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
