"""A rigid arm cannot clear its own torch; bracket translation must do it."""

import unittest
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics

from hj90.arm_kinematics import ArmKinematics
from hj90.mount_offset import TOOL,apply_usd_offset,torch_frustum_audit
from hj90.optics import load_calibration


class MountOffsetTests(unittest.TestCase):
    def stage(self):
        stage=Usd.Stage.Open('assets/scenes/industrial-welding/industrial_welding.usdc')
        stage.SetEditTarget(stage.GetSessionLayer())
        return stage

    def test_joint_motion_does_not_remove_rigidly_mounted_torch(self):
        stage=self.stage();calibration=load_calibration()
        before=torch_frustum_audit(stage,calibration)
        arm=ArmKinematics(stage);q=arm.initial.copy();q[0]+=.2;arm.apply(q)
        after=torch_frustum_audit(stage,calibration)
        self.assertGreater(before[0]['triangles_potentially_intersecting_frustum'],0)
        self.assertEqual(before,after)

    def test_shifted_housing_and_left_eye_clear_torch_without_hiding_it(self):
        stage=self.stage();tool=stage.GetPrimAtPath(TOOL)
        names=('hj90','hj90_housing','hj90_camera_seat','hj90_envelope_collision')
        old={n:np.array(UsdGeom.Xformable(tool.GetChild(n)).GetLocalTransformation()).T.copy() for n in names}
        torch=tool.GetChild('e05l_welding_torch_visual_0')
        visible=UsdGeom.Imageable(torch).ComputeVisibility()
        transform=np.array(UsdGeom.Xformable(torch).GetLocalTransformation())
        record=apply_usd_offset(stage,.05)
        for name in names:
            new=np.array(UsdGeom.Xformable(tool.GetChild(name)).GetLocalTransformation()).T
            np.testing.assert_allclose(new[:3,3]-old[name][:3,3],record['translation_in_torch_m'],atol=1e-8)
            np.testing.assert_allclose(new[:3,:3],old[name][:3,:3],atol=1e-8)
        np.testing.assert_array_equal(transform,np.array(UsdGeom.Xformable(torch).GetLocalTransformation()))
        self.assertEqual(visible,UsdGeom.Imageable(torch).ComputeVisibility())
        self.assertTrue(all(r['triangles_potentially_intersecting_frustum']==0 for r in torch_frustum_audit(stage,load_calibration())))
        mass=UsdPhysics.MassAPI(tool)
        self.assertAlmostEqual(mass.GetMassAttr().Get(),6.056,places=5)
        self.assertTrue(all(v>0 for v in mass.GetDiagonalInertiaAttr().Get()))


if __name__=='__main__':unittest.main()
