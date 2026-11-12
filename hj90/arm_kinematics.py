"""Six-joint USD arm kinematics; the HJ-90 remains rigidly on the torch."""

import numpy as np
from scipy.spatial.transform import Rotation


def joint_frame(joint, side):
    position = getattr(joint, f'GetLocalPos{side}Attr')().Get()
    quaternion = getattr(joint, f'GetLocalRot{side}Attr')().Get()
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix()
    matrix[:3, 3] = position
    return matrix


class ArmKinematics:
    def __init__(self, stage):
        from pxr import UsdGeom, UsdPhysics
        self.stage = stage
        cache = UsdGeom.XformCache()
        world = lambda path: np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T
        self.base = world('/World/Robot/e05l_base_link')
        self.joints = [UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f'/World/Robot/Joints/e05l_joint{i}')) for i in range(1, 7)]
        self.frames = [(joint_frame(j, 0), np.linalg.inv(joint_frame(j, 1))) for j in self.joints]
        self.children = [str(j.GetBody1Rel().GetTargets()[0]) for j in self.joints]
        self.axes = [{'X': np.array([1., 0, 0]), 'Y': np.array([0., 1, 0]), 'Z': np.array([0., 0, 1])}[j.GetAxisAttr().Get()] for j in self.joints]
        self.bounds = np.radians([[j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()] for j in self.joints])
        self.initial = np.radians([j.GetPrim().GetAttribute('state:angular:physics:position').Get() for j in self.joints])
        last = world(self.children[-1])
        self.attachments = {path: np.linalg.inv(last) @ world(path) for path in (
            '/World/Robot/e05l_end_Link', '/World/Robot/e05l_welding_torch')}
        self.torch_to_camera = np.linalg.inv(world('/World/Robot/e05l_welding_torch')) @ world(
            '/World/Robot/e05l_welding_torch/hj90') @ np.diag([1., -1., -1., 1.])
        for path, transform in self.forward(self.initial).items():
            if not np.allclose(transform, world(path), atol=2e-6, rtol=0):
                raise ValueError('USD joint frames disagree with authored arm pose: '+path)

    def forward(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all() or np.any(q < self.bounds[:, 0]) or np.any(q > self.bounds[:, 1]):
            raise ValueError('Arm pose violates joint limits')
        transform = self.base.copy()
        links = {}
        for path, axis, (before, after), angle in zip(self.children, self.axes, self.frames, q):
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec(axis*angle).as_matrix()
            transform = transform @ before @ rotation @ after
            links[path] = transform.copy()
        links.update({path: transform @ local for path, local in self.attachments.items()})
        return links

    def camera(self, q):
        return self.forward(q)['/World/Robot/e05l_welding_torch'] @ self.torch_to_camera

    def apply(self, q):
        from pxr import Gf, UsdGeom
        links = self.forward(q)
        cache = UsdGeom.XformCache()
        for path, world in links.items():
            prim = self.stage.GetPrimAtPath(path)
            parent_world = np.array(cache.GetLocalToWorldTransform(prim.GetParent())).T
            local = np.linalg.inv(parent_world) @ world
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(local.T.tolist()))
        for joint, angle in zip(self.joints, q):
            joint.GetPrim().GetAttribute('state:angular:physics:position').Set(float(np.degrees(angle)))
            joint.GetPrim().GetAttribute('drive:angular:physics:targetPosition').Set(float(np.degrees(angle)))
        return links
