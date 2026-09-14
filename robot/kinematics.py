"""Serial-chain FK and bounded numerical IK. No Isaac import on the CPU path."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def pose_error(actual, target):
    return np.r_[target[:3, 3] - actual[:3, 3],
                 Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()]


@dataclass
class SerialKinematics:
    names: list[str]
    parent_frames: np.ndarray
    child_frames_inverse: np.ndarray
    axes: np.ndarray
    limits: np.ndarray
    world_from_base: np.ndarray
    last_link_from_tcp: np.ndarray
    base_path: str = ""
    last_link_path: str = ""
    tcp_path: str = ""

    def forward(self, q, *, links=False):
        transform = self.world_from_base.copy()
        frames = [transform.copy()]
        for angle, parent, axis, child_inv in zip(q, self.parent_frames, self.axes, self.child_frames_inverse):
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix()
            transform = transform @ parent @ rotation @ child_inv
            frames.append(transform.copy())
        tcp = transform @ self.last_link_from_tcp
        return (tcp, frames) if links else tcp

    def inverse(self, target, seed, *, position_tolerance=0.0002, orientation_tolerance=0.002):
        if not np.isfinite(target).all():
            raise ValueError("Non-finite IK target")
        lower, upper = self.limits.T
        def residual(q):
            error = pose_error(self.forward(q), target)
            return np.r_[error[:3], error[3:] * 0.25]
        solution = least_squares(residual, np.clip(seed, lower + 1e-8, upper - 1e-8),
                                 bounds=(lower, upper), max_nfev=160,
                                 ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error = pose_error(self.forward(solution.x), target)
        if np.linalg.norm(error[:3]) > position_tolerance or np.linalg.norm(error[3:]) > orientation_tolerance:
            raise RuntimeError(f"IK failed at {target[:3, 3].tolist()}: "
                               f"position={np.linalg.norm(error[:3]):.6f} m, "
                               f"orientation={np.linalg.norm(error[3:]):.6f} rad")
        return solution.x

    def save(self, path):
        np.savez(path, names=self.names, parent_frames=self.parent_frames,
                 child_frames_inverse=self.child_frames_inverse, axes=self.axes,
                 limits=self.limits, world_from_base=self.world_from_base,
                 last_link_from_tcp=self.last_link_from_tcp)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(**{key: data[key].tolist() if key == "names" else data[key] for key in data.files})

    @classmethod
    def from_stage(cls, stage, config):
        from pxr import Usd, UsdGeom, UsdPhysics

        def matrix(position, quaternion):
            result = np.eye(4)
            result[:3, :3] = Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix()
            result[:3, 3] = position
            return result

        def world(path):
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise RuntimeError(f"Missing robot frame: {path}")
            return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T

        joints = {p.GetName(): UsdPhysics.RevoluteJoint(p) for p in stage.Traverse()
                  if p.IsA(UsdPhysics.RevoluteJoint) and p.GetPath().HasPrefix(config.robot_prim_path)}
        parents, children, axes, limits = [], [], [], []
        previous = f"{config.robot_prim_path}/{config.robot_base_frame}"
        base_path = previous
        for name in config.joint_names:
            if name not in joints:
                raise RuntimeError(f"Missing configured revolute joint {name}")
            joint = joints[name]
            body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not body0 or not body1 or str(body0[0]) != previous:
                raise RuntimeError(f"{name}: expected serial parent {previous}; got {body0}")
            previous = str(body1[0])
            parents.append(matrix(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()))
            children.append(np.linalg.inv(matrix(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())))
            axes.append(np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())])
            bounds = np.deg2rad([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()])
            limits.append(np.where(np.isfinite(bounds), bounds, [-2 * np.pi, 2 * np.pi]))
        tcp_path = f"{previous}/{config.end_effector_frame}/TCP"
        return cls(list(config.joint_names), np.array(parents), np.array(children), np.array(axes),
                   np.array(limits), world(base_path), np.linalg.inv(world(previous)) @ world(tcp_path),
                   base_path, previous, tcp_path)
