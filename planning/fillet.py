"""Fillet-weld poses along an observed concave seam between two faces, and their joint plan."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from planning.waypoint_generator import CartesianPath, resample_polyline


@dataclass
class FaceObstacles:
    """Observed faces as solid half-spaces behind their planes; normals point into free space.

    A face with a finite cap is solid only below that height along `up` (an observed web top).
    """

    points: np.ndarray
    normals: np.ndarray
    up: np.ndarray | None = None
    caps: np.ndarray | None = None

    def clearance(self, points):
        points = np.atleast_2d(points)
        offsets = points[:, None, :] - self.points[None]
        distance = np.einsum("pkj,kj->pk", offsets, self.normals)
        if self.caps is not None:
            above = np.einsum("pkj,j->pk", offsets, self.up) - self.caps
            distance = np.where(above > 0, np.maximum(distance, above), distance)
        return distance.min(axis=1)

    def intersects(self, points, radius):
        return bool(np.any(self.clearance(points) < radius))


def seam_rotation(normal_a, normal_b, direction, roll=0.0):
    """Torch +Z along the inward bisector of the corner, +X along the seam, then rolled about +Z."""
    inward = -(np.asarray(normal_a) + np.asarray(normal_b))
    if np.linalg.norm(inward) < 1e-6:
        raise ValueError("Faces are coplanar; a fillet needs a corner")
    inward /= np.linalg.norm(inward)
    x = np.asarray(direction, dtype=float) - (np.asarray(direction) @ inward) * inward
    if np.linalg.norm(x) < 1e-6:
        raise ValueError("Seam direction is parallel to the torch axis")
    x /= np.linalg.norm(x)
    rotation = np.column_stack((x, np.cross(inward, x), inward))
    return rotation @ Rotation.from_rotvec([0., 0., roll]).as_matrix()


def fillet_path(seam_points, normal_a, normal_b, settings, roll=0.0):
    """Approach, descend, trace and retreat with the TCP standing off along the bisector."""
    surface = resample_polyline(seam_points, settings.waypoint_spacing_m, closed=False)
    rotation = seam_rotation(normal_a, normal_b, surface[-1] - surface[0], roll)
    back = -rotation[:, 2]
    trace = surface + settings.tcp_offset_m * back
    count = int(np.ceil((settings.approach_height_m - settings.tcp_offset_m) / settings.waypoint_spacing_m))
    if count < 1:
        raise ValueError("Approach distance must exceed TCP clearance")
    approach = surface[0] + settings.approach_height_m * back
    descent = np.linspace(approach, trace[0], count + 1)[1:]
    retreat = np.linspace(trace[-1], surface[-1] + settings.approach_height_m * back, count + 1)[1:]
    positions = np.vstack((approach, descent, trace[1:], retreat))
    phases = ["approach"] + ["descend"] * len(descent) + ["trace"] * (len(trace) - 1) + ["retreat"] * len(retreat)
    poses = np.repeat(np.eye(4)[None], len(positions), axis=0)
    poses[:, :3, :3] = rotation
    poses[:, :3, 3] = positions
    return CartesianPath(surface, poses, phases, closed=False)


def clip_seam(points, centre, length):
    """The sub-segment of an ordered polyline within half the length (arc) of the sample nearest centre."""
    points = np.asarray(points, dtype=float)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    middle = arc[np.argmin(np.linalg.norm(points - centre, axis=1))]
    start, stop = np.clip([middle - length / 2, middle + length / 2], 0, arc[-1])
    if stop - start < length:
        start, stop = (0., length) if start == 0 else (arc[-1] - length, arc[-1])
    keep = (arc > start) & (arc < stop)
    resampled = [np.interp(value, arc, points[:, axis]) for value in (start, stop) for axis in range(3)]
    return np.vstack((resampled[:3], points[keep], resampled[3:]))


def _arm_clear(model, q, obstacles, link_radius=0.035, tool_radius=0.006):
    tcp, links = model.forward(q, links=True)
    points = [link[:3, 3] for link in links[1:]] + [tcp[:3, 3]]
    for index, (a, b) in enumerate(zip(points, points[1:])):
        segment = np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b - a) / 0.01)) + 1))
        if obstacles.intersects(segment, tool_radius if index == len(points) - 2 else link_radius):
            return False
    return True


def plan_fillet(model, seam_points, normal_a, normal_b, home_q, config, settings, *, tool_points=None,
                tool_clearance_m=0.002, rolls_deg=range(0, 360, 15), seeds=24, seed=0, face_a_top_m=None,
                pre_approach_m=0.15):
    """Choose the torch roll and IK branch; joint-space transfer from home, then a Cartesian seam pass.

    The joint-space move ends `pre_approach_m` further back along the torch axis than the approach,
    away from the faces; the descent from there is a straight Cartesian line.
    `face_a_top_m` caps face A at that height above the seam along normal B (an observed web top).
    Returns joints, poses and phases in the controller's format, the Cartesian path and the obstacles.
    """
    from robot.robot_controller import plan_joint_path

    seam_points = np.asarray(seam_points, dtype=float)
    caps = None if face_a_top_m is None else np.array([face_a_top_m, np.inf])
    obstacles = FaceObstacles(np.repeat(seam_points.mean(0)[None], 2, axis=0), np.vstack((normal_a, normal_b)),
                              up=np.asarray(normal_b, dtype=float), caps=caps)
    rng = np.random.default_rng(seed)
    lower, upper = model.limits.T
    starts = np.vstack(([home_q], rng.uniform(np.maximum(lower, -np.pi), np.minimum(upper, np.pi), (seeds, len(home_q)))))
    best, reasons = None, {}
    for roll in np.radians(list(rolls_deg)):
        path = fillet_path(seam_points, normal_a, normal_b, settings, roll)
        if tool_points is not None:
            work = [pose for pose, phase in zip(path.poses, path.phases) if phase in ("descend", "trace")]
            gap = min(obstacles.clearance(np.asarray(tool_points) @ pose[:3, :3].T + pose[:3, 3]).min() for pose in work)
            if gap < tool_clearance_m:
                reasons["torch_body_hits_face"] = reasons.get("torch_body_hits_face", 0) + 1
                continue
        pre_approach = path.poses[0].copy()
        pre_approach[:3, 3] -= pre_approach_m * pre_approach[:3, 2]
        branches = []
        for start in starts:
            try:
                q = model.inverse(pre_approach, start, position_tolerance=config.ik_position_tolerance_m,
                                  orientation_tolerance=config.ik_orientation_tolerance_rad)
            except RuntimeError:
                continue
            if not any(np.allclose(q, other, atol=0.05) for other in branches):
                branches.append(q)
        if not branches:
            reasons["approach_unreachable"] = reasons.get("approach_unreachable", 0) + 1
        for q0 in sorted(branches, key=lambda q: np.abs(q - home_q).max()):
            transfer = home_q + np.linspace(0, 1, 30)[:, None] * (q0 - home_q)
            if not all(_arm_clear(model, q, obstacles) for q in transfer):
                reasons["transfer_hits_face"] = reasons.get("transfer_hits_face", 0) + 1
                continue
            try:
                joints, poses, phases = plan_joint_path(model, path, q0, config, surface=obstacles)
            except RuntimeError as exc:
                key = "pass_" + str(exc).split(" at ")[0].split(" near ")[0][:40]
                reasons[key] = reasons.get(key, 0) + 1
                continue
            margin = float(np.min(np.minimum(joints - lower, upper - joints)))
            if best is None or margin > best[0]:
                best = (margin, roll, joints, poses, phases, path)
            break
    if best is None:
        raise RuntimeError(f"No feasible fillet pass: {reasons}")
    margin, roll, joints, poses, phases, path = best
    joints = np.vstack(([home_q], joints))
    poses = np.concatenate(([model.forward(home_q)], poses))
    phases = ["home", "transfer"] + list(phases[1:])
    return joints, poses, phases, path, obstacles, {"torch_roll_deg": float(np.degrees(roll)),
                                                     "min_joint_limit_margin_rad": margin}


def plan_reachable_fillet(model, seam_points, normals, home_q, config, settings, length_m, **options):
    """Weld the longest of length, 3/4 and 1/2 of it, centred on the seam point nearest the arm base."""
    failures = []
    for length in (length_m, length_m * .75, length_m * .5):
        segment = clip_seam(seam_points, np.zeros(3), length)
        try:
            *plan, info = plan_fillet(model, segment, normals[0], normals[1], home_q, config, settings, **options)
        except RuntimeError as exc:
            failures.append(f"{length:.2f} m: {exc}")
            continue
        return (*plan, {**info, "weld_length_m": float(length), "shorter_after": failures})
    raise RuntimeError("No reachable fillet segment; " + "; ".join(failures))
