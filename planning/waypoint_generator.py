from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class CartesianPath:
    surface_points: np.ndarray
    poses: np.ndarray
    phases: list[str]
    closed: bool = True


def resample_polyline(points, spacing, *, closed):
    points = np.asarray(points, dtype=float)
    if spacing <= 0 or len(points) < (3 if closed else 2) or not np.isfinite(points).all():
        raise ValueError("Need a finite path and positive spacing")
    if closed and np.linalg.norm(points[-1] - points[0]) < 1e-10:
        points = points[:-1]
    if not closed and np.linalg.norm(points[-1] - points[0]) < 1e-10:
        raise ValueError("An open path must have distinct endpoints")
    sampled = []
    ends = np.roll(points, -1, axis=0) if closed else points[1:]
    for start, end in zip(points if closed else points[:-1], ends):
        length = np.linalg.norm(end - start)
        if length < 1e-9:
            continue
        count = max(1, int(np.ceil(length / spacing)))
        sampled.extend(start + (end - start) * t for t in np.arange(count) / count)
    if len(sampled) < (3 if closed else 1):
        raise ValueError("Degenerate path")
    return np.vstack((sampled, sampled[0] if closed else points[-1]))


def resample_closed_polygon(points, spacing):
    return resample_polyline(points, spacing, closed=True)


def generate_waypoints(contour_points, config, plane_normal=(0, 0, 1), *, closed=True):
    """Generate poses in the same coordinate frame as the contour and normal."""
    surface = resample_polyline(contour_points, config.waypoint_spacing_m, closed=closed)
    normal = np.asarray(plane_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    z_axis = -normal  # Tool +Z points towards the plate.
    x_axis = np.array([1., 0., 0.])
    x_axis -= x_axis @ z_axis * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([0., 1., 0.])
        x_axis -= x_axis @ z_axis * z_axis
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.column_stack((x_axis, np.cross(z_axis, x_axis), z_axis))
    trace = surface + config.tcp_offset_m * normal
    approach = surface[0] + config.approach_height_m * normal
    count = int(np.ceil((config.approach_height_m - config.tcp_offset_m) / config.waypoint_spacing_m))
    if count < 1:
        raise ValueError("Approach height must exceed TCP clearance")
    descent = np.linspace(approach, trace[0], count + 1)[1:]
    retreat_target = surface[-1] + config.approach_height_m * normal
    retreat = np.linspace(trace[-1], retreat_target, count + 1)[1:]
    positions = np.vstack((approach, descent, trace[1:], retreat))
    phases = ["approach"] + ["descend"] * len(descent) + ["trace"] * (len(trace) - 1) + ["retreat"] * len(retreat)
    poses = np.repeat(np.eye(4)[None], len(positions), axis=0)
    poses[:, :3, :3] = rotation
    poses[:, :3, 3] = positions
    return CartesianPath(surface, poses, phases, closed)


def interpolate_pose(start, end, fraction):
    pose = np.eye(4)
    pose[:3, 3] = start[:3, 3] + fraction * (end[:3, 3] - start[:3, 3])
    delta = Rotation.from_matrix(end[:3, :3] @ start[:3, :3].T).as_rotvec()
    pose[:3, :3] = Rotation.from_rotvec(fraction * delta).as_matrix() @ start[:3, :3]
    return pose
