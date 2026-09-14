"""Static coarse paths from registered RGB-D and calibration; no simulator/scene input."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from planning.waypoint_generator import CartesianPath, generate_waypoints
from vision.contour_detector import detect_contour, save_overlay


@dataclass
class RGBDCalibration:
    image_size: tuple
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    depth_scale_to_m: float
    base_from_camera: np.ndarray

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("depth_convention") != "optical_z":
            raise ValueError("Depth must be registered optical-axis Z, not radial distance")
        matrix = np.asarray(data["camera_matrix"], dtype=float)
        transform = np.asarray(data["base_from_camera"], dtype=float)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], [0, 0, 1]) or matrix[0, 1] != 0 or matrix[1, 0] != 0):
            raise ValueError("Invalid camera intrinsics")
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1)):
            raise ValueError("base_from_camera must be a rigid transform")
        scale = float(data["depth_scale_to_m"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid depth scale")
        size = tuple(data["image_size"])
        distortion = np.asarray(data["distortion_coefficients"], dtype=float)
        if len(size) != 2 or any(not isinstance(value, int) or value <= 0 for value in size):
            raise ValueError("image_size must contain positive integer width and height")
        if distortion.ndim != 1 or distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
            raise ValueError("Invalid OpenCV distortion coefficients")
        return cls(size, matrix, distortion, scale, transform)

    def rays(self, pixels):
        xy = cv2.undistortPoints(np.asarray(pixels, dtype=float).reshape(-1, 1, 2),
                                 self.camera_matrix, self.distortion_coefficients).reshape(-1, 2)
        return np.column_stack((xy, np.ones(len(xy))))


@dataclass(frozen=True)
class RGBDSettings:
    waypoint_spacing_m: float = 0.01
    tcp_offset_m: float = 0.01
    approach_height_m: float = 0.10
    dark_threshold: int = 80
    min_contour_area_px: float = 300.0
    min_depth_m: float = 0.1
    max_depth_m: float = 5.0
    plane_inlier_distance_m: float = 0.003
    min_inlier_fraction: float = 0.7
    min_valid_depth_fraction: float = 0.4
    min_plane_points: int = 100
    max_fit_points: int = 3000
    ransac_iterations: int = 160


@dataclass
class ObservedSurface:
    point: np.ndarray
    normal: np.ndarray
    basis: np.ndarray
    footprint_equations: np.ndarray

    @classmethod
    def from_points(cls, points, normal):
        point = points.mean(axis=0)
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        u = np.cross(normal, axis)
        u /= np.linalg.norm(u)
        basis = np.column_stack((u, np.cross(normal, u)))
        hull = ConvexHull((points - point) @ basis)
        return cls(point, normal, basis, hull.equations)

    def intersects(self, points, radius):
        local = (points - self.point) @ self.basis
        inside = np.all(local @ self.footprint_equations[:, :2].T + self.footprint_equations[:, 2] <= radius, axis=1)
        # Treat the observed top surface as a solid half-space within its footprint.
        return np.any(inside & ((points - self.point) @ self.normal < radius))


def fit_plane(points, settings):
    rng = np.random.default_rng(0)
    if len(points) < settings.min_plane_points:
        raise RuntimeError("Insufficient valid depth points in the contour interior")
    if len(points) > settings.max_fit_points:
        points = points[rng.choice(len(points), settings.max_fit_points, replace=False)]
    best = np.zeros(len(points), dtype=bool)
    for _ in range(settings.ransac_iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-9:
            continue
        normal /= np.linalg.norm(normal)
        inliers = np.abs((points - a) @ normal) < settings.plane_inlier_distance_m
        if inliers.sum() > best.sum():
            best = inliers
    if best.mean() < settings.min_inlier_fraction:
        raise RuntimeError("Depth does not support a dominant planar workpiece")
    for _ in range(2):
        center = points[best].mean(axis=0)
        _, singular, vt = np.linalg.svd(points[best] - center, full_matrices=False)
        if singular[1] < 1e-4:
            raise RuntimeError("Plane support is degenerate")
        normal = vt[-1]
        best = np.abs((points - center) @ normal) < settings.plane_inlier_distance_m
    if best.mean() < settings.min_inlier_fraction:
        raise RuntimeError("Unstable plane fit")
    # Surface normal faces the observing camera; no world-up prior.
    if normal @ center > 0:
        normal = -normal
    rms = np.sqrt(np.mean(((points[best] - center) @ normal) ** 2))
    return center, normal, {"plane_inlier_fraction": float(best.mean()), "plane_rms_m": float(rms),
                            "fit_points": len(points)}


@dataclass
class RGBDTrajectory:
    contour_pixel: np.ndarray
    contour_camera: np.ndarray
    contour_base: np.ndarray
    path: CartesianPath
    surface: ObservedSurface
    diagnostics: dict
    mask: np.ndarray


def generate_rgbd_trajectory(rgb, depth, calibration, settings=RGBDSettings()):
    width, height = calibration.image_size
    if rgb.shape != (height, width, 3) or depth.shape != (height, width):
        raise ValueError("RGB and registered depth must match the calibration image size")
    contour, mask = detect_contour(rgb, threshold=settings.dark_threshold, min_area=settings.min_contour_area_px)
    interior = np.zeros((height, width), np.uint8)
    cv2.fillPoly(interior, [np.rint(contour).astype(np.int32)], 255)
    interior = cv2.erode(interior, np.ones((9, 9), np.uint8)) > 0
    depth_m = np.asarray(depth, dtype=float) * calibration.depth_scale_to_m
    valid = np.isfinite(depth_m) & (depth_m > settings.min_depth_m) & (depth_m < settings.max_depth_m)
    if not interior.any() or (valid & interior).sum() / interior.sum() < settings.min_valid_depth_fraction:
        raise RuntimeError("Not enough registered depth inside the detected contour; no scene-coordinate fallback")
    v, u = np.nonzero(valid & interior)
    cloud = calibration.rays(np.column_stack((u, v))) * depth_m[v, u, None]
    center, normal, diagnostics = fit_plane(cloud, settings)
    rays = calibration.rays(contour)
    denominator = rays @ normal
    if np.any(np.abs(denominator) < 1e-6):
        raise RuntimeError("Contour rays are parallel to the estimated surface")
    z = (normal @ center) / denominator
    if np.any(z <= settings.min_depth_m) or np.any(z >= settings.max_depth_m):
        raise RuntimeError("Estimated contour lies outside the depth range")
    camera_points = rays * z[:, None]
    rotation, translation = calibration.base_from_camera[:3, :3], calibration.base_from_camera[:3, 3]
    base_points = camera_points @ rotation.T + translation
    normal_base = rotation @ normal
    # Use observed coplanar depth pixels for the footprint, not configured plate dimensions.
    v, u = np.nonzero(valid)
    v, u = v[::4], u[::4]
    cloud = calibration.rays(np.column_stack((u, v))) * depth_m[v, u, None]
    cloud = cloud[np.abs((cloud - center) @ normal) < settings.plane_inlier_distance_m]
    # Keep the collision/clearance plane identical to the fitted trajectory plane.
    cloud = cloud - ((cloud - center) @ normal)[:, None] * normal
    surface = ObservedSurface.from_points(cloud @ rotation.T + translation, normal_base)
    path = generate_waypoints(base_points, settings, normal_base)
    diagnostics.update(coordinate_frame="robot_base", depth_source="registered_optical_z",
                       surface_source="RGB-D RANSAC plane", surface_point_base=surface.point.tolist(),
                       surface_normal_base=normal_base.tolist(), observed_surface_points=len(cloud),
                       contour_vertices=len(contour), waypoint_count=len(path.poses))
    return RGBDTrajectory(contour, camera_points, base_points, path, surface, diagnostics, mask)


def save_rgbd_trajectory(result, rgb, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_overlay(rgb, result.contour_pixel, output_dir / "contour_overlay.png")
    if not cv2.imwrite(str(output_dir / "contour_mask.png"), result.mask):
        raise OSError("Could not save contour mask")
    np.savez(output_dir / "rgbd_trajectory.npz", contour_pixel=result.contour_pixel,
             contour_camera=result.contour_camera, contour_base=result.contour_base,
             surface_points=result.path.surface_points, poses_base=result.path.poses,
             phases=result.path.phases, surface_normal_base=result.surface.normal,
             surface_point_base=result.surface.point, coordinate_frame="robot_base")
    np.savetxt(output_dir / "waypoints_base.csv", result.path.poses[:, :3, 3], delimiter=",",
               header="tcp_x_m,tcp_y_m,tcp_z_m", comments="")
    (output_dir / "vision_report.json").write_text(json.dumps(result.diagnostics, indent=2) + "\n")
