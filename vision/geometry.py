"""Column-vector transforms; USD camera looks along -Z, image v points down."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PlaneCamera:
    width: int
    height: int
    fx: float
    fy: float
    world_from_camera: np.ndarray
    plane_point: np.ndarray
    plane_normal: np.ndarray

    def pixel_to_world(self, pixels):
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        rays = np.column_stack(((pixels[:, 0] - self.width / 2) / self.fx,
                                -(pixels[:, 1] - self.height / 2) / self.fy,
                                -np.ones(len(pixels))))
        directions = rays @ self.world_from_camera[:3, :3].T
        origin = self.world_from_camera[:3, 3]
        denominator = directions @ self.plane_normal
        if np.any(np.abs(denominator) < 1e-8):
            raise ValueError("Camera ray is parallel to the plate")
        distance = ((self.plane_point - origin) @ self.plane_normal) / denominator
        if np.any(distance <= 0):
            raise ValueError("Plate intersection is behind the camera")
        return origin + directions * distance[:, None]

    def world_to_pixel(self, points):
        local = (np.asarray(points) - self.world_from_camera[:3, 3]) @ self.world_from_camera[:3, :3]
        if np.any(local[:, 2] >= -1e-8):
            raise ValueError("Point is behind the camera")
        return np.column_stack((self.width / 2 + self.fx * local[:, 0] / -local[:, 2],
                                self.height / 2 - self.fy * local[:, 1] / -local[:, 2]))

    @classmethod
    def from_stage(cls, stage, config):
        from pxr import Usd, UsdGeom

        camera = UsdGeom.Camera(stage.GetPrimAtPath(config.camera_prim_path))
        transform = np.array(camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        plate = UsdGeom.Xformable(stage.GetPrimAtPath(config.plate_frame_path))
        plane = np.array(plate.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        width, height = config.camera_resolution
        # Ratios use authored USD optical units, avoiding mm/metre conversion ambiguity.
        fx = width * camera.GetFocalLengthAttr().Get() / camera.GetHorizontalApertureAttr().Get()
        fy = height * camera.GetFocalLengthAttr().Get() / camera.GetVerticalApertureAttr().Get()
        return cls(width, height, fx, fy, transform, plane[:3, 3], plane[:3, 2])
