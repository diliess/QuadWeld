"""Isaac Sim 6 body camera and registered RGB-D capture."""

from pathlib import Path

import numpy as np
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
from PIL import Image
from scipy.spatial.transform import Rotation

from config import SceneConfig


def create_body_camera(config, calibration):
    """Author the sensor from installation calibration, independently of the plate."""
    width, height = calibration.image_size
    k = calibration.camera_matrix
    if (calibration.image_size != config.camera_resolution or calibration.depth_scale_to_m != 1.0
            or np.any(calibration.distortion_coefficients != 0)
            or not np.allclose(k, [[k[0, 0], 0, width / 2], [0, k[1, 1], height / 2], [0, 0, 1]])):
        raise ValueError("Simulation fixture requires matching resolution, centred undistorted intrinsics and metre depth")
    camera = RtxCamera(config.camera_prim_path, tick_rate=30.0)
    local_pose = calibration.base_from_camera @ np.diag([1., -1., -1., 1.])
    focal_mm = config.camera_focal_length_m * 1000
    camera.camera.set_focal_lengths(focal_mm)
    camera.camera.set_apertures(horizontal_apertures=width * focal_mm / k[0, 0],
                                vertical_apertures=height * focal_mm / k[1, 1])
    camera.camera.set_clipping_ranges(config.camera_near_m, config.camera_far_m)
    quaternion = Rotation.from_matrix(local_pose[:3, :3]).as_quat(scalar_first=True)
    camera.camera.set_local_poses(translations=[local_pose[:3, 3]], orientations=[quaternion])
    return camera


def create_rgbd_sensor(camera: RtxCamera, config: SceneConfig) -> CameraSensor:
    width, height = config.camera_resolution
    return CameraSensor(camera, resolution=(height, width), annotators=["rgb", "distance_to_image_plane"])


def save_rgbd_frame(sensor, simulation_app, output_dir, max_render_steps=120):
    """Capture registered RGB and optical-axis depth from one static render product."""
    output_dir = Path(output_dir)
    for _ in range(max_render_steps):
        simulation_app.update()
        rgb, _ = sensor.get_data("rgb")
        depth, _ = sensor.get_data("distance_to_image_plane")
        if rgb is None or depth is None:
            continue
        rgb, depth = rgb.numpy(), depth.numpy().squeeze(-1)
        if rgb.shape[:2] != sensor.resolution or depth.shape != sensor.resolution or np.ptp(rgb) == 0:
            continue
        rgb = rgb[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        if not np.any(np.isfinite(depth) & (depth > 0)):
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(output_dir / "body_rgb.png")
        np.save(output_dir / "body_depth_m.npy", depth)
        valid = np.isfinite(depth) & (depth > 0)
        low, high = np.percentile(depth[valid], [2, 98])
        preview = np.zeros(depth.shape, np.uint8)
        preview[valid] = np.clip((depth[valid] - low) / max(float(high - low), 1e-6) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(preview).save(output_dir / "body_depth_preview.png")
        return rgb.copy(), depth.copy()
    raise RuntimeError("RGB-D camera did not produce registered populated frames")
