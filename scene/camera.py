"""Isaac Sim 6 camera authoring and raw RGB capture; no image processing."""

from pathlib import Path

import numpy as np
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
from PIL import Image

from config import SceneConfig


def create_overhead_camera(config: SceneConfig) -> RtxCamera:
    # USD cameras look along local -Z with local +Y pointing upward. With an
    # identity rotation at this position, the camera looks straight down.
    camera = RtxCamera(
        config.camera_prim_path,
        tick_rate=30.0,
        positions=[list(config.camera_position_world)],
        orientations=[[1.0, 0.0, 0.0, 0.0]],
    )
    # The 6.0 optical API uses the USD camera convention shown in millimetres
    # in NVIDIA's examples. Keep the project configuration in SI metres.
    camera.camera.set_focal_lengths(config.camera_focal_length_m * 1000.0)
    camera.camera.set_apertures(
        horizontal_apertures=config.camera_horizontal_aperture_m * 1000.0,
        vertical_apertures=config.camera_horizontal_aperture_m
        * 1000.0
        * config.camera_resolution[1]
        / config.camera_resolution[0],
    )
    camera.camera.set_clipping_ranges(config.camera_near_m, config.camera_far_m)
    return camera


def create_rgb_sensor(camera: RtxCamera, config: SceneConfig) -> CameraSensor:
    width, height = config.camera_resolution
    return CameraSensor(camera, resolution=(height, width), annotators=["rgb"])


def save_rgb_frame(sensor: CameraSensor, simulation_app, output_path: Path, max_render_steps: int = 120) -> None:
    """Save the first populated camera frame for scene inspection."""
    for _ in range(max_render_steps):
        simulation_app.update()
        frame, _ = sensor.get_data("rgb")
        if frame is None:
            continue
        image_rgb = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
        if image_rgb.ndim != 3 or image_rgb.shape[:2] != sensor.resolution or image_rgb.shape[2] < 3:
            continue
        image_rgb = image_rgb[:, :, :3]
        if np.ptp(image_rgb) == 0:
            continue
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0.0, 1.0) * 255.0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.ascontiguousarray(image_rgb, dtype=np.uint8)).save(output_path)
        return
    raise RuntimeError("RGB camera did not produce a populated frame after rendering")
