"""Demo frames from extra cameras during a run; never used by perception or control."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

# Eye and target in the robot mount frame (x forward, z up; z measured from the ground) and focal
# length in mm, for a station facing a seam about 1 m ahead (the ship-grid demo).
DEMO_VIEWS = {
    "overview": ((-1.3, -2.3, 1.55), (0.55, 0.0, 0.2), 16.0, (1080, 1920)),
    "closeup": ((0.55, -0.8, 0.5), (0.97, 0.0, 0.06), 28.0, (540, 960)),
}


def look_at(eye, target):
    """USD camera orientation (looks along -Z, +Y up) as a w-first quaternion."""
    forward = np.asarray(target, dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, np.cross(right, forward), -forward))
    return Rotation.from_matrix(rotation).as_quat(scalar_first=True)


class DemoRecorder:
    """Saves every `every`-th controller tick from each view, with the controller phase."""

    def __init__(self, directory, world_from_mount, views=DEMO_VIEWS, every=2):
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

        self.directory, self.every, self.ticks, self.timeline = Path(directory), every, 0, []
        self.views, self.sensors = list(views), {}
        for name, (eye, target, focal, resolution) in views.items():
            eye_world = world_from_mount[:3, :3] @ np.r_[eye[:2], 0.] + world_from_mount[:3, 3] * [1, 1, 0] + [0, 0, eye[2]]
            target_world = (world_from_mount[:3, :3] @ np.r_[target[:2], 0.] + world_from_mount[:3, 3] * [1, 1, 0]
                            + [0, 0, target[2]])
            camera = RtxCamera(f"/World/DemoCameras/{name}", tick_rate=30.0)
            camera.camera.set_local_poses(translations=[eye_world], orientations=[look_at(eye_world, target_world)])
            camera.camera.set_focal_lengths(focal)
            camera.camera.set_apertures(horizontal_apertures=36.0, vertical_apertures=36.0 * resolution[0] / resolution[1])
            camera.camera.set_clipping_ranges(0.02, 50.0)
            self.sensors[name] = CameraSensor(camera, resolution=resolution, annotators=["rgb"])
            (self.directory / name).mkdir(parents=True, exist_ok=True)

    def __call__(self, controller):
        self.ticks += 1
        if self.ticks % self.every:
            return
        frames = {}
        for name, sensor in self.sensors.items():
            rgb, _ = sensor.get_data("rgb")
            if rgb is None or np.ptp(rgb.numpy()) == 0:
                return  # The render product is still warming up.
            frames[name] = rgb.numpy()[:, :, :3].astype(np.uint8)
        index = len(self.timeline)
        for name, rgb in frames.items():
            Image.fromarray(rgb).save(self.directory / name / f"{index:05d}.jpg", quality=90)
        self.timeline.append({"index": index, "sim_time_s": controller.manager.get_simulation_time(),
                              "phase": controller.phase})

    def close(self):
        for sensor in self.sensors.values():
            sensor.detach_annotators(sensor.annotators)
        self.sensors = {}
        (self.directory / "timeline.json").write_text(json.dumps(
            {"every_ticks": self.every, "views": self.views, "frames": self.timeline}, indent=1) + "\n")
