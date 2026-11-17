"""Industrial welding scene manifest and RGB capture."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "assets/scenes/industrial-welding"
STEREO_SCENE = BUNDLE / "industrial_welding_full.usda"
INITIAL_ARM_Q = (0., 1.2, 2.3, 0., 1.3, 0.)
RESOLUTION = (640, 1100)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def scene_setup(rgb_profile="legacy", rgb_resolution=None, pose_name=None):
    """Historical benchmark callers retain legacy optics; CLI selects vendor RGB explicitly."""
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    scene = manifest["full_scene"]
    pose_name = pose_name or scene["default_pose"]
    if pose_name not in scene["poses"]:
        raise ValueError(f"Pose {pose_name!r} is unavailable for this scene layout; "
                         "replan the workstation placement before capturing it")
    pose = scene["poses"][pose_name]
    if len(pose["arm_q_rad"]) != 6 or any(
            not math.isclose(a, b, abs_tol=1e-8) for a, b in zip(pose["arm_q_rad"], INITIAL_ARM_Q)):
        raise ValueError("The default capture pose must retain the normal initial joint positions")
    if rgb_profile not in ("vendor", "legacy"):
        raise ValueError("Unknown HJ-90 RGB profile")
    vendor = None
    if rgb_profile == "vendor":
        from weld_scene.vendor_calibration import load_rgb_calibration, resize_rgb_calibration
        vendor = resize_rgb_calibration(load_rgb_calibration(), rgb_resolution)
        sensor = manifest["full_scene"]["active_stereo_laser"]
        if vendor["source_sha256"] != sensor["rgb_calibration"]["source_sha256"]:
            raise ValueError("RGB calibration changed; rebuild with weld_scene/build_sensor.py")
        if pose_name != scene["default_pose"]:
            raise ValueError("The vendor RGB layer is authored for the bundle's default pose")
        asset_name = scene["asset"]
        camera = sensor["rgb_camera"]
        resolution = vendor["image_size_wh"]
    else:
        asset_name = pose["asset"]
        camera = manifest["full_scene"]["hj90_camera"]
        resolution = list(rgb_resolution or RESOLUTION)
    asset = (BUNDLE / asset_name).resolve()
    if not asset.is_relative_to(BUNDLE.resolve()) or not asset.is_file():
        raise FileNotFoundError(f"Pose asset not found in the scene bundle: {asset}")
    return {"pose": pose_name, "usd": str(asset), "camera": camera,
            "context_camera": pose["context_camera"], "station": pose.get("station"),
            "rgb_profile": rgb_profile, "rgb_calibration": vendor,
            "resolution_wh": resolution, "base_xy_yaw": pose["base_xy_yaw"],
            "base_height_m": pose["base_height_m"], "arm_q_rad": pose["arm_q_rad"],
            "placement": pose.get("placement", "source-authored static pose; no movement trajectory"),
            "recorded_static_checks": pose["static_checks"]}


def inspect_stage(stage, setup):
    """Read pose and camera metadata only; no seam coordinates enter perception."""
    import numpy as np
    from pxr import UsdGeom, UsdUtils

    if not stage or UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise ValueError("The imported scene must use metres and Z-up")
    camera = UsdGeom.Camera(stage.GetPrimAtPath(setup["camera"]))
    if not camera:
        raise ValueError(f"HJ-90 camera missing: {setup['camera']}")
    _, _, missing = UsdUtils.ComputeAllDependencies(setup["usd"])
    if missing:
        raise FileNotFoundError(f"Missing scene dependencies: {missing}")
    actual_q = []
    for index in range(1, 7):
        joint = stage.GetPrimAtPath(f"/World/Robot/Joints/e05l_joint{index}")
        value = joint.GetAttribute("state:angular:physics:position").Get() if joint else None
        if value is None:
            raise ValueError(f"Initial arm joint state missing: {index}")
        actual_q.append(math.radians(value))
    if not np.allclose(actual_q, INITIAL_ARM_Q, atol=1e-6, rtol=0):
        raise ValueError(f"Arm pose differs from the normal initial position: {actual_q}")
    base = stage.GetPrimAtPath("/World/Robot/base_link")
    if not base:
        raise ValueError("B2 base link missing from scene")
    cache = UsdGeom.XformCache()
    position = list(cache.GetLocalToWorldTransform(base).ExtractTranslation())
    expected = [*setup["base_xy_yaw"][:2], setup["base_height_m"]]
    if not np.allclose(position, expected, atol=1e-6, rtol=0):
        raise ValueError(f"B2 is not at the recorded scene position: {position}")
    if camera.GetProjectionAttr().Get() != UsdGeom.Tokens.perspective:
        raise ValueError("HJ-90 must be a perspective camera")
    focal = float(camera.GetFocalLengthAttr().Get())
    horizontal = float(camera.GetHorizontalApertureAttr().Get())
    vertical = float(camera.GetVerticalApertureAttr().Get())
    if not all(math.isfinite(v) and v > 0 for v in (focal, horizontal, vertical)):
        raise ValueError("Invalid camera focal length or aperture")
    from weld_scene.usd import camera_intrinsics
    width, height = setup["resolution_wh"]
    calibration = {"image_size": [width, height],
                   "camera_matrix": camera_intrinsics(camera, (width, height)).tolist(),
                   "pixel_convention": "edge origin; pixel centres at index + 0.5",
                   "distortion_coefficients": [0, 0, 0, 0, 0]}
    if setup.get("rgb_calibration"):
        profile = setup["rgb_calibration"]
        if not np.allclose(calibration["camera_matrix"], profile["intrinsics"], atol=.001, rtol=0):
            raise ValueError("USD RGB frustum disagrees with the vendor calibration")
        calibration.update(camera_matrix_opencv=profile["intrinsics_opencv"],
                           vendor_calibration=profile)
    world_from_camera = np.asarray(cache.GetLocalToWorldTransform(camera.GetPrim())).T @ np.diag([1., -1., -1., 1.])
    return calibration, {"arm_q_rad": actual_q, "base_position_world_m": position,
                         "world_from_camera_optical": world_from_camera.tolist()}
