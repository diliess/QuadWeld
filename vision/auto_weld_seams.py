"""Seams from scripts/auto_weld.py, rescaled by the robot's support plane, in the robot base frame.

auto_weld.py returns monocular DA3 estimates in camera coordinates. A legged robot knows the plane
it stands on (foot contacts and gravity); fitting that plane in the image fixes the depth scale.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The model workers need their own interpreters and libraries, not the simulator's environment.
PASSED_ENVIRONMENT = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "TERM", "HF_HOME")


@dataclass
class ObservedSeam:
    points_base: np.ndarray
    normals_base: np.ndarray  # The two faces' normals, pointing into free space.
    scale: float
    diagnostics: dict = field(default_factory=dict)
    upright_top_m: float | None = None  # Observed top of the upright face above the seam.


def default_python():
    configured = os.environ.get("QUADWELD_DA3_PYTHON")
    for candidate in ([Path(configured)] if configured else []) + [
            Path.home() / "miniconda3/envs/depth-anything-3/bin/python"]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("depth-anything-3 environment not found; set QUADWELD_DA3_PYTHON or pass --auto-weld-python")


def run_auto_weld(image, output_dir, calibration_path, *, python=None, gpu=0):
    """Run the unchanged image pipeline (DA3, SAM3, geometry) with the calibrated intrinsics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [str(python or default_python()), str(ROOT / "scripts/auto_weld.py"), str(image),
               "--output-dir", str(output_dir), "--gpu", str(gpu), "--intrinsics", str(calibration_path)]
    environment = {key: os.environ[key] for key in PASSED_ENVIRONMENT if key in os.environ}
    result = subprocess.run(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    (output_dir / "auto_weld.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError("auto_weld.py failed:\n" + result.stdout[-6000:])
    return output_dir


def support_plane(foot_points_base, foot_radius_m, up_base):
    """The plane under the feet: contact centres lowered by the foot radius along gravity."""
    up = np.asarray(up_base, dtype=float) / np.linalg.norm(up_base)
    return np.mean(foot_points_base, axis=0) - foot_radius_m * up, up


def _unit_plane(equation):
    equation = np.asarray(equation, dtype=float)
    norm = np.linalg.norm(equation[:3])
    normal, offset = equation[:3] / norm, equation[3] / norm
    return (normal, offset) if offset > 0 else (-normal, -offset)  # Normal faces the camera.


def support_scale(planes, normal_camera, distance_m, max_angle_deg=10.):
    """Ratio of the known camera height over the support plane to its fitted monocular distance."""
    matches = []
    for plane in planes:
        normal, offset = _unit_plane(plane["equation_camera"])
        angle = np.degrees(np.arccos(np.clip(normal @ normal_camera, -1, 1)))
        if angle <= max_angle_deg:
            matches.append((plane["support_pixels"], plane["id"], offset, angle))
    if not matches:
        raise RuntimeError("No observed plane matches the support plane; the monocular depth scale is unknown")
    pixels, plane_id, offset, angle = max(matches)
    return distance_m / offset, {"support_plane_id": plane_id, "support_plane_pixels": pixels,
                                 "support_plane_angle_error_deg": float(angle),
                                 "support_plane_monocular_distance_m": float(offset),
                                 "support_plane_known_distance_m": float(distance_m)}


def _tilt_deg(normal, up):
    return float(np.degrees(np.arccos(np.clip(abs(normal @ up), -1, 1))))


def load_seam(report_dir, base_from_camera, support_point_base, support_normal_base, *,
              min_length_m=0.15, upright_deg=35., lower_deg=45.):
    """Pick the longest valid interval where an upright face meets a lower face; a horizontal fillet.

    Monocular depth bends the faces (a stiffener web leans, its corner rounds off), so only the seam
    line comes from the image. The returned faces are the measured support plane and an upright
    face through the seam: the 2F fillet geometry a welding procedure states its torch angles in.
    """
    report_dir = Path(report_dir)
    report = json.loads((report_dir / "data/auto_report.json").read_text())
    if len(report["results"]) != 1:
        raise ValueError("Expected the result of one camera image")
    result = report["results"][0]
    rotation, translation = base_from_camera[:3, :3], base_from_camera[:3, 3]
    up = np.asarray(support_normal_base, dtype=float) / np.linalg.norm(support_normal_base)
    up_camera = rotation.T @ up
    camera_height = float(up_camera @ (np.zeros(3) - rotation.T @ (support_point_base - translation)))
    scale, diagnostics = support_scale(result["planes"], up_camera, camera_height)
    candidates, rejected = [], {}
    prefix = result["array_prefix"]
    with np.load(report_dir / "data/auto_results.npz", allow_pickle=False) as stored:
        arrays = dict(stored)
    for path in result["paths"]:
        if path["kind"] != "plate_intersection":
            continue
        joint = result["joints"][path["joint_index"]]
        faces = sorted(((rotation @ _unit_plane(result["planes"][i]["equation_camera"])[0], i) for i in joint["planes"]),
                       key=lambda face: abs(face[0] @ up))
        (upright, upright_id), (lower, _) = faces
        if 90 - _tilt_deg(upright, up) > upright_deg or _tilt_deg(lower, up) > lower_deg:
            rejected["not_upright_on_lower_face"] = rejected.get("not_upright_on_lower_face", 0) + 1
            continue
        points = arrays[path["xyz_key"]].astype(float) * scale @ rotation.T + translation
        for run in path["geometry_runs"]:
            segment = points[run["start_sample"]:run["stop_sample"]]
            length = float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum())
            if length < min_length_m:
                rejected["short_interval"] = rejected.get("short_interval", 0) + 1
                continue
            candidates.append((length, path["id"], run, segment, upright, upright_id, lower, path["state"]))
    if not candidates:
        raise RuntimeError(f"No weldable seam in the auto_weld result; rejected: {rejected}")
    length, path_id, run, points, upright, upright_id, lower, state = max(candidates, key=lambda item: item[0])
    upright_top = None
    labels = arrays.get(prefix + "_plane_labels")
    if labels is not None and (labels == upright_id).any():
        from vision.weld_detector import camera_points

        cloud = camera_points(arrays[prefix + "_depth_m"], arrays[prefix + "_intrinsics"])[labels == upright_id]
        heights = (cloud * scale @ rotation.T + translation - points.mean(0)) @ up
        upright_top = float(np.percentile(heights, 98))
    centre = points.mean(0)
    direction = np.linalg.svd(points - centre)[2][0]
    seam_tilt = 90 - _tilt_deg(direction, up)
    # A horizontal fillet runs parallel to the plate: keep the observed line, levelled.
    direction -= (direction @ up) * up
    direction /= np.linalg.norm(direction)
    points = centre + np.outer((points - centre) @ direction, direction)
    across = np.cross(up, direction)
    across *= np.sign(across @ upright)  # Upright face normal: horizontal, towards the camera side.
    across /= np.linalg.norm(across)
    diagnostics.update(depth_scale=float(scale), path_id=path_id, geometry_run=run, seam_length_m=length,
                       weld_state=state, candidates=len(candidates), rejected=rejected,
                       camera_height_over_support_m=camera_height,
                       face_model="support plane + upright face through the seam (2F fillet)",
                       monocular_upright_tilt_from_vertical_deg=90 - _tilt_deg(upright, up),
                       monocular_lower_tilt_from_support_deg=_tilt_deg(lower, up),
                       monocular_seam_tilt_from_support_deg=seam_tilt,
                       upright_top_above_seam_m=upright_top)
    return ObservedSeam(points, np.vstack((across, up)), float(scale), diagnostics, upright_top)
