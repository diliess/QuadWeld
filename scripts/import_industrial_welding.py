"""Import a validated welding hall and rebuild QuadWeld's independent sensor layer."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "robosimstudio/examples/live/twins/industrial_welding"
DEFAULT_DESTINATION = ROOT / "assets/scenes/industrial-welding"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"External bundle path: {relative}")
    return path


def validate_layers(bundle, names):
    from pxr import Usd, UsdGeom, UsdUtils

    for name in names:
        path = local_path(bundle, name)
        stage = Usd.Stage.Open(str(path))
        if (not stage or not stage.GetDefaultPrim() or UsdGeom.GetStageUpAxis(stage) != "Z"
                or UsdGeom.GetStageMetersPerUnit(stage) != 1.):
            raise ValueError(f"Invalid metre/Z-up scene: {path}")
        layers, assets, missing = UsdUtils.ComputeAllDependencies(str(path))
        if missing:
            raise FileNotFoundError(str(missing))
        if any(not Path(p).resolve().is_relative_to(bundle.resolve())
               for p in [*(layer.realPath for layer in layers), *assets]):
            raise ValueError(f"External dependency in {path}")


def build_bundle(source, destination):
    from pxr import Usd, UsdGeom

    usd = source / "exports/usd"
    source_manifest = json.loads((usd / "manifest.json").read_text())
    if digest(usd / "industrial_welding.usdc") != source_manifest["stage_sha256"]:
        raise ValueError("Source USDC differs from its manifest")
    if digest(source / "config.json") != source_manifest["source_config_sha256"]:
        raise ValueError("Source configuration differs from its exported stage")
    physical = json.loads((source / "exports/hj90_physical_validation.json").read_text())
    if (not physical["passed"] or physical["source_stage_sha256"] != source_manifest["stage_sha256"]
            or physical["config_sha256"] != source_manifest["source_config_sha256"]):
        raise ValueError("Source HJ-90 validation is failed or stale")
    previews_file = source / "exports/hj90_previews/manifest.json"
    previews = json.loads(previews_file.read_text()) if previews_file.is_file() else {}
    if previews and previews["source_stage_sha256"] != source_manifest["stage_sha256"]:
        raise ValueError("Source camera previews refer to another base stage")

    # Remove only the previously generated payload in the staging copy. Unknown
    # user files and README remain intact; the original bundle is backed up below.
    previous_manifest = destination / "manifest.json"
    if previous_manifest.is_file():
        previous = json.loads(previous_manifest.read_text())
        for record in previous["files"]:
            local_path(destination, record["path"]).unlink(missing_ok=True)
    copies = {
        "industrial_welding.usdc": usd / "industrial_welding.usdc",
        "source/config.json": source / "config.json",
        "source/usd_manifest.json": usd / "manifest.json",
        "source/hj90_physical_validation.json": source / "exports/hj90_physical_validation.json",
        "source/hj90_datasheet.md": source / "ref/hj90_datasheet.md",
        "licenses/RoboSimStudio.LICENSE": source / "LICENSE",
        "licenses/RoboSimStudio.NOTICE": source / "NOTICE",
        "licenses/robot_models.LICENSE": source / "assets/assembled/source/LICENSE",
    }
    for name in ("scene_sync_validation.json", "multilayer_validation.json"):
        if (source / "exports" / name).is_file():
            copies["source/" + name] = source / "exports" / name
    for name in ("seams.json", "material_manifest.json", "hj90_physical_validation.json"):
        copies["metadata/" + name] = usd / "metadata" / name
    copies.update({"textures/" + p.name: p for p in (usd / "textures").glob("*.png")})
    # Only import current, provenance-checked previews. Archived old poses must
    # never silently override the newly imported robot or workpiece layout.
    for preview in previews.get("previews", {}).values():
        original = local_path(source, preview["preview"])
        copies[original.name] = original
    for relative, original in copies.items():
        target = local_path(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)

    environment = destination / "industrial_welding.usda"
    environment.write_text('''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    subLayers = [@industrial_welding.usdc@]
)

over "World"
{
    over "Robot" (active = false) {}
    over "Cameras" (active = false) {}
    over "Lights" (active = false) {}
    over "PhysicsScene" (active = false) {}
}
''')
    import math
    stage = Usd.Stage.Open(str(destination / "industrial_welding.usdc"))
    base = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Robot/base_link"))
    position = list(base.ExtractTranslation())
    forward = base.TransformDir((1, 0, 0))
    poses = {"original": {
        "asset": "industrial_welding.usdc", "context_camera": "/World/Cameras/robot",
        "base_xy_yaw": [*position[:2], math.atan2(forward[1], forward[0])],
        "base_height_m": position[2], "arm_q_rad": source_manifest["initial_pose"]["arm_q_rad"],
        "placement": "source normal initial standing pose; no work-station placement or aiming",
        "station": None,
        "static_checks": {"source_initial_pose": source_manifest["initial_pose"],
                          "hj90_physical_validation_passed": physical["passed"],
                          "scope": "source initial pose and camera physical model; no work-stance IK or dynamics"},
    }}
    for name, preview in previews.get("previews", {}).items():
        asset = Path(preview["preview"]).name
        stage = Usd.Stage.Open(str(destination / asset))
        context_cameras = [str(p.GetPath()) for p in stage.Traverse()
                           if p.IsA(UsdGeom.Camera) and p.GetName().endswith("photo_context")]
        if len(context_cameras) != 1:
            raise ValueError(f"Expected one context camera: {asset}")
        poses[name] = {
            "asset": asset, "context_camera": context_cameras[0],
            "base_xy_yaw": preview["base_xy_yaw"], "base_height_m": preview["base_height_m"],
            "arm_q_rad": preview["arm_q_rad"], "static_checks": preview["static_checks"],
            "placement": "current source-authored static workstation preview; no movement trajectory",
            "station": "A02" if name == "A02" else "A04" if name.startswith("A04") else None,
        }
    default_pose = "A04_initial_arm" if "A04_initial_arm" in poses else "original"
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from hj90.optics import load_calibration
    from hj90.usd import export_layer
    active_sensor = export_layer(destination, load_calibration(), base_layer=poses[default_pose]["asset"])
    full_scene = destination / "industrial_welding_full.usda"
    validate_layers(destination, {environment.name, full_scene.name, active_sensor["layer"],
                                  *(p["asset"] for p in poses.values())})
    environment_stage = Usd.Stage.Open(str(environment))
    if environment_stage.GetPrimAtPath("/World/Robot").IsActive():
        raise ValueError("Environment-only layer must disable the imported robot")
    payload = sorted({*copies, environment.name, full_scene.name, *active_sensor["extra_files"]})
    config = json.loads((source / "config.json").read_text())
    manifest = {
        "asset": environment.name, "default_prim": "/World", "units": "metres", "up_axis": "Z",
        "origin": "original hall floor centre; floor top Z=0; no recentering",
        "robots_included": False, "environment_children": ["Fixtures", "Looks"],
        "source": {"project": "RoboSimStudio", "commit": commit,
                   "twin": "examples/live/twins/industrial_welding",
                   "stage_sha256": source_manifest["stage_sha256"],
                   "config_sha256": source_manifest["source_config_sha256"]},
        "layout": {"hall_size_m": config["hall"]["size_m"], "A_stations": len(config["stations"])},
        "full_scene": {"asset": full_scene.name, "robots_included": True,
                       "robot": "B2 + E05L-Pro + welding torch + SVersion-HJ-90",
                       "default_pose": default_pose, "poses": poses,
                       "active_stereo_laser": {k: v for k, v in active_sensor.items() if k != "extra_files"},
                       "hj90_camera": "/World/Robot/e05l_welding_torch/hj90",
                       "counts": {"rigid_bodies": source_manifest["counts"]["robot_bodies"],
                                  "revolute_joints": source_manifest["counts"]["revolute_joints"],
                                  "colliders": source_manifest["counts"]["fixture_colliders"]
                                               + source_manifest["counts"]["robot_colliders"]}},
        "hj90_camera": source_manifest["hj90_camera"],
        "sensor_scope": "90 mm simulated stereo and central RTX laser; separate vendor-calibrated rectified RGB; "
                        "stereo intrinsics and laser calibration remain simulation assumptions",
        "scope": "static scene; only current source poses included; no QuadWeld robot controller, "
                 "navigation, gait, dynamics or welding feasibility certification",
        "files": [{"path": name, "sha256": digest(destination / name),
                   "bytes": (destination / name).stat().st_size} for name in payload],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def import_scene(source, destination):
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination must be separate directories")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".welding-import-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "bundle"
        if destination.exists():
            shutil.copytree(destination, staging)
        else:
            staging.mkdir()
        manifest = build_bundle(source, staging)
        # Publish only after every generated layer and dependency has validated.
        # Preserve a complete runnable copy of the previous layout, not old
        # photo layers pointing at the newly replaced base stage.
        if destination.exists():
            stamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M%S_%f")
            backup = ROOT / "outputs/imports" / f"{destination.name}_{stamp}"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(destination, backup)
            previous = Path(temporary) / "previous"
            os.replace(destination, previous)
            try:
                os.replace(staging, destination)
            except BaseException:
                os.replace(previous, destination)
                raise
            print(f"PREVIOUS_BUNDLE {backup}")
        else:
            os.replace(staging, destination)
    print(f"IMPORTED {destination}: {len(manifest['files'])} files, "
          f"{len(manifest['full_scene']['poses'])} current poses, "
          f"default={manifest['full_scene']['default_pose']}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    import_scene(args.source, args.destination)


if __name__ == "__main__":
    main()
