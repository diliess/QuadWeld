"""Import the committed RoboSimStudio welding hall as a local USD bundle."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "robosimstudio/examples/live/twins/industrial_welding"
DEFAULT_DESTINATION = ROOT / "assets/scenes/industrial-welding"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_scene(source, destination):
    from pxr import Usd, UsdGeom, UsdUtils

    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    usd = source / "exports/usd"
    source_manifest = json.loads((usd / "manifest.json").read_text())
    if digest(usd / "industrial_welding.usdc") != source_manifest["stage_sha256"]:
        raise ValueError("Source USDC differs from its manifest")
    if digest(source / "config.json") != source_manifest["source_config_sha256"]:
        raise ValueError("Source configuration differs from its exported stage")
    previews = json.loads((source / "exports/hj90_previews/manifest.json").read_text())
    if previews["source_stage_sha256"] != source_manifest["stage_sha256"]:
        raise ValueError("Source camera previews refer to another base stage")
    destination.mkdir(parents=True, exist_ok=True)
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
    copies.update({"textures/" + p.name: p for p in (usd / "textures").glob("*.png")})
    copies.update({p.name: p for p in usd.glob("industrial_welding_*_photo.usda")})
    for relative, original in copies.items():
        target = destination / relative
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
    full_scene = destination / "industrial_welding_full.usda"
    full_scene.write_text('''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    subLayers = [@industrial_welding_A04_initial_arm_photo.usda@]
)
''')
    poses = {"original": {"asset": "industrial_welding.usdc", "context_camera": "/World/Cameras/robot"}}
    for name, preview in previews["previews"].items():
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
        }
    for path in (environment, full_scene, *(destination / p["asset"] for p in poses.values())):
        stage = Usd.Stage.Open(str(path))
        if not stage or not stage.GetDefaultPrim():
            raise ValueError(f"Cannot open imported stage: {path}")
        layers, assets, missing = UsdUtils.ComputeAllDependencies(str(path))
        if missing:
            raise FileNotFoundError(str(missing))
        if any(not Path(p).resolve().is_relative_to(destination)
               for p in [*(layer.realPath for layer in layers), *assets]):
            raise ValueError(f"External dependency in {path}")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    payload = sorted([*copies, environment.name, full_scene.name])
    manifest = {
        "asset": environment.name, "default_prim": "/World", "units": "metres", "up_axis": "Z",
        "origin": "original hall floor centre; floor top Z=0; no recentering",
        "robots_included": False, "environment_children": ["Fixtures", "Looks"],
        "source": {"project": "RoboSimStudio", "commit": commit,
                   "twin": "examples/live/twins/industrial_welding",
                   "stage_sha256": source_manifest["stage_sha256"],
                   "config_sha256": source_manifest["source_config_sha256"]},
        "full_scene": {"asset": full_scene.name, "robots_included": True,
                       "robot": "B2 + E05L-Pro + welding torch + SVersion-HJ-90",
                       "default_pose": "A04_initial_arm", "poses": poses,
                       "hj90_camera": "/World/Robot/e05l_welding_torch/hj90",
                       "counts": {"rigid_bodies": source_manifest["counts"]["robot_bodies"],
                                  "revolute_joints": source_manifest["counts"]["revolute_joints"],
                                  "colliders": source_manifest["counts"]["fixture_colliders"]
                                               + source_manifest["counts"]["robot_colliders"]}},
        "hj90_camera": source_manifest["hj90_camera"],
        "scope": "static scene and camera preview; imported robot has no QuadWeld controller profile; "
                 "no navigation, gait, dynamics or welding feasibility certification",
        "files": [{"path": name, "sha256": digest(destination / name),
                   "bytes": (destination / name).stat().st_size} for name in payload],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"IMPORTED {destination}: {len(payload)} local files, {len(poses)} poses")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    import_scene(args.source, args.destination)


if __name__ == "__main__":
    main()
