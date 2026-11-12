"""Load A04 with the normal initial arm pose, capture one HJ-90 RGB frame, and run auto_weld."""

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "assets/scenes/industrial-welding"
POSE = "A04_initial_arm"
INITIAL_ARM_Q = (0., 1.2, 2.3, 0., 1.3, 0.)
RESOLUTION = (640, 1100)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def child_environment():
    """Each interpreter loads its own Python and native libraries."""
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "VIRTUAL_ENV",
                "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL", "CUDA_VISIBLE_DEVICES"):
        environment.pop(key, None)
    environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    return environment


def scene_setup(rgb_profile="legacy", rgb_resolution=None):
    """Historical benchmark callers retain legacy optics; CLI selects vendor RGB explicitly."""
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    pose = manifest["full_scene"]["poses"][POSE]
    if len(pose["arm_q_rad"]) != 6 or any(
            not math.isclose(a, b, abs_tol=1e-8) for a, b in zip(pose["arm_q_rad"], INITIAL_ARM_Q)):
        raise ValueError("A04_initial_arm must retain the normal initial joint positions")
    if rgb_profile not in ("vendor", "legacy"):
        raise ValueError("Unknown HJ-90 RGB profile")
    vendor = None
    if rgb_profile == "vendor":
        from hj90.vendor_calibration import load_rgb_calibration, resize_rgb_calibration
        vendor = resize_rgb_calibration(load_rgb_calibration(), rgb_resolution)
        sensor = manifest["full_scene"]["active_stereo_laser"]
        if vendor["source_sha256"] != sensor["rgb_calibration"]["source_sha256"]:
            raise ValueError("RGB calibration changed; rebuild with scripts/build_hj90_sensor.py")
        asset_name = manifest["full_scene"]["asset"]
        camera = sensor["rgb_camera"]
        resolution = vendor["image_size_wh"]
    else:
        asset_name = pose["asset"]
        camera = manifest["full_scene"]["hj90_camera"]
        resolution = list(rgb_resolution or RESOLUTION)
    asset = (BUNDLE / asset_name).resolve()
    if not asset.is_relative_to(BUNDLE.resolve()) or not asset.is_file():
        raise FileNotFoundError(f"A04 pose asset not found in the scene bundle: {asset}")
    return {"pose": POSE, "usd": str(asset), "camera": camera,
            "rgb_profile": rgb_profile, "rgb_calibration": vendor,
            "resolution_wh": resolution, "base_xy_yaw": pose["base_xy_yaw"],
            "base_height_m": pose["base_height_m"], "arm_q_rad": pose["arm_q_rad"],
            "placement": "authored static A04 placement; normal initial arm; no aiming or walking",
            "recorded_static_checks": pose["static_checks"]}


def python_candidates(role, explicit=None, da3_python=None):
    configured = explicit or os.environ.get(f"QUADWELD_{role.upper()}_PYTHON")
    if configured:
        return [Path(configured).expanduser().absolute()]
    candidates = []
    if role == "isaac":
        candidates.append(ROOT / ".venv/bin/python")
    if role == "sam" and da3_python:
        candidates.append(Path(da3_python).parent.parent.parent / "sam3/bin/python")
    candidates += [Path(sys.executable), ROOT / f".venv-{role}/bin/python"]
    name = {"isaac": "isaacsim", "da3": "depth-anything-3", "sam": "sam3"}[role]
    for directory in ("miniconda3", "miniforge3", "anaconda3"):
        candidates.append(Path.home() / directory / "envs" / name / "bin/python")
    registry = Path.home() / ".conda/environments.txt"
    if registry.is_file():
        candidates += [Path(line.strip()) / "bin/python" for line in registry.read_text().splitlines()
                       if line.strip() and not line.lstrip().startswith("#")]
    return list(dict.fromkeys(path.absolute() for path in candidates))


def find_python(role, explicit=None, da3_python=None):
    modules = {"isaac": ["isaacsim", "PIL", "numpy"],
               "da3": ["depth_anything_3", "torch", "cv2", "scipy", "matplotlib", "PIL", "safetensors"],
               "sam": ["sam3", "torch", "torchvision", "cv2", "PIL"]}[role]
    probe = ("import importlib.util,json,sys; "
             "print(json.dumps({'version':list(sys.version_info[:2]),"
             "'missing':[name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]}))")
    failures = []
    for executable in python_candidates(role, explicit, da3_python):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            if explicit or os.environ.get(f"QUADWELD_{role.upper()}_PYTHON"):
                failures.append(f"Not an executable file: {executable}")
            continue
        try:
            result = subprocess.run([str(executable), "-c", probe, *modules], cwd=ROOT,
                                    env=child_environment(), capture_output=True, text=True, timeout=15)
            info = json.loads(result.stdout.strip()) if result.returncode == 0 else None
            if info and info["version"] == [3, 12] and not info["missing"]:
                return str(executable), None
            failures.append(f"{executable}: {info or result.stderr.strip()[-400:]}")
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            failures.append(f"{executable}: {error}")
    detail = "; ".join(failures)
    return None, (f"Set --{role}-python or QUADWELD_{role.upper()}_PYTHON to a Python 3.12 environment "
                  f"containing {', '.join(modules)}. {detail}").strip()


def preflight(args, setup):
    interpreters, issues = {}, []
    sensor_only = args.capture_only or args.sensor == "hj90"
    roles = ("isaac",) if sensor_only else ("isaac", "da3", "sam")
    for role in roles:
        executable, error = find_python(role, getattr(args, f"{role}_python"), interpreters.get("da3"))
        interpreters[role] = executable
        if error:
            issues.append(error)
    weights = [] if sensor_only else [
        ROOT / "thirdparty/sam3/assets/model/sam3.1_multiplex.pt",
        ROOT / "thirdparty/depth-anything-3/weights/DA3NESTED-GIANT-LARGE-1.1/config.json",
        ROOT / "thirdparty/depth-anything-3/weights/DA3NESTED-GIANT-LARGE-1.1/model.safetensors"]
    for path in weights:
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"Missing model file: {path}")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        if gpu.returncode or str(args.gpu) not in gpu.stdout.split():
            issues.append(f"NVIDIA GPU {args.gpu} unavailable: {gpu.stderr.strip() or gpu.stdout.strip()}")
    except (OSError, subprocess.TimeoutExpired) as error:
        issues.append(f"NVIDIA GPU check failed: {error}")
    return {"ready": not issues, "scene": setup, "interpreters": interpreters, "issues": issues}


def run_process(command, log_path):
    print(f"Running {Path(command[1]).name}; log: {log_path}", flush=True)
    with Path(log_path).open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=child_environment(), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        if code:
            raise RuntimeError(f"Process exited with status {code}; see {log_path}")


def inspect_stage(stage, setup):
    """Read pose and camera metadata only; no seam coordinates enter perception."""
    import numpy as np
    from pxr import UsdGeom, UsdUtils

    if not stage or UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise ValueError("The imported A04 scene must use metres and Z-up")
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
        raise ValueError(f"B2 is not at the recorded A04 station position: {position}")
    if camera.GetProjectionAttr().Get() != UsdGeom.Tokens.perspective:
        raise ValueError("HJ-90 must be a perspective camera")
    focal = float(camera.GetFocalLengthAttr().Get())
    horizontal = float(camera.GetHorizontalApertureAttr().Get())
    vertical = float(camera.GetVerticalApertureAttr().Get())
    if not all(math.isfinite(v) and v > 0 for v in (focal, horizontal, vertical)):
        raise ValueError("Invalid camera focal length or aperture")
    from hj90.usd import camera_intrinsics
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


def capture_worker(args, setup):
    """Separate process: capture RGB with the authored static pose, then release Isaac/GPU."""
    output = args.output_dir
    report_path = output / "capture.json"
    report = {"passed": False, "scene": setup, "physics_executed": False, "aim_adjusted": False}
    write_json(report_path, report)
    from isaacsim import SimulationApp

    resolution = tuple(setup["resolution_wh"])
    app = SimulationApp({"headless": args.headless, "renderer": "RaytracedLighting",
                         "active_gpu": args.gpu, "multi_gpu": False, "open_usd": setup["usd"],
                         "width": resolution[0], "height": resolution[1], "sync_loads": True,
                         "extra_args": ["--/exts/omni.kit.async_engine/keep_loop_running=false",
                                        "--/app/window/title=QuadWeld | A04 initial arm | HJ-90"]})
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from PIL import Image

        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        for _ in range(60):
            app.update()
        stage = omni.usd.get_context().get_stage()
        # RGB perception uses ambient imagery. Opening the full scene separately
        # still previews the central laser; this session edit never writes assets.
        stage.SetEditTarget(stage.GetSessionLayer())
        from hj90.usd import RIG_PATH
        laser = stage.GetPrimAtPath(RIG_PATH + "/laser")
        if laser:
            laser.GetAttribute("inputs:intensity").Set(0.)
        report["laser_enabled"] = False
        calibration, pose = inspect_stage(stage, setup)
        if not args.headless:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport:
                viewport.camera_path = setup["camera"]
                viewport.set_texture_resolution(resolution)
        product = rep.create.render_product(setup["camera"], resolution)
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(product)
        try:
            for _ in range(32):
                app.update()
            for _ in range(5):
                rep.orchestrator.step(rt_subframes=16, pause_timeline=True, delta_time=0.)
                data = np.asarray(rgb.get_data()).copy()
                if data.ndim != 3 or data.shape[:2] != (resolution[1], resolution[0]) or data.shape[2] < 3:
                    continue
                data = data[..., :3]
                if not np.isfinite(data).all():
                    continue
                if data.dtype != np.uint8:
                    data = (np.clip(data, 0, 1)*255).astype(np.uint8)
                if float(data.std()) >= 1.:
                    break
            else:
                raise RuntimeError("HJ-90 did not return a valid, nonconstant RGB frame")
            if timeline.is_playing():
                raise RuntimeError("The timeline unexpectedly advanced during static capture")
            Image.fromarray(data).save(output / "rgb.png")
            write_json(output / "camera.json", calibration)
            report.update(passed=True, actual_pose=pose, rgb="rgb.png", calibration="camera.json",
                          image_sha256=hashlib.sha256((output / "rgb.png").read_bytes()).hexdigest(),
                          rgb_std=float(data.std()), depth_source="DA3 from captured RGB; no renderer depth")
            write_json(report_path, report)
        finally:
            rgb.detach(product)
            product.destroy()
    except BaseException as error:
        report.update(passed=False, error=str(error))
        write_json(report_path, report)
        raise
    finally:
        # Kit may terminate this worker during close; persist the report first.
        app.close()
    return 0


def execute(args, setup, checked):
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Use an empty output directory to avoid mixing runs: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "run_report.json"
    report = {"completed": False, "status": "preflight", "scene": setup, "preflight": checked,
              "started_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat()}
    write_json(report_path, report)
    try:
        if not checked["ready"]:
            raise RuntimeError("\n".join(checked["issues"]))
        report["status"] = "capturing_rgb"
        write_json(report_path, report)
        command = [checked["interpreters"]["isaac"], str(Path(__file__).resolve()), "--_capture-worker",
                   "--output-dir", str(output), "--gpu", str(args.gpu), "--rgb-profile", args.rgb_profile]
        if args.rgb_resolution:
            command += ["--rgb-resolution", *map(str, args.rgb_resolution)]
        if args.headless:
            command.append("--headless")
        run_process(command, output / "capture.log")
        capture = json.loads((output / "capture.json").read_text())
        if not capture.get("passed"):
            raise RuntimeError(f"RGB capture failed: {capture.get('error', 'see capture.log')}")
        image = output / "rgb.png"
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest != capture["image_sha256"]:
            raise RuntimeError("Captured image differs from its capture report")
        report["capture"] = capture
        if args.capture_only:
            report.update(completed=True, status="capture_only")
        else:
            report["status"] = "recognizing"
            write_json(report_path, report)
            command = [checked["interpreters"]["da3"], str(ROOT / "scripts/auto_weld.py"), str(image),
                       "--sam-python", checked["interpreters"]["sam"], "--gpu", str(args.gpu),
                       "--intrinsics", str(output / "camera.json"), "--output-dir", str(output / "auto_weld"),
                       "--semantic-threshold", str(args.semantic_threshold)]
            if args.model_cache_dir:
                command += ["--model-cache-dir", str(args.model_cache_dir)]
            run_process(command, output / "auto_weld.log")
            result = json.loads((output / "auto_weld/data/auto_report.json").read_text())
            if len(result["results"]) != 1 or result["results"][0]["image_sha256"] != digest:
                raise RuntimeError("Auto-weld report does not match the captured RGB frame")
            image_result = result["results"][0]
            report.update(completed=True, status=image_result["status"],
                          recognition={"report": "auto_weld/data/auto_report.json", "paths": "auto_weld/seam_paths.csv",
                                       "candidate_paths": len(image_result["paths"]),
                                       "valid_intervals": sum(len(path["geometry_runs"]) for path in image_result["paths"]),
                                       "coordinate_frame": "estimated optical camera; metres"})
        write_json(report_path, report)
        print(f"Completed: {report['status']}\nResults: {output}", flush=True)
        return 0
    except BaseException as error:
        report.update(completed=False, status="failed", error=str(error))
        write_json(report_path, report)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New, empty run directory")
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index")
    parser.add_argument("--headless", action="store_true", help="Capture without a GUI")
    parser.add_argument("--isaac-python", type=Path, help="Isaac Sim Python executable")
    parser.add_argument("--da3-python", type=Path, help="Depth Anything 3 Python executable")
    parser.add_argument("--sam-python", type=Path, help="SAM3 Python executable")
    parser.add_argument("--model-cache-dir", type=Path, help="Optional DA3/SAM model-output cache")
    parser.add_argument("--semantic-threshold", type=float, default=.25)
    parser.add_argument("--capture-only", action="store_true", help="Capture RGB without running recognition")
    parser.add_argument("--sensor", choices=("rgb", "hj90"), default="rgb",
                        help="rgb: existing DA3/SAM workflow; hj90: active stereo/laser scan and metric point cloud")
    parser.add_argument("--rgb-profile", choices=("vendor", "legacy"), default="vendor",
                        help="vendor: measured rectified RGB K/T; legacy: original simulated midpoint camera")
    parser.add_argument("--rgb-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Resize the full RGB raster and K; default is vendor native 3120 x 4096")
    parser.add_argument("--scan-lines", type=int, default=640, help="HJ-90 static laser scan samples")
    parser.add_argument("--check", action="store_true", help="Check environments/assets/GPU without launching Isaac")
    parser.add_argument("--_capture-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.gpu < 0 or args.scan_lines < 1 or not 0 < args.semantic_threshold <= 1:
        parser.error("Use --gpu >= 0 and 0 < --semantic-threshold <= 1")
    if args.rgb_resolution and min(args.rgb_resolution) < 1:
        parser.error("RGB width and height must be positive")
    stamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M%S_%f")
    args.output_dir = (args.output_dir or ROOT / "outputs/runs" / f"a04_{stamp}").expanduser().resolve()
    if args.model_cache_dir:
        args.model_cache_dir = args.model_cache_dir.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    setup = scene_setup(args.rgb_profile if args.sensor == "rgb" else "legacy", args.rgb_resolution)
    if args._capture_worker:
        return capture_worker(args, setup)
    checked = preflight(args, setup)
    if args.check:
        print(json.dumps(checked, ensure_ascii=False, indent=2))
        return 0 if checked["ready"] else 1
    if args.sensor == "hj90":
        if not checked["ready"]:
            raise RuntimeError("\n".join(checked["issues"]))
        command = [sys.executable, str(ROOT / "scripts/scan_hj90.py"), "--pose", POSE,
                   "--lines", str(args.scan_lines), "--gpu", str(args.gpu),
                   "--isaac-python", checked["interpreters"]["isaac"],
                   "--output-dir", str(args.output_dir)]
        if args.headless:
            command.append("--headless")
        subprocess.run(command, cwd=ROOT, env=child_environment(), check=True)
        return 0
    return execute(args, setup, checked)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"QuadWeld failed: {error}", file=sys.stderr)
        raise SystemExit(1)
