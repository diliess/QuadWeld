"""Measure raw DA3 depth on independent industrial-scene RGB views."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import find_python, inspect_stage, run_process, scene_setup, write_json

STATIONS = {
    "A01": ("T_joint", ("A01_base", "A01_web")),
    "A02": ("butt_joint", ("A02_plate_negative", "A02_plate_positive", "A02_prepared_negative", "A02_prepared_positive")),
    "A04": ("vertical_corner", ("A04_base", "A04_back", "A04_return")),
    "A06": ("inclined_T", ("A06_inclined_base", "A06_inclined_web")),
}
BASELINE = ROOT / "outputs/runs/a04_20261107_135232_048087"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_record(directory):
    checkpoint = directory / "model.safetensors"
    return {"model_dir": str(directory.resolve()), "checkpoint_sha256": digest(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size, "config_sha256": digest(directory / "config.json"),
            "predictor_sha256": digest(ROOT / "scripts/predict_depth.py"),
            "source_commit": subprocess.check_output(["git", "-C", str(ROOT / "thirdparty/depth-anything-3"),
                                                       "rev-parse", "HEAD"], text=True).strip()}


def transform(points, matrix):
    import numpy as np
    return (np.c_[points, np.ones(len(points))] @ matrix.T)[:, :3]


def plan(output):
    import numpy as np
    from pxr import Usd, UsdGeom

    setup = scene_setup()
    stage = Usd.Stage.Open(setup["usd"])
    cache = UsdGeom.XformCache()
    cases = []
    for station, (kind, meshes) in STATIONS.items():
        fixture = stage.GetPrimAtPath(f"/World/Fixtures/{station}")
        curves = [p for p in Usd.PrimRange(fixture) if p.IsA(UsdGeom.BasisCurves)]
        points = np.concatenate([transform(np.array(UsdGeom.BasisCurves(p).GetPointsAttr().Get()),
                                          np.array(cache.GetLocalToWorldTransform(p)).T) for p in curves])
        target = points.mean(axis=0)
        rotation = np.array(cache.GetLocalToWorldTransform(fixture)).T[:3, :3]
        directions = ((1., -1., .3), (.65, -1., .5), (1., -.65, .4)) if station == "A04" else (
            (0., -1., .55), (.35, -1., .75), (-.4, -1., .55))
        for distance, direction in zip((.8, 1.2, 1.8), directions):
            direction = rotation @ np.array(direction)
            eye = target + distance * direction / np.linalg.norm(direction)
            forward = (target-eye) / np.linalg.norm(target-eye)
            right = np.cross(forward, [0., 0., 1.])
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            pose = np.eye(4)
            pose[:3, :3] = np.column_stack((right, down, forward))
            pose[:3, 3] = eye
            name = f"{station}_d{round(distance*100):03d}"
            cases.append({"id": name, "station": station, "kind": kind, "distance_to_target_m": distance,
                          "world_from_camera_optical": pose.tolist(), "workpiece_meshes": list(meshes),
                          "rgb": f"images/{name}.png", "truth": f"truth/{name}.npz",
                          "capture": f"captures/{name}.json", "camera_type": "independent_evaluation_camera"})
    if (BASELINE / "evaluation_sim_truth/da3_depth_evaluation.npz").is_file():
        cases.append({"id": "A04_original_HJ90", "station": "A04", "kind": "vertical_corner",
                      "camera_type": "cached_normal_arm_HJ90", "source": str(BASELINE),
                      "rgb": "images/A04_original_HJ90.png"})
    payload = {"scene": setup, "resolution_wh": [640, 1100], "cases": cases,
               "base_usd_sha256": digest(ROOT / "assets/scenes/industrial-welding/industrial_welding.usdc"),
               "scope": "DA3 raw depth only; independent single-image inference; no SAM or weld fitting; no robot motion",
               "ground_truth_usage": "Evaluation only. Predictor receives RGB filenames and GPU index; no ground-truth inputs."}
    write_json(output / "plan.json", payload)
    return payload


def capture(args):
    import numpy as np
    from isaacsim import SimulationApp

    data = json.loads((args.output_dir / "plan.json").read_text())
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "active_gpu": args.gpu,
                         "multi_gpu": False, "open_usd": data["scene"]["usd"], "sync_loads": True,
                         "width": 640, "height": 1100,
                         "extra_args": ["--/exts/omni.kit.async_engine/keep_loop_running=false"]})
    try:
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from PIL import Image
        from pxr import Gf, UsdGeom

        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        for _ in range(60):
            app.update()
        stage = omni.usd.get_context().get_stage()
        inspect_stage(stage, data["scene"])
        source = UsdGeom.Camera(stage.GetPrimAtPath(data["scene"]["camera"]))
        for case in data["cases"]:
            if case["camera_type"].startswith("cached"):
                continue
            camera = UsdGeom.Camera.Define(stage, f"/World/DA3Evaluation/{case['id']}")
            for getter in ("GetFocalLengthAttr", "GetHorizontalApertureAttr", "GetVerticalApertureAttr"):
                getattr(camera, getter)().Set(getattr(source, getter)().Get())
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(.02, 100.))
            matrix = np.array(case["world_from_camera_optical"]) @ np.diag([1., -1., -1., 1.])
            UsdGeom.Xformable(camera).MakeMatrixXform().Set(Gf.Matrix4d(matrix.T.tolist()))
            product = rep.create.render_product(str(camera.GetPath()), tuple(data["resolution_wh"]))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            rgb.attach(product)
            depth.attach(product)
            try:
                for _ in range(20):
                    app.update()
                for _ in range(5):
                    rep.orchestrator.step(rt_subframes=16, pause_timeline=True, delta_time=0.)
                    image = np.asarray(rgb.get_data()).copy()[..., :3]
                    z = np.asarray(depth.get_data()).copy().squeeze()
                    if image.shape == (1100, 640, 3) and z.shape == (1100, 640) and image.std() > 1:
                        break
                else:
                    raise RuntimeError(f"Invalid RGB/depth capture: {case['id']}")
                if timeline.is_playing():
                    raise RuntimeError("Static evaluation capture advanced the timeline")
                if image.dtype != np.uint8:
                    image = (np.clip(image, 0, 1)*255).astype(np.uint8)
                Image.fromarray(image).save(args.output_dir / case["rgb"])
                focal = float(camera.GetFocalLengthAttr().Get())
                K = np.array([[640*focal/camera.GetHorizontalApertureAttr().Get(), 0, 320],
                              [0, 1100*focal/camera.GetVerticalApertureAttr().Get(), 550], [0, 0, 1]])
                np.savez_compressed(args.output_dir / case["truth"], optical_z_m=z, intrinsics=K,
                                    world_from_camera_optical=np.array(case["world_from_camera_optical"]))
                write_json(args.output_dir / case["capture"], {**case, "image_sha256": digest(args.output_dir / case["rgb"]),
                           "truth_sha256": digest(args.output_dir / case["truth"]), "resolution_wh": [640, 1100],
                           "physics_executed": False, "arm_pose_changed": False})
                print(f"Captured {case['id']}", flush=True)
            finally:
                rgb.detach(product)
                depth.detach(product)
                product.destroy()
            stage.RemovePrim(str(camera.GetPath()))
        inspect_stage(stage, data["scene"])
        write_json(args.output_dir / "capture_status.json", {"completed": True,
                   "samples": sum(not c["camera_type"].startswith("cached") for c in data["cases"])})
    finally:
        app.close()


def triangle_depth(rays, vertices):
    import numpy as np
    a, b, c = vertices
    first, second = b-a, c-a
    cross = np.cross(rays, second)
    det = cross @ first
    inverse = np.divide(1., det, out=np.zeros(len(rays)), where=np.abs(det) > 1e-12)
    u = (cross @ -a)*inverse
    q = np.cross(-a, first)
    v = (rays @ q)*inverse
    z = (second @ q)*inverse
    hit = (np.abs(det) > 1e-12) & (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (z > 1e-8)
    return np.where(hit, z, np.inf)


def workpiece_faces(stage, case, K, shape):
    import numpy as np
    from pxr import Usd, UsdGeom
    yy, xx = np.indices(shape)
    rays = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)] @ np.linalg.inv(K).T
    nearest = np.full(xx.size, np.inf)
    labels = np.zeros(xx.size, dtype=np.int32)
    inverse = np.linalg.inv(np.array(case["world_from_camera_optical"]))
    cache = UsdGeom.XformCache()
    face_id = 0
    fixture = stage.GetPrimAtPath(f"/World/Fixtures/{case['station']}")
    for prim in Usd.PrimRange(fixture):
        if not prim.IsA(UsdGeom.Mesh) or prim.GetName() not in case["workpiece_meshes"]:
            continue
        mesh = UsdGeom.Mesh(prim)
        vertices = transform(np.array(mesh.GetPointsAttr().Get()), inverse @ np.array(cache.GetLocalToWorldTransform(prim)).T)
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        start = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face_id += 1
            face = indices[start:start+count]
            start += count
            for corner in range(1, count-1):
                values = triangle_depth(rays, vertices[face[[0, corner, corner+1]]])
                update = values < nearest
                nearest[update] = values[update]
                labels[update] = face_id
    return nearest.reshape(shape), labels.reshape(shape)


def measures(pred, true):
    import numpy as np
    error = pred-true
    absolute = np.abs(error)
    fit, _, _, _ = np.linalg.lstsq(np.c_[true, np.ones(len(true))], pred, rcond=None)
    return {"pixels": len(true), "true_mean_m": float(true.mean()), "predicted_mean_m": float(pred.mean()),
            "mae_m": float(absolute.mean()), "rmse_m": float(np.sqrt(np.mean(error**2))), "bias_m": float(error.mean()),
            "abs_rel_percent": float(np.mean(absolute/true)*100), "p95_error_m": float(np.percentile(absolute, 95)),
            "max_error_m": float(absolute.max()), "median_depth_ratio": float(np.median(pred/true)),
            "diagnostic_pred_vs_truth_slope": float(fit[0]), "diagnostic_pred_vs_truth_offset_m": float(fit[1])}


def evaluate(output, data):
    import cv2
    import numpy as np
    from pxr import Usd
    from scipy.ndimage import binary_erosion
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quadweld-benchmark-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    stage = Usd.Stage.Open(data["scene"]["usd"])
    rows, pooled_pred, pooled_true = [], [], []
    figure, axes = plt.subplots(3, 5, figsize=(15, 14))
    for axis in axes.flat:
        axis.axis("off")
    for axis, case in zip(axes.flat, data["cases"]):
        if case["camera_type"].startswith("cached"):
            source = Path(case["source"])
            with np.load(source / "evaluation_sim_truth/da3_depth_evaluation.npz") as stored:
                pred = stored["raw_da3_depth_m"].astype(float)
                true = stored["a04_geometry_depth_m"].astype(float)
                mask = stored["interior_mask"]
                labels = stored["face_labels"]
            rgb_path = source / "rgb.png"
            image_hash = digest(rgb_path)
            expected_hash = json.loads((source / "capture.json").read_text())["image_sha256"]
            if image_hash != expected_hash:
                raise ValueError("Baseline RGB changed since its prediction")
            shutil.copy2(rgb_path, output / case["rgb"])
        else:
            rgb_path = output / case["rgb"]
            recorded = json.loads((output / case["capture"]).read_text())
            image_hash = digest(rgb_path)
            if recorded["image_sha256"] != image_hash or recorded["truth_sha256"] != digest(output / case["truth"]):
                raise ValueError(f"Capture hash mismatch: {case['id']}")
            with np.load(output / "predictions" / f"{case['id']}_depth.npz") as stored:
                if str(stored["image_sha256"]) != image_hash or not bool(stored["is_metric"]):
                    raise ValueError(f"Prediction mismatch: {case['id']}")
                pred = stored["depth_m"].astype(float)
            with np.load(output / case["truth"]) as stored:
                full_truth, K = stored["optical_z_m"], stored["intrinsics"].copy()
            h, w = pred.shape
            sy, sx = h/full_truth.shape[0], w/full_truth.shape[1]
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] = (K[0, 2]+.5)*sx-.5
            K[1, 2] = (K[1, 2]+.5)*sy-.5
            yy, xx = np.indices(pred.shape, dtype=np.float32)
            true = cv2.remap(full_truth, (xx+.5)/sx-.5, (yy+.5)/sy-.5, cv2.INTER_NEAREST)
            cpu_depth, labels = workpiece_faces(stage, case, K, pred.shape)
            # Renderer depth rejects occlusions from all scene objects. The CPU
            # mesh labels select only the intended workpiece, excluding fixtures.
            finite = np.isfinite(true) & np.isfinite(cpu_depth) & (true > 0)
            disagreement = np.full(true.shape, np.inf)
            np.subtract(cpu_depth, true, out=disagreement, where=finite)
            visible = finite & (np.abs(disagreement) < np.maximum(.005, true*.005))
            mask = np.zeros(pred.shape, dtype=bool)
            for face in np.unique(labels[labels > 0]):
                mask |= binary_erosion((labels == face) & visible, structure=np.ones((7, 7)))
            if mask.sum() < 100:
                raise ValueError(f"Too few verified workpiece pixels: {case['id']}: {mask.sum()}")
        if not np.isfinite(pred[mask]).all() or not (pred[mask] > 0).all():
            raise ValueError(f"Invalid predicted depth: {case['id']}")
        result = measures(pred[mask], true[mask])
        row = {"sample": case["id"], "station": case["station"], "kind": case["kind"],
               "camera_type": case["camera_type"], "distance_to_target_m": case.get("distance_to_target_m"),
               "image_sha256": image_hash, **result}
        rows.append(row)
        pooled_pred.append(pred[mask])
        pooled_true.append(true[mask])
        np.savez_compressed(output / "evaluation" / f"{case['id']}.npz", raw_depth_m=pred.astype(np.float32),
                            true_depth_m=true.astype(np.float32), interior_mask=mask, face_labels=labels)
        axis.imshow(Image.open(rgb_path))
        axis.set_title(f"{case['id']}\nMAE {result['mae_m']:.3f} m | AbsRel {result['abs_rel_percent']:.1f}%", fontsize=9)
        print(f"{case['id']}: {result['pixels']} pixels, MAE {result['mae_m']:.4f} m, "
              f"AbsRel {result['abs_rel_percent']:.2f}%", flush=True)
    figure.suptitle("Independent RGB samples | raw DA3 vs simulation depth | no scale alignment", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, .97))
    figure.savefig(output / "samples_overview.jpg", dpi=130)
    plt.close(figure)
    aggregate = measures(np.concatenate(pooled_pred), np.concatenate(pooled_true))
    fresh = [r for r in rows if not r["camera_type"].startswith("cached")]
    payload = {"completed": True, "samples": len(rows), "new_samples": sum(not c["camera_type"].startswith("cached") for c in data["cases"]),
               "pooled_pixel_metrics": aggregate, "mean_frame_mae_m": float(np.mean([r["mae_m"] for r in rows])),
               "median_frame_mae_m": float(np.median([r["mae_m"] for r in rows])),
               "new_samples_mean_frame_mae_m": float(np.mean([r["mae_m"] for r in fresh])),
               "new_samples_mean_frame_abs_rel_percent": float(np.mean([r["abs_rel_percent"] for r in fresh])),
               "new_samples_mae_range_m": [min(r["mae_m"] for r in fresh), max(r["mae_m"] for r in fresh)],
               "definitions": {"error": "Raw optical-Z prediction minus true optical Z, metres; no geometric fit or scale/offset correction.",
                               "mask": "Visible target workpiece face interiors, eroded by 3 processed pixels; renderer depth checked against USD mesh rays.",
                               "baseline": "Existing A04 normal-arm HJ90 RGB prediction and independently verified CPU geometric truth.",
                               "diagnostic_fit": "Regression coefficients describe bias only; they are NOT used to correct predictions or reported raw errors."},
               "cases": rows}
    if (output / "model.json").is_file():
        payload["model"] = json.loads((output / "model.json").read_text())
    write_json(output / "summary.json", payload)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for station in STATIONS:
        selected = [r for r in rows if r["station"] == station]
        axes[0].scatter([r["true_mean_m"] for r in selected], [r["predicted_mean_m"] for r in selected], label=station)
        axes[1].scatter([r["true_mean_m"] for r in selected], [r["bias_m"] for r in selected], label=station)
    lower = min(r["true_mean_m"] for r in rows)
    upper = max(max(r["true_mean_m"], r["predicted_mean_m"]) for r in rows)
    axes[0].plot([lower, upper], [lower, upper], "k--", linewidth=1, label="Perfect depth")
    axes[0].set(xlabel="Mean true optical Z (m)", ylabel="Mean raw DA3 optical Z (m)", title="One dot per independent image")
    axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Mean true optical Z (m)", ylabel="Mean signed depth error (m)", title="Positive values mean depth is overestimated")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
    figure.savefig(output / "depth_errors.png", dpi=150)
    plt.close(figure)
    lines = ["# DA3 多样本深度测试", "", f"共 {len(rows)} 帧，4 类工件，12 张新采集 RGB 加 1 张已有 A04 基线。",
             "新样本使用 HJ-90 相同内参的独立评测相机，目标距离为 0.8、1.2、1.8 m，各距离采用不同方位；距离与视角同时变化，不能单独归因于某一个因素。机械臂保持正常初始姿态。", "",
             "每张 RGB 独立调用 DA3，模型只接收图片，没有输入真值深度或相机内参。仿真光轴深度仅用于离线评价。未运行 SAM3 或焊缝拟合。", "",
             "评价可见焊板各面内部，排除 3 个处理像素宽的边界、遮挡和夹具。原始 MAE、RMSE 与相对误差均未作尺度或偏移校正。", "",
             f"逐帧 MAE 平均值：{payload['mean_frame_mae_m']:.4f} m；像素汇总 MAE：{aggregate['mae_m']:.4f} m；像素汇总 AbsRel：{aggregate['abs_rel_percent']:.2f}%。", "",
             f"仅 12 张新采集样本的逐帧 MAE 平均值：{payload['new_samples_mean_frame_mae_m']:.4f} m；"
             f"各帧 MAE 范围：{payload['new_samples_mae_range_m'][0]:.4f}–{payload['new_samples_mae_range_m'][1]:.4f} m。", "",
             "| 样本 | 有效像素 | 真值均深 m | DA3 均深 m | MAE m | RMSE m | AbsRel |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['sample']} | {row['pixels']} | {row['true_mean_m']:.3f} | {row['predicted_mean_m']:.3f} | "
                     f"{row['mae_m']:.3f} | {row['rmse_m']:.3f} | {row['abs_rel_percent']:.1f}% |")
    lines += ["", "[样本总览](samples_overview.jpg) · [误差图](depth_errors.png) · [逐帧 CSV](summary.csv) · [完整 JSON](summary.json)", "",
              "这些结果评价当前模型与部署在同一仿真场景中的表现，不代表 DA3 在真实照片或公开基准上的普遍精度。",
              "该测试不评价 SAM3 分割、接缝检出率或机器人运动可行性。不同帧独立推理，未做多视图融合。", ""]
    lines += ["## 复跑", "", "```bash", ".venv/bin/python scripts/benchmark_da3.py --gpu 0",
              "# 使用同一目录中经图片哈希校验的采集和预测结果恢复执行", f".venv/bin/python scripts/benchmark_da3.py --gpu 0 --resume --output-dir {output.relative_to(ROOT)}",
              "```", ""]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--isaac-python", type=Path)
    parser.add_argument("--da3-python", type=Path)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "thirdparty/depth-anything-3/weights/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--resume", action="store_true", help="Resume verified captures/predictions in this output directory")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--_capture-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M%S")
    args.output_dir = (args.output_dir or ROOT / "outputs/benchmarks" / f"da3_{stamp}").resolve()
    output = args.output_dir
    if args._capture_worker:
        capture(args)
        return
    if args.evaluate_only:
        evaluate(output, json.loads((output / "plan.json").read_text()))
        return
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError(f"Use an empty output directory: {output}")
    for directory in ("images", "truth", "captures", "predictions", "evaluation"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    if args.resume:
        data = json.loads((output / "plan.json").read_text())
        if data["base_usd_sha256"] != digest(ROOT / "assets/scenes/industrial-welding/industrial_welding.usdc"):
            raise ValueError("Scene changed since this capture; use a new benchmark directory")
    else:
        data = plan(output)
    for filename in ("config.json", "model.safetensors"):
        if not (args.model_dir / filename).is_file():
            raise FileNotFoundError(f"Missing model file: {args.model_dir / filename}")
    record = model_record(args.model_dir)
    if (output / "model.json").is_file() and json.loads((output / "model.json").read_text()) != record:
        raise ValueError("Model or predictor changed; use a new benchmark directory")
    write_json(output / "model.json", record)
    isaac, error = find_python("isaac", args.isaac_python)
    if error:
        raise RuntimeError(error)
    da3, error = find_python("da3", args.da3_python)
    if error:
        raise RuntimeError(error)
    subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], check=True)
    write_json(output / "status.json", {"completed": False, "stage": "capture", "samples_planned": len(data["cases"])})
    if not (output / "capture_status.json").is_file() or not json.loads((output / "capture_status.json").read_text()).get("completed"):
        run_process([isaac, str(Path(__file__).resolve()), "--_capture-worker", "--output-dir", str(output), "--gpu", str(args.gpu)], output / "capture.log")
    if not json.loads((output / "capture_status.json").read_text())["completed"]:
        raise RuntimeError("Incomplete RGB/ground-truth capture")
    import numpy as np
    images = []
    for case in data["cases"]:
        if case["camera_type"].startswith("cached"):
            continue
        image = output / case["rgb"]
        recorded = json.loads((output / case["capture"]).read_text())
        if recorded["image_sha256"] != digest(image) or recorded["truth_sha256"] != digest(output / case["truth"]):
            raise ValueError(f"Capture modified: {case['id']}")
        cached = output / "predictions" / f"{case['id']}_depth.npz"
        valid_cache = False
        if cached.is_file():
            with np.load(cached, allow_pickle=False) as stored:
                valid_cache = str(stored["image_sha256"]) == digest(image) and bool(stored["is_metric"])
        if not valid_cache:
            images.append(str(image))
    write_json(output / "status.json", {"completed": False, "stage": "DA3_inference"})
    if images:
        run_process([da3, str(ROOT / "scripts/predict_depth.py"), *images, "--gpu", str(args.gpu),
                     "--model-dir", str(args.model_dir.resolve()), "--output-dir", str(output / "predictions")], output / "da3.log")
    write_json(output / "status.json", {"completed": False, "stage": "evaluation"})
    evaluate(output, data)
    write_json(output / "status.json", {"completed": True, "stage": "complete", "report": "README.md"})
    print(f"Benchmark completed: {output}", flush=True)


if __name__ == "__main__":
    main()
