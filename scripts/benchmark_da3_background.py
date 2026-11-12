"""Hold A01, optics and lights fixed; measure DA3 sensitivity to background."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import benchmark_da3 as bench
from main import find_python, inspect_stage, run_process, write_json

ROOT = bench.ROOT
CONDITIONS = {
    "original": "原始车间",
    "empty": "移除其他工位，保留地面和厂房",
    "gray": "空背景加灰色平面",
    "grid": "相同平面加 0.1 m 棋盘格",
    "cube": "空背景加 0.3 m 蓝色立方体",
    "original_repeat": "恢复原始车间并重新拍摄",
}


def plan(output, repeats):
    data = bench.plan(output)
    fixed = next(c for c in data["cases"] if c["id"] == "A01_d080")
    data["cases"] = []
    for repeat in range(1, repeats+1):
        for condition, description in CONDITIONS.items():
            name = f"{condition}_r{repeat}"
            data["cases"].append({**fixed, "id": name, "condition": condition, "group": "rendered",
                                  "replicate": repeat, "description": description, "rgb": f"images/{name}.png",
                                  "truth": f"truth/{name}.npz", "capture": f"captures/{name}.json"})
    data.update(scope="Single fixed A01 viewpoint; background ablation; raw metric DA3 only",
                foreground_control="Also restore the original A01 fixture RGB pixels with a 4-pixel safety margin. "
                                   "These composites isolate visual context from shadows/reflections.",
                checkpoint_seed_per_image=0, render_repeats_per_condition=repeats,
                reference_usage="Grid/cube dimensions are NOT passed to DA3. They are visual cues only.")
    data["scene_sha256"] = bench.digest(data["scene"]["usd"])
    write_json(output / "plan.json", data)
    return data


def invariant_record(stage, camera):
    """Record composed properties, bindings and transforms, not just file names."""
    from pxr import Usd, UsdGeom
    result = {}
    for root in ("/World/Fixtures/A01", "/World/Lights", "/World/Robot", str(camera.GetPath())):
        entries = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            entries.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                            "attributes": {a.GetName(): str(a.Get()) for a in prim.GetAttributes()},
                            "relationships": {r.GetName(): list(map(str, r.GetTargets())) for r in prim.GetRelationships()}})
        result[root] = entries
    # Materials live outside the A01 subtree.
    result["looks"] = [{"path": str(p.GetPath()),
                        "attributes": {a.GetName(): str(a.Get()) for a in p.GetAttributes()}}
                       for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Looks"))]
    result["meters_per_unit"] = UsdGeom.GetStageMetersPerUnit(stage)
    result["up_axis"] = str(UsdGeom.GetStageUpAxis(stage))
    return result


def add_background(stage, condition):
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    def material(name, color):
        mat = UsdShade.Material.Define(stage, f"/World/BackgroundStudy/Materials/{name}")
        shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.9)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    # A plane behind the entire A01 fixture (which ends at world y=3.57 m).
    if condition in ("gray", "grid"):
        points, indices, colors = [], [], []
        for iz in range(20):
            for ix in range(40):
                x, z = -14.5 + ix*.1, iz*.1
                start = len(points)
                points.extend([(x, 4.5, z), (x+.1, 4.5, z), (x+.1, 4.5, z+.1), (x, 4.5, z+.1)])
                indices.extend(range(start, start+4))
                value = (.15 if (ix+iz) % 2 else .55) if condition == "grid" else .35
                colors.append(Gf.Vec3f(value, value, value))
        mesh = UsdGeom.Mesh.Define(stage, "/World/BackgroundStudy/Panel")
        mesh.CreatePointsAttr(Vt.Vec3fArray(points))
        mesh.CreateFaceVertexCountsAttr([4]*len(colors))
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(colors)
        # Vertex color is consumed by a standard surface shader.
        mat = UsdShade.Material.Define(stage, "/World/BackgroundStudy/PanelMaterial")
        reader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Color")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), "result")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.9)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    elif condition == "cube":
        cube = UsdGeom.Cube.Define(stage, "/World/BackgroundStudy/ReferenceCube")
        cube.CreateSizeAttr(.3)
        UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(-12.15, 4.2, .15))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material("Blue", (.035, .18, .65)))


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
        # All ablations are session-layer opinions; source USD files are untouched.
        stage.SetEditTarget(stage.GetSessionLayer())
        inspect_stage(stage, data["scene"])
        camera = UsdGeom.Camera.Define(stage, "/World/DA3BackgroundCamera")
        source = UsdGeom.Camera(stage.GetPrimAtPath(data["scene"]["camera"]))
        for getter in ("GetFocalLengthAttr", "GetHorizontalApertureAttr", "GetVerticalApertureAttr"):
            getattr(camera, getter)().Set(getattr(source, getter)().Get())
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(.02, 100.))
        pose = np.array(data["cases"][0]["world_from_camera_optical"])
        matrix = pose @ np.diag([1., -1., -1., 1.])
        UsdGeom.Xformable(camera).MakeMatrixXform().Set(Gf.Matrix4d(matrix.T.tolist()))
        focal = float(camera.GetFocalLengthAttr().Get())
        K = np.array([[640*focal/camera.GetHorizontalApertureAttr().Get(), 0, 320],
                      [0, 1100*focal/camera.GetVerticalApertureAttr().Get(), 550], [0, 0, 1]])
        product = rep.create.render_product(str(camera.GetPath()), (640, 1100))
        rgb, depth = [rep.AnnotatorRegistry.get_annotator(n) for n in ("rgb", "distance_to_image_plane")]
        rgb.attach(product)
        depth.attach(product)
        # Replicator authors camera/render-product relationships on first use.
        # Record the settled scene before changing any background condition.
        for _ in range(30):
            app.update()
        rep.orchestrator.step(rt_subframes=32, pause_timeline=True, delta_time=0.)
        fixed = invariant_record(stage, camera)
        write_json(args.output_dir / "invariants.json", fixed)
        others = [p for p in stage.GetPrimAtPath("/World/Fixtures").GetChildren() if p.GetName() not in ("A01", "hall")]
        for case in data["cases"]:
            condition = case["condition"]
            stage.RemovePrim("/World/BackgroundStudy")
            for prim in others:
                UsdGeom.Imageable(prim).GetVisibilityAttr().Set("inherited" if condition.startswith("original") else "invisible")
            add_background(stage, condition)
            for _ in range(30):
                app.update()
            for _ in range(2):
                rep.orchestrator.step(rt_subframes=32, pause_timeline=True, delta_time=0.)
            image = np.asarray(rgb.get_data()).copy()[..., :3]
            z = np.asarray(depth.get_data()).copy().squeeze()
            if image.shape != (1100, 640, 3) or image.dtype != np.uint8 or z.shape != (1100, 640) or image.std() < 1:
                raise ValueError(f"Invalid capture {case['id']}")
            current = invariant_record(stage, camera)
            if timeline.is_playing() or current != fixed:
                write_json(args.output_dir / "invariants_failed.json", current)
                raise ValueError("Workpiece, camera, robot, materials or lighting changed")
            Image.fromarray(image).save(args.output_dir / case["rgb"])
            np.savez_compressed(args.output_dir / case["truth"], optical_z_m=z, intrinsics=K,
                                world_from_camera_optical=pose)
            write_json(args.output_dir / case["capture"], {**case, "image_sha256": bench.digest(args.output_dir / case["rgb"]),
                       "truth_sha256": bench.digest(args.output_dir / case["truth"]), "invariants_equal": True,
                       "physics_executed": False, "invariants_sha256": bench.digest(args.output_dir / "invariants.json")})
            print(f"Captured fixed-camera background condition: {condition}", flush=True)
        rgb.detach(product)
        depth.detach(product)
        product.destroy()
        write_json(args.output_dir / "capture_status.json", {"completed": True, "invariants_equal": True})
        for stale in ("capture_failure.json", "invariants_failed.json"):
            (args.output_dir / stale).unlink(missing_ok=True)
    except BaseException as error:
        import traceback
        traceback.print_exc()
        write_json(args.output_dir / "capture_failure.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        app.close()


def fixture_mask(stage, case, K, shape, truth):
    """Ray-project the complete unchanged fixture, including table and clamps."""
    import numpy as np
    from pxr import Usd, UsdGeom
    from scipy.ndimage import binary_dilation
    yy, xx = np.indices(shape)
    rays = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)] @ np.linalg.inv(K).T
    inverse = np.linalg.inv(case["world_from_camera_optical"])
    cache = UsdGeom.XformCache()
    near = np.full(xx.size, np.inf)
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Fixtures/A01")):
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            vertices = np.array(mesh.GetPointsAttr().Get())
            indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
            faces, start = [], 0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                faces.append(indices[start:start+count])
                start += count
        elif prim.IsA(UsdGeom.Cube):
            size = UsdGeom.Cube(prim).GetSizeAttr().Get()/2
            vertices = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                                 [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])*size
            faces = [[0,4,6,2],[1,3,7,5],[0,1,5,4],[2,6,7,3],[0,2,3,1],[4,5,7,6]]
        else:
            continue
        vertices = bench.transform(vertices, inverse @ np.array(cache.GetLocalToWorldTransform(prim)).T)
        for face in faces:
            for i in range(1, len(face)-1):
                near = np.minimum(near, bench.triangle_depth(rays, vertices[[face[0], face[i], face[i+1]]]))
    near = near.reshape(shape)
    finite = np.isfinite(near) & np.isfinite(truth) & (truth > 0)
    difference = np.full(shape, np.inf)
    np.subtract(near, truth, out=difference, where=finite)
    visible = finite & (np.abs(difference) < np.maximum(.005, truth*.005))
    return binary_dilation(visible, structure=np.ones((9, 9)))


def prepare_controls(output, data):
    import numpy as np
    from PIL import Image
    from pxr import Usd
    base = data["cases"][0]
    original = np.asarray(Image.open(output / base["rgb"]))
    with np.load(output / base["truth"]) as stored:
        truth, K = stored["optical_z_m"], stored["intrinsics"]
    mask = fixture_mask(Usd.Stage.Open(data["scene"]["usd"]), base, K, truth.shape, truth)
    np.savez_compressed(output / "foreground_control.npz", protected_mask=mask)
    Image.fromarray((mask*255).astype(np.uint8)).save(output / "protected_fixture_mask.png")
    cases = [c for c in data["cases"] if c["group"] == "rendered"]
    for case in cases[1:]:
        image = np.array(Image.open(output / case["rgb"]))
        delta = np.abs(image.astype(float)-original)
        case["rendered_foreground_rgb_mae_255"] = float(delta[mask].mean())
        case["rendered_foreground_rgb_max_255"] = float(delta[mask].max())
        image[mask] = original[mask]
        name = case["id"] + "_locked"
        target = f"images/{name}.png"
        Image.fromarray(image).save(output / target)
        data["cases"].append({**case, "id": name, "group": "foreground_locked", "rgb": target,
                              "image_sha256": bench.digest(output / target), "foreground_rgb_max_delta_255": 0})
    # An exact input repeat separates model nondeterminism from recapture noise.
    name = "identical_input_repeat"
    Image.fromarray(original).save(output / f"images/{name}.png")
    data["cases"].append({**base, "id": name, "condition": "identical_input_repeat", "group": "repeat_control",
                          "rgb": f"images/{name}.png", "image_sha256": bench.digest(output / f"images/{name}.png")})
    data["protected_fixture_pixels"] = int(mask.sum())
    write_json(output / "plan.json", data)


def evaluate(output, data):
    import cv2
    import numpy as np
    from PIL import Image
    from pxr import Usd
    from scipy.ndimage import binary_erosion
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quadweld-background-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    stage = Usd.Stage.Open(data["scene"]["usd"])
    rows, items = [], []
    original_image = np.array(Image.open(output / data["cases"][0]["rgb"]))
    protected = np.load(output / "foreground_control.npz")["protected_mask"]
    mask = None
    for case in data["cases"]:
        capture = json.loads((output / case["capture"]).read_text())
        if bench.digest(output / case["truth"]) != capture["truth_sha256"]:
            raise ValueError("Ground truth changed")
        expected = capture["image_sha256"] if case["group"] == "rendered" else case["image_sha256"]
        if bench.digest(output / case["rgb"]) != expected:
            raise ValueError("RGB changed")
        with np.load(output / "predictions" / f"{case['id']}_depth.npz") as stored:
            if str(stored["image_sha256"]) != expected or not bool(stored["is_metric"]):
                raise ValueError("Prediction hash/units mismatch")
            pred, predicted_K, processed = stored["depth_m"], stored["intrinsics"], stored["processed_rgb"]
        with np.load(output / case["truth"]) as stored:
            full_truth, K = stored["optical_z_m"], stored["intrinsics"].copy()
            if not np.array_equal(stored["world_from_camera_optical"], case["world_from_camera_optical"]):
                raise ValueError("Camera pose changed")
        sy, sx = pred.shape[0]/full_truth.shape[0], pred.shape[1]/full_truth.shape[1]
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] = (K[0, 2]+.5)*sx-.5
        K[1, 2] = (K[1, 2]+.5)*sy-.5
        yy, xx = np.indices(pred.shape, dtype=np.float32)
        true = cv2.remap(full_truth, (xx+.5)/sx-.5, (yy+.5)/sy-.5, cv2.INTER_NEAREST)
        if mask is None:
            cpu, labels = bench.workpiece_faces(stage, case, K, pred.shape)
            finite = np.isfinite(cpu) & np.isfinite(true) & (true > 0)
            difference = np.full(true.shape, np.inf)
            np.subtract(cpu, true, out=difference, where=finite)
            visible = finite & (np.abs(difference) < np.maximum(.005, true*.005))
            mask = np.zeros(pred.shape, dtype=bool)
            for face in np.unique(labels[labels > 0]):
                mask |= binary_erosion((labels == face) & visible, structure=np.ones((7, 7)))
            original_truth, original_K, original_pred, original_processed = true.copy(), K.copy(), pred.copy(), processed.copy()
            if mask.sum() < 100:
                raise ValueError("Insufficient target pixels")
        truth_delta = float(np.max(np.abs(true[mask]-original_truth[mask])))
        if truth_delta > 1e-5 or not np.array_equal(K, original_K):
            raise ValueError("Target depth/visibility or intrinsics changed")
        if not np.isfinite(pred[mask]).all() or not (pred[mask] > 0).all():
            raise ValueError("Invalid prediction")
        image = np.array(Image.open(output / case["rgb"]))
        target_rgb_delta = int(np.max(np.abs(image.astype(int)-original_image.astype(int))[protected]))
        processed_delta = int(np.max(np.abs(processed.astype(int)-original_processed.astype(int))[mask]))
        if case["group"] in ("foreground_locked", "repeat_control") and (target_rgb_delta or processed_delta):
            raise ValueError("Locked target RGB changed")
        score = bench.measures(pred[mask].astype(float), original_truth[mask].astype(float))
        row = {"sample": case["id"], "condition": case["condition"], "group": case["group"],
               "replicate": case.get("replicate", 1), **score,
               "predicted_focal_ratio": float(predicted_K[0, 0]+predicted_K[1, 1])/float(K[0, 0]+K[1, 1]),
               "foreground_rgb_max_delta_255": target_rgb_delta, "processed_workpiece_rgb_max_delta_255": processed_delta,
               "target_truth_max_delta_m": truth_delta, "mae_change_from_original_m": score["mae_m"]-rows[0]["mae_m"] if rows else 0.,
               "predicted_depth_max_change_m": float(np.max(np.abs(pred[mask]-original_pred[mask]))) }
        rows.append(row)
        items.append((case, image, pred))
        np.savez_compressed(output / "evaluation" / f"{case['id']}.npz", raw_depth_m=pred,
                            true_depth_m=original_truth, interior_mask=mask)
        print(f"{case['id']}: MAE={score['mae_m']:.6f} m, AbsRel={score['abs_rel_percent']:.2f}%, target RGB delta={target_rgb_delta}", flush=True)
    # All controlled inputs share the first original fixture pixels, including
    # subsequent original-background captures. Replicates measure render noise,
    # not independent workpieces or viewpoints.
    aggregates = []
    for group in ("rendered", "foreground_locked"):
        for condition in CONDITIONS:
            selected_rows = [r for i, r in enumerate(rows) if r["condition"] == condition
                             and (r["group"] == group or (i == 0 and group == "foreground_locked"))]
            values = np.array([r["mae_m"] for r in selected_rows])
            aggregates.append({"group": group, "condition": condition, "n": len(values),
                               "mae_mean_m": float(values.mean()),
                               "mae_std_m": float(values.std(ddof=1)) if len(values) > 1 else None,
                               "mae_min_m": float(values.min()), "mae_max_m": float(values.max()),
                               "predicted_mean_m": float(np.mean([r["predicted_mean_m"] for r in selected_rows])),
                               "abs_rel_mean_percent": float(np.mean([r["abs_rel_percent"] for r in selected_rows])),
                               "focal_ratio_mean": float(np.mean([r["predicted_focal_ratio"] for r in selected_rows]))})
    write_json(output / "summary.json", {"completed": True, "scope": data["scope"], "evaluation_pixels": int(mask.sum()),
               "model": json.loads((output / "model.json").read_text()), "cases": rows,
               "aggregates": aggregates, "render_repeats_per_condition": data.get("render_repeats_per_condition", 1),
               "interpretation": "One workpiece and viewpoint. Locked composites isolate RGB context, not physically rerendered surfaces. "
                                 "Object dimensions and true calibration were never supplied to DA3. No scale/offset correction."})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    selected = [0] + [next(i for i, row in enumerate(rows) if row["condition"] == condition
                          and row["group"] == "foreground_locked") for condition in list(CONDITIONS)[1:]]
    fig, axes = plt.subplots(2, len(selected), figsize=(18, 9), constrained_layout=True)
    error_max = max(float(np.percentile(np.abs(items[i][2][mask]-original_truth[mask]), 99)) for i in selected)
    for col, i in enumerate(selected):
        case, image, pred = items[i]
        axes[0, col].imshow(image)
        axes[0, col].set_title(CONDITIONS[case["condition"]], fontsize=10, wrap=True)
        err = np.where(mask, np.abs(pred-original_truth), np.nan)
        im = axes[1, col].imshow(err, vmin=0, vmax=error_max, cmap="magma")
        axes[1, col].set_title(f"MAE {rows[i]['mae_m']:.3f} m\nAbsRel {rows[i]['abs_rel_percent']:.1f}%", fontsize=10)
        for axis in axes[:, col]:
            axis.axis("off")
    fig.colorbar(im, ax=axes[1].tolist(), shrink=.7, label="绝对光轴深度误差 (m)，统一色标")
    example = "第 1 轮示例｜" if data.get("render_repeats_per_condition", 1) > 1 else ""
    fig.suptitle(f"{example}A01 固定相机与工件｜工位像素完全相同｜DA3 仅输入 RGB", fontsize=14)
    fig.savefig(output / "comparison.png", dpi=145)
    plt.close(fig)
    figure, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    names = ["原始", "移除其他工位", "灰平面", "棋盘格平面", "立方体", "恢复原始"]
    raw = [a for a in aggregates if a["group"] == "rendered"]
    locked = [a for a in aggregates if a["group"] == "foreground_locked"]
    x = np.arange(6)
    ax.bar(x-.18, [r["mae_mean_m"] for r in raw], .36, yerr=[r["mae_std_m"] or 0 for r in raw], capsize=3, label="直接渲染")
    ax.bar(x+.18, [r["mae_mean_m"] for r in locked], .36, yerr=[r["mae_std_m"] or 0 for r in locked], capsize=3, label="锁定原始工位 RGB")
    ax.set_xticks(x, names)
    ax.set_ylabel("工件内部原始深度 MAE (m)")
    ax.set_title("背景变化与工位像素变化的对照" + ("（误差线为重拍标准差）" if raw[0]["n"] > 1 else "（单轮采集，尚无重复统计）"))
    ax.legend()
    figure.savefig(output / "errors.png", dpi=150)
    plt.close(figure)
    lines = ["# A01 背景与参照物控制实验", "", "固定 A01 工件、相机位姿、640×1100 分辨率、HJ-90 内参、全部灯光和机器人初始状态。仅在内存 session layer 修改背景；源 USD 不写回。", "",
             "使用此前 A01_d080 的同一相机视角，目标距离 0.8 m。灰平面和棋盘格平面的几何、材质粗糙度相同，只有颜色图案不同。立方体边长 0.3 m，棋盘格边长 0.1 m；这些尺寸没有告诉 DA3，不构成显式尺度标定。", "",
             "每种条件保留直接渲染图；另外将第一张原始 A01 整个工位（板件、工作台和夹具）像素恢复，留 4 个原图像素的保护边缘。后者是背景替换对照图，用于隔离阴影、反射及渲染噪声。锁定图工位 RGB 和评价区预处理 RGB 都逐像素验证完全相同。", "",
             f"全部条件评价同一组 {int(mask.sum())} 个板面内部像素（每面腐蚀 3 个处理像素，排除夹具与遮挡）。渲染真值光轴深度与真实内参均相同；DA3 仅输入 RGB，每帧重设随机种子 0，独立推理。未进行尺度、偏移校正，未运行 SAM3 或焊缝拟合。", "",
             f"每个背景重拍 {data.get('render_repeats_per_condition', 1)} 次，每轮末尾恢复原始背景。重复采集评价渲染微小像素波动，不能当成独立工件样本。另用完全相同的原始图片副本验证推理重复性。", "",
             "| 条件 | 直接渲染 MAE 均值 ± 标准差 m | 锁定工位像素 MAE 均值 ± 标准差 m | 锁定图 MAE 范围 m |",
             "| --- | ---: | ---: | ---: |"]
    for a, b in zip(raw, locked):
        def cell(item):
            return (f"{item['mae_mean_m']:.4f} ± {item['mae_std_m']:.4f}" if item["mae_std_m"] is not None
                    else f"{item['mae_mean_m']:.4f}（单次）")
        lines.append(f"| {CONDITIONS[a['condition']]} | {cell(a)} | {cell(b)} | {b['mae_min_m']:.4f}–{b['mae_max_m']:.4f} |")
    repeat = next(r for r in rows if r["group"] == "repeat_control")
    lines += ["", f"完全相同图片重复推理：评价区深度最大差值 {repeat['predicted_depth_max_change_m']:.9f} m。", "",
             "| 条件 | 输入 | 真值均深 m | DA3 均深 m | MAE m | AbsRel | 估计焦距/真值 | 工位 RGB 最大变化/255 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        label = CONDITIONS.get(row["condition"], "完全相同输入重复推理")
        group = {"rendered": "直接渲染", "foreground_locked": "锁定工位像素", "repeat_control": "原始输入副本"}[row["group"]]
        lines.append(f"| {label} r{row['replicate']} | {group} | {row['true_mean_m']:.4f} | {row['predicted_mean_m']:.4f} | {row['mae_m']:.4f} | "
                     f"{row['abs_rel_percent']:.2f}% | {row['predicted_focal_ratio']:.3f} | {row['foreground_rgb_max_delta_255']} |")
    lines += ["", "本实验只支持这一工件和固定视角下的背景敏感性分析，不能概括全部工位，也不能证明加入任意参照物都能提高绝对深度精度。", "",
              "[输入与误差对比](comparison.png) · [直接渲染与锁定像素比较](errors.png) · [完整数据](summary.json) · [CSV](summary.csv)", "",
              "## 复跑", "", "```bash", ".venv/bin/python scripts/benchmark_da3_background.py --gpu 0",
              f".venv/bin/python scripts/benchmark_da3_background.py --evaluate-only --output-dir {output.relative_to(ROOT)}", "```", ""]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3, help="Render repeats for each background (default: 3)")
    parser.add_argument("--wait-for-gpu", action="store_true", help="Wait until at least 12000 MiB GPU memory is free")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "thirdparty/depth-anything-3/weights/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--isaac-python", type=Path)
    parser.add_argument("--da3-python", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--_capture-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    stamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M%S")
    output = args.output_dir = (args.output_dir or ROOT / "outputs/benchmarks" / f"da3_background_{stamp}").resolve()
    if args._capture_worker:
        capture(args)
        return
    if args.evaluate_only:
        evaluate(output, json.loads((output / "plan.json").read_text()))
        return
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Use an empty output directory, or --resume")
    for folder in ("images", "truth", "captures", "predictions", "evaluation"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    data = json.loads((output / "plan.json").read_text()) if args.resume else plan(output, args.repeats)
    if data["scene_sha256"] != bench.digest(data["scene"]["usd"]) or data["base_usd_sha256"] != bench.digest(ROOT / "assets/scenes/industrial-welding/industrial_welding.usdc"):
        raise ValueError("Scene changed")
    while True:
        free_mib = int(subprocess.check_output(["nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.free",
                                               "--format=csv,noheader,nounits"], text=True).strip())
        if free_mib >= 12000:
            break
        write_json(output / "status.json", {"completed": False, "stage": "waiting_for_GPU",
                   "free_gpu_memory_mib": free_mib, "required_free_memory_mib": 12000,
                   "checked_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
                   "background_pid": os.getpid() if args.wait_for_gpu else None})
        if not args.wait_for_gpu:
            raise RuntimeError(f"GPU {args.gpu} has only {free_mib} MiB free; need 12000 MiB for capture and DA3. Resume when available.")
        print(f"Waiting for GPU {args.gpu}: {free_mib} MiB free; require 12000 MiB", flush=True)
        time.sleep(20)
    record = bench.model_record(args.model_dir)
    if (output / "model.json").exists() and json.loads((output / "model.json").read_text()) != record:
        raise ValueError("Model or predictor changed")
    write_json(output / "model.json", record)
    isaac, error = find_python("isaac", args.isaac_python)
    if error:
        raise RuntimeError(error)
    da3, error = find_python("da3", args.da3_python)
    if error:
        raise RuntimeError(error)
    write_json(output / "status.json", {"completed": False, "stage": "capture"})
    status = output / "capture_status.json"
    if not status.exists() or not json.loads(status.read_text()).get("completed"):
        run_process([isaac, str(Path(__file__).resolve()), "--_capture-worker", "--output-dir", str(output), "--gpu", str(args.gpu)], output / "capture.log")
    if not status.exists() or not json.loads(status.read_text()).get("completed"):
        raise RuntimeError("Capture incomplete")
    if not any(c["group"] == "foreground_locked" for c in data["cases"]):
        prepare_controls(output, data)
    import numpy as np
    images = []
    for case in data["cases"]:
        cache = output / "predictions" / f"{case['id']}_depth.npz"
        valid = False
        if cache.exists():
            with np.load(cache) as stored:
                valid = str(stored["image_sha256"]) == bench.digest(output / case["rgb"]) and bool(stored["is_metric"])
        if not valid:
            images.append(str(output / case["rgb"]))
    write_json(output / "status.json", {"completed": False, "stage": "DA3_inference"})
    if images:
        run_process([da3, str(ROOT / "scripts/predict_depth.py"), *images, "--gpu", str(args.gpu),
                     "--model-dir", str(args.model_dir.resolve()), "--output-dir", str(output / "predictions")], output / "da3.log")
    write_json(output / "status.json", {"completed": False, "stage": "evaluation"})
    evaluate(output, data)
    write_json(output / "status.json", {"completed": True, "stage": "complete", "report": "README.md"})
    print(f"Background study completed: {output}", flush=True)


if __name__ == "__main__":
    main()
