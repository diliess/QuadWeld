"""Recognize welds and plate-junction candidates from whole RGB images."""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def semantic_worker(images, output, gpu, threshold):
    """Fixed category prompts, identical for every image; no human coordinates."""
    if gpu == "auto":
        rows = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
        gpu, free_mib = max((tuple(map(int, row.split(","))) for row in rows.splitlines()),
                            key=lambda item: item[1])
        print(f"SAM3 using physical GPU {gpu}, {free_mib} MiB free", flush=True)
        if free_mib < 8000:
            raise RuntimeError("SAM3 needs an available GPU with at least 8000 MiB free")
        gpu = str(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from sam3.model_builder import build_sam3_predictor

    with contextlib.redirect_stdout(io.StringIO()):
        predictor = build_sam3_predictor(
            checkpoint_path=str(Path("thirdparty/sam3/assets/model/sam3.1_multiplex.pt").resolve()),
            version="sam3.1", use_fa3=False, use_rope_real=False, compile=False, async_loading_frames=False)
    predictor.model.score_threshold_detection = min(0.4, threshold)
    predictor.model.image_only_det_thresh = threshold
    arrays = {"metadata_json": np.array(json.dumps({
        "threshold": threshold, "prompts": {"plate": ["metal plate"], "weld": ["weld", "weld bead"]},
        "precision": "bfloat16 autocast", "model": "SAM3.1 multiplex"}))}
    for index, image in enumerate(images):
        started = time.monotonic()
        with Image.open(image) as source:
            height, width = source.height, source.width
        scale = min(1, 504 / max(height, width))
        target_hw = (round(height * scale), round(width * scale))
        arrays[f"{index}_image_sha256"] = np.array(hashlib.sha256(image.read_bytes()).hexdigest())
        for category, prompts in (("plate", ("metal plate",)), ("weld", ("weld", "weld bead"))):
            masks, scores = [], []
            for prompt in prompts:
                session = predictor.handle_request({
                    "type": "start_session", "resource_path": str(image.resolve())})["session_id"]
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        result = predictor.handle_request({
                            "type": "add_prompt", "session_id": session, "frame_index": 0,
                            "text": prompt})["outputs"]
                    for mask, score in zip(result["out_binary_masks"], result["out_probs"]):
                        mask = cv2.resize(np.asarray(mask, dtype=np.uint8), target_hw[::-1],
                                          interpolation=cv2.INTER_NEAREST).astype(bool)
                        if any(np.logical_and(mask, previous).sum() /
                               max(1, np.logical_or(mask, previous).sum()) > 0.7 for previous in masks):
                            continue
                        masks.append(mask)
                        scores.append(float(score))
                finally:
                    predictor.handle_request({"type": "close_session", "session_id": session})
            arrays[f"{index}_{category}_masks"] = (np.stack(masks) if masks else
                                                         np.zeros((0, *target_hw), dtype=bool))
            arrays[f"{index}_{category}_scores"] = np.asarray(scores, dtype=np.float32)
        print(json.dumps({"image": str(image), "automatic_semantics_seconds": round(time.monotonic()-started, 2),
                          "plate_instances": len(arrays[f"{index}_plate_scores"]),
                          "weld_instances": len(arrays[f"{index}_weld_scores"])}), flush=True)
    np.savez_compressed(output, **arrays)


def input_images(paths):
    images = []
    for path in paths:
        if path.is_dir():
            images.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}))
        elif path.is_file():
            images.append(path)
        else:
            raise ValueError(f"Input not found: {path}")
    images = list(dict.fromkeys(p.resolve() for p in images))
    if not images:
        raise ValueError("No input images")
    if len({p.stem for p in images}) != len(images):
        raise ValueError("Input images must have distinct stems for depth inference")
    return images


def run_worker(command):
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"Model worker failed ({command[1]}):\n" + result.stdout[-12000:])
    for line in result.stdout.splitlines():
        if line.startswith('{"image":') or line.startswith("SAM3 using") or line.startswith("Using GPU"):
            print(line, flush=True)


def to_original(xy, scale):
    import numpy as np
    return (np.asarray(xy) + .5) / scale - .5


def serialize_result(index, image, data, result, band_width):
    import cv2
    import numpy as np

    depth, rgb = data["depth_m"], data["processed_rgb"]
    original_hw = data["original_hw"]
    scale = np.array([depth.shape[1] / original_hw[1], depth.shape[0] / original_hw[0]])
    prefix = f"image_{index:03d}"
    class_map = np.zeros(tuple(original_hw), dtype=np.uint8)
    arrays = {prefix + "_depth_m": depth, prefix + "_processed_rgb": rgb,
              prefix + "_intrinsics": data["intrinsics"], prefix + "_plane_labels": result["labels"],
              prefix + "_plate_semantic_mask": result["semantic_plate_mask"]}
    arrays[prefix + "_depth_quality_mask"] = result["depth_quality_mask"]
    if "confidence" in data:
        arrays[prefix + "_depth_confidence"] = data["confidence"]
    overlay = rgb.copy()
    joints, paths = [], []
    joint_fallback_count = 0
    for candidate in result["joints"]:
        number = len(joints)
        xy = to_original(candidate["xy"], scale)
        joint_prefix = f"{prefix}_joint_{number}"
        arrays[joint_prefix + "_centerline_xy"] = xy.astype(np.float32)
        arrays[joint_prefix + "_xyz_camera_estimated_m"] = candidate["xyz_camera_estimated_m"]
        arrays[joint_prefix + "_observed_xy_original"] = to_original(candidate["observed_xy"], scale)
        for field in ("relative_depth_disagreement", "normal_snap_distance_processed_px"):
            arrays[joint_prefix + "_" + field] = candidate[field]
        joint_report = {key: candidate[key] for key in (
            "planes", "acute_angle_deg", "rank", "median_relative_depth_disagreement",
            "concavity_signed_distances_estimated_m", "plate_semantic_support", "overlaps_weld")}
        joint_report.update(kind="plate_joint_candidate", weld_state="unknown", segments=[],
                            centerline_key=joint_prefix + "_centerline_xy",
                            xyz_key=joint_prefix + "_xyz_camera_estimated_m",
                            origin_camera_estimated_m=candidate["origin_camera_estimated_m"].tolist(),
                            direction_camera=candidate["direction_camera"].tolist())
        for segment_index, segment in enumerate(candidate["segments"]):
            first, last = segment["start_sample"], segment["stop_sample"]
            segment_prefix = f"{joint_prefix}_segment_{segment_index}"
            arrays[segment_prefix + "_xy_original"] = xy[first:last].astype(np.float32)
            arrays[segment_prefix + "_xyz_camera_estimated_m"] = candidate["xyz_camera_estimated_m"][first:last]
            arrays[segment_prefix + "_geometry_supported"] = segment["geometry_supported"]
            entry = {key: value for key, value in segment.items() if key != "geometry_supported"}
            entry.update(xy_key=segment_prefix + "_xy_original", xyz_key=segment_prefix + "_xyz_camera_estimated_m",
                         geometry_key=segment_prefix + "_geometry_supported", points=last-first,
                         geometry_support_fraction=float(segment["geometry_supported"].mean()))
            joint_report["segments"].append(entry)
            if segment["branch"] == "plate_intersection" and last-first >= 2:
                # Display only fallback intervals. Covered intervals keep their
                # state and bead association in the report, without root paths.
                cv2.polylines(class_map, [np.rint(xy[first:last]).astype(np.int32)], False, 1, band_width)
                color = (255, 175, 0) if segment["state"] == "no_weld_detected" else (255, 100, 0)
                cv2.polylines(overlay, [np.rint(candidate["xy"][first:last]).astype(np.int32)], False, color, 2)
                paths.append({**entry, "id": f"joint_{number}_segment_{segment_index}",
                              "kind": "plate_intersection", "joint_index": number,
                              "geometry_source": "analytic_plane_intersection", "closed": False})
                joint_fallback_count += 1
        joints.append(joint_report)
    welds = []
    for number, region in enumerate(result["welds"]):
        mask = cv2.resize(region["mask"].astype(np.uint8), tuple(original_hw[::-1]), interpolation=cv2.INTER_NEAREST).astype(bool)
        class_map[mask] = 2
        overlay[region["mask"]] = (overlay[region["mask"]] * .5 + np.array([30, 235, 65]) * .5).astype(np.uint8)
        centerline = to_original(region["centerline"], scale)
        centerline_key = f"{prefix}_weld_{number}_centerline_xy"
        arrays[centerline_key] = centerline.astype(np.float32)
        boundary_keys = []
        for part, boundary in enumerate(region["boundaries"]):
            cv2.polylines(overlay, [boundary.astype(np.int32)], True, (30, 255, 60), 1)
            key = f"{prefix}_weld_{number}_boundary_{part}_xy"
            arrays[key] = to_original(boundary, scale).astype(np.float32)
            boundary_keys.append(key)
        welds.append({"kind": "welded_region_candidate", "sam_detection_score": region["score"],
                      "geometric_contact_support": region["contact_support"], "centerline_key": centerline_key,
                      "boundary_keys": boundary_keys, "elongation": region["elongation"]})
        toes = region.get("toe_paths", {"status": "not_extracted", "segments": []})
        welds[-1].update(toe_status=toes["status"], toe_segments=[],
                         toe_method=toes.get("method"),
                         toe_search_radius_processed_px=toes.get("search_radius_processed_px"))
        for segment_index, segment in enumerate(toes["segments"]):
            segment_report = {"closed": False, "sides": {}}
            for side_name, color in (("side_a", (0, 220, 255)), ("side_b", (255, 50, 210))):
                side = segment[side_name]
                side_prefix = f"{prefix}_weld_{number}_toe_{segment_index}_{side_name}"
                xy_key = side_prefix + "_xy_original"
                arrays[xy_key] = to_original(side["xy"], scale).astype(np.float32)
                arrays[side_prefix + "_seed_xy_original"] = to_original(side["seed_xy"], scale).astype(np.float32)
                for field in ("xyz_camera_estimated_m", "plane_xyz_camera_estimated_m", "plane_projection_valid",
                              "depth_supported", "geometry_supported", "normal_contrast_deg", "plane_distance_estimated_m",
                              "offset_from_seed_processed_px", "surface_xyz_camera_estimated_m", "surface_valid",
                              "surface_kind", "surface_fit_rmse_estimated_m", "surface_validation_error_estimated_m",
                              "depth_quality_valid", "surface_continuity_valid"):
                    arrays[side_prefix + "_" + field] = np.asarray(side[field])
                for field in ("offsets", "normal_score", "rgb_score", "plane_score"):
                    arrays[side_prefix + "_profile_" + field] = side["cross_section"][field]
                cv2.polylines(overlay, [np.rint(side["xy"]).astype(np.int32)], False, color, 1)
                plane = side["plane"]
                segment_report["sides"][side_name] = {
                    "array_prefix": side_prefix, "xy_key": xy_key,
                    "seed_xy_key": side_prefix + "_seed_xy_original",
                    "xyz_key": side_prefix + "_xyz_camera_estimated_m",
                    "plane_xyz_key": side_prefix + "_plane_xyz_camera_estimated_m",
                    "surface_xyz_key": side_prefix + "_surface_xyz_camera_estimated_m",
                    "points": len(side["xy"]), "depth_support_fraction": float(side["depth_supported"].mean()),
                    "plane_projection_valid_fraction": float(side["plane_projection_valid"].mean()),
                    "geometry_support_fraction": float(side["geometry_supported"].mean()),
                    "surface_valid_fraction": float(side["surface_valid"].mean()),
                    "surface_kind_counts": {str(kind): int(np.sum(side["surface_kind"] == kind)) for kind in range(4)},
                    "median_shift_from_sam_processed_px": float(np.median(np.abs(side["offset_from_seed_processed_px"]))),
                    "profile_selected_offset": side["cross_section"]["selected_offset"],
                    "profile_seed_offset": side["cross_section"]["seed_offset"],
                    "mother_plane": ({**plane, "equation": plane["equation"].tolist()} if plane is not None else None),
                }
                paths.append({"id": f"weld_{number}_segment_{segment_index}_{side_name}", "kind": "weld_toe",
                              "branch": "existing_bead_edges", "state": "weld_detected", "weld_state": "welded_candidate",
                              "weld_index": number, "segment_index": segment_index, "side": side_name,
                              "xy_key": xy_key, "xyz_key": side_prefix + "_surface_xyz_camera_estimated_m",
                              "geometry_key": side_prefix + "_geometry_supported", "points": len(side["xy"]),
                              "surface_kind_key": side_prefix + "_surface_kind", "closed": False,
                              "geometry_source": "mother_plane_or_local_surface",
                              "geometry_support_fraction": float(side["geometry_supported"].mean())})
            arrays[f"{prefix}_weld_{number}_toe_{segment_index}_width_processed_px"] = segment["width_processed_px"]
            welds[-1]["toe_segments"].append(segment_report)
    arrays[prefix + "_class_map_original"] = class_map
    for path in paths:
        support = arrays[path["geometry_key"]]
        transitions = np.diff(np.r_[False, support, False].astype(int))
        path["geometry_runs"] = [{"start_sample": int(a), "stop_sample": int(b)}
                                 for a, b in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)) if b-a >= 2]
    planes = [{"id": i, "equation_camera": plate.equation.tolist(), "support_pixels": int(plate.mask.sum()),
               "fit_rmse_estimated_m": plate.residual, "semantic_instance": plate.semantic_instance,
               "sam_mask_index": (result["plate_instance_indices"][plate.semantic_instance]
                                  if plate.semantic_instance is not None else None),
               "sampling_source": plate.sampling_source} for i, plate in enumerate(result["plates"])]
    report = {"image": str(image.relative_to(ROOT)) if image.is_relative_to(ROOT) else str(image),
              "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "array_prefix": prefix,
              "original_hw": original_hw.tolist(), "processed_hw": list(depth.shape),
              "automatic_plate_instance_count": result.get("plate_instance_count", 0),
              "selected_sam_plate_indices": result["plate_instance_indices"],
              "status": "candidates_found" if joints or welds else "no_candidate",
              "plate_sampling_mode": result["plate_sampling_mode"],
              "depth_confidence_threshold": result["confidence_threshold"],
              "planes": planes, "joints": joints, "welds": welds, "paths": paths,
              "plate_joint_candidates": len(joints), "joint_candidates": joint_fallback_count,
              "welded_region_candidates": len(welds)}
    return arrays, report, overlay


def save_overview(panels, reports, path):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quadweld-da3-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = min(3, len(panels))
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, panel, report in zip(axes.flat, panels, reports):
        axis.imshow(panel)
        axis.set_title(f"{Path(report['image']).name}\n{report['welded_region_candidates']} weld regions | "
                       f"{report['joint_candidates']} joint candidates", fontsize=10)
    figure.suptitle("Automatic inference | Green: bead | Cyan / magenta: two toe candidates | Orange: joint (state unknown)", fontsize=12)
    figure.tight_layout(rect=(0, .015, 1, .97))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_toe_outputs(arrays, reports, directory):
    """One toe-only figure and one table for all detected bead instances."""
    import numpy as np
    import matplotlib.pyplot as plt

    data_directory = directory / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    entries = [(report, index, weld) for report in reports for index, weld in enumerate(report["welds"])
               if weld["toe_segments"]]
    fields = ["image", "weld", "segment", "side", "sample", "u_original_px", "v_original_px",
              "x_camera_est_m", "y_camera_est_m", "z_camera_est_m",
              "plane_x_camera_est_m", "plane_y_camera_est_m", "plane_z_camera_est_m",
              "depth_supported", "plane_projection_valid", "geometry_supported", "normal_contrast_deg",
              "surface_x_camera_est_m", "surface_y_camera_est_m", "surface_z_camera_est_m",
              "surface_valid", "surface_kind", "depth_quality_valid", "surface_continuity_valid"]
    with (data_directory / "weld_toes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report, weld_index, weld in entries:
            for segment_index, segment in enumerate(weld["toe_segments"]):
                for side_name, side in segment["sides"].items():
                    prefix = side["array_prefix"]
                    for sample_index, (xy, xyz, plane_xyz) in enumerate(zip(
                            arrays[side["xy_key"]], arrays[side["xyz_key"]], arrays[side["plane_xyz_key"]])):
                        values = [report["image"], weld_index, segment_index, side_name, sample_index,
                            *map(float, xy), *map(float, xyz), *map(float, plane_xyz),
                            bool(arrays[prefix + "_depth_supported"][sample_index]),
                            bool(arrays[prefix + "_plane_projection_valid"][sample_index]),
                            bool(arrays[prefix + "_geometry_supported"][sample_index]),
                            float(arrays[prefix + "_normal_contrast_deg"][sample_index]),
                            *map(float, arrays[side["surface_xyz_key"]][sample_index]),
                            bool(arrays[prefix + "_surface_valid"][sample_index]),
                            int(arrays[prefix + "_surface_kind"][sample_index]),
                            bool(arrays[prefix + "_depth_quality_valid"][sample_index]),
                            bool(arrays[prefix + "_surface_continuity_valid"][sample_index])]
                        writer.writerow(dict(zip(fields, values)))
    if not entries:
        figure, axis = plt.subplots(figsize=(10, 3))
        axis.text(.5, .5, "No paired weld-toe candidates in this run", ha="center", va="center")
        axis.axis("off")
        figure.savefig(directory / "weld_toes_overview.jpg", dpi=150)
        plt.close(figure)
        return
    figure, axes = plt.subplots(len(entries), 3, figsize=(16, 3.8 * len(entries)), squeeze=False)
    for row, (report, weld_index, weld) in enumerate(entries):
        prefix = report["array_prefix"]
        rgb = arrays[prefix + "_processed_rgb"]
        scale = np.array([rgb.shape[1] / report["original_hw"][1], rgb.shape[0] / report["original_hw"][0]])
        boundaries = [(arrays[key] + .5) * scale - .5 for key in weld["boundary_keys"]]
        vertices = np.concatenate(boundaries)
        extent = np.ptp(vertices, axis=0)
        rotated = extent[1] > 1.4 * extent[0]

        def display_xy(xy):
            # Rotate tall beads for inspection only; exported coordinates stay
            # in the original image frame.
            return np.column_stack((xy[:, 1], rgb.shape[1]-1-xy[:, 0])) if rotated else xy

        boundaries = [display_xy(boundary) for boundary in boundaries]
        vertices = np.concatenate(boundaries)
        view = np.rot90(rgb) if rotated else rgb
        low, high = vertices.min(0), vertices.max(0)
        padding = max(12, min(high-low) * .8)
        for axis in axes[row, :2]:
            axis.imshow(view)
            axis.set_xlim(max(0, low[0]-padding), min(view.shape[1], high[0]+padding))
            axis.set_ylim(min(view.shape[0], high[1]+padding), max(0, low[1]-padding))
            axis.axis("off")
        for boundary in boundaries:
            axes[row, 0].plot(*np.vstack((boundary, boundary[0])).T, color="lime", lw=1.2)
        view_note = " | view rotated 90 deg" if rotated else ""
        axes[row, 0].set_title(f"{Path(report['image']).name} | bead {weld_index}{view_note}\nSAM bead seed (closed contour)", fontsize=10)
        fractions = []
        for segment in weld["toe_segments"]:
            for side_name, side in segment["sides"].items():
                color = "deepskyblue" if side_name == "side_a" else "deeppink"
                xy = display_xy((arrays[side["xy_key"]] + .5) * scale - .5)
                supported = arrays[side["array_prefix"] + "_geometry_supported"]
                axes[row, 1].plot(*xy.T, color=color, lw=1.6)
                axes[row, 1].scatter(xy[~supported, 0], xy[~supported, 1], s=5, c="yellow", zorder=4)
                fractions.append(f"{side_name[-1].upper()} {side['geometry_support_fraction']:.0%}")
        axes[row, 1].set_title("Two OPEN toe candidates | Cyan A / Magenta B\nModel checks: " + ", ".join(fractions) + " | Yellow: unconfirmed", fontsize=10)
        axis = axes[row, 2]
        for side_name, side in weld["toe_segments"][0]["sides"].items():
            side_prefix = side["array_prefix"]
            offsets = arrays[side_prefix + "_profile_offsets"]
            for field, color, label_name in (("normal_score", "darkorange", "DA3 normal change"),
                                              ("rgb_score", "gray", "RGB edge"),
                                              ("plane_score", "forestgreen", "DA3 plane transition")):
                axis.plot(offsets, arrays[side_prefix + "_profile_" + field], color=color,
                          lw=1.2, label=label_name if side_name == "side_a" else None)
            color = "deepskyblue" if side_name == "side_a" else "deeppink"
            axis.axvline(side["profile_seed_offset"], color="black", lw=.8, ls=":")
            axis.axvline(side["profile_selected_offset"], color=color, lw=1.6)
        axis.set_ylim(-.05, 1.15)
        axis.set_title("Representative transverse section\nDotted: SAM seed | Colored: fused side", fontsize=10)
        axis.set_xlabel("Transverse offset (processed pixels)")
        axis.set_ylabel("Normalized evidence (not probability)")
        axis.legend(fontsize=8, loc="upper right")
        axis.grid(alpha=.2)
    figure.suptitle("Existing bead: two weld-toe candidates from SAM + DA3 + RGB", fontsize=14)
    figure.text(.5, .012, "Checks: depth transition, host surface, confidence, 3D continuity. Fractions are not accuracy; camera geometry is uncalibrated.", ha="center", fontsize=10)
    figure.tight_layout(rect=(0, .035, 1, .97))
    figure.savefig(directory / "weld_toes_overview.jpg", dpi=160)
    plt.close(figure)


def save_paths(arrays, reports, directory):
    """One indexed table for both branches; every path is an independent candidate."""
    import numpy as np

    fields = ["image", "path_id", "kind", "state", "sample", "u_original_px", "v_original_px",
              "x_camera_est_m", "y_camera_est_m", "z_camera_est_m", "geometry_supported", "surface_kind", "geometry_run"]
    with (directory / "seam_paths.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for path in report["paths"]:
                xy, xyz = arrays[path["xy_key"]], arrays[path["xyz_key"]]
                valid = arrays[path["geometry_key"]]
                runs = np.full(len(xy), -1, dtype=int)
                for run_index, run in enumerate(path["geometry_runs"]):
                    runs[run["start_sample"]:run["stop_sample"]] = run_index
                kinds = arrays[path["surface_kind_key"]] if "surface_kind_key" in path else ["analytic_intersection"] * len(xy)
                for index, (uv, point) in enumerate(zip(xy, xyz)):
                    values = [report["image"], path["id"], path["kind"], path["state"], index,
                              *map(float, uv), *map(float, point), bool(valid[index]), str(kinds[index]), int(runs[index])]
                    writer.writerow(dict(zip(fields, values)))


def save_sam_outputs(arrays, reports, directory):
    """Render saved SAM candidates before any plate or bead geometry filtering."""
    import cv2
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_directory = directory / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    for category, label, prompt in (("plate", "Metal plate candidates", "metal plate"),
                                    ("weld", "Weld candidates", "weld / weld bead")):
        columns = min(3, len(reports))
        rows = (len(reports)+columns-1)//columns
        figure, axes = plt.subplots(rows, columns, figsize=(6*columns, 5.7*rows), squeeze=False)
        colors = plt.get_cmap("tab20")
        for axis in axes.flat:
            axis.axis("off")
        for axis, report in zip(axes.flat, reports):
            prefix = report["array_prefix"]
            masks = arrays[f"{prefix}_sam_{category}_masks"]
            scores = arrays[f"{prefix}_sam_{category}_scores"]
            if len(masks) != len(scores):
                raise ValueError("SAM mask/score counts differ")
            source = ROOT / report["image"]
            height, width = masks.shape[1:]
            with Image.open(source) as image:
                rgb = np.array(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR))
            overlay = rgb.copy()
            # Paint large masks first so smaller overlapping instances remain
            # visible. All instance contours and source scores are retained.
            for index in sorted(range(len(masks)), key=lambda k: masks[k].sum(), reverse=True):
                color = np.array(colors(index % 20)[:3])*255
                overlay[masks[index]] = (overlay[masks[index]]*.72+color*.28).astype(np.uint8)
            axis.imshow(overlay)
            occupied = np.zeros((height, width), dtype=np.uint8)
            for index, mask in enumerate(masks):
                color = colors(index % 20)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    xy = contour[:, 0]
                    if len(xy) >= 3:
                        axis.plot(*np.vstack((xy, xy[0])).T, color=color, lw=1.1)
                if mask.any():
                    distances = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
                    if (mask & ~occupied.astype(bool)).any():
                        distances[occupied.astype(bool)] = -1
                    distances[~mask] = -np.inf
                    y, x = np.unravel_index(distances.argmax(), distances.shape)
                    axis.text(x, y, str(index+1), color="white", fontsize=9, ha="center", va="center",
                              bbox={"facecolor": color, "edgecolor": "black", "alpha": .9, "pad": 1.5})
                    cv2.circle(occupied, (int(x), int(y)), 14, 1, -1)
            if not len(masks):
                axis.text(.04, .93, "NO MASK RETURNED", transform=axis.transAxes, va="top", color="red",
                          fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
            numbered = [f"{i+1}:{float(score):.3f}" for i, score in enumerate(scores)]
            score_lines = ["  ".join(numbered[i:i+5]) for i in range(0, len(numbered), 5)]
            axis.set_title(f"{Path(report['image']).name} | {len(masks)} SAM mask(s)\n" +
                           ("\n".join(score_lines) if score_lines else "No detection score"), fontsize=9)
            axis.set_xlim(-.5, width-.5)
            axis.set_ylim(height-.5, -.5)
        figure.suptitle(f"SAM3 only | {label} | Prompt: {prompt}\n"
                        "Saved text-prompt masks; instance IDs match detection scores", fontsize=14)
        figure.text(.5, .012, "All saved candidates before DA3 geometry filtering. Overlap paint order is for display; scores are not measured accuracy.",
                    ha="center", fontsize=10)
        figure.tight_layout(rect=(0, .035, 1, .96))
        figure.savefig(data_directory / f"sam_{category}_overview.jpg", dpi=150)
        plt.close(figure)


def save_result_index(reports, directory):
    """Keep one short reading guide beside the primary figures and path table."""
    welds = [weld for report in reports for weld in report["welds"]]
    paired = sum(bool(weld.get("toe_segments")) for weld in welds)
    paths = sum(len(report["paths"]) for report in reports)
    lines = [
        "# DA3 与 SAM3 焊缝实验结果", "",
        f"当前汇总：{len(reports)} 张图片，{len(welds)} 个焊道区域候选，"
        f"其中 {paired} 个有两侧边缘；共 {paths} 条候选路径。", "",
        "## 从这里查看", "",
        "1. [全图结果](auto_overview.jpg)：先确认工件上的候选位置。",
        "2. [焊道两侧边缘](weld_toes_overview.jpg)：放大检查边缘与黄色待确认点。",
        "3. [统一三维路径表](seam_paths.csv)：按图片、路径和连续有效区间读取坐标。", "",
        "绿色是焊道区域候选，青色和品红色是两侧边缘，橙色是焊接状态未确认的交接线。",
        "黄色点没有通过全部模型几何检查；图中比例不代表定位准确率。", "",
        "## 详细数据", "",
        "| 文件 | 内容 |", "| --- | --- |",
        "| [data/auto_report.json](data/auto_report.json) | 图片、区段状态、曲线数组索引和几何检查 |",
        "| [data/auto_results.npz](data/auto_results.npz) | 深度、掩码、内参和三维曲线数组 |",
        "| [data/weld_toes.csv](data/weld_toes.csv) | 焊趾原始深度、平面与局部表面坐标及标记 |", "",
        "`seam_paths.csv` 按 `image,path_id,geometry_run` 分组，排除 `geometry_run=-1`，"
        "按 `sample` 排序。不同组不要直接连接。坐标单位为米，位于估计相机坐标系。", "",
        "自动识别复跑会替换上述当前结果并更新本说明。单独深度推理和手工选区实验默认写入 `data/`。", "",
    ]
    if all((directory / "data" / f"sam_{category}_overview.jpg").is_file() for category in ("weld", "plate")):
        lines += ["## SAM3 单独分割结果", "",
                  "- [SAM3 焊道候选](data/sam_weld_overview.jpg)",
                  "- [SAM3 工件候选](data/sam_plate_overview.jpg)", "",
                  "展示本轮全部文字提示掩码及模型分数，尚未经过 DA3 几何筛选；工件候选可能包含桌面和背景。", ""]
    review = ROOT / "docs/weld_image_results.md"
    if review.is_file():
        link = os.path.relpath(review, directory).replace(os.sep, "/")
        lines += ["## 逐张效果检查", "",
                  f"[原始图片数量与逐张识别效果]({link})：包含原图、结果叠加和漏检案例。",
                  "这是人工看图整理的快照；更新识别结果后需要同步更新报告。", ""]
    if (directory / "archive.zip").is_file():
        lines += ["## 早期实验", "",
                  "[archive.zip](archive.zip) 保存三张单图深度实验及手工选区几何实验，文件保留原名。",
                  "压缩包内的旧报告保留当时的路径记录；需要复现时可解压到临时目录。", ""]
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")


def pipeline(images, args):
    import numpy as np
    from vision.weld_detector import detect

    started = time.monotonic()
    arrays, reports, panels = {}, [], []
    if args.model_cache_dir:
        args.model_cache_dir.mkdir(parents=True, exist_ok=True)
    context = (contextlib.nullcontext(str(args.model_cache_dir)) if args.model_cache_dir else
               tempfile.TemporaryDirectory(prefix="quadweld_auto_"))
    with context as temporary:
        temp_dir = Path(temporary)
        depth_dir = temp_dir / "depth"
        depth_dir.mkdir(exist_ok=True)
        missing, depth_paths = [], {}
        for image in images:
            cached = (args.reuse_depth_dir or depth_dir) / f"{image.stem}_depth.npz"
            valid_cache = False
            if cached and cached.is_file():
                with np.load(cached, allow_pickle=False) as data:
                    valid_cache = ("image_sha256" in data and str(data["image_sha256"]) ==
                                   hashlib.sha256(image.read_bytes()).hexdigest())
            if valid_cache:
                depth_paths[image] = cached
            else:
                missing.append(image)
                depth_paths[image] = depth_dir / f"{image.stem}_depth.npz"
        if missing:
            print(f"DA3: predicting {len(missing)} independent whole images", flush=True)
            run_worker([sys.executable, str(ROOT / "scripts/predict_depth.py"), *map(str, missing),
                        "--output-dir", str(depth_dir), "--gpu", args.gpu])
        semantic_path = args.reuse_semantics or temp_dir / "semantics.npz"
        semantics_valid = False
        if semantic_path.is_file():
            with np.load(semantic_path, allow_pickle=False) as cache:
                semantics_valid = ("metadata_json" in cache and json.loads(str(cache["metadata_json"]))["threshold"] == args.semantic_threshold
                                   and all(f"{index}_image_sha256" in cache and str(cache[f"{index}_image_sha256"]) ==
                                           hashlib.sha256(image.read_bytes()).hexdigest() for index, image in enumerate(images)))
        if not semantics_valid:
            if args.reuse_semantics:
                raise ValueError("Semantic cache does not match the input images or threshold")
            sam_python = args.sam_python or Path(sys.executable).resolve().parents[2] / "sam3/bin/python"
            if not sam_python.is_file():
                raise ValueError("SAM3 environment not found; specify --sam-python")
            print(f"SAM3: fixed category inference for {len(images)} images", flush=True)
            run_worker([str(sam_python), str(Path(__file__).resolve()), *map(str, images),
                        "--semantic-worker", str(semantic_path), "--gpu", args.gpu,
                        "--semantic-threshold", str(args.semantic_threshold)])
        with np.load(semantic_path, allow_pickle=False) as semantics:
            metadata = json.loads(str(semantics["metadata_json"]))
            if metadata["threshold"] != args.semantic_threshold:
                raise ValueError("Semantic cache threshold differs from this run")
            for index, image in enumerate(images):
                expected_hash = hashlib.sha256(image.read_bytes()).hexdigest()
                if str(semantics[f"{index}_image_sha256"]) != expected_hash:
                    raise ValueError("Semantic cache does not match input images and order")
                with np.load(depth_paths[image], allow_pickle=False) as data:
                    if not bool(data["is_metric"]):
                        raise ValueError("Expected metric DA3 output")
                    result = detect(data["depth_m"], data["intrinsics"], data["processed_rgb"],
                                    plate_masks=semantics[f"{index}_plate_masks"],
                                    weld_masks=semantics[f"{index}_weld_masks"],
                                    weld_scores=semantics[f"{index}_weld_scores"],
                                    confidence=data["confidence"] if "confidence" in data else None)
                    output, report, panel = serialize_result(index, image, data, result, args.joint_band_width)
                arrays.update(output)
                for key in ("plate_masks", "plate_scores", "weld_masks", "weld_scores"):
                    arrays[f"image_{index:03d}_sam_{key}"] = semantics[f"{index}_{key}"]
                reports.append(report)
                panels.append(panel)
                print(json.dumps({k: report[k] for k in ("image", "status", "joint_candidates", "welded_region_candidates")}), flush=True)
        data_directory = args.output_dir / "data"
        data_directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data_directory / "auto_results.npz", **arrays)
        payload = {"schema_version": 2,
                   "pipeline": "SAM-interior planes + per-joint interval states + analytic 3D roots or paired 3D toes",
                   "human_region_or_point_inputs": False, "human_weld_state_input": False,
                   "semantics": metadata, "seconds": round(time.monotonic() - started, 2),
                   "depth_cache_hits": len(images) - len(missing), "semantic_cache_reused": semantics_valid,
                   "detector_source_sha256": hashlib.sha256((ROOT / "vision/weld_detector.py").read_bytes()).hexdigest(),
                   "toe_source_sha256": hashlib.sha256((ROOT / "vision/weld_toes.py").read_bytes()).hexdigest(),
                   "surface_source_sha256": hashlib.sha256((ROOT / "vision/weld_surfaces.py").read_bytes()).hexdigest(),
                   "surface_kinds": {"0": "unconfirmed_raw_depth", "1": "mother_plane", "2": "local_plane", "3": "local_quadratic"},
                   "states": {"no_weld_detected": "plate-intersection fallback; manufacturing state unknown",
                              "uncertain_weld_evidence": "SAM evidence not accepted as a bead; review the fallback",
                              "weld_detected": "accepted bead candidate; use its two edge paths"},
                   "class_map": {"0": "background_or_undetected", "1": "plate_joint_candidate_state_unknown", "2": "welded_region_candidate"},
                   "joint_band_width_original_px": args.joint_band_width,
                   "limitations": ["Geometry cannot establish welding intention; joint candidates include non-weld contacts.",
                                   "No seam boundary ground truth; misses and false positives remain possible.",
                                   "Joint mask width is chosen, not measured; single-image DA3 has no metrology calibration.",
                                   "Mask and curve details are limited by the approximately 504-pixel model outputs.",
                                   "Missing SAM bead detections do not prove an unwelded joint; states are local to each contact interval.",
                                   "Confidence filtering uses an image-relative percentile, not a calibrated uncertainty probability.",
                                   "Plane intersection assumes non-parallel planar faces; coplanar butt joints need a different model.",
                                   "Toe support combines depth transition, host surface, depth quality and 3D continuity, not measured accuracy.",
                                   "geometry_runs index separate valid intervals; never concatenate across an invalid gap.",
                                   "Local curved-host fits extrapolate DA3 samples; unknown surfaces retain flagged raw-depth placeholders.",
                                   "Exported camera coordinates are DA3 estimates, not robot-frame motion commands or process offsets.",
                                   "Each geometric segment is returned independently; branching path ordering is not inferred."],
                   "results": reports}
        (data_directory / "auto_report.json").write_text(json.dumps(payload, indent=2) + "\n")
        save_overview(panels, reports, args.output_dir / "auto_overview.jpg")
        save_toe_outputs(arrays, reports, args.output_dir)
        save_paths(arrays, reports, args.output_dir)
        save_sam_outputs(arrays, reports, args.output_dir)
        save_result_index(reports, args.output_dir)
    print(f"Saved automatic results to {args.output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/offline/depth_anything_3"))
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--semantic-threshold", type=float, default=0.25)
    parser.add_argument("--sam-python", type=Path, help="Python executable of the local SAM3 environment")
    parser.add_argument("--joint-band-width", type=int, default=5, help="Rasterized joint-line width in original pixels")
    parser.add_argument("--reuse-depth-dir", type=Path, help="Reuse only depth files with matching input SHA256")
    parser.add_argument("--model-cache-dir", type=Path, help="Optional model-output cache; inputs are verified by SHA256")
    parser.add_argument("--reuse-semantics", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--semantic-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.semantic_threshold <= 1 or args.joint_band_width < 1:
        parser.error("Use 0 < semantic-threshold <= 1 and joint-band-width >= 1")
    images = input_images(args.images)
    if args.semantic_worker:
        semantic_worker(images, args.semantic_worker, args.gpu, args.semantic_threshold)
        return
    pipeline(images, args)


if __name__ == "__main__":
    main()
