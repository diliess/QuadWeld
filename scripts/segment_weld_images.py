"""Run local SAM 3.1 text or point prompts on independent welding images."""

import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("assets/weld_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/offline/sam3"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("thirdparty/sam3/assets/model/sam3.1_multiplex.pt"))
    parser.add_argument("--prompt", action="append", help="Repeat to compare text prompts")
    parser.add_argument("--points-csv", type=Path,
                        help="Point prompts: image,case,x,y,label; pixel coordinates, labels 1/0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gpu", default="auto", help="Physical GPU index or auto")
    parser.add_argument("--save-details", action="store_true",
                        help="Also save model logs, instance arrays and JSON reports")
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    if args.points_csv and args.prompt:
        parser.error("Use either --points-csv or --prompt; this model routes them separately")
    point_cases = {}
    if args.points_csv:
        with args.points_csv.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                name, case = row["image"], row["case"]
                if Path(name).name != name or not re.fullmatch(r"[a-z0-9_]+", case):
                    parser.error("Point image must be a filename and case must use a-z, 0-9, _")
                point = (float(row["x"]), float(row["y"]), int(row["label"]))
                if point[2] not in (0, 1):
                    parser.error("Point labels must be 0 (exclude) or 1 (include)")
                point_cases.setdefault(case, {}).setdefault(name, []).append(point)
        if not point_cases:
            parser.error("--points-csv contains no points")
    if args.gpu == "auto":
        rows = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits",
        ], text=True)
        devices = [tuple(int(v.strip()) for v in row.split(","))
                   for row in rows.strip().splitlines()]
        args.gpu, free_mib = max(devices, key=lambda item: item[1])
        print(f"Using GPU {args.gpu}, {free_mib} MiB free", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    import torch
    from sam3.model_builder import build_sam3_predictor

    images = sorted(p for p in args.input_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if point_cases:
        requested = {name for case in point_cases.values() for name in case}
        available = {p.name for p in images}
        if missing := requested - available:
            parser.error(f"Point images missing from input directory: {sorted(missing)}")
        images = [p for p in images if p.name in requested]
        for case in point_cases.values():
            for name, points in case.items():
                with Image.open(args.input_dir / name) as source:
                    if not all(0 <= x < source.width and 0 <= y < source.height
                               for x, y, _ in points):
                        parser.error(f"Point outside image: {name}")
                if not any(label == 1 for _, _, label in points):
                    parser.error(f"At least one positive point is required for {name}")
    if not images:
        raise RuntimeError(f"No images in {args.input_dir}")
    prompts = list(point_cases) if point_cases else args.prompt or ["weld"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading local SAM 3.1 checkpoint...", flush=True)
    with ((args.output_dir / "model_load.log").open("w")
          if args.save_details else io.StringIO()) as log:
        with contextlib.redirect_stdout(log):
            predictor = build_sam3_predictor(
                checkpoint_path=str(args.checkpoint.resolve()), version="sam3.1",
                use_fa3=False, use_rope_real=False, compile=False,
                async_loading_frames=False,
            )
    # The multiplex single-image path uses these thresholds for detection;
    # its add_prompt(output_prob_thresh=...) does not apply this filtering.
    predictor.model.score_threshold_detection = min(0.4, args.threshold)
    predictor.model.image_only_det_thresh = args.threshold
    print("Checkpoint loaded", flush=True)
    report = {
        "checkpoint": str(args.checkpoint), "gpu": str(args.gpu),
        "threshold": args.threshold, "manual_prompts": bool(point_cases), "results": [],
        "detection_threshold": predictor.model.score_threshold_detection,
        "image_detection_threshold": predictor.model.image_only_det_thresh,
    }
    failures = 0
    panels = []
    for prompt in prompts:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt).strip("_").lower()
        run_tag = (f"points_{slug}" if point_cases else
                   f"{slug}__t{args.threshold:g}".replace(".", "p"))
        prompt_dir = args.output_dir
        (prompt_dir / "overlays").mkdir(parents=True, exist_ok=True)
        (prompt_dir / "masks").mkdir(parents=True, exist_ok=True)
        for image_path in images:
            if point_cases and image_path.name not in point_cases[prompt]:
                continue
            output_stem = f"{image_path.stem}__{run_tag}"
            target = prompt_dir / "details" / output_stem
            if args.save_details:
                target.mkdir(parents=True, exist_ok=True)
            session_id = None
            started = time.monotonic()
            row = {"image": str(image_path), "prompt": "" if point_cases else prompt,
                   "threshold": "" if point_cases else args.threshold,
                   "prompt_type": "points" if point_cases else "text",
                   "case": prompt if point_cases else "",
                   "scores_note": ("interactive object score is assigned 1.0; not confidence"
                                   if point_cases else "detection scores")}
            try:
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                session_id = predictor.handle_request({
                    "type": "start_session", "resource_path": str(image_path.resolve()),
                })["session_id"]
                request = {
                    "type": "add_prompt", "session_id": session_id,
                    "frame_index": 0,
                    "output_prob_thresh": args.threshold,
                }
                if point_cases:
                    points = point_cases[prompt][image_path.name]
                    row["points_px"] = [[x, y] for x, y, _ in points]
                    row["point_labels"] = [label for _, _, label in points]
                    request.update(
                        points=[[x / image.width, y / image.height] for x, y, _ in points],
                        point_labels=row["point_labels"], obj_id=1, rel_coordinates=True,
                    )
                else:
                    request["text"] = prompt
                response = predictor.handle_request(request)
                output = response["outputs"]
                masks = np.asarray(output["out_binary_masks"], dtype=bool)
                scores = np.asarray(output["out_probs"])
                object_ids = np.asarray(output["out_obj_ids"])
                boxes = np.asarray(output["out_boxes_xywh"])
                assert masks.shape == (len(object_ids), image.height, image.width)
                assert np.isfinite(scores).all()
                union = masks.any(axis=0)
                mask_path = prompt_dir / "masks" / (output_stem + ".png")
                overlay_path = prompt_dir / "overlays" / (output_stem + ".png")
                Image.fromarray(union.astype(np.uint8) * 255).save(mask_path)
                if args.save_details:
                    np.savez_compressed(target / "instances.npz", masks=masks,
                                        scores=scores, object_ids=object_ids, boxes_xywh=boxes)
                overlay = np.array(image).copy()
                palette = [(255, 65, 65), (40, 220, 70), (40, 150, 255),
                           (255, 200, 40), (200, 70, 255)]
                if point_cases:
                    palette = [(40, 150, 255)]
                for index, mask in enumerate(masks):
                    color = np.array(palette[index % len(palette)])
                    overlay[mask] = (overlay[mask] * 0.55 + color * 0.45).astype(np.uint8)
                    edge = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                                      .filter(ImageFilter.FIND_EDGES)) > 0
                    overlay[edge & mask] = color
                preview = Image.fromarray(overlay)
                if point_cases:
                    draw = ImageDraw.Draw(preview)
                    radius = max(3, round(min(image.size) * 0.006))
                    for x, y, label in points:
                        color = (30, 220, 60) if label else (255, 60, 60)
                        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                                     fill=color, outline="white", width=max(1, radius // 3))
                        draw.line((x - radius / 2, y, x + radius / 2, y),
                                  fill="black", width=max(1, radius // 4))
                        if label:
                            draw.line((x, y - radius / 2, x, y + radius / 2),
                                      fill="black", width=max(1, radius // 4))
                preview.save(overlay_path)
                row.update(status="ok", objects=len(masks),
                           scores=scores.tolist(), object_ids=object_ids.tolist(),
                           mask_pixels=[int(m.sum()) for m in masks],
                           union_fraction=float(union.mean()),
                           mask=str(mask_path), overlay=str(overlay_path))
                preview.thumbnail((480, 540))
                panel = Image.new("RGB", (500, 585), "white")
                panel.paste(preview, ((500 - preview.width) // 2, 40))
                ImageDraw.Draw(panel).text((10, 10),
                    f"{image_path.stem} | {prompt} | {len(masks)} regions", fill="black")
                panels.append(panel)
            except Exception as error:
                failures += 1
                row.update(status="error", error=str(error), traceback=traceback.format_exc())
                traceback.print_exc()
            finally:
                if session_id is not None:
                    predictor.handle_request({"type": "close_session", "session_id": session_id})
            torch.cuda.synchronize()
            row["seconds"] = round(time.monotonic() - started, 2)
            report["results"].append(row)
            if args.save_details:
                (target / "result.json").write_text(json.dumps(row, indent=2) + "\n")
            with (args.output_dir / "latest.csv").open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=[
                    "image", "prompt", "threshold", "status", "objects", "scores",
                    "mask", "overlay", "seconds", "error", "prompt_type", "case",
                    "points_px", "point_labels", "mask_pixels", "union_fraction",
                    "scores_note",
                ], extrasaction="ignore")
                writer.writeheader()
                writer.writerows(report["results"])
            print(json.dumps({k: v for k, v in row.items() if k != "traceback"}), flush=True)
    if panels:
        columns = min(4, len(panels))
        rows = (len(panels) + columns - 1) // columns
        sheet = Image.new("RGB", (500 * columns, 585 * rows), "white")
        for index, panel in enumerate(panels):
            sheet.paste(panel, (500 * (index % columns), 585 * (index // columns)))
        sheet.save(args.output_dir / "latest.jpg", quality=92)
    report["peak_gpu_memory_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    if args.save_details:
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{failures} inference calls failed; see latest.csv")


if __name__ == "__main__":
    main()
