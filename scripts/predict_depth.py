"""Estimate single-image metric depth with the locally deployed Depth Anything 3."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="+", help="Independent input RGB images")
    parser.add_argument("--model-dir", type=Path, default=Path(
        "thirdparty/depth-anything-3/weights/DA3NESTED-GIANT-LARGE-1.1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/offline/depth_anything_3/data"))
    parser.add_argument("--gpu", default="auto", help="Physical GPU index or auto")
    parser.add_argument("--process-res", type=int, default=504, help="Longest processed image side")
    args = parser.parse_args()
    for image in args.image:
        if not image.is_file():
            parser.error(f"Image not found: {image}")
    for filename in ("config.json", "model.safetensors"):
        if not (args.model_dir / filename).is_file():
            parser.error(f"Missing local model file: {args.model_dir / filename}")
    if args.process_res < 28:
        parser.error("--process-res must be at least 28")

    if args.gpu == "auto":
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits",
        ], text=True)
        devices = [tuple(int(value.strip()) for value in line.split(","))
                   for line in output.strip().splitlines()]
        gpu_index, free_mib = max(devices, key=lambda device: device[1])
        args.gpu = str(gpu_index)
        print(f"Using GPU {args.gpu}, {free_mib} MiB free", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quadweld-da3-matplotlib")

    import cv2
    import numpy as np
    from PIL import Image
    from safetensors import safe_open
    import torch
    from depth_anything_3.api import DepthAnything3

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this deployment")
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    print("Loading local DA3NESTED-GIANT-LARGE-1.1 checkpoint...", flush=True)
    started = time.monotonic()
    model = DepthAnything3.from_pretrained(str(args.model_dir.resolve()), local_files_only=True)
    state = model.state_dict(keep_vars=True)
    shared_aliases = {}
    with safe_open(args.model_dir / "model.safetensors", framework="pt", device="cpu") as file:
        checkpoint_keys = set(file.keys())
        # Safetensors saves a shared parameter once. DA3 reuses one LayerNorm
        # in four auxiliary heads, so several state-dict names are aliases.
        names_by_object = {id(value): name for name, value in state.items()
                           if name in checkpoint_keys}
        for name in set(state) - checkpoint_keys:
            alias = names_by_object.get(id(state[name]))
            if alias is not None and torch.equal(state[name].detach().cpu(), file.get_tensor(alias)):
                shared_aliases[name] = alias
    missing = set(state) - checkpoint_keys - set(shared_aliases)
    unexpected = checkpoint_keys - set(state)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint keys mismatch: missing={sorted(missing)[:10]}, "
                           f"unexpected={sorted(unexpected)[:10]}")
    model_state_keys = len(state)
    del state
    model = model.to(device="cuda").eval()
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started

    model_info = {
        "load_seconds": round(load_seconds, 3), "checkpoint_keys_verified": len(checkpoint_keys),
        "model_state_keys": model_state_keys, "shared_parameter_aliases": shared_aliases,
    }
    for image in args.image:
        predict_image(model, args, image, model_info)


def predict_image(model, args, image, model_info):
    import cv2
    import numpy as np
    from PIL import Image
    import torch

    started = time.monotonic()
    # Unrelated photographs must be inferred independently, not as one view set.
    prediction = model.inference([str(image.resolve())], process_res=args.process_res)
    torch.cuda.synchronize()
    inference_seconds = time.monotonic() - started
    depth = np.asarray(prediction.depth[0], dtype=np.float32)
    if not bool(prediction.is_metric):
        raise RuntimeError("The selected model did not return metric depth")
    if not np.isfinite(depth).all() or not (depth > 0).all():
        raise RuntimeError("Depth contains nonfinite or nonpositive values")
    rgb = np.asarray(prediction.processed_images[0], dtype=np.uint8)
    if depth.shape != rgb.shape[:2]:
        raise RuntimeError("Processed RGB and depth shapes differ")
    arrays = {
        "depth_m": depth, "processed_rgb": rgb, "is_metric": np.array(True),
        "intrinsics": np.asarray(prediction.intrinsics[0]),
        "extrinsics_w2c": np.asarray(prediction.extrinsics[0]),
    }
    if prediction.conf is not None:
        arrays["confidence"] = np.asarray(prediction.conf[0])
    if prediction.sky is not None:
        arrays["sky"] = np.asarray(prediction.sky[0])
    image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    arrays["image_sha256"] = np.array(image_sha256)
    with Image.open(image) as original:
        arrays["original_hw"] = np.array([original.height, original.width])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / f"{image.stem}_depth.npz"
    preview_path = args.output_dir / f"{image.stem}_depth.jpg"
    report_path = args.output_dir / f"{image.stem}_report.json"
    np.savez_compressed(data_path, **arrays)

    near, far = np.percentile(depth, [2, 98])
    normalized = np.clip((depth - near) / max(float(far - near), 1e-6), 0, 1)
    colors = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    height, width = depth.shape
    preview = np.full((height + 56, width * 2, 3), 255, dtype=np.uint8)
    preview[56:, :width] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    preview[56:, width:] = colors
    cv2.putText(preview, "RGB (processed)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(preview, "Estimated depth (m)", (width + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(preview, f"Colour scale: {near:.3f} to {far:.3f} m (2-98%)",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Failed to save {preview_path}")

    report = {
        "image": str(image), "image_sha256": image_sha256, "model_dir": str(args.model_dir),
        "source_commit": subprocess.check_output([
            "git", "-C", "thirdparty/depth-anything-3", "rev-parse", "HEAD",
        ], text=True).strip(),
        "python": platform.python_version(),
        "depth_anything_3": importlib.metadata.version("depth-anything-3"),
        "xformers": importlib.metadata.version("xformers"),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_index": args.gpu, "gpu_name": torch.cuda.get_device_name(0),
        **model_info,
        "is_metric": True, "unit": "m", "original_hw": arrays["original_hw"].tolist(),
        "processed_hw": [height, width], "process_res": args.process_res,
        "depth_min_m": float(depth.min()), "depth_median_m": float(np.median(depth)),
        "depth_max_m": float(depth.max()), "all_depth_values_finite_positive": True,
        "inference_seconds": round(inference_seconds, 3),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "data": str(data_path), "preview": str(preview_path),
        "alignment": "Depth, RGB and intrinsics use the processed resolution; extrinsics are world-to-camera.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
