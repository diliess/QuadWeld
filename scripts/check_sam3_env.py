"""Check SAM 3 dependencies and CUDA; optionally run without pretrained weights."""

import argparse
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-smoke", action="store_true",
                        help="Run text and point prompts with random weights; does not test segmentation quality")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    import numpy as np
    from PIL import Image
    import torch
    import torchvision
    import sam3
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    for module in ("cv2", "scipy", "skimage", "pycocotools.mask",
                   "ipywidgets", "ipycanvas", "ipympl", "jupyterlab"):
        importlib.import_module(module)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check GPU access and the NVIDIA driver")

    with torch.inference_mode():
        matrix = torch.eye(16, device="cuda")
        torch.testing.assert_close(matrix @ matrix, matrix)
        kept = torchvision.ops.nms(
            torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]], device="cuda"),
            torch.tensor([0.9, 0.8], device="cuda"), 0.5,
        )
        assert kept.tolist() == [0], "torchvision CUDA NMS failed"

    report = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "packages": {name: version(name) for name in (
            "torch", "torchvision", "sam3", "numpy", "opencv-python",
            "setuptools", "scipy", "scikit-image", "huggingface-hub",
        )},
        "sam3_source": sam3.__file__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_ops": "passed",
        "pretrained_weights_loaded": False,
    }

    if args.forward_smoke:
        print("Building SAM 3 with random weights for an environment-only check...", flush=True)
        model = build_sam3_image_model(
            device="cuda", load_from_HF=False, enable_inst_interactivity=True,
        )
        processor = Sam3Processor(model)
        image = Image.new("RGB", (640, 480), color=(128, 128, 128))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            masks, scores, logits = model.predict_inst(
                state, point_coords=np.array([[320., 240.]]),
                point_labels=np.array([1]), multimask_output=True,
            )
            assert masks.shape[-2:] == (480, 640)
            assert np.isfinite(scores).all() and np.isfinite(logits).all()
            output = processor.set_text_prompt(state=state, prompt="metal plate")
            assert torch.isfinite(output["scores"]).all()
        torch.cuda.synchronize()
        report["random_weight_forward"] = {
            "point_mask_shape": list(masks.shape),
            "text_mask_shape": list(output["masks"].shape),
            "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "status": "passed; segmentation quality was not evaluated",
        }

    report["status"] = "passed"
    serialized = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
