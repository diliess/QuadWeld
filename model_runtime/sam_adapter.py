"""SAM3.1 worker for plate candidates in calibrated stereo captures."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

from .runtime import SAM_CHECKPOINT

PROMPTS = ("metal panel", "metal plate", "steel plate")


def build_predictor(detection_threshold=None, threshold=None, **options):
    from sam3.model_builder import build_sam3_predictor

    with contextlib.redirect_stdout(io.StringIO()):
        predictor = build_sam3_predictor(
            checkpoint_path=str(SAM_CHECKPOINT), version="sam3.1",
            use_fa3=False, use_rope_real=False, compile=False, async_loading_frames=False, **options)
    if detection_threshold is not None:
        predictor.model.score_threshold_detection = detection_threshold
    if threshold is not None:
        predictor.model.image_only_det_thresh = threshold
    return predictor


def segment_plates(inputs, output_dir, gpu, threshold=.25, prompts=PROMPTS):
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['HF_HUB_OFFLINE'] = '1'
    import numpy as np
    import torch
    predictor = build_predictor(threshold, threshold)
    for file in inputs:
        spec = json.loads(file.read_text())
        image = (file.parent / spec['views'][0]['image']).resolve()
        width, height = spec['views'][0]['image_size_wh']
        masks, scores, prompt_ids = [], [], []
        for index, prompt in enumerate(prompts):
            session = predictor.handle_request({'type': 'start_session', 'resource_path': str(image)})['session_id']
            try:
                torch.manual_seed(0)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = predictor.handle_request({'type': 'add_prompt', 'session_id': session, 'frame_index': 0,
                                                       'text': prompt})['outputs']
                found = np.asarray(result['out_binary_masks'], dtype=bool).reshape(-1, height, width)
                masks += list(found)
                scores += [float(v) for v in np.asarray(result['out_probs'], dtype=np.float32)]
                prompt_ids += [index] * len(found)
            finally:
                predictor.handle_request({'type': 'close_session', 'session_id': session})
        name = file.parent.name
        np.savez_compressed(output_dir / (name + '.npz'),
                            masks=np.stack(masks) if masks else np.zeros((0, height, width), bool),
                            scores=np.array(scores, np.float32), prompts=np.array(prompt_ids, np.int16),
                            prompt_texts=np.array(prompts))
        print('PLATE_CANDIDATES', name, len(masks), [round(s, 2) for s in scores], flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Segment plate candidates in the left images of stereo captures")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts", nargs="+", default=PROMPTS)
    parser.add_argument("--threshold", type=float, default=.25)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)
    segment_plates(args.inputs, args.output_dir, args.gpu, args.threshold, args.prompts)


if __name__ == "__main__":
    main()
