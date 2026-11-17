"""Measure SAM3.1 text prompts on saved RGB, preserving every raw candidate."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


PROMPTS = [
    'metal plate', 'steel plate', 'metal sheet',
    'vertical metal plate', 'horizontal metal plate',
    'upright metal plate', 'flat metal plate',
    'vertical steel plate', 'horizontal steel plate',
    'vertical metal panel', 'horizontal metal panel',
    'vertical metal surface', 'horizontal metal surface',
    'vertical metal sheet', 'horizontal metal sheet',
    'metal base plate', 'vertical panel', 'horizontal panel',
    'foreground vertical metal plate', 'foreground horizontal metal plate',
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--pilot', action='store_true', help='Use the three challenging centre-left images')
    p.add_argument('--prompts', nargs='+', default=PROMPTS)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stream-blocks', action='store_true', help='Offload visual transformer blocks between forwards without changing weights/dtypes')
    a = p.parse_args()
    a.run_dir = a.run_dir.resolve()
    a.output_dir = a.output_dir.resolve()
    if a.output_dir.exists() and any(a.output_dir.iterdir()) and not a.resume:
        p.error('Use an empty directory or --resume')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    import cv2
    from PIL import Image
    import torch
    from model_runtime.sam_adapter import build_predictor

    records = json.loads((a.run_dir / 'seams/plan.json').read_text())['records']
    if a.pilot:
        records = [r for r in records if r['patch'] == 'center' and r['view'] == 'left']
    checkpoint = Path('thirdparty/sam3/assets/model/sam3.1_multiplex.pt').resolve()
    with checkpoint.open('rb') as f:
        checkpoint_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    config = {'model': 'SAM3.1 multiplex', 'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
              'prompts': a.prompts, 'threshold': .25, 'records': records,
              'prompt_scope': 'Text only on full uncropped RGB; no boxes, points, depth or CAD supplied to SAM',
              'native_masks_saved': True, 'visual_block_cpu_offload': a.stream_blocks,
              'seed_per_query': 0}
    (a.output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    predictor = build_predictor(.25, .25)
    hooks = []
    if a.stream_blocks:
        def load_block(module, inputs):
            module.to('cuda')
        def offload_block(module, inputs, result):
            module.to('cpu')
        for block in predictor.model.detector.backbone.vision_backbone.trunk.blocks:
            block.to('cpu')
            hooks.append(block.register_forward_pre_hook(load_block))
            hooks.append(block.register_forward_hook(offload_block))
        torch.cuda.empty_cache()
    for record in records:
        destination = a.output_dir / (record['id'] + '.npz')
        if destination.exists() and a.resume:
            continue
        image = Path(record['image'])
        if hashlib.sha256(image.read_bytes()).hexdigest() != record['image_sha256']:
            raise ValueError('Source RGB changed')
        with Image.open(image) as rgb:
            original_wh = rgb.size
            size = (round(rgb.width * 504 / max(rgb.size)), round(rgb.height * 504 / max(rgb.size)))
        arrays = {'image_sha256': np.array(record['image_sha256'])}
        results = []
        for i, prompt in enumerate(a.prompts):
            # A fresh image session makes each text query independent.
            session = predictor.handle_request({'type': 'start_session', 'resource_path': str(image)})['session_id']
            try:
                started = time.monotonic()
                torch.manual_seed(0)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = predictor.handle_request({'type': 'add_prompt', 'session_id': session,
                                                       'frame_index': 0, 'text': prompt})['outputs']
                native = np.asarray(result['out_binary_masks'], dtype=bool)
                if not len(native):
                    native = np.zeros((0, original_wh[1], original_wh[0]), bool)
                if native.shape != (len(native), original_wh[1], original_wh[0]):
                    raise ValueError('Unexpected native SAM mask shape')
                masks = [cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
                         for m in native]
                scores = np.asarray(result['out_probs'], dtype=np.float32)
                if len(native) != len(scores):
                    raise ValueError('Mask/score count mismatch')
                arrays[f'{i}_native_masks'] = native
                arrays[f'{i}_masks'] = np.stack(masks) if masks else np.zeros((0, size[1], size[0]), bool)
                arrays[f'{i}_scores'] = scores
                row = {'prompt': prompt, 'count': len(masks), 'scores': scores.tolist(),
                       'mask_pixels': [int(m.sum()) for m in masks], 'seconds': time.monotonic() - started}
                results.append(row)
                print(record['id'], json.dumps(row), flush=True)
            finally:
                predictor.handle_request({'type': 'close_session', 'session_id': session})
        arrays['queries_json'] = np.array(json.dumps(results))
        np.savez_compressed(destination, **arrays)
        (a.output_dir / (record['id'] + '.json')).write_text(json.dumps(results, indent=2) + '\n')
    print('SAM_PROMPT_BENCHMARK_COMPLETE', a.output_dir, flush=True)


if __name__ == '__main__':
    main()
