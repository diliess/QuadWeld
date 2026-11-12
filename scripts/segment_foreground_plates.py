"""Separate foreground plate parts with automatic depth-guided SAM3.1 point prompts."""

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vision.plate_parts import make_plate_prompts, resolve_plate_masks


COLORS = {'horizontal': (255, 150, 0), 'vertical': (0, 190, 255)}
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def write_preview(record, masks, destination):
    rgb = np.array(Image.open(record['image']).convert('RGB'))
    height, width = rgb.shape[:2]
    for role in ('horizontal', 'vertical'):
        mask = cv2.resize(masks[role].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        rgb[mask] = np.rint(rgb[mask] * .6 + np.asarray(COLORS[role]) * .4).astype(np.uint8)
    tile = Image.new('RGB', (width, height + 92), 'white')
    tile.paste(Image.fromarray(rgb), (0, 92))
    draw = ImageDraw.Draw(tile)
    draw.text((12, 5), record['patch'].capitalize() + ' pose | ' + record['view'].capitalize() + ' eye',
              font=ImageFont.truetype(FONT, 30), fill='black')
    draw.text((12, 48), 'Orange: horizontal  |  Cyan: vertical', font=ImageFont.truetype(FONT, 24), fill='black')
    tile.save(destination / (record['id'] + '.png'))
    return tile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--distance-range', type=float, nargs=2, default=(.3, .8), metavar=('MIN', 'MAX'))
    p.add_argument('--contact-pair', action='store_true', help='Choose horizontal support near the predicted vertical plate foot rather than the largest floor plane')
    a = p.parse_args()
    if not 0 < a.distance_range[0] < a.distance_range[1]:
        p.error('Invalid foreground distance range')
    run, output = a.run_dir.resolve(), a.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not a.resume:
        p.error('Use an empty output directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir(exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    from sam3.model_builder import build_sam3_predictor

    records = json.loads((run / 'seams/plan.json').read_text())['records']
    with contextlib.redirect_stdout(io.StringIO()):
        predictor = build_sam3_predictor(checkpoint_path=str(ROOT / 'thirdparty/sam3/assets/model/sam3.1_multiplex.pt'),
                                        version='sam3.1', use_fa3=False, use_rope_real=False,
                                        compile=False, async_loading_frames=False, max_num_objects=1)
    # Preserve checkpoint dtypes and arithmetic while sharing VRAM with the GUI.
    # The visual transformer executes blocks sequentially, so only the current
    # block needs to reside on the GPU during this offline inference.
    hooks = []
    def load_block(module, inputs):
        module.to('cuda')
    def offload_block(module, inputs, result):
        module.to('cpu')
    blocks = predictor.model.detector.backbone.vision_backbone.trunk.blocks
    for block in blocks:
        block.to('cpu')
        hooks.append(block.register_forward_pre_hook(load_block))
        hooks.append(block.register_forward_hook(offload_block))
    torch.cuda.empty_cache()
    metadata = {'model': 'SAM3.1 multiplex', 'mode': 'automatic_depth_guided_positive_negative_points',
                'threshold': .25, 'role_order': ['horizontal', 'vertical'], 'precision': 'bfloat16 autocast',
                'threshold_note': 'Legacy semantic-cache detection threshold; point-prompt instances bypass text detection scoring',
                'mask_probability_threshold': .5,
                'prompt_source': 'Predicted Nested depth, camera calibration, and camera gravity only; no CAD/truth/hand annotations',
                'point_model_score_note': 'SAM3.1 assigns prompted objects score 1.0; it is not a segmentation accuracy measure',
                'visual_block_cpu_offload': True, 'weight_dtype': str(next(predictor.model.parameters()).dtype),
                'distance_range_m': list(a.distance_range),
                'select_contact_pair': a.contact_pair,
                'overlap_rule': 'Assign SAM overlap to the nearer fitted predicted-depth plane; retain connected support in configured predicted-distance range'}
    arrays = {'metadata_json': np.array(json.dumps(metadata))}
    tiles, reports = {}, []
    for index, record in enumerate(records):
        record_dir = output / record['id']
        record_dir.mkdir(exist_ok=True)
        if hashlib.sha256(Path(record['image']).read_bytes()).hexdigest() != record['image_sha256']:
            raise ValueError('Source RGB changed')
        with np.load(record['depth']) as d:
            depth, k = d['depth_m'], d['intrinsics']
            inputs = json.loads((run / 'predictions' / record['station'] / 'inputs.json').read_text())
            pose_index = [r['id'] for r in inputs['records']].index(record['id'])
            world_pose = np.asarray(inputs['world_from_camera_optical'][pose_index])
            prompts, foreground, points, k_index = make_plate_prompts(depth, k, d.get('confidence'), world_pose,
                                                                   distance_range=a.distance_range,
                                                                   select_contact_pair=a.contact_pair)
        raw_k = np.asarray(record['intrinsics']).copy()
        raw_k[:2, 2] -= .5
        width, height = record['image_size_wh']
        raw, details = {}, {}
        for role in ('horizontal', 'vertical'):
            positive, negative = (prompts[role][key] for key in ('positive_xy_processed', 'negative_xy_processed'))
            xy = np.concatenate((positive, negative))
            rays = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(k_index).T
            native = rays @ raw_k.T
            native = native[:, :2] / native[:, 2:]
            normalized = np.clip((native + .5) / [width, height], 0, 1)
            labels = [1] * len(positive) + [0] * len(negative)
            session = predictor.handle_request({'type': 'start_session', 'resource_path': record['image']})['session_id']
            try:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = predictor.handle_request({'type': 'add_prompt', 'session_id': session, 'frame_index': 0,
                                                       'points': normalized.tolist(), 'point_labels': labels,
                                                       'obj_id': 1, 'rel_coordinates': True})['outputs']
                if result is None or len(result['out_binary_masks']) != 1:
                    raise RuntimeError('Expected one prompted plate mask')
                raw[role] = np.asarray(result['out_binary_masks'][0], dtype=bool)
            finally:
                predictor.handle_request({'type': 'close_session', 'session_id': session})
            details[role] = {'points_normalized': normalized.tolist(), 'point_labels': labels,
                             'plane_equation_camera': prompts[role]['equation'].tolist(),
                             'raw_sam_pixels': int(raw[role].sum())}
        resolved = resolve_plate_masks(raw, prompts, foreground, points)
        np.savez_compressed(record_dir / 'masks.npz', **{f'raw_{r}': raw[r] for r in raw},
                            **{f'final_{r}': resolved[r] for r in resolved}, foreground=foreground)
        for role in details:
            details[role]['final_pixels'] = int(resolved[role].sum())
        details.update(id=record['id'], image_sha256=record['image_sha256'])
        (record_dir / 'prompts.json').write_text(json.dumps(details, indent=2) + '\n')
        reports.append(details)
        # Preserve the existing semantic-cache interface, keeping roles as distinct instances.
        sam_size = (round(width * 504 / max(width, height)), round(height * 504 / max(width, height)))
        arrays[f'{index}_image_sha256'] = np.array(record['image_sha256'])
        arrays[f'{index}_plate_masks'] = np.stack([cv2.resize(resolved[r].astype(np.uint8), sam_size, interpolation=cv2.INTER_NEAREST).astype(bool)
                                                   for r in ('horizontal', 'vertical')])
        arrays[f'{index}_plate_scores'] = np.ones(2, dtype=np.float32)
        arrays[f'{index}_weld_masks'] = np.zeros((0, sam_size[1], sam_size[0]), bool)
        arrays[f'{index}_weld_scores'] = np.zeros(0, np.float32)
        tiles[record['id']] = write_preview(record, resolved, output / 'images')
        print('FOREGROUND_PARTS', record['id'], {r: details[r]['final_pixels'] for r in ('horizontal', 'vertical')}, flush=True)
    np.savez_compressed(output / 'semantics.npz', **arrays)
    (output / 'summary.json').write_text(json.dumps({'method': metadata, 'views': reports}, indent=2) + '\n')
    lines = ['# 前方横板与竖板分割', '',
             '橙色为水平板，青色为竖直板。使用DA3预测深度自动生成SAM3.1正负点提示，并用预测板面消解重叠；未使用CAD真值或人工选区。', '']
    for station in dict.fromkeys(r['station'] for r in records):
        example = next(tiles[r['id']] for r in records if r['station'] == station)
        gutter = 18
        canvas = Image.new('RGB', (example.width * 3 + gutter * 2, example.height * 2 + gutter + 60), 'white')
        ImageDraw.Draw(canvas).text((12, 8), station + ' | Foreground plate parts', font=ImageFont.truetype(FONT, 36), fill='black')
        for row, side in enumerate(('left', 'right')):
            for col, patch in enumerate(('left', 'center', 'right')):
                rec = next(r for r in records if r['station'] == station and r['view'] == side and r['patch'] == patch)
                canvas.paste(tiles[rec['id']], (col * (example.width + gutter), 60 + row * (example.height + gutter)))
        filename = station + '_parts.png'
        canvas.save(output / 'images' / filename)
        lines += [f'## {station}', '', f'![{station} 前方横板与竖板](images/{filename})', '']
    (output / 'README.md').write_text('\n'.join(lines))
    print('FOREGROUND_PLATE_SEGMENTATION_COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
