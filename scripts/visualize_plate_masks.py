"""Show all cached SAM plate candidates on uncropped source RGB images."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PALETTE = [(255, 145, 0), (0, 190, 255), (245, 65, 160),
           (110, 220, 30), (175, 100, 255), (255, 220, 40)]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def render_candidates(record, masks, scores):
    """Resize only the binary masks; the full original RGB remains uncropped."""
    with Image.open(record['image']) as source:
        rgb = np.array(source.convert('RGB'))
    height, width = rgb.shape[:2]
    if len(masks) != len(scores):
        raise ValueError('Mask/score count mismatch')
    masks = [cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
             for m in masks]
    overlay = rgb.copy()
    for i in sorted(range(len(masks)), key=lambda i: masks[i].sum(), reverse=True):
        color = np.asarray(PALETTE[i % len(PALETTE)])
        overlay[masks[i]] = np.rint(overlay[masks[i]] * .58 + color * .42).astype(np.uint8)
    body = Image.fromarray(overlay)
    draw = ImageDraw.Draw(body)
    number_font = ImageFont.truetype(FONT, 28)
    occupied = np.zeros((height, width), np.uint8)
    for i, mask in enumerate(masks):
        color = PALETTE[i % len(PALETTE)]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            xy = [tuple(map(int, p)) for p in contour[:, 0]]
            if len(xy) >= 3:
                draw.line(xy + xy[:1], fill=color, width=3)
        if mask.any():
            distances = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            if (mask & ~occupied.astype(bool)).any():
                distances[occupied.astype(bool)] = -1
            distances[~mask] = -np.inf
            y, x = np.unravel_index(distances.argmax(), mask.shape)
            x = int(np.clip(x, 25, width - 26))
            y = int(np.clip(y, 25, height - 26))
            text = str(i + 1)
            box = draw.textbbox((int(x), int(y)), text, font=number_font, anchor='mm')
            draw.rounded_rectangle((box[0] - 7, box[1] - 5, box[2] + 7, box[3] + 5),
                                   radius=5, fill=color, outline='black', width=2)
            draw.text((int(x), int(y)), text, font=number_font, anchor='mm', fill='black')
            cv2.circle(occupied, (int(x), int(y)), 30, 1, -1)
    if not masks:
        font = ImageFont.truetype(FONT, 29)
        text = 'NO PLATE CANDIDATE'
        box = draw.textbbox((16, 18), text, font=font)
        draw.rectangle((box[0] - 8, box[1] - 7, box[2] + 8, box[3] + 7), fill='white')
        draw.text((16, 18), text, font=font, fill=(205, 25, 25))
    header, footer = 88, max(96, 16 + 40 * ((len(scores) + 2) // 3))
    tile = Image.new('RGB', (width, header + height + footer), 'white')
    tile.paste(body, (0, header))
    labels = ImageDraw.Draw(tile)
    patch = record.get('patch') or record['case'].rsplit('_', 1)[-1]
    labels.text((12, 7), patch.capitalize() + ' pose | ' + record['view'].capitalize() + ' eye',
                font=ImageFont.truetype(FONT, 31), fill='black')
    labels.text((12, 48), f'{len(masks)} plate candidate(s)', font=ImageFont.truetype(FONT, 25), fill='black')
    score_font = ImageFont.truetype(FONT, 25)
    for i, score in enumerate(scores):
        x = 12 + (i % 3) * (width // 3)
        y = header + height + 10 + (i // 3) * 40
        labels.rectangle((x, y + 4, x + 22, y + 26), fill=PALETTE[i % len(PALETTE)])
        labels.text((x + 29, y), f'{i + 1}: {float(score):.3f}', font=score_font, fill='black')
    return tile


def render(run):
    plan = json.loads((run / 'seams/plan.json').read_text())
    destination = run / 'plate_masks'
    destination.mkdir(exist_ok=True)
    records = plan['records']
    stations = list(dict.fromkeys(record['station'] for record in records))
    metadata = {'source_cache_sha256': hashlib.sha256((run / 'seams/semantics.npz').read_bytes()).hexdigest(),
                'mask_selection': 'Every cached plate mask, before any geometry filtering; no additions or removals',
                'views': []}
    tiles = {}
    with np.load(run / 'seams/semantics.npz', allow_pickle=False) as cache:
        metadata['sam'] = json.loads(str(cache['metadata_json']))
        for i, record in enumerate(records):
            image_hash = hashlib.sha256(Path(record['image']).read_bytes()).hexdigest()
            if image_hash != str(cache[f'{i}_image_sha256']):
                raise ValueError('RGB does not match the SAM cache: ' + record['id'])
            masks, scores = cache[f'{i}_plate_masks'], cache[f'{i}_plate_scores']
            tile = render_candidates(record, masks, scores)
            tile.save(destination / (record['id'] + '.png'))
            tiles[record['id']] = tile
            metadata['views'].append({'id': record['id'], 'image_sha256': image_hash,
                                      'mask_count': len(masks), 'scores': scores.tolist(),
                                      'mask_pixels_at_sam_resolution': [int(m.sum()) for m in masks]})
    lines = ['# SAM3 焊板候选掩码', '',
             '展示这轮缓存的全部 metal plate 候选。不同颜色和编号对应不同候选，编号后的数值是SAM得分；NO PLATE CANDIDATE表示未返回候选。', '']
    for station in stations:
        canvas = None
        gutter = 18
        for row, side in enumerate(('left', 'right')):
            for column, patch in enumerate(('left', 'center', 'right')):
                record = next(r for r in records if r['station'] == station and r['view'] == side
                              and (r.get('patch') or r['case'].rsplit('_', 1)[-1]) == patch)
                tile = tiles[record['id']]
                if canvas is None:
                    canvas = Image.new('RGB', (tile.width * 3 + gutter * 2, tile.height * 2 + gutter + 64), 'white')
                    ImageDraw.Draw(canvas).text((12, 9), station + ' | SAM3 metal plate candidates',
                                               font=ImageFont.truetype(FONT, 36), fill='black')
                canvas.paste(tile, (column * (tile.width + gutter), 64 + row * (tile.height + gutter)))
        filename = station + '_candidates.png'
        canvas.save(destination / filename)
        lines += [f'## {station}', '', f'![{station} 焊板候选掩码]({filename})', '']
    (destination / 'README.md').write_text('\n'.join(lines))
    (destination / 'candidates.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print('PLATE_MASKS_RENDERED', destination, len(records), 'images', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    render(p.parse_args().run_dir.resolve())


if __name__ == '__main__':
    main()
