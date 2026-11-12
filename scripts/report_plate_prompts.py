"""Display all unfiltered native SAM text candidates beside their source RGB."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(root):
    config = json.loads((root / 'config.json').read_text())
    prompts = config['prompts']
    if not prompts:
        raise ValueError('At least one text prompt is required')
    images = root / 'images'
    images.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    columns = len(prompts) + 1
    centre_count = sum(r['patch'] == 'center' and r['view'] == 'left' for r in config['records'])
    preview = Image.new('RGB', (480 * columns, max(1, centre_count) * 370), 'white')
    preview_draw = ImageDraw.Draw(preview)
    summaries, totals = [], [0] * len(prompts)
    centre_index = 0
    for record in config['records']:
        name = record['id']
        original = Image.open(record['image']).convert('RGB')
        assert sha256(Path(record['image'])) == record['image_sha256']
        width, height = original.size
        figure = Image.new('RGB', (width * columns, height + 110), 'white')
        figure.paste(original, (0, 110))
        labels = ImageDraw.Draw(figure)
        labels.text((10, 8), name, font=font, fill='black')
        labels.text((10, 45), 'Original full RGB', font=font, fill='black')
        counts, boxes, rendered = [], [], [original]
        with np.load(root / (name + '.npz'), allow_pickle=False) as data:
            assert str(data['image_sha256']) == record['image_sha256']
            queries = json.loads(str(data['queries_json']))
            for i, prompt in enumerate(prompts):
                masks, scores = data[f'{i}_native_masks'], data[f'{i}_scores']
                assert masks.shape == (len(scores), height, width)
                assert queries[i]['prompt'] == prompt and queries[i]['count'] == len(masks)
                counts.append(len(masks))
                totals[i] += len(masks) > 0
                rgb = np.array(original).copy()
                palette = ((0, 190, 255), (255, 150, 0), (60, 220, 100), (210, 90, 255))
                color = np.array(palette[i % len(palette)])
                if len(masks):
                    union = masks.any(0)
                    rgb[union] = np.rint(rgb[union] * .55 + color * .45).astype(np.uint8)
                overlay = Image.fromarray(rgb)
                draw = ImageDraw.Draw(overlay)
                instance_boxes = []
                for j, (mask, score) in enumerate(zip(masks, scores)):
                    yy, xx = np.nonzero(mask)
                    box = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())] if len(xx) else None
                    instance_boxes.append({'instance': j + 1, 'score': float(score), 'bbox_xyxy': box,
                                           'native_pixels': int(mask.sum())})
                    if box is not None:
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            coordinates = [tuple(x) for x in contour[:, 0]]
                            if len(coordinates) > 1:
                                draw.line(coordinates + coordinates[:1], fill=tuple(color.tolist()), width=2)
                        x, y = box[:2]
                        text = f'{j+1}: {score:.2f}'
                        draw.rectangle((x, max(0, y - 23), min(width, x + 100), y), fill='white')
                        draw.text((x + 2, max(0, y - 23)), text, fill='black', font=small)
                figure.paste(overlay, ((i + 1) * width, 110))
                labels.text(((i + 1) * width + 10, 8), prompt, font=font, fill='black')
                labels.text(((i + 1) * width + 10, 45), f'Instances: {len(masks)} | threshold {config["threshold"]}', font=font, fill='black')
                boxes.append(instance_boxes)
                rendered.append(overlay)
        figure.save(images / (name + '.png'))
        summaries.append({'id': name, 'station': record['station'], 'patch': record['patch'], 'eye': record['view'],
                          'image_sha256': record['image_sha256'], 'counts': counts, 'instances_by_prompt': boxes})
        if record['patch'] == 'center' and record['view'] == 'left':
            top = centre_index * 370
            for col, img in enumerate(rendered):
                # Display crop only; inference and the linked per-view figures
                # always use the complete native image without an ROI.
                crop = img.crop((0, 420, width, 780)).resize((480, 270), Image.Resampling.LANCZOS)
                preview.paste(crop, (col * 480, top + 80))
                title = 'Original RGB' if col == 0 else prompts[col - 1]
                preview_draw.text((col * 480 + 8, top + 8), record['station'] + ' | ' + title, font=small, fill='black')
                if col:
                    preview_draw.text((col * 480 + 8, top + 40), f'All-image detections: {counts[col - 1]}', font=small, fill='black')
            centre_index += 1
    preview.save(images / 'centres_preview.png')
    summary = {'prompts': prompts, 'frames': len(summaries), 'nonempty_frames_by_prompt': totals,
               'visualization': 'Native masks and all instances above model threshold; no depth/range/ROI/geometry filtering',
               'views': summaries}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# 指定文字提示的 SAM3 分割测试', '', '实际使用的提示：', '', '```python',
             'prompts = [', *[f'    "{p}",' for p in prompts], ']', '```', '',
             f'对原有{len(summaries)}张完整 RGB 分别独立查询，检测阈值0.25。不输入正负点、框、深度或CAD，也不按距离、前景位置或平面方向筛选输出。以下保留模型返回的全部候选；分数是模型检测分数，不是分割准确率。', '',
             '| 文字提示 | 有非空结果的图像数 |', '|---|---:|',
             *[f'| `{prompt}` | {total}/{len(summaries)} |' for prompt, total in zip(prompts, totals)], '',
             '下图为中心左目局部放大，便于观察；各视角完整原图和完整输出在后面直接展示。', '',
             '![三工件文字提示分割结果](images/centres_preview.png)', '']
    for station in dict.fromkeys(r['station'] for r in config['records']):
        lines += [f'## {station}', '']
        for record in (r for r in config['records'] if r['station'] == station):
            lines += [f"### {record['patch']} 臂姿 / {record['view']} 目", '',
                      f"![原图与各条文字提示的完整候选](images/{record['id']}.png)", '']
    (root / 'README.md').write_text('\n'.join(lines))
    validation = {'completed': True, 'images_verified': len(summaries), 'text_queries_verified': len(summaries) * len(prompts),
                  'prompts': prompts, 'points_supplied': False, 'boxes_supplied': False, 'depth_supplied': False,
                  'cad_supplied': False, 'native_masks_preserved': True, 'post_filtering': False,
                  'checkpoint_sha256': config['checkpoint_sha256']}
    (root / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('frames', 'prompts', 'nonempty_frames_by_prompt')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    render(p.parse_args().output_dir.resolve())
