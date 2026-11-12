"""Locate a T/fillet/corner seam in 3D from one calibrated stereo pair and two plate masks.

    python scripts/stereo_seam.py pair/input.json --masks plate_parts/masks.npz --output-dir out

The pair uses the ``quadweld-stereo-v1`` input.json.  Masks are boolean arrays
on the left image (any segmenter; ``segment_foreground_plates.py`` writes
``raw_horizontal`` / ``raw_vertical``).  No depth map is required; a DA3 depth
file may be passed as a starting guess only.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from vision import stereo_seam as ss


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_pair(input_json, masks=(), half_pixel_centres=True):
    """Images, masks, index-convention intrinsics, baseline, and rectified->left-camera rotation.

    ``half_pixel_centres``: the calibration puts the centre of pixel i at i + 0.5
    (USD / DA3 / this project's captures).  OpenCV calibrations use i: pass False.
    A pair that is not already rectified is rectified here, masks included.
    """
    input_json = Path(input_json)
    data = json.loads(input_json.read_text())
    views = data['views']
    if [v.get('id') for v in views] != ['left', 'right']:
        raise ValueError('Stereo views must be ordered left, right')
    left, right = (read_rgb(input_json.parent / v['image']) for v in views)
    k = [np.array(v['intrinsics'], float) for v in views]
    if half_pixel_centres:
        for matrix in k:
            matrix[:2, 2] -= .5
    d = [np.array(v.get('distortion_coefficients', [0] * 5), float) for v in views]
    e = [np.vstack((np.array(v['extrinsics_w2c'], float)[:3], [0, 0, 0, 1])) for v in views]
    relative = e[1] @ np.linalg.inv(e[0])            # left-camera points -> right camera
    baseline = float(np.linalg.norm(relative[:3, 3]))
    rectified = (not np.any(d[0]) and not np.any(d[1]) and np.allclose(k[0], k[1], atol=1e-6) and
                 np.allclose(relative[:3, :3], np.eye(3), atol=1e-7) and
                 np.allclose(relative[:3, 3], [-baseline, 0, 0], atol=1e-7 + 1e-6 * baseline))
    if rectified:
        return left, right, [np.asarray(m, bool) for m in masks], k[0], baseline, np.eye(3)
    return ss.rectify_pair(left, right, k[0], d[0], k[1], d[1], relative[:3, :3], relative[:3, 3], masks)


def load_init_depth(path, shape):
    """A depth map (metres) from another model, resized to the stereo image; None if absent."""
    if path is None or not Path(path).is_file():
        return None
    with np.load(path) as data:
        depth = data['depth_m'].astype(np.float32)
    return cv2.resize(depth, shape[::-1], interpolation=cv2.INTER_LINEAR)


def describe(result, rotation=np.eye(3), runtime_s=None):
    """JSON-ready record. Vectors are rotated from the rectified frame into the left camera frame."""
    record = {'status': result['status'], 'frame': 'left camera optical (x right, y down, z forward), metres',
              'runtime_s': runtime_s, 'checks': result['checks']}
    if 'fits' in result:
        record['plates'] = [{'disparity_plane_abc': f['params'].tolist(), 'normal': (rotation @ n).tolist(),
                             'offset_m': h, 'pixels': f['pixels']}
                            for f, (n, h) in zip(result['fits'], result['planes'])]
        record['neighbourhood_px'] = result['neighbourhood_px']
    if result['status'] == 'ok':
        record.update(
            joint_ends_xyz=(result['joint_ends_xyz'] @ rotation.T).tolist(), length_m=result['length_m'],
            direction=(rotation @ result['direction']).tolist(), end_is_open=result['end_is_open'],
            torch_axis=(rotation @ result['torch_axis']).tolist(), opening_deg=result['opening_deg'],
            face_normals=[(rotation @ n).tolist() for n in result['face_normals']],
            image_segment_left=result['image_segment'].tolist(),
            image_ends_right=result['xy_right'][[0, -1]].tolist(), points=len(result['xyz']))
    return record


def draw_overlay(left, right, result, path, reference=None):
    """Left/right crops with the located seam; ``reference`` adds evaluation curves (left xy, right xy)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout='constrained')
    for axis, image, key, name in zip(axes, (left, right), ('xy', 'xy_right'), ('left', 'right')):
        axis.imshow(image)
        focus = []
        if reference is not None:
            curve = reference[0 if name == 'left' else 1]
            axis.plot(*curve.T, '--', color='magenta', lw=1., zorder=4, label='reference (evaluation only)')
            focus.append(curve)
        if result['status'] == 'ok':
            axis.plot(*result[key].T, color='lime', lw=1.4, label='stereo plane intersection')
            axis.plot(*result[key][[0, -1]].T, 'o', color='lime', ms=5, mfc='none')
            focus.append(result[key])
        if focus:
            points = np.concatenate(focus)
            lo, hi = points.min(0), points.max(0)
            pad = max(.12 * (hi[0] - lo[0]), 25)
            axis.set_xlim(max(lo[0] - pad, -.5), min(hi[0] + pad, image.shape[1] - .5))
            axis.set_ylim(min(hi[1] + 3.5 * pad, image.shape[0] - .5), max(lo[1] - 3.5 * pad, -.5))
            axis.legend(fontsize=8, loc='lower right')
        axis.set_title(f'{name} | {result["status"]}')
        axis.axis('off')
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', type=Path, help='quadweld-stereo-v1 input.json of one synchronized pair')
    p.add_argument('--masks', type=Path, required=True, help='npz with two boolean plate masks on the left image')
    p.add_argument('--mask-keys', nargs=2, default=['raw_horizontal', 'raw_vertical'])
    p.add_argument('--init-depth', type=Path, help='Optional *_depth.npz (e.g. DA3) used as a starting guess only')
    p.add_argument('--depth-range', nargs=2, type=float, default=[.25, 4.], metavar=('NEAR', 'FAR'))
    p.add_argument('--end-margin', type=float, default=0., help='Start/stop inset from each joint end, metres')
    p.add_argument('--spacing', type=float, default=.002, help='Way-point spacing, metres')
    p.add_argument('--opencv-pixel-centres', action='store_true',
                   help='Intrinsics place pixel centres at integer coordinates (OpenCV calibration)')
    p.add_argument('--output-dir', type=Path, default=Path('outputs/stereo_seam'))
    a = p.parse_args()
    with np.load(a.masks) as stored:
        masks = [stored[key] for key in a.mask_keys]
    left, right, masks, k, baseline, rotation = load_pair(a.input, masks, not a.opencv_pixel_centres)
    if a.init_depth and not np.allclose(rotation, np.eye(3)):
        p.error('--init-depth needs an already rectified pair')
    start = time.time()
    result = ss.locate_seam(left, right, k, baseline, masks[0], masks[1], depth_range=tuple(a.depth_range),
                            init_depth=load_init_depth(a.init_depth, left.shape[:2]),
                            end_margin=a.end_margin, spacing=a.spacing)
    record = describe(result, rotation, time.time() - start)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / 'seam.json').write_text(json.dumps(record, indent=2) + '\n')
    with (a.output_dir / 'seam_path.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample', 'x_m', 'y_m', 'z_m', 'torch_x', 'torch_y', 'torch_z'])
        if result['status'] == 'ok':
            axis = rotation @ result['torch_axis']
            for i, point in enumerate(result['xyz'] @ rotation.T):
                writer.writerow([i, *(f'{v:.6f}' for v in point), *(f'{v:.6f}' for v in axis)])
    draw_overlay(left, right, result, a.output_dir / 'overlay.png')
    print('STEREO_SEAM', result['status'], json.dumps(record['checks']), a.output_dir, flush=True)
    return 0 if result['status'] == 'ok' else 2


if __name__ == '__main__':
    raise SystemExit(main())
