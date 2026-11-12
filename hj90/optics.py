"""RGB-only laser detection and calibrated stereo triangulation; no scene depth input."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_calibration(path=None):
    c = json.loads(Path(path or ROOT / 'calibration/hj90_sim.json').read_text())
    k = np.asarray(c['intrinsics'], float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Invalid stereo intrinsics')
    if c['baseline_m'] <= 0 or not 0 < c['working_distance_m'][0] < c['working_distance_m'][1]:
        raise ValueError('Invalid baseline or working distance')
    if any(c[name] != [0, 0, 0, 0, 0] for name in ('distortion_left', 'distortion_right')):
        raise ValueError('This renderer requires rectified zero-distortion calibration')
    if not c['max_line_rate_hz'] > 0 or not 0 < c['max_full_frame_fps'] <= 6:
        raise ValueError('Invalid acquisition limits')
    return c


def stripe_centres(on, off, settings):
    """One unambiguous subpixel blue stripe per epipolar row; absent rows are NaN.

    Laser-on minus laser-off suppresses ambient textures. Multiple disconnected
    peaks, broad saturated regions and weak returns are rejected, never filled.
    """
    if on.shape != off.shape or on.ndim != 3 or on.shape[2] != 3:
        raise ValueError('Expected aligned RGB on/off images')
    d = on.astype(float) - off.astype(float)
    score = np.maximum(0, d[..., 2] - .1 * (np.maximum(d[..., 0], 0) + np.maximum(d[..., 1], 0)))
    peak = score.argmax(axis=1)
    strength = score[np.arange(len(peak)), peak]
    radius = int(settings['centroid_radius_px'])
    columns = np.arange(on.shape[1])[None, :]
    near = np.abs(columns - peak[:, None]) <= radius
    weights = np.where(near, np.maximum(0, score - settings['threshold_255']), 0)
    total = weights.sum(axis=1)
    centre = (weights * (columns + .5)).sum(axis=1) / np.maximum(total, 1e-12)
    second = np.where(np.abs(columns - peak[:, None]) > 2*radius, score, 0).max(axis=1)
    support = score > np.maximum(settings['threshold_255'], strength[:, None]*.25)
    width = (support & (np.abs(columns - peak[:, None]) <= 2*radius)).sum(axis=1)
    valid = ((strength > settings['threshold_255']) & (total > 0)
             & (second < strength * settings['peak_ratio'])
             & (width <= settings['max_stripe_width_px'])
             & (peak >= radius) & (peak < on.shape[1]-radius))
    centre[~valid] = np.nan
    return centre, strength, score


def triangulate(left_u, right_u, calibration, angle_rad=None):
    """Rectified stereo: Z=fx*baseline/disparity; output in midpoint optical XYZ.

    Image coordinates refer to pixel centres (index + 0.5). Laser-plane
    consistency is only a rejection check, not a source of metric depth.
    """
    l, r = np.asarray(left_u), np.asarray(right_u)
    if l.shape != r.shape or l.ndim != 1:
        raise ValueError('Expected one correspondence per epipolar row')
    k = np.asarray(calibration['intrinsics'])
    b = calibration['baseline_m']
    disparity = l-r
    z = np.full(l.shape, np.nan)
    np.divide(k[0, 0]*b, disparity, out=z, where=np.isfinite(disparity) & (disparity > 0))
    x = (l-k[0, 2])*z/k[0, 0]-b/2
    y = (np.arange(len(l))+.5-k[1, 2])*z/k[1, 1]
    xyz = np.c_[x, y, z]
    near, far = calibration['working_distance_m']
    valid = np.isfinite(xyz).all(1) & (z >= near-1e-9) & (z <= far+1e-9)
    if angle_rad is not None:
        normal = np.array([np.cos(angle_rad), 0., -np.sin(angle_rad)])
        residual = (xyz-np.asarray(calibration['laser_origin_optical_m'])) @ normal
        valid &= np.abs(residual) <= calibration['detector']['plane_tolerance_m']
    xyz[~valid] = np.nan
    return xyz, valid


def reconstruct(left_on, left_off, right_on, right_off, calibration, angle_rad):
    l, ls, _ = stripe_centres(left_on, left_off, calibration['detector'])
    r, rs, _ = stripe_centres(right_on, right_off, calibration['detector'])
    xyz, valid = triangulate(l, r, calibration, angle_rad)
    return xyz[valid], np.c_[l[valid], np.flatnonzero(valid)+.5], np.minimum(ls, rs)[valid]


def rasterize(points, calibration):
    """Sparse midpoint depth; no interpolation across holes or occlusions."""
    w, h = calibration['image_size_wh']
    k = np.asarray(calibration['intrinsics'])
    depth = np.full((h, w), np.inf, np.float32)
    if len(points):
        uv = points[:, :2]/points[:, 2:3] @ k[:2, :2].T + k[:2, 2]
        xy = np.floor(uv).astype(int)
        keep = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        np.minimum.at(depth, (xy[keep, 1], xy[keep, 0]), points[keep, 2])
    depth[~np.isfinite(depth)] = np.nan
    return depth


def write_ply(path, points):
    with Path(path).open('w') as stream:
        stream.write('ply\nformat ascii 1.0\nelement vertex %d\n' % len(points))
        stream.write('property float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(stream, points, fmt='%.7f')


def write_pcd(path, points):
    """Standard ASCII PCD in metres, no fabricated intensity or normals."""
    with Path(path).open('w') as stream:
        stream.write('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n')
        stream.write('WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n' % (len(points),len(points)))
        np.savetxt(stream,points,fmt='%.7f')
