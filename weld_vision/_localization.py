"""Shared capture adapter and coordinate helpers for the two localization stages."""
import json

import cv2
import numpy as np

from . import bead_seam, stereo_seam
from .stereo_io import load_pair

SPACING = .001


def validate_measurement(left, right, intrinsics, baseline, mask_a, mask_b, world):
    """Validate an already rectified stereo observation, in metres."""
    if np.asarray(left).shape[:2] != np.asarray(right).shape[:2]:
        raise ValueError('Left and right images must have the same rectified size')
    shape = np.asarray(left).shape[:2]
    if len(shape) != 2 or any(np.asarray(m).shape != shape for m in (mask_a, mask_b)):
        raise ValueError('Both plate masks must match the rectified left image')
    k = np.asarray(intrinsics, float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Expected a finite 3x3 rectified camera matrix')
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError('Stereo baseline must be positive, in metres')
    world = np.asarray(world, float)
    if (world.shape != (4, 4) or not np.isfinite(world).all()
            or not np.allclose(world[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(world[:3, :3].T @ world[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(world[:3, :3]), 1., atol=1e-6)):
        raise ValueError('world_from_left must be a rigid 4x4 optical-camera pose')
    return world


def transform_points(points, world):
    return np.asarray(points) @ world[:3, :3].T + world[:3, 3]



def line_error(points, origin, direction):
    offset = points - origin
    return np.linalg.norm(offset - np.outer(offset @ direction, direction), axis=1)



def pick_scan_pair(masks, scores, intrinsics, world, observation):
    """Match segmentation candidates to projected patches of the two coarse planes."""
    if len(masks) < 2:
        return None
    a, b = np.asarray(observation['root_world_m'])
    normals = np.asarray(observation['face_normals_world'])
    roots = a + np.linspace(.04, .96, 61)[:, None] * (b - a)
    offsets = np.array([.035, .06, .09, .12])
    camera = np.linalg.inv(world)
    height, width = masks[0].shape
    support = []
    for index, normal in enumerate(normals):
        neighbour = normals[1 - index]
        across = neighbour - (normal @ neighbour) * normal
        across /= np.linalg.norm(across)
        patch = (roots[:, None, :] + offsets[None, :, None] * across).reshape(-1, 3)
        local = transform_points(patch, camera)
        xy = np.rint(stereo_seam.project(local[local[:, 2] > 0], intrinsics)).astype(int)
        x, y = xy.T
        visible = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if visible.sum() < 10:
            return None
        support.append(masks[:, y[visible], x[visible]].mean(1))
    support = np.asarray(support)
    quality = support - .6 * support[::-1] + .02 * scores
    pair_score = quality[0, :, None] + quality[1, None, :]
    compatible = (support[0, :, None] >= .30) & (support[1, None, :] >= .30)
    np.fill_diagonal(compatible, False)
    if not compatible.any():
        return None
    return np.unravel_index(np.where(compatible, pair_score, -np.inf).argmax(), pair_score.shape)



def read_capture(run, phase, ident, overview=None):
    directory = run / phase
    path = directory / 'captures' / ident / 'input.json'
    pair = json.loads(path.read_text())
    world = np.asarray(pair['world_from_left_optical'])
    with np.load(directory / 'sam' / (ident + '.npz')) as data:
        masks, scores = data['masks'], data['scores']
    left, right, masks, intrinsics, baseline, original_from_rectified = load_pair(path, masks)
    world[:3, :3] = world[:3, :3] @ original_from_rectified
    indices = None if overview is None else pick_scan_pair(np.asarray(masks), scores, intrinsics, world, overview)
    if indices is None:
        indices = bead_seam.pick_plate_pair(masks, scores)
    if indices is None:
        return None
    return left, right, intrinsics, baseline, *(masks[i] for i in indices), world



def as_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)



def overview_overlay(run, output, observation, fused):
    folder = run / 'overview' / 'captures' / observation['id']
    left, _, _, intrinsics, _, rotation = load_pair(folder / 'input.json')
    pose = np.asarray(json.loads((folder / 'input.json').read_text())['world_from_left_optical'])
    pose[:3, :3] = pose[:3, :3] @ rotation
    inverse = np.linalg.inv(pose)
    image = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)

    def draw(points, color, width):
        local = transform_points(np.asarray(points), inverse)
        xy = np.rint(stereo_seam.project(local, intrinsics)).astype(np.int32)
        if len(xy) >= 2:
            cv2.polylines(image, [xy], False, color, width, cv2.LINE_AA)
        elif len(xy):
            cv2.circle(image, tuple(xy[0]), width, color, -1, cv2.LINE_AA)

    draw(observation['root_world_m'], (0, 220, 255), 2)
    draw(fused['root_world_m'], (255, 255, 255), 1)
    for name, color in (('base', (255, 0, 255)), ('web', (255, 255, 0))):
        for segment in fused.get('toes', {}).get(name, {}).get('segments_world_m', []):
            draw(segment, color, 1)
    cv2.imwrite(str(output / 'overview_overlay.png'), image)
