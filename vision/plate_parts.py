"""Automatic geometric prompts for separate foreground horizontal/vertical plates."""

import cv2
import numpy as np

from .weld_detector import camera_points, depth_quality, extract_plates


def sample_interior(mask, count=5):
    """Spread prompts across a face while avoiding its mixed-depth boundary."""
    distances = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    eligible = mask & (distances >= min(6., max(1., float(distances.max()) * .25)))
    yx = np.argwhere(eligible)
    if not len(yx):
        raise ValueError('No interior pixels for a plate prompt')
    chosen = [yx[np.argmax(distances[eligible])]]
    while len(chosen) < min(count, len(yx)):
        separation = ((yx[:, None] - np.asarray(chosen)[None]) ** 2).sum(axis=2).min(axis=1)
        score = separation * np.sqrt(distances[eligible])
        if score.max() <= 0:
            break
        chosen.append(yx[score.argmax()])
    return np.asarray(chosen)[:, ::-1].astype(float)


def make_plate_prompts(depth, intrinsics, confidence, world_from_camera, *, distance_range=(.3, .8),
                       select_contact_pair=False):
    k = np.asarray(intrinsics).copy()
    k[:2, 2] -= .5
    points = camera_points(depth, k)
    distance = np.linalg.norm(points, axis=2)
    low, high = distance_range
    if not 0 < low < high:
        raise ValueError('Invalid foreground distance range')
    foreground = (distance >= low) & (distance <= high)
    quality, _ = depth_quality(depth, confidence)
    planes, _, _ = extract_plates(depth, k, quality_mask=quality & foreground)
    up_camera = np.asarray(world_from_camera)[2, :3]
    choices = {'horizontal': [], 'vertical': []}
    for plane in planes:
        alignment = abs(plane.equation[:3] @ up_camera)
        if alignment >= (.97 if select_contact_pair else .85):
            choices['horizontal'].append(plane)
        if alignment <= .35:
            choices['vertical'].append(plane)
    if any(not candidates for candidates in choices.values()):
        raise ValueError('Both foreground plate orientations are required')
    selected = {role: max(candidates, key=lambda plane: plane.mask.sum()) for role, candidates in choices.items()}
    if select_contact_pair:
        # At overview distances the floor can dominate horizontal pixel area.
        # Select the horizontal face nearest the observed vertical face's foot
        # in gravity height, using predicted depth alone (no known plate height).
        foot_height = np.percentile(points[selected['vertical'].mask] @ up_camera, 5)
        selected['horizontal'] = min(choices['horizontal'], key=lambda plane:
            abs(np.median(points[plane.mask] @ up_camera) - foot_height))
    prompts = {}
    for role, other in (('horizontal', 'vertical'), ('vertical', 'horizontal')):
        positives = sample_interior(selected[role].mask)
        negatives = sample_interior(selected[other].mask)
        background = distance > high + .1
        if background.sum() >= 100:
            negatives = np.concatenate((negatives, sample_interior(background, 3)))
        prompts[role] = {'positive_xy_processed': positives, 'negative_xy_processed': negatives,
                         'equation': selected[role].equation, 'support_mask': selected[role].mask}
    return prompts, foreground, points, k


def resolve_plate_masks(raw, prompts, foreground, points):
    """Keep SAM support in range; assign overlaps to the closer predicted plane."""
    height, width = foreground.shape
    masks = {role: cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool) & foreground
             for role, mask in raw.items()}
    overlap = masks['horizontal'] & masks['vertical']
    if overlap.any():
        distances = {}
        for role in masks:
            equation = prompts[role]['equation']
            distances[role] = np.abs(points @ equation[:3] + equation[3])
        horizontal_wins = distances['horizontal'] <= distances['vertical']
        masks['horizontal'][overlap & ~horizontal_wins] = False
        masks['vertical'][overlap & horizontal_wins] = False
    for role, mask in masks.items():
        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        if count > 1:
            support = prompts[role]['support_mask']
            selected = max(range(1, count), key=lambda i: int((support & (labels == i)).sum()))
            masks[role] = labels == selected
    assert not (masks['horizontal'] & masks['vertical']).any()
    return masks
