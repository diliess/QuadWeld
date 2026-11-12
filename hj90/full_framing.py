"""Smallest stereo standoff that frames complete workpiece geometry."""

import numpy as np

from .overview_planning import framing, look_at, transform_points


def fit_stereo_distance(fixture, target, vertices, intrinsics, size, elevation_deg,
                        azimuth_deg=0., baseline=.09, margin=.08):
    if not 0 <= margin < .4 or baseline <= 0:
        raise ValueError('Invalid stereo framing margin or baseline')
    elevation, azimuth = np.radians([elevation_deg, azimuth_deg])
    direction = np.array([np.cos(elevation) * np.sin(azimuth),
                          -np.cos(elevation) * np.cos(azimuth), np.sin(elevation)])
    world_target = transform_points(fixture, np.asarray(target)[None])[0]
    world_vertices = transform_points(fixture, vertices)

    def evaluate(distance):
        eye = transform_points(fixture, (target + direction * distance)[None])[0]
        pose = look_at(eye, world_target)
        fraction, boxes = framing(pose, world_vertices, intrinsics, size, baseline, margin)
        return fraction == 1., pose, boxes

    low, high = .1, 1.
    while not evaluate(high)[0]:
        high *= 1.5
        if high > 10:
            raise ValueError('Cannot frame workpiece within 10 metres')
    for _ in range(45):
        middle = (low + high) / 2
        if evaluate(middle)[0]:
            high = middle
        else:
            low = middle
    distance = float(np.ceil(high * 100) / 100)  # Round outward to centimetres.
    valid, pose, boxes = evaluate(distance)
    if not valid:
        raise ValueError('Rounded full-frame distance is invalid')
    return distance, pose, boxes
