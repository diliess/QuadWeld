"""Find one ordered dark path in RGB, preserving whether it is open or closed."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectedPath:
    points: np.ndarray
    closed: bool
    mask: np.ndarray
    target_mask: np.ndarray


def _dark_mask(image_rgb, threshold):
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]


def _closed_outline(mask, min_area):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            area = abs(cv2.contourArea(contour))
            x, y, w, h = cv2.boundingRect(contour)
            child = hierarchy[0, i, 2]
            # A closed stroke encloses a bright region; border-touching regions do not qualify.
            if (area >= min_area and child >= 0 and x > 1 and y > 1
                    and x + w < mask.shape[1] - 1 and y + h < mask.shape[0] - 1
                    and abs(cv2.contourArea(contours[child])) / area > 0.5):
                candidates.append((area, contour))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[1][0] > 0.5 * candidates[0][0]:
        raise RuntimeError("Multiple plausible outlines; target selection is ambiguous")
    # Closed paths retain the visible stroke's outer edge used by the original pipeline.
    polygon = cv2.approxPolyDP(candidates[0][1], 1.0, True).reshape(-1, 2).astype(float)
    if len(polygon) < 3:
        raise RuntimeError("Detected outline is degenerate")
    area2 = np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1))
    if area2 < 0:
        polygon = polygon[::-1]
    start = np.argmin(np.linalg.norm(polygon - polygon.min(axis=0), axis=1))
    return np.roll(polygon, -start, axis=0)


def _thin_stroke(stroke):
    """Zhang-Suen thinning on a cropped binary stroke; no contrib OpenCV dependency."""
    image = np.pad((stroke > 0).astype(np.uint8), 1)
    for _ in range(128):
        changed = False
        for phase in (0, 1):
            center = image[1:-1, 1:-1]
            p2, p3, p4 = image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:]
            p5, p6, p7 = image[2:, 2:], image[2:, 1:-1], image[2:, :-2]
            p8, p9 = image[1:-1, :-2], image[:-2, :-2]
            neighbours = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(neighbours)
            transitions = np.zeros_like(center)
            for first, second in zip(neighbours, neighbours[1:] + neighbours[:1]):
                transitions += ((first == 0) & (second == 1)).astype(np.uint8)
            if phase == 0:
                preserve = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                preserve = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (center == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & preserve
            if np.any(remove):
                center[remove] = 0
                changed = True
        if not changed:
            break
    return image[1:-1, 1:-1]


def _ordered_open_centerline(component, x, y, w, h):
    skeleton = _thin_stroke(component[y:y + h, x:x + w])
    local_y, local_x = np.nonzero(skeleton)
    pixels = {(int(row + y), int(col + x)) for row, col in zip(local_y, local_x)}
    if len(pixels) < 2:
        return None
    neighbours = {}
    for row, col in pixels:
        adjacent = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0 or (row + dr, col + dc) not in pixels:
                    continue
                # A diagonal beside an orthogonal skeleton edge is a corner shortcut.
                if dr and dc and ((row + dr, col) in pixels or (row, col + dc) in pixels):
                    continue
                adjacent.append((row + dr, col + dc))
        neighbours[row, col] = adjacent
    endpoints = sorted(pixel for pixel, adjacent in neighbours.items() if len(adjacent) == 1)
    if len(endpoints) != 2 or any(len(adjacent) > 2 for adjacent in neighbours.values()):
        return None  # Branches and loops have no unambiguous endpoint-to-endpoint order.
    ordered = [endpoints[0]]
    previous = None
    while True:
        choices = [pixel for pixel in neighbours[ordered[-1]] if pixel != previous]
        if not choices:
            break
        previous, next_pixel = ordered[-1], choices[0]
        ordered.append(next_pixel)
    if len(ordered) != len(pixels) or ordered[-1] != endpoints[1]:
        return None
    return np.array([(col, row) for row, col in ordered], dtype=np.float32)


def _open_stroke(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for label in range(1, count):
        x, y, w, h, pixels = map(int, stats[label])
        if (pixels < max(40, min_area / 4) or x <= 1 or y <= 1
                or x + w >= mask.shape[1] - 1 or y + h >= mask.shape[0] - 1):
            continue
        component = (labels == label).astype(np.uint8)
        line = _ordered_open_centerline(component, x, y, w, h)
        if line is None:
            continue
        length = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
        if length >= 20:
            candidates.append((length, line, component * 255))
    if not candidates:
        raise RuntimeError("No unbranched open dark stroke found; check visibility, endpoints and threshold")
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[1][0] > 0.5 * candidates[0][0]:
        raise RuntimeError("Multiple plausible open strokes; target selection is ambiguous")
    line = cv2.approxPolyDP(candidates[0][1].reshape(-1, 1, 2), 1.0, False).reshape(-1, 2).astype(float)
    return line, candidates[0][2]


def detect_path(image_rgb, *, threshold=80, min_area=300.0, mode="auto"):
    if mode not in ("auto", "closed", "open"):
        raise ValueError("Path mode must be auto, closed or open")
    mask = _dark_mask(image_rgb, threshold)
    if mode != "open":
        polygon = _closed_outline(mask, min_area)
        if polygon is not None:
            return DetectedPath(polygon, True, mask, np.zeros_like(mask))
        if mode == "closed":
            raise RuntimeError("No closed dark outline found; check RGB image, lighting, occlusion and threshold")
    line, component = _open_stroke(mask, min_area)
    return DetectedPath(line, False, mask, component)


def detect_contour(image_rgb, *, threshold=80, min_area=300.0):
    """Compatibility entry point for callers requiring a closed contour."""
    detected = detect_path(image_rgb, threshold=threshold, min_area=min_area, mode="closed")
    return detected.points, detected.mask


def save_overlay(image_rgb, points, path, *, closed=True):
    overlay = image_rgb.copy()
    pixels = np.rint(points).astype(np.int32)
    cv2.polylines(overlay, [pixels], closed, (0, 220, 80), 2)
    cv2.circle(overlay, tuple(pixels[0]), 5, (255, 50, 0), -1)
    if not closed:
        cv2.circle(overlay, tuple(pixels[-1]), 5, (0, 100, 255), -1)
    cv2.arrowedLine(overlay, tuple(pixels[0]), tuple(pixels[1]), (255, 50, 0), 2, tipLength=0.1)
    if not cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save {path}")
