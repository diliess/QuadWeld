"""Detect a dark closed outline from RGB; no synthetic rectangle coordinates."""

import cv2
import numpy as np


def detect_contour(image_rgb, *, threshold=80, min_area=300.0):
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            area = abs(cv2.contourArea(contour))
            x, y, w, h = cv2.boundingRect(contour)
            child = hierarchy[0, i, 2]
            # A closed dark stroke surrounds a bright interior; reject shadows and image edges.
            if (area >= min_area and child >= 0 and x > 1 and y > 1
                    and x + w < gray.shape[1] - 1 and y + h < gray.shape[0] - 1
                    and abs(cv2.contourArea(contours[child])) / area > 0.5):
                candidates.append((area, contour))
    if not candidates:
        raise RuntimeError("No closed dark outline found; check RGB image, lighting, occlusion and threshold")
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[1][0] > 0.5 * candidates[0][0]:
        raise RuntimeError("Multiple plausible outlines; target selection is ambiguous")
    # Follow the outer edge of the visible stroke, not its centreline.
    polygon = cv2.approxPolyDP(candidates[0][1], 1.0, True).reshape(-1, 2).astype(float)
    if len(polygon) < 3:
        raise RuntimeError("Detected outline is degenerate")
    # Positive signed area in image coordinates is clockwise.
    area2 = np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1))
    if area2 < 0:
        polygon = polygon[::-1]
    # Start at the vertex nearest the image-space top-left bounding-box corner.
    start = np.argmin(np.linalg.norm(polygon - polygon.min(axis=0), axis=1))
    return np.roll(polygon, -start, axis=0), mask


def save_overlay(image_rgb, polygon, path):
    overlay = image_rgb.copy()
    points = np.rint(polygon).astype(np.int32)
    cv2.polylines(overlay, [points], True, (0, 220, 80), 2)
    cv2.circle(overlay, tuple(points[0]), 5, (255, 50, 0), -1)
    cv2.arrowedLine(overlay, tuple(points[0]), tuple(points[1]), (255, 50, 0), 2, tipLength=0.1)
    if not cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save {path}")
