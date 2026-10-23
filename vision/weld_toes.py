"""Two open weld-toe candidates, seeded by a bead mask and refined with depth.

Image coordinates use the processed RGB/depth resolution. DA3 camera points
are estimates, not calibrated robot coordinates. The two side names are local
to an ordered bead centerline; they do not identify physical sides across views.
"""

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from .weld_surfaces import continuous_support, local_host_surface


def _sample(array, xy):
    return cv2.remap(array.astype(np.float32), xy[..., 0].astype(np.float32),
                     xy[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-10)


def _xyz(xy, depth, intrinsics):
    pixels = np.concatenate((xy, np.ones((*xy.shape[:-1], 1))), axis=-1)
    return (pixels @ np.linalg.inv(intrinsics).T) * depth[..., None]


def _frame(centerline):
    curve = np.asarray(centerline, dtype=float)
    if len(curve) < 8:
        return None
    if tuple(curve[0, ::-1]) > tuple(curve[-1, ::-1]):
        curve = curve[::-1]
    lengths = np.r_[0, np.linalg.norm(np.diff(curve, axis=0), axis=1).cumsum()]
    keep = np.r_[True, np.diff(lengths) > 1e-6]
    curve, lengths = curve[keep], lengths[keep]
    if lengths[-1] < 12:
        return None
    stations = np.arange(1.5, lengths[-1] - 1.5, 1.5)
    centers = np.column_stack([np.interp(stations, lengths, curve[:, axis]) for axis in range(2)])
    window = min(9, len(centers) if len(centers) % 2 else len(centers)-1)
    smooth = savgol_filter(centers, window, 2, axis=0) if window >= 5 else centers
    tangent = _unit(np.gradient(smooth, axis=0))
    across = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return centers, across, stations


def _seed_edges(mask, centers, across):
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    radius = _sample(distance, centers[None])[0]
    extent = min(max(10., np.percentile(radius, 95) * 2.5 + 6), min(mask.shape) * .25)
    offsets = np.arange(-np.ceil(extent * 2), np.ceil(extent * 2) + 1) * .5
    strip = centers[None] + offsets[:, None, None] * across[None]
    inside = _sample(mask.astype(np.float32), strip) >= .5
    middle = int(np.argmin(abs(offsets)))
    lower = np.full(len(centers), np.nan)
    upper = lower.copy()
    valid = inside[middle].copy()
    for column in range(len(centers)):
        if not valid[column]:
            continue
        a = b = middle
        while a > 0 and inside[a-1, column]:
            a -= 1
        while b < len(offsets)-1 and inside[b+1, column]:
            b += 1
        if a == 0 or b == len(offsets)-1:
            valid[column] = False
            continue
        lower[column] = offsets[a] - .25
        upper[column] = offsets[b] + .25
    valid &= np.isfinite(lower) & np.isfinite(upper) & (upper-lower >= 3)
    return lower, upper, valid


def _runs(valid, minimum=8):
    padded = np.r_[False, valid, False].astype(int)
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b-a >= minimum]


def _fit_flank(centers, across, seed, sign, depth, intrinsics, normals, plates, quality_mask):
    """Select an observed mother-material face outside this mask edge."""
    offsets = seed[None] + sign * np.array([3., 4., 5., 6., 8.])[:, None]
    xy = centers[None] + offsets[..., None] * across[None]
    inside = ((xy[..., 0] >= 2) & (xy[..., 0] < depth.shape[1]-2)
              & (xy[..., 1] >= 2) & (xy[..., 1] < depth.shape[0]-2)
              & (_sample(quality_mask, xy) >= .99))
    if inside.sum() < 30:
        return None
    point_map = _xyz(xy, _sample(depth, xy), intrinsics)
    flank_points = point_map[inside]
    best = None
    for index, plate in enumerate(plates):
        support = _sample(plate.mask.astype(np.float32), xy)[inside] > .5
        fraction = float(support.mean())
        if support.sum() < 20 or fraction < .15:
            continue
        distances = np.abs(flank_points @ plate.equation[:3] + plate.equation[3])
        agreement = float(np.mean(distances < max(plate.residual * 3, np.median(flank_points[:, 2]) * .003)))
        score = fraction * agreement
        if best is None or score > best[0]:
            best = (score, {"plane_id": index, "equation": plate.equation.copy(),
                            "fit_rmse_estimated_m": plate.residual, "flank_support_fraction": fraction})
    if best is not None and best[0] >= .15:
        return best[1]
    # A global plane can be fragmented or absent; robustly refit the visible flank.
    flank_normals = _sample(normals, xy)[inside]
    reference = _unit(np.median(flank_normals, axis=0))
    selected = np.sum(flank_normals * reference, axis=1) > np.cos(np.deg2rad(20))
    for _ in range(4):
        if selected.sum() < 30:
            return None
        points = flank_points[selected]
        center = np.median(points, axis=0)
        _, singular, vectors = np.linalg.svd(points-center, full_matrices=False)
        if singular[1] < 1e-7:
            return None
        normal = vectors[-1]
        offset = -normal @ center
        if offset < 0:
            normal, offset = -normal, -offset
        distances = np.abs(flank_points @ normal + offset)
        selected = distances < np.median(flank_points[:, 2]) * .004
    rmse = float(np.sqrt(np.mean(distances[selected] ** 2))) if selected.any() else np.inf
    if selected.mean() < .5 or not np.isfinite(rmse):
        return None
    return {"plane_id": None, "equation": np.r_[normal, offset],
            "fit_rmse_estimated_m": rmse, "flank_support_fraction": float(selected.mean())}


def _normalize(values, valid):
    sample = values[valid]
    if not len(sample):
        return np.zeros_like(values)
    high = float(np.percentile(sample, 95))
    return np.clip(values / max(high, 1e-7), 0, 1)


def _continuous_path(score, positions, valid):
    """Optimize a side independently, penalizing transverse changes in pixels."""
    if not valid.any(axis=0).all():
        return None
    score = np.where(valid, score, -1e9)
    previous = score[:, 0].copy()
    rows = np.arange(len(score))
    parents = np.zeros(score.shape, dtype=np.int16)
    for column in range(1, score.shape[1]):
        # A small candidate grid permits an explicit absolute-offset penalty.
        movement = np.abs(positions[:, column, None] - positions[None, :, column-1])
        alternatives = previous[None] - .12 * movement
        alternatives[movement > 3.] = -1e9
        choice = alternatives.argmax(axis=1)
        parents[:, column] = choice
        previous = score[:, column] + alternatives[rows, choice]
    last = int(previous.argmax())
    if previous[last] < -1e8:
        return None
    path = [last]
    for column in range(score.shape[1]-1, 0, -1):
        path.append(int(parents[path[-1], column]))
    path = np.array(path[::-1])
    if not valid[path, np.arange(score.shape[1])].all():
        return None
    return path


def _side(centers, across, seed, sign, depth, intrinsics, normals, rgb, plates, radius, probe, quality_mask):
    seed = median_filter(seed, size=5, mode="nearest")
    deviations = np.arange(-radius * 2, radius * 2 + 1) * .5
    positions = seed[None] + deviations[:, None]
    xy = centers[None] + positions[..., None] * across[None]
    valid = ((xy[..., 0] >= probe+1) & (xy[..., 0] < depth.shape[1]-probe-1)
             & (xy[..., 1] >= probe+1) & (xy[..., 1] < depth.shape[0]-probe-1)
             & (sign * positions >= 1.))
    before = _unit(_sample(normals, xy - probe * across[None]))
    after = _unit(_sample(normals, xy + probe * across[None]))
    contrast = np.rad2deg(np.arccos(np.clip(np.sum(before * after, axis=-1), -1, 1)))
    color_before = _sample(rgb, xy - across[None])
    color_after = _sample(rgb, xy + across[None])
    color_gradient = np.linalg.norm(color_after-color_before, axis=-1)
    plane = _fit_flank(centers, across, seed, sign, depth, intrinsics, normals, plates, quality_mask)
    plane_gradient = np.zeros_like(contrast)
    if plane is not None:
        equation = plane["equation"]
        inner_xy, outer_xy = xy-sign*across[None], xy+sign*across[None]
        inner = _xyz(inner_xy, _sample(depth, inner_xy), intrinsics)
        outer = _xyz(outer_xy, _sample(depth, outer_xy), intrinsics)
        plane_gradient = np.maximum(np.abs(inner @ equation[:3]+equation[3]) -
                                    np.abs(outer @ equation[:3]+equation[3]), 0)
    prior = (deviations[:, None] / max(radius, 1)) ** 2
    score = (.45 * _normalize(contrast, valid) + .2 * _normalize(color_gradient, valid)
             + .2 * _normalize(plane_gradient, valid) - .65 * prior)
    path = _continuous_path(score, positions, valid)
    if path is None:
        return None
    columns = np.arange(len(centers))
    curve = xy[path, columns]
    curve_depth = _sample(depth, curve[None])[0]
    points = _xyz(curve, curve_depth, intrinsics)
    contrast_selected = contrast[path, columns]
    plane_change = plane_gradient[path, columns]
    depth_supported = contrast_selected >= 5.
    projection = np.full_like(points, np.nan)
    projected_valid = np.zeros(len(points), dtype=bool)
    plane_distance = np.full(len(points), np.nan)
    if plane is not None:
        equation = plane["equation"]
        depth_supported |= plane_change > max(plane["fit_rmse_estimated_m"] * 3, float(np.median(curve_depth)) * 1e-4)
        rays = np.column_stack((curve, np.ones(len(curve)))) @ np.linalg.inv(intrinsics).T
        denominator = rays @ equation[:3]
        z = np.divide(-equation[3], denominator, out=np.full(len(curve), -1.), where=np.abs(denominator)>1e-8)
        projected_valid = (z > 0) & (np.abs(z-curve_depth)/curve_depth < .03)
        # Keep finite arrays; invalid projections retain the observed DA3 point,
        # and a separate mask records whether a plane-based estimate exists.
        projection = points.copy()
        projection[projected_valid] = rays[projected_valid] * z[projected_valid, None]
        plane_distance = np.abs(points @ equation[:3] + equation[3])
    else:
        projection = points.copy()
        plane_distance = np.zeros(len(points))
    surface = local_host_surface(curve, centers, across, seed, sign, depth, intrinsics, normals, quality_mask)
    # Use the mother plane where it agrees and the local samples do not justify
    # curvature. Curved neighbours and missing planes keep their local model.
    use_plane = projected_valid & (surface["kind"] != 3)
    surface["xyz"][use_plane] = projection[use_plane]
    surface["kind"][use_plane] = 1
    surface["valid"] |= use_plane
    if plane is not None:
        surface["fit_rmse_estimated_m"][use_plane] = plane["fit_rmse_estimated_m"]
        surface["validation_error_estimated_m"][use_plane] = np.abs(projection[use_plane, 2]-curve_depth[use_plane])
    depth_quality_valid = _sample(quality_mask, curve[None])[0] >= .99
    continuity_valid = continuous_support(surface["xyz"], surface["valid"] & depth_quality_valid)
    return {"xy": curve, "seed_xy": centers + seed[:, None]*across,
            "xyz_camera_estimated_m": points, "plane_xyz_camera_estimated_m": projection,
            "plane_projection_valid": projected_valid, "depth_supported": depth_supported,
            "surface_xyz_camera_estimated_m": surface["xyz"], "surface_valid": surface["valid"],
            "surface_kind": surface["kind"], "surface_fit_rmse_estimated_m": surface["fit_rmse_estimated_m"],
            "surface_validation_error_estimated_m": surface["validation_error_estimated_m"],
            "depth_quality_valid": depth_quality_valid,
            "surface_continuity_valid": continuity_valid,
            "geometry_supported": depth_supported & surface["valid"] & depth_quality_valid & continuity_valid,
            "normal_contrast_deg": contrast_selected, "plane_distance_estimated_m": plane_distance,
            "plane": plane, "offset_from_seed_processed_px": positions[path, columns]-seed,
            "cross_section": {"offsets": positions[:, len(columns)//2],
                              "normal_score": _normalize(contrast, valid)[:, len(columns)//2],
                              "rgb_score": _normalize(color_gradient, valid)[:, len(columns)//2],
                              "plane_score": _normalize(plane_gradient, valid)[:, len(columns)//2],
                              "selected_offset": float(positions[path[len(columns)//2], len(columns)//2]),
                              "seed_offset": float(seed[len(columns)//2])}}


def extract_weld_toes(mask, centerline, depth, intrinsics, rgb, normals, plates=(), *, quality_mask=None):
    """Return synchronized pairs of OPEN curves, splitting missing cross-sections."""
    if mask.shape != depth.shape or rgb.shape != (*depth.shape, 3) or normals.shape != rgb.shape:
        raise ValueError("Toe inputs must share the processed image resolution")
    if not np.isfinite(depth).all() or not (depth > 0).all():
        raise ValueError("Toe extraction needs finite positive depth")
    quality_mask = np.ones(depth.shape, dtype=bool) if quality_mask is None else np.asarray(quality_mask, dtype=bool)
    if quality_mask.shape != depth.shape:
        raise ValueError("Toe depth-quality mask must match depth resolution")
    frame = _frame(centerline)
    if frame is None:
        return {"status": "insufficient_length", "segments": []}
    centers, across, stations = frame
    seed_a, seed_b, valid = _seed_edges(mask, centers, across)
    median_width = float(np.nanmedian((seed_b-seed_a)[valid])) if valid.any() else 0.
    radius = float(np.clip(median_width * .25, 2, 5))
    radius = round(radius * 2) / 2
    probe = float(np.clip(median_width / 6, .75, 2))
    segments = []
    for first, last in _runs(valid):
        a = _side(centers[first:last], across[first:last], seed_a[first:last], -1,
                  depth, intrinsics, normals, rgb, plates, radius, probe, quality_mask)
        b = _side(centers[first:last], across[first:last], seed_b[first:last], 1,
                  depth, intrinsics, normals, rgb, plates, radius, probe, quality_mask)
        if a is None or b is None:
            continue
        separation = np.sum((b["xy"]-a["xy"]) * across[first:last], axis=1)
        if (separation < 2).any():
            continue
        segments.append({"side_a": a, "side_b": b, "stations_processed_px": stations[first:last],
                         "centerline_xy": centers[first:last], "across_xy": across[first:last],
                         "width_processed_px": separation})
    return {"status": "paired_candidates" if segments else "insufficient_boundary_support",
            "method": "SAM cross-section seeds + DA3 normal change + mother-plane residual + RGB edge + continuity",
            "search_radius_processed_px": radius, "normal_probe_processed_px": probe,
            "segments": segments}
