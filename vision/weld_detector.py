"""Whole-image plate geometry and automatic weld/junction candidates.

No region annotations, reference points, filenames or weld-state labels are
inputs. Semantic masks are produced by fixed SAM category prompts. Geometry
alone establishes a possible plate contact, not the manufacturing intention.
"""

from dataclasses import dataclass
import heapq

import cv2
import numpy as np
from scipy.ndimage import binary_closing, label
from scipy.spatial import cKDTree

from .contour_detector import _thin_stroke
from .weld_toes import extract_weld_toes


@dataclass
class Plate:
    equation: np.ndarray
    mask: np.ndarray
    residual: float
    semantic_instance: int | None = None
    sampling_source: str = "geometry_fallback"


def camera_points(depth, intrinsics):
    y, x = np.indices(depth.shape)
    return np.stack((x, y, np.ones_like(x)), axis=-1) @ np.linalg.inv(intrinsics).T * depth[..., None]


def normals_from_depth(depth, intrinsics):
    points = camera_points(cv2.GaussianBlur(depth, (0, 0), 1.0), intrinsics)
    dy, dx = np.gradient(points, axis=(0, 1))
    normals = np.cross(dx, dy)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    normals[np.sum(normals * points, axis=-1) > 0] *= -1
    return normals.astype(np.float32)


def least_squares_plane(points):
    center = points.mean(axis=0)
    _, singular, vectors = np.linalg.svd(points - center, full_matrices=False)
    if len(singular) < 3 or singular[1] < 1e-8:
        raise ValueError("Degenerate plane samples")
    normal = vectors[-1]
    offset = -normal @ center
    if offset < 0:
        normal, offset = -normal, -offset
    return np.r_[normal, offset]


def extract_plates(depth, intrinsics, *, seed=27, max_passes=16,
                   plate_instances=(), exclude_mask=None, quality_mask=None):
    """Fit interior SAM samples, splitting each object into connected planar faces.

    Without SAM instances, retain an explicitly labelled geometry fallback.
    The sampling domain never grows outside an instance or into excluded beads.
    """
    if depth.ndim != 2 or min(depth.shape) < 20 or not np.isfinite(depth).all() or not (depth > 0).all():
        raise ValueError("Expected finite positive, aligned optical-Z depth")
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or np.linalg.det(intrinsics) <= 0:
        raise ValueError("Invalid intrinsics")
    points = camera_points(depth, intrinsics).reshape(-1, 3)
    normal_map = normals_from_depth(depth, intrinsics)
    normals = normal_map.reshape(-1, 3)
    eligible = np.ones(depth.shape, dtype=bool)
    eligible[:3] = eligible[-3:] = False
    eligible[:, :3] = eligible[:, -3:] = False
    if quality_mask is not None:
        eligible &= quality_mask
    if exclude_mask is not None:
        eligible &= ~cv2.dilate(exclude_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    domains = []
    if len(plate_instances):
        # Smaller objects own overlapping pixels before assembly/table masks.
        # Erosion removes mixed-depth boundaries and also shrinks hole edges.
        owned = np.zeros(depth.shape, dtype=bool)
        for index in sorted(range(len(plate_instances)), key=lambda k: plate_instances[k].sum()):
            interior = cv2.erode(plate_instances[index].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            domain = interior & ~owned & eligible
            domains.append((index, domain))
            owned |= interior
    else:
        domains.append((None, eligible))
    plates = []
    for instance, domain in domains:
        plates.extend(_peel_planes(points, normals, domain, seed + (instance or 0), max_passes, instance))
    labels = np.full(depth.shape, -1, dtype=np.int16)
    for index, plate in enumerate(plates):
        labels[plate.mask] = index
    return plates, labels, normal_map


def _peel_planes(points, normals, eligible, seed, max_passes, instance):
    depth_shape = eligible.shape
    remaining = np.flatnonzero(eligible.reshape(-1))
    rng = np.random.default_rng(seed)
    plates = []
    normal_cosine = np.cos(np.deg2rad(15))
    min_pixels = max(60, int(eligible.sum() * .01), int(eligible.size * .001))
    for _ in range(max_passes):
        if len(remaining) < min_pixels:
            break
        subset = remaining[rng.choice(len(remaining), min(12000, len(remaining)), replace=False)]
        sample, sample_normals = points[subset], normals[subset]
        best = np.zeros(len(subset), dtype=bool)
        # Local normals supply minimal hypotheses; consensus is over the whole
        # remaining image, followed by orthogonal least-squares refinement.
        for _ in range(400):
            index = int(rng.integers(len(subset)))
            normal = sample_normals[index]
            offset = -normal @ sample[index]
            inliers = ((np.abs(np.einsum("ij,j->i", sample, normal) + offset) < sample[:, 2] * 0.004)
                       & (np.einsum("ij,j->i", sample_normals, normal) > normal_cosine))
            if inliers.sum() > best.sum():
                best = inliers
        if best.sum() < max(60, len(subset) * 0.008):
            break
        selected = sample[best]
        degenerate = False
        for _ in range(3):
            try:
                equation = least_squares_plane(selected)
            except ValueError:
                remaining = remaining[~np.isin(remaining, subset[best])]
                degenerate = True
                break
            distance = np.abs(np.einsum("ij,j->i", points[remaining], equation[:3]) + equation[3])
            same_normal = np.einsum("ij,j->i", normals[remaining], equation[:3]) > normal_cosine
            inliers = (distance < points[remaining, 2] * 0.004) & same_normal
            selected = points[remaining[inliers]]
            if len(selected) < min_pixels:
                break
        if degenerate:
            continue
        if len(selected) < min_pixels:
            break
        mask = np.zeros(eligible.size, dtype=np.uint8)
        mask[remaining[inliers]] = 1
        remaining = remaining[~inliers]
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask.reshape(depth_shape), connectivity=8)
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] < min_pixels:
                continue
            region = components == component
            yx = np.argwhere(region)
            minor_radius = float(np.sqrt(max(0, np.linalg.eigvalsh(np.cov(yx.T))[0])))
            if minor_radius < max(2.5, min(depth_shape) * .008):
                continue  # Thin plate side faces and image borders are not host plate surfaces.
            local_points = points[region.reshape(-1)]
            local_equation = least_squares_plane(local_points)
            residual = float(np.sqrt(np.mean((local_points @ local_equation[:3] + local_equation[3]) ** 2)))
            plates.append(Plate(local_equation, region, residual, instance,
                                "sam_interior" if instance is not None else "geometry_fallback"))
    return plates


def remap(array, xy):
    return cv2.remap(array.astype(np.float32), xy[..., 0].astype(np.float32),
                     xy[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def intersection_line(a, b, intrinsics):
    angle = float(np.rad2deg(np.arccos(np.clip(abs(a[:3] @ b[:3]), 0, 1))))
    if angle < 25:
        return None
    line = np.linalg.inv(intrinsics).T @ (b[3] * a[:3] - a[3] * b[:3])
    if np.linalg.norm(line[:2]) < 1e-10:
        return None
    return line / np.linalg.norm(line[:2]), angle


def intersection_geometry(a, b):
    """Point nearest the camera origin and unit direction of a 3D plane line."""
    direction = np.cross(a[:3], b[:3])
    norm = np.linalg.norm(direction)
    if norm < 1e-5:
        raise ValueError("Parallel planes have no unique intersection line")
    direction /= norm
    origin = np.linalg.solve(np.vstack((a[:3], b[:3], direction)), [-a[3], -b[3], 0.])
    return origin, direction


def refine_contact_curve(xy, line, normals):
    """Snap an analytic contact to a continuous local normal-change ridge."""
    offsets = np.arange(-8, 9, dtype=float)
    strip = xy[None] + offsets[:, None, None] * line[:2]
    before = remap(normals, strip - 3 * line[:2])
    after = remap(normals, strip + 3 * line[:2])
    before /= np.maximum(np.linalg.norm(before, axis=-1, keepdims=True), 1e-8)
    after /= np.maximum(np.linalg.norm(after, axis=-1, keepdims=True), 1e-8)
    angle = np.arccos(np.clip(np.sum(before * after, axis=-1), -1, 1))
    valid = ((strip[..., 0] >= 3) & (strip[..., 0] < normals.shape[1]-3)
             & (strip[..., 1] >= 3) & (strip[..., 1] < normals.shape[0]-3))
    if not valid.any(axis=0).all():
        return xy
    score = angle / max(float(np.percentile(angle[valid], 95)), 1e-6) - .25 * (offsets[:, None] / 5) ** 2
    score = np.where(valid, score, -1e9)
    previous = score[:, 0].copy()
    parents = np.zeros(score.shape, dtype=np.int16)
    rows = np.arange(len(offsets))
    for column in range(1, score.shape[1]):
        alternatives = []
        for shift in range(-2, 3):
            source = rows + shift
            alternatives.append(np.where((source >= 0) & (source < len(rows)),
                                         previous[np.clip(source, 0, len(rows)-1)] - .12 * abs(shift), -1e9))
        alternatives = np.stack(alternatives)
        choice = alternatives.argmax(axis=0)
        parents[:, column] = rows + choice - 2
        previous = score[:, column] + alternatives[choice, rows]
    path = [int(previous.argmax())]
    for column in range(score.shape[1]-1, 0, -1):
        path.append(int(parents[path[-1], column]))
    return strip[np.array(path[::-1]), np.arange(len(xy))]


def joint_candidates(depth, intrinsics, plates, plate_semantics=None, plate_instances=(), normals=None):
    """Retain finite contact intervals with nearby support on opposite sides."""
    height, width = depth.shape
    gap = min(28., max(12., min(depth.shape) * 0.065))
    corners = np.array([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])
    distances = [cv2.distanceTransform((~plate.mask).astype(np.uint8), cv2.DIST_L2, 5) for plate in plates]
    pixel_positions = [np.argwhere(plate.mask)[:, ::-1].astype(float) for plate in plates]
    points = camera_points(depth, intrinsics)
    candidates = []
    for i, a in enumerate(plates):
        for j in range(i + 1, len(plates)):
            b = plates[j]
            projection = intersection_line(a.equation, b.equation, intrinsics)
            if projection is None:
                continue
            line, angle = projection
            tangent = np.array([-line[1], line[0]])
            if tangent[0] < 0:
                tangent *= -1
            origin = -line[2] * line[:2]
            s = corners @ tangent
            xy = origin + np.arange(s.min(), s.max() + 1)[:, None] * tangent
            inside = ((xy[:, 0] > 3) & (xy[:, 0] < width-4) & (xy[:, 1] > 3) & (xy[:, 1] < height-4))
            if inside.sum() < 20:
                continue
            xy = xy[inside]
            da = remap(distances[i], xy[None])[0]
            db = remap(distances[j], xy[None])[0]
            support = (da < gap) & (db < gap) & (da + db < gap * 1.5)
            if support.sum() < 20:
                continue
            near_a = pixel_positions[i] @ line[:2] + line[2]
            near_b = pixel_positions[j] @ line[:2] + line[2]
            near_a = near_a[np.abs(near_a) < gap * 1.5]
            near_b = near_b[np.abs(near_b) < gap * 1.5]
            if not len(near_a) or not len(near_b) or np.median(near_a) * np.median(near_b) >= 0:
                continue
            rays = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(intrinsics).T
            denominator = rays @ a.equation[:3]
            expected_z = np.divide(-a.equation[3], denominator, out=np.full(len(xy), -1.),
                                   where=np.abs(denominator) > 1e-8)
            actual_z = remap(depth, xy[None])[0]
            error = np.abs(expected_z - actual_z) / np.maximum(actual_z, 1e-6)
            support &= (expected_z > 0) & (error < .035)
            # Close only short gaps caused by texture/noisy depth.
            support = binary_closing(support, structure=np.ones(5)) & (expected_z > 0)
            labels, count = label(support)
            for component in range(1, count + 1):
                selected = np.flatnonzero(labels == component)
                if len(selected) < max(20, max(depth.shape) * .06):
                    continue
                segment = xy[selected]
                local_mask = np.zeros(depth.shape, dtype=np.uint8)
                cv2.polylines(local_mask, [np.rint(segment).astype(np.int32)], False, 1, round(gap * 2))
                semantic_support = None
                # With both normals facing the camera, visible concave contacts
                # put each face on the positive side of the other face's plane.
                # Convex plate outlines and table front edges have the opposite sign.
                signed_contact = []
                for plate, other in ((a, b), (b, a)):
                    local = plate.mask & local_mask.astype(bool)
                    signed_contact.append(float(np.median(
                        points[local] @ other.equation[:3] + other.equation[3])) if local.any() else -1.)
                noise = max(a.residual, b.residual, float(np.median(actual_z[selected])) * .0005)
                if min(signed_contact) < noise:
                    continue
                if plate_semantics is not None and plate_semantics.any():
                    supports = []
                    for plate in (a, b):
                        local = plate.mask & local_mask.astype(bool)
                        supports.append(float(plate_semantics[local].mean()) if local.any() else 0.)
                    semantic_support = supports
                    if min(supports) < .12 or max(supports) < .45:
                        continue
                assigned_instances = []
                if len(plate_instances) >= 2:
                    for plate in (a, b):
                        local = plate.mask & local_mask.astype(bool)
                        coverage = np.array([instance[local].mean() if local.any() else 0.
                                             for instance in plate_instances])
                        eligible = np.flatnonzero(coverage >= .55)
                        if not len(eligible):
                            break
                        # Prefer the smallest object that explains this face,
                        # rather than an encompassing assembly/table mask.
                        assigned_instances.append(int(min(eligible, key=lambda k: plate_instances[k].sum())))
                    if len(assigned_instances) != 2 or assigned_instances[0] == assigned_instances[1]:
                        continue
                    first, second = [plate_instances[k] for k in assigned_instances]
                    nested = np.logical_and(first, second).sum() / max(1, min(first.sum(), second.sum()))
                    if nested > .8:
                        continue  # Two visible faces of the same semantic object.
                rank = (len(segment) / np.sqrt(depth.size) * np.sin(np.deg2rad(angle))
                        * max(.1, 1 - float(np.median(error[selected])) / .035))
                refined = refine_contact_curve(segment, line, normals) if normals is not None else segment
                # The result stays on BOTH fitted planes. Normal snapping is
                # an image observation used to check the fit, not a new depth.
                xyz = rays[selected] * expected_z[selected, None]
                origin_3d, direction_3d = intersection_geometry(a.equation, b.equation)
                candidates.append({"planes": [i, j], "line": line, "xy": segment,
                                   "observed_xy": refined, "xyz_camera_estimated_m": xyz,
                                   "origin_camera_estimated_m": origin_3d,
                                   "direction_camera": direction_3d,
                                   "relative_depth_disagreement": error[selected],
                                   "normal_snap_distance_processed_px": np.linalg.norm(refined-segment, axis=1),
                                   "geometry_supported": ((error[selected] < .035) & (expected_z[selected] > 0)
                                                          & (np.linalg.norm(refined-segment, axis=1) <= 5)),
                                   "acute_angle_deg": angle, "rank": rank,
                                   "median_relative_depth_disagreement": float(np.median(error[selected])),
                                   "concavity_signed_distances_estimated_m": signed_contact,
                                   "plate_semantic_support": semantic_support})
    # Merge duplicate detections from fragmented fits of the same physical face.
    kept = []
    for candidate in sorted(candidates, key=lambda item: item["rank"], reverse=True):
        duplicate = False
        for previous in kept:
            if abs(candidate["line"][:2] @ previous["line"][:2]) < np.cos(np.deg2rad(10)):
                continue
            distance = cKDTree(previous["xy"]).query(candidate["xy"])[0]
            if np.mean(distance < 7) > .6:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def split_joint_states(joint, welds, tentative_mask, quality_mask):
    """Label contiguous intervals locally; absence of a mask is not an unwelded label."""
    xy = joint["xy"]
    hits = np.zeros((len(welds), len(xy)), dtype=bool)
    for index, weld in enumerate(welds):
        band = cv2.dilate(weld["mask"].astype(np.uint8), np.ones((7, 7), np.uint8))
        hits[index] = remap(band, xy[None])[0] > .5
    covered = hits.any(axis=0) if len(hits) else np.zeros(len(xy), dtype=bool)
    weak = remap(cv2.dilate(tentative_mask.astype(np.uint8), np.ones((5, 5), np.uint8)), xy[None])[0] > .5
    states = np.where(covered, 1, np.where(weak, 2, 0))
    quality = remap(quality_mask.astype(np.float32), xy[None])[0] >= .99
    cuts = np.r_[0, np.flatnonzero(np.diff(states)) + 1, len(states)]
    segments = []
    for first, last in zip(cuts[:-1], cuts[1:]):
        state = ("no_weld_detected", "weld_detected", "uncertain_weld_evidence")[states[first]]
        segments.append({"start_sample": int(first), "stop_sample": int(last), "state": state,
                         "weld_state": "welded_candidate" if state == "weld_detected" else "unknown",
                         "branch": "existing_bead_edges" if state == "weld_detected" else "plate_intersection",
                         "weld_indices": np.flatnonzero(hits[:, first:last].any(axis=1)).tolist(),
                         "geometry_supported": joint["geometry_supported"][first:last] & quality[first:last]})
    joint["segments"] = segments
    joint["overlaps_weld"] = bool(covered.mean() > .4)  # Diagnostic only; never suppress a whole joint.
    return joint


def depth_quality(depth, confidence):
    """Reject invalid and lowest-decile confidence samples; scores are uncalibrated."""
    if confidence is None:
        return np.ones(depth.shape, dtype=bool), None
    confidence = np.asarray(confidence)
    if confidence.shape != depth.shape:
        raise ValueError("Depth confidence must match the processed depth resolution")
    finite = np.isfinite(confidence)
    if not finite.any():
        return finite, None
    threshold = float(np.percentile(confidence[finite], 10))
    return finite & (confidence >= threshold), threshold


def mask_centerline(mask):
    """Main geodesic path of a thinned mask; bent paths keep their bends."""
    skeleton = _thin_stroke(mask.astype(np.uint8))
    yx = np.argwhere(skeleton)
    if len(yx) < 2:
        return np.empty((0, 2), dtype=float)
    lookup = {tuple(point): i for i, point in enumerate(yx)}
    graph = [[] for _ in yx]
    for index, (y, x) in enumerate(yx):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx):
                    continue
                neighbour = lookup.get((y + dy, x + dx))
                if neighbour is not None:
                    graph[index].append((neighbour, float(np.hypot(dy, dx))))

    def furthest(start):
        distance = np.full(len(graph), np.inf)
        parent = np.full(len(graph), -1, dtype=int)
        distance[start] = 0
        heap = [(0., start)]
        while heap:
            cost, vertex = heapq.heappop(heap)
            if cost != distance[vertex]:
                continue
            for neighbour, length in graph[vertex]:
                next_cost = cost + length
                if next_cost < distance[neighbour]:
                    distance[neighbour] = next_cost
                    parent[neighbour] = vertex
                    heapq.heappush(heap, (next_cost, neighbour))
        return int(np.argmax(np.where(np.isfinite(distance), distance, -1))), parent

    first, _ = furthest(0)
    last, parent = furthest(first)
    ordered = [last]
    while ordered[-1] != first and parent[ordered[-1]] >= 0:
        ordered.append(int(parent[ordered[-1]]))
    return yx[ordered, ::-1].astype(float)


def welded_regions(masks, scores, shape):
    regions = []
    for mask, score in zip(masks, scores):
        resized = cv2.resize(mask.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST)
        count, components, stats, _ = cv2.connectedComponentsWithStats(resized, connectivity=8)
        for index in range(1, count):
            area = stats[index, cv2.CC_STAT_AREA]
            if area < max(18, np.prod(shape) * .00012):
                continue
            region = components == index
            yx = np.argwhere(region)
            eigenvalues = np.linalg.eigvalsh(np.cov(yx.T))
            elongation = float(np.sqrt(eigenvalues[-1] / max(eigenvalues[0], .1)))
            # Compact bolts/tack spots should not be promoted to long weld seams.
            if elongation < 2 or np.sqrt(eigenvalues[-1]) < max(shape) * .018:
                continue
            if any(np.logical_and(region, previous["mask"]).sum() /
                   max(1, np.logical_or(region, previous["mask"]).sum()) > .65 for previous in regions):
                continue
            contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            regions.append({"mask": region, "score": float(score), "centerline": mask_centerline(region),
                            "boundaries": [contour[:, 0].astype(float) for contour in contours],
                            "elongation": elongation})
    return regions


def detect(depth, intrinsics, rgb, *, plate_masks=(), weld_masks=(), weld_scores=(), confidence=None):
    if rgb.shape != (*depth.shape, 3):
        raise ValueError("RGB must match depth resolution")
    instances, source_indices = [], []
    for source_index, mask in enumerate(plate_masks):
        resized = cv2.resize(mask.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        if resized.any():
            instances.append(resized)
            source_indices.append(source_index)
    interior, interior_indices = [], []
    margin = max(3, round(min(depth.shape) * .02))
    for index, instance in enumerate(instances):
        yy, xx = np.nonzero(instance)
        if xx.min() >= margin and xx.max() < depth.shape[1]-margin and yy.min() >= margin and yy.max() < depth.shape[0]-margin:
            interior.append(instance)
            interior_indices.append(source_indices[index])
    # When complete objects are available, exclude masks clipped by the frame
    # (usually the workbench). Cropped-object scenes retain a geometry fallback.
    if len(interior) >= 2:
        instances = interior
        source_indices = interior_indices
    semantic_plate = np.logical_or.reduce(instances) if instances else np.zeros(depth.shape, dtype=bool)
    if len(weld_masks) != len(weld_scores):
        raise ValueError("Every SAM weld mask needs a detection score")
    tentative = np.zeros(depth.shape, dtype=bool)
    for mask, score in zip(weld_masks, weld_scores):
        if score >= .25:
            tentative |= cv2.resize(mask.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    quality, confidence_threshold = depth_quality(depth, confidence)
    plates, labels, normals = extract_plates(depth, intrinsics, plate_instances=instances,
                                             exclude_mask=tentative, quality_mask=quality)
    joints = joint_candidates(depth, intrinsics, plates, semantic_plate, instances, normals=normals)
    welds = welded_regions(weld_masks, weld_scores, depth.shape)
    contact_band = np.zeros(depth.shape, dtype=np.uint8)
    for joint in joints:
        cv2.polylines(contact_band, [np.rint(joint["xy"]).astype(np.int32)], False, 1,
                      max(9, round(min(depth.shape) * .07)))
    accepted_welds = []
    for weld in welds:
        support = float(contact_band[weld["mask"]].mean())
        weld["contact_support"] = support
        # Weak text matches on scratches/table edges need geometric evidence.
        if (support >= .2 and weld["score"] >= .4) or weld["score"] >= .65:
            accepted_welds.append(weld)
    welds = accepted_welds
    for weld in welds:
        weld["toe_paths"] = extract_weld_toes(weld["mask"], weld["centerline"], depth,
                                              intrinsics, rgb, normals, plates, quality_mask=quality)
    for joint in joints:
        split_joint_states(joint, welds, tentative, quality)
    return {"plates": plates, "labels": labels, "normals": normals, "joints": joints,
            "welds": welds, "semantic_plate_mask": semantic_plate,
            "depth_quality_mask": quality, "confidence_threshold": confidence_threshold,
            "plate_sampling_mode": "sam_interiors" if instances else "geometry_fallback_no_sam_plate",
            "plate_instance_indices": source_indices,
            "plate_instance_count": len(instances), "interior_plate_instance_count": len(interior)}
