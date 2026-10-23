"""Test plane intersections and seam candidates using saved DA3 depth and RGB.

The CSV supplies plate sampling polygons and a broad junction search polygon,
all in original-image pixels. This is an assisted geometry experiment, not an
automatic workpiece detector. Manual ROIs use prior knowledge of the target
location; the reference-point check is not an independent detection evaluation.
SAM point prompts are not read by the fitting or curve extraction functions.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def backproject(depth, intrinsics):
    y, x = np.indices(depth.shape)
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    return (pixels @ np.linalg.inv(intrinsics).T) * depth[..., None]


def fit_plane(points, relative_threshold=0.005, seed=7):
    """RANSAC followed by orthogonal least squares; distances use depth units."""
    if len(points) < 30 or not np.isfinite(points).all():
        raise ValueError("Each plate needs at least 30 finite depth samples")
    rng = np.random.default_rng(seed)
    threshold = float(np.median(points[:, 2]) * relative_threshold)
    sample = points[rng.choice(len(points), min(len(points), 6000), replace=False)]
    best = np.zeros(len(sample), dtype=bool)
    for _ in range(400):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        inliers = np.abs(sample @ normal - normal @ a) < threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 30:
        raise ValueError("No supported plate plane")
    selected = sample[best]
    for _ in range(4):
        center = selected.mean(axis=0)
        _, singular, vectors = np.linalg.svd(selected - center, full_matrices=False)
        if singular[1] < 1e-8:
            raise ValueError("Plate samples are collinear")
        normal = vectors[-1]
        offset = -normal @ center
        inliers = np.abs(points @ normal + offset) < threshold
        if inliers.sum() < 30:
            raise ValueError("Plane refinement lost its support")
        selected = points[inliers]
    # Face the camera, consistent with the local normal convention below.
    if offset < 0:
        normal, offset = -normal, -offset
    residual = np.abs(points @ normal + offset)
    return np.r_[normal, offset], {
        "sample_count": len(points), "inlier_fraction": float(inliers.mean()),
        "ransac_threshold_estimated_m": threshold,
        "inlier_rmse_estimated_m": float(np.sqrt(np.mean(residual[inliers] ** 2))),
        "all_sample_p90_residual_estimated_m": float(np.percentile(residual, 90)),
    }


def project_intersection(plane_a, plane_b, intrinsics):
    """Return image line ax+by+c=0 and the corresponding 3D intersection."""
    a, da = plane_a[:3], plane_a[3]
    b, db = plane_b[:3], plane_b[3]
    direction = np.cross(a, b)
    length = np.linalg.norm(direction)
    if length < np.sin(np.deg2rad(5)):
        raise ValueError("Plate planes are almost parallel")
    direction /= length
    point = np.linalg.lstsq(np.stack((a, b)), -np.array([da, db]), rcond=None)[0]
    line = np.linalg.inv(intrinsics).T @ (db * a - da * b)
    if np.linalg.norm(line[:2]) < 1e-10:
        raise ValueError("Intersection has a degenerate image projection")
    return line / np.linalg.norm(line[:2]), point, direction


def surface_normals(depth, intrinsics):
    smooth_depth = cv2.GaussianBlur(depth, (0, 0), 1.0)
    points = backproject(smooth_depth, intrinsics)
    dy, dx = np.gradient(points, axis=(0, 1))
    normals = np.cross(dx, dy)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    normals[np.sum(normals * points, axis=-1) > 0] *= -1
    return normals.astype(np.float32)


def trace_path(scores, valid, penalty=0.12, max_step=2):
    """Find a continuous transverse offset at each longitudinal strip sample."""
    rows, columns = scores.shape
    if not valid.any(axis=0).all():
        raise ValueError("Search strip contains an empty cross-section")
    energy = np.where(valid, scores, -1e9)
    previous = energy[:, 0].copy()
    parents = np.zeros((rows, columns), dtype=np.int32)
    indices = np.arange(rows)
    for column in range(1, columns):
        alternatives = []
        for shift in range(-max_step, max_step + 1):
            source = indices + shift
            allowed = (source >= 0) & (source < rows)
            alternatives.append(np.where(
                allowed, previous[np.clip(source, 0, rows - 1)] - penalty * abs(shift), -1e9))
        alternatives = np.stack(alternatives)
        choice = alternatives.argmax(axis=0)
        parents[:, column] = indices + choice - max_step
        previous = energy[:, column] + alternatives[choice, indices]
    path = np.zeros(columns, dtype=np.int32)
    path[-1] = previous.argmax()
    if previous[path[-1]] < -1e8:
        raise ValueError("No continuous curve through the search polygon")
    for column in range(columns - 1, 0, -1):
        path[column - 1] = parents[path[column], column]
    if not valid[path, np.arange(columns)].all():
        raise ValueError("Curve escaped the search polygon")
    return path


def scale_pixels(xy, scale):
    return (np.asarray(xy, dtype=float) + 0.5) * scale - 0.5


def polygon_mask(polygon, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
    return mask.astype(bool)


def normalize_signal(signal, valid):
    values = signal[valid]
    low, high = np.percentile(values, [10, 95])
    return np.clip((signal - low) / max(float(high - low), 1e-8), 0, 1)


def make_strip(polygons, shape):
    polygon = polygons["search"]
    origin = polygon.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((polygon - origin).T))
    tangent = vectors[:, -1]
    if tangent[0] < 0:
        tangent *= -1
    across = np.array([-tangent[1], tangent[0]])
    if across @ (polygons["plate_b"].mean(0) - polygons["plate_a"].mean(0)) < 0:
        across *= -1
    longitudinal = (polygon - origin) @ tangent
    transverse = (polygon - origin) @ across
    # Trim polygon tips, where rotated pixel sampling produces empty sections.
    low, high = longitudinal.min(), longitudinal.max()
    low, high = low + 0.08 * (high - low), high - 0.08 * (high - low)
    s = np.arange(low, high + 0.1, 1.0)
    q = np.arange(transverse.min() + 1, transverse.max(), 1.0)
    xy = origin + s[None, :, None] * tangent + q[:, None, None] * across
    map_x, map_y = xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32)
    search = polygon_mask(polygon, shape).astype(np.uint8)
    valid = cv2.remap(search, map_x, map_y, cv2.INTER_NEAREST).astype(bool)
    valid &= (map_x >= 3) & (map_x < shape[1] - 3) & (map_y >= 3) & (map_y < shape[0] - 3)
    columns = np.flatnonzero(valid.any(axis=0))
    if len(columns) < 10 or len(q) < 8:
        raise ValueError("Junction ROI is too small at the depth resolution")
    return xy[:, columns], valid[:, columns], across


def sample(array, xy):
    return cv2.remap(array.astype(np.float32), xy[..., 0].astype(np.float32),
                     xy[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def extract_curves(rgb, depth, intrinsics, polygons, line, planes, plane_reports, state):
    normals = surface_normals(depth, intrinsics)
    xy, valid, across = make_strip(polygons, depth.shape)
    minus, plus = sample(normals, xy - 3 * across), sample(normals, xy + 3 * across)
    minus /= np.maximum(np.linalg.norm(minus, axis=-1, keepdims=True), 1e-8)
    plus /= np.maximum(np.linalg.norm(plus, axis=-1, keepdims=True), 1e-8)
    contrast = np.rad2deg(np.arccos(np.clip(np.sum(minus * plus, axis=-1), -1, 1)))
    normal_score = normalize_signal(contrast, valid)
    ridge = trace_path(normal_score, valid)
    columns = np.arange(xy.shape[1])
    curves = {"normal_ridge": xy[ridge, columns]}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    intensities = sample(gray, xy)
    gradient = np.gradient(intensities, axis=0)
    details = {
        "normal_ridge_median_contrast_deg": float(np.median(contrast[ridge, columns])),
        "normal_probe_offset_processed_px": 3,
    }
    points = backproject(depth, intrinsics)
    residual = np.min(np.stack([
        np.abs(points @ plane[:3] + plane[3])
        for plane in planes]), axis=0)
    threshold = 3 * max(report["inlier_rmse_estimated_m"] for report in plane_reports)
    threshold = max(threshold, float(np.median(depth)) * 1e-4)
    residual_strip = sample(residual, xy)
    details["plane_residual_threshold_estimated_m"] = threshold
    details["junction_fraction_above_plane_noise"] = float(
        np.mean(residual_strip[valid] > threshold))
    if state == "unwelded":
        distance = np.abs(xy @ line[:2] + line[2])
        span = max(xy.shape[0] / 2, 1)
        score = (0.6 * normalize_signal(np.abs(gradient), valid) + 0.4 * normal_score
                 - 0.5 * (distance / span) ** 2)
        path = trace_path(score, valid)
        curves["rgb_refined"] = xy[path, columns]
    else:
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks

        normal_profile = np.array([np.median(row[keep]) if keep.any() else 0
                                   for row, keep in zip(contrast, valid)])
        normal_profile = gaussian_filter1d(normal_profile, 1.0)
        first_valid = valid.argmax(axis=0)
        last_valid = len(valid) - 1 - valid[::-1].argmax(axis=0)
        peaks, properties = find_peaks(normal_profile, distance=4, prominence=3)
        usable = (peaks > first_valid.max() + 1) & (peaks < last_valid.min() - 1)
        peaks = peaks[usable]
        prominences = properties["prominences"][usable]
        details["depth_two_ridges_found"] = bool(len(peaks) >= 2)
        if len(peaks) >= 2:
            pair = np.sort(peaks[np.argsort(prominences)[-2:]])
            split = float(pair.mean())
            row_indices = np.arange(len(valid))[:, None]
            upper = trace_path(normal_score, valid & (row_indices < split))
            lower = trace_path(normal_score, valid & (row_indices > split))
            curves["depth_upper"] = xy[upper, columns]
            curves["depth_lower"] = xy[lower, columns]
            details["depth_ridge_width_median_processed_px"] = float(np.median(lower - upper))
            details["depth_ridge_profile_peaks_deg"] = normal_profile[pair].tolist()
        # Appearance assumption for this demo: a brighter bead between darker plates.
        # Find the mean brightness peak, then trace its rising and falling edges.
        profile = np.array([np.median(row[keep]) if keep.any() else -1
                            for row, keep in zip(intensities, valid)])
        candidates = np.arange(first_valid.max() + 2, last_valid.min() - 1)
        if not len(candidates):
            raise ValueError("Weld search polygon has no shared interior cross-section")
        peak = int(candidates[np.argmax(profile[candidates])])
        upper_valid = valid & (np.arange(len(profile))[:, None] < peak)
        lower_valid = valid & (np.arange(len(profile))[:, None] > peak)
        upper_score = normalize_signal(np.maximum(gradient, 0), valid) + 0.15 * normal_score
        lower_score = normalize_signal(np.maximum(-gradient, 0), valid) + 0.15 * normal_score
        upper = trace_path(upper_score, upper_valid)
        lower = trace_path(lower_score, lower_valid)
        curves["rgb_upper"] = xy[upper, columns]
        curves["rgb_lower"] = xy[lower, columns]
        details["rgb_candidate_width_median_processed_px"] = float(np.median(lower - upper))
        details["rgb_boundary_assumption"] = "Brighter bead between darker plates; candidates, no boundary ground truth."
        # Width above the plane-fit noise is diagnostic, not a bead segmentation.
        widths = []
        for column in columns:
            selected = np.flatnonzero(valid[:, column] & (residual_strip[:, column] > threshold))
            if len(selected):
                widths.append(int(selected[-1] - selected[0] + 1))
        details["depth_residual_span_median_processed_px"] = float(np.median(widths)) if widths else None
    return curves, normals, xy, contrast, details


def reference_errors(image, points_csv, original_line, original_curves):
    with points_csv.open(encoding="utf-8-sig", newline="") as stream:
        points = np.array([[float(row["x"]), float(row["y"])] for row in csv.DictReader(stream)
                           if row["image"] == image and row["case"] == "three_positive"
                           and row["label"] == "1"])
    if not len(points):
        return {}
    result = {"reference_source": str(points_csv), "reference_points_original_xy": points.tolist(),
              "plane_line_mean_distance_original_px": float(np.abs(
                  points @ original_line[:2] + original_line[2]).mean())}
    for name, curve in original_curves.items():
        distances = np.linalg.norm(points[:, None] - curve[None], axis=-1).min(axis=1)
        result[name + "_mean_distance_original_px"] = float(distances.mean())
    return result


def analyze(image, case, args):
    data_path = args.depth_dir / f"{Path(image).stem}_depth.npz"
    with np.load(data_path, allow_pickle=False) as data:
        depth, rgb = data["depth_m"], data["processed_rgb"]
        intrinsics, original_hw = data["intrinsics"], data["original_hw"]
        if not bool(data["is_metric"]):
            raise ValueError("This experiment expects the deployed metric DA3 model")
    if not np.isfinite(depth).all() or not (depth > 0).all() or depth.shape != rgb.shape[:2]:
        raise ValueError(f"Invalid or misaligned depth: {data_path}")
    scale = np.array([depth.shape[1] / original_hw[1], depth.shape[0] / original_hw[0]])
    polygons = {key: scale_pixels(vertices, scale) for key, vertices in case["polygons"].items()}
    points = backproject(depth, intrinsics)
    planes, plane_reports = [], []
    for key in ("plate_a", "plate_b"):
        plane, report = fit_plane(points[polygon_mask(polygons[key], depth.shape)],
                                  relative_threshold=args.plane_threshold)
        planes.append(plane)
        plane_reports.append(report)
    line, point, direction = project_intersection(*planes, intrinsics)
    curves, normals, strip_xy, contrast, details = extract_curves(
        rgb, depth, intrinsics, polygons, line, planes, plane_reports, case["state"])
    transform = np.array([[scale[0], 0, (scale[0] - 1) / 2],
                          [0, scale[1], (scale[1] - 1) / 2], [0, 0, 1]])
    original_line = transform.T @ line
    original_line /= np.linalg.norm(original_line[:2])
    original_curves = {key: scale_pixels(value, 1 / scale) for key, value in curves.items()}
    # Project strip samples orthogonally onto the analytic image line, then clip
    # to the manually specified search region. This finite support is not a weld mask.
    centers = strip_xy.mean(axis=0)
    intersection_xy = centers - (centers @ line[:2] + line[2])[:, None] * line[:2]
    keep = sample(polygon_mask(polygons["search"], depth.shape).astype(np.float32),
                  intersection_xy[None])[0] > 0.5
    intersection_original = scale_pixels(intersection_xy[keep], 1 / scale)
    mask = np.zeros(tuple(original_hw), dtype=np.uint8)
    if case["state"] == "unwelded":
        if len(intersection_original) > 1:
            cv2.polylines(mask, [np.rint(intersection_original).astype(np.int32)], False,
                          255, args.band_width, cv2.LINE_8)
            mask &= polygon_mask(case["polygons"]["search"], tuple(original_hw)).astype(np.uint8) * 255
        mask_kind = "Fixed-width band around the calculated plane intersection, not measured gap width."
    else:
        boundary = np.concatenate((original_curves["rgb_upper"], original_curves["rgb_lower"][::-1]))
        cv2.fillPoly(mask, [np.rint(boundary).astype(np.int32)], 255)
        mask_kind = "RGB-assisted candidate bead region; no boundary ground truth."
    angle = float(np.rad2deg(np.arccos(np.clip(abs(planes[0][:3] @ planes[1][:3]), 0, 1))))
    report = {
        "image": image, "configured_state": case["state"],
        "depth": str(data_path), "processed_hw": list(depth.shape),
        "original_hw": original_hw.tolist(), "plane_a": plane_reports[0], "plane_b": plane_reports[1],
        "plane_equations_camera": np.array(planes).tolist(), "acute_plane_angle_deg": angle,
        "intersection_image_line_original_abc": original_line.tolist(),
        "intersection_point_camera_estimated_m": point.tolist(), "intersection_direction_camera": direction.tolist(),
        "intersection_has_search_roi_support": bool(len(intersection_original) > 1),
        "mask_kind": mask_kind, "band_width_original_px": args.band_width if case["state"] == "unwelded" else None,
        **details,
    }
    if case["state"] == "unwelded" and args.reference_points.is_file():
        report["approximate_reference_check"] = reference_errors(
            image, args.reference_points, original_line, original_curves)
    arrays = {"plane_parameters": np.array(planes), "normals": normals,
              "intersection_line_original_abc": original_line,
              "intersection_xy_original": intersection_original,
              "normal_contrast_strip_deg": contrast, "strip_xy_processed": strip_xy,
              "candidate_mask_original": mask, **{key + "_xy_original": value for key, value in original_curves.items()}}
    plot = {"image": image, "rgb": rgb, "scale": scale, "polygons": polygons,
            "line_xy": intersection_xy[keep], "curves": curves, "strip_xy": strip_xy,
            "contrast": contrast, "state": case["state"], "report": report}
    return report, arrays, plot


def render(plots, output):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quadweld-da3-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(plots), 3, figsize=(15, 4.2 * len(plots)), squeeze=False)
    for row, plot in enumerate(plots):
        rgb, polygons = plot["rgb"], plot["polygons"]
        all_vertices = np.concatenate(list(polygons.values()))
        low, high = all_vertices.min(0), all_vertices.max(0)
        padding = max(high - low) * 0.12
        for axis in axes[row]:
            axis.imshow(rgb)
            axis.set_xlim(max(0, low[0] - padding), min(rgb.shape[1], high[0] + padding))
            axis.set_ylim(min(rgb.shape[0], high[1] + padding), max(0, low[1] - padding))
            axis.set_aspect("equal")
            axis.axis("off")
        axis = axes[row, 0]
        for key, color in (("plate_a", "lime"), ("plate_b", "dodgerblue"), ("search", "white")):
            polygon = np.vstack((polygons[key], polygons[key][0]))
            axis.plot(*polygon.T, color=color, lw=1.3, ls="--")
        if len(plot["line_xy"]):
            axis.plot(*plot["line_xy"].T, color="yellow", lw=1.8)
        check = plot["report"].get("approximate_reference_check", {})
        suffix = f" | ref error {check['plane_line_mean_distance_original_px']:.1f} original px" if check else ""
        axis.set_title(plot["image"] + "\nPlane intersection" + suffix, fontsize=10)

        axis = axes[row, 1]
        xy = plot["strip_xy"]
        axis.pcolormesh(xy[..., 0], xy[..., 1], plot["contrast"], cmap="magma", shading="nearest",
                        alpha=0.78, vmin=0, vmax=45)
        if "depth_upper" in plot["curves"]:
            axis.plot(*plot["curves"]["depth_upper"].T, color="cyan", lw=1.3)
            axis.plot(*plot["curves"]["depth_lower"].T, color="lime", lw=1.3)
            title = "Depth normal change (0-45 deg)\nCyan + green: two depth ridge candidates"
        else:
            axis.plot(*plot["curves"]["normal_ridge"].T, color="cyan", lw=1.3)
            title = "Depth normal change (0-45 deg)\nCyan: continuous normal ridge"
        axis.set_title(title, fontsize=10)

        axis = axes[row, 2]
        if len(plot["line_xy"]):
            axis.plot(*plot["line_xy"].T, color="yellow", lw=1, alpha=0.8)
        if plot["state"] == "unwelded":
            axis.plot(*plot["curves"]["rgb_refined"].T, color="lime", lw=1.5)
            title = "RGB + normal + plane prior\nGreen: refined junction candidate"
        else:
            upper, lower = plot["curves"]["rgb_upper"], plot["curves"]["rgb_lower"]
            polygon = np.concatenate((upper, lower[::-1]))
            axis.fill(*polygon.T, color="lime", alpha=0.2)
            axis.plot(*upper.T, color="lime", lw=1.5)
            axis.plot(*lower.T, color="deepskyblue", lw=1.5)
            title = "RGB-assisted bead boundary candidates\nGreen: upper | Blue: lower (unverified)"
        axis.set_title(title, fontsize=10)
    figure.suptitle("DA3 geometry experiment: manually supplied plate and junction ROIs", fontsize=14)
    figure.text(0.5, 0.012, "Yellow = fitted plane intersection. Estimated depth has no metric ground truth. "
                "Candidate curves do not prove weld boundaries.", ha="center", fontsize=10)
    figure.tight_layout(rect=(0, 0.03, 1, 0.965))
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=Path, default=Path("assets/weld_images/geometry_regions.csv"))
    parser.add_argument("--depth-dir", type=Path, default=Path("outputs/offline/depth_anything_3/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/offline/depth_anything_3/data"))
    parser.add_argument("--reference-points", type=Path, default=Path("assets/weld_images/point_prompts.csv"))
    parser.add_argument("--plane-threshold", type=float, default=0.005,
                        help="RANSAC distance threshold as a fraction of median estimated Z")
    parser.add_argument("--band-width", type=int, default=5, help="Unwelded line mask width in original pixels")
    args = parser.parse_args()
    if not 0 < args.plane_threshold < 0.1 or args.band_width < 1:
        parser.error("Use 0 < plane-threshold < 0.1 and band-width >= 1")
    cases = {}
    with args.regions.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            case = cases.setdefault(row["image"], {"state": row["state"], "polygons": {}})
            if row["state"] != case["state"] or row["state"] not in {"unwelded", "welded"}:
                raise ValueError("Inconsistent or unknown weld state")
            case["polygons"].setdefault(row["region"], []).append([float(row["x"]), float(row["y"])])
    reports, arrays, plots = [], {}, []
    for image, case in cases.items():
        if set(case["polygons"]) != {"plate_a", "plate_b", "search"}:
            raise ValueError(f"Missing or extra region in {image}")
        if any(len(polygon) < 3 for polygon in case["polygons"].values()):
            raise ValueError(f"Each polygon needs at least three vertices: {image}")
        report, result_arrays, plot = analyze(image, case, args)
        reports.append(report)
        plots.append(plot)
        arrays.update({Path(image).stem + "_" + key: value for key, value in result_arrays.items()})
        print(json.dumps(report, ensure_ascii=False), flush=True)
    if not cases:
        parser.error("No image regions supplied")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "geometry_results.npz", **arrays)
    payload = {"method": "DA3 -> camera points -> RANSAC plate planes -> analytic intersection; local normals and RGB curves",
               "manual_input": str(args.regions), "evaluation_points_used_for_fitting": False,
               "limitations": ["Plate and search polygons and weld state are supplied manually.",
                               "Manual ROI selection uses known target location; reference checking is not independent.",
                               "DA3 predicted scale and fine surface geometry have no measured ground truth.",
                               "Unwelded masks use a chosen rasterization width, not measured gap width.",
                               "Welded boundaries additionally assume a bright bead; they are unverified RGB-assisted candidates."],
               "cases": reports}
    (args.output_dir / "geometry_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    render(plots, args.output_dir / "geometry_overview.jpg")


if __name__ == "__main__":
    main()
