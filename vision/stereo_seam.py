"""Metric seam localisation from a calibrated, rectified stereo pair.

A plate face is a plane, and a plane seen by a rectified stereo pair has a
disparity that is affine in the pixel coordinates.  Each plate therefore needs
only three numbers, estimated from *all* of its pixels by direct left/right
image alignment.  The seam is where two such planes meet.

Segmentation (SAM3 or anything else) only has to say roughly where each plate
is, and monocular depth (DA3) is at most a starting guess.  Metric scale comes
exclusively from the calibrated baseline.

Pixel convention: integer index ``i`` has image coordinate ``i``; pass an
intrinsic matrix in that convention (subtract 0.5 from cx, cy when the
calibration places pixel centres at ``i + 0.5``).
"""

import cv2
import numpy as np


# ------------------------------------------------------------------- images

def rectify_pair(left, right, k_left, d_left, k_right, d_right, rotation, translation, masks=()):
    """Rectify a real pair. ``rotation``/``translation`` map left-camera points into the right camera.

    Returns rectified images, rectified left-image masks, the shared intrinsics
    (pixel centres at integer indices), the baseline, and the rotation taking
    rectified-left coordinates back to the original left camera frame.
    """
    size = (left.shape[1], left.shape[0])
    k_left, k_right = np.asarray(k_left, float), np.asarray(k_right, float)
    d_left, d_right = np.asarray(d_left, float), np.asarray(d_right, float)
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        k_left, d_left, k_right, d_right, size, np.asarray(rotation, float), np.asarray(translation, float),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    if abs(p2[1, 3]) > abs(p2[0, 3]):
        raise ValueError('Vertical stereo rig: rotate both images by 90 degrees first')
    baseline = float(-p2[0, 3] / p2[0, 0])
    if baseline <= 0:
        raise ValueError('Expected the second camera to the right of the first')
    maps = [cv2.initUndistortRectifyMap(k, d, r, p, size, cv2.CV_32FC1)
            for k, d, r, p in ((k_left, d_left, r1, p1), (k_right, d_right, r2, p2))]
    images = [cv2.remap(im, mx, my, cv2.INTER_LINEAR) for im, (mx, my) in zip((left, right), maps)]
    masks = [cv2.remap(np.asarray(m).astype(np.uint8), *maps[0], cv2.INTER_NEAREST).astype(bool) for m in masks]
    return images[0], images[1], masks, p1[:, :3].copy(), baseline, r1.T


def _gray(image):
    image = np.asarray(image)
    scale = 255. if image.dtype == np.uint8 else (65535. if image.dtype == np.uint16 else 1.)
    if image.ndim == 3:
        image = cv2.cvtColor(np.ascontiguousarray(image[..., :3]), cv2.COLOR_RGB2GRAY)
    return image.astype(np.float32) / scale


def _normalise(gray, blur, window):
    """Band-pass and local contrast normalisation: insensitive to exposure and smooth shading."""
    if blur > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur)
    mean = cv2.GaussianBlur(gray, (0, 0), window)
    var = cv2.GaussianBlur(gray * gray, (0, 0), window) - mean * mean
    return (gray - mean) / np.sqrt(np.maximum(var, 0) + 1e-4)


def _sample(image, x, y):
    """Bilinear lookup at sub-pixel positions; NaN outside the image."""
    height, width = image.shape
    inside = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x0 = np.clip(np.floor(x).astype(np.intp), 0, width - 2)
    y0 = np.clip(np.floor(y).astype(np.intp), 0, height - 2)
    fx, fy = x - x0, y - y0
    value = ((image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx) * (1 - fy) +
             (image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx) * fy)
    return np.where(inside, value, np.nan)


class StereoImages:
    """Left/right images, normalised once for every scale of the coarse-to-fine schedule."""

    def __init__(self, left, right, scales=(6., 3., 1.5, .75)):
        self.left, self.right = _gray(left), _gray(right)
        if self.left.shape != self.right.shape:
            raise ValueError('Left and right images differ in size')
        self.shape, self.scales = self.left.shape, tuple(scales)
        self.levels = []
        for sigma in self.scales:
            l, r = (_normalise(im, sigma, max(4 * sigma, 6.)) for im in (self.left, self.right))
            self.levels.append((l, r, cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)))

    def reduced(self, factor):
        size = (self.shape[1] // factor, self.shape[0] // factor)
        crop = (slice(0, size[1] * factor), slice(0, size[0] * factor))
        return StereoImages(*(cv2.resize(im[crop], size, interpolation=cv2.INTER_AREA)
                              for im in (self.left, self.right)), self.scales)


def clean_mask(mask, close=1):
    """Remove speckle from a raw segmentation: close pin holes, keep the main components."""
    mask = np.asarray(mask).astype(np.uint8)
    if close:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2 * close + 1,) * 2, np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 2:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return np.isin(labels, 1 + np.flatnonzero(areas >= .2 * areas.max()))


# ------------------------------------------------------- photometric fitting

def _tukey(residual, floor):
    finite = residual[np.isfinite(residual)]
    scale = max(1.4826 * float(np.median(np.abs(finite - np.median(finite)))), floor) if len(finite) else floor
    t = residual / (4.685 * scale)
    return np.where(np.abs(t) < 1, (1 - t * t) ** 2, 0.), scale


def _refine(x, y, basis, p, level, iterations=30):
    """Robust Gauss-Newton for ``disparity = basis @ p`` on one scale."""
    left, right, right_dx = level
    target = _sample(left, x, y)
    residual, scale = np.full(len(x), np.nan), np.inf
    for _ in range(iterations):
        xr = x - basis @ p
        residual = _sample(right, xr, y) - target
        gradient = _sample(right_dx, xr, y)
        ok = np.isfinite(residual) & np.isfinite(gradient)
        if ok.sum() < 50:
            return p, np.inf, np.full(len(x), np.nan)
        weight, scale = _tukey(np.where(ok, residual, np.nan), .02)
        weight = np.where(ok, weight, 0.)
        jac = np.where(ok, -gradient, 0.)[:, None] * basis
        hessian = jac.T @ (jac * weight[:, None])
        hessian += (1e-3 * np.trace(hessian) / len(p) + 1e-12) * np.eye(len(p))
        step = np.linalg.solve(hessian, -jac.T @ (weight * np.where(ok, residual, 0.)))
        p = p + step
        if np.abs(step).max() < 2e-4:
            break
    return p, scale, residual


class _Region:
    """Pixels of one plate with conditioned coordinates: d = p0*xn + p1*yn + p2."""

    def __init__(self, mask, max_pixels, seed=0):
        ys, xs = np.nonzero(mask)
        if len(xs) > max_pixels:
            keep = np.random.default_rng(seed).choice(len(xs), max_pixels, replace=False)
            xs, ys = xs[keep], ys[keep]
        self.x, self.y = xs.astype(np.float64), ys.astype(np.float64)
        self.centre = np.array([self.x.mean(), self.y.mean()])
        self.half = np.array([max(np.ptp(self.x) / 2, 1.), max(np.ptp(self.y) / 2, 1.)])
        self.basis = np.stack(((self.x - self.centre[0]) / self.half[0],
                               (self.y - self.centre[1]) / self.half[1], np.ones(len(xs))), axis=1)

    def to_pixel_params(self, p):
        a, b = p[0] / self.half[0], p[1] / self.half[1]
        return np.array([a, b, p[2] - a * self.centre[0] - b * self.centre[1]])

    def from_pixel_params(self, q):
        return np.array([q[0] * self.half[0], q[1] * self.half[1],
                         q[2] + q[0] * self.centre[0] + q[1] * self.centre[1]])


def _coarse_search(region, level, d_range, step, max_slope, keep=3):
    """Exhaustive correlation search over offset and slant on the most blurred scale."""
    left, right, _ = level
    target = _sample(left, region.x, region.y)
    target = target - target.mean()
    offsets = np.arange(d_range[0], d_range[1] + step, step)
    counts = [int(np.ceil(slope * half / step)) for slope, half in zip(max_slope, region.half)]
    found = []
    for i in range(-counts[0], counts[0] + 1):
        for j in range(-counts[1], counts[1] + 1):
            tilt = region.basis[:, 0] * i * step + region.basis[:, 1] * j * step
            disparity = tilt[:, None] + offsets[None]
            value = _sample(right, region.x[:, None] - disparity, region.y[:, None])
            ok = np.isfinite(value)
            usable = (ok.mean(0) > .6) & (disparity.min(0) > 0)
            value = np.where(ok, value, 0.)
            value = np.where(ok, value - value.sum(0) / np.maximum(ok.sum(0), 1), 0.)
            score = (value * target[:, None]).sum(0) / (
                np.linalg.norm(value, axis=0) * np.sqrt((ok * target[:, None] ** 2).sum(0)) + 1e-9)
            found.extend((float(s), i * step, j * step, float(o)) for s, o, u in zip(score, offsets, usable) if u)
    found.sort(reverse=True)
    picked = []
    for item in found:
        if all(max(abs(item[k] - q[k]) for k in (1, 2, 3)) > 2 * step for q in picked):
            picked.append(item)
        if len(picked) == keep:
            break
    return [np.array(q[1:]) for q in picked]


def fit_disparity_plane(images, mask, d_range, init=None, search=True, margin=None, max_pixels=30000,
                        max_slope=(.15, .3)):
    """Fit ``d(u, v) = a*u + b*v + c`` (left -> right disparity, px) to one planar region.

    ``margin`` grows the region so that the plate's own outline, usually its
    strongest stereo feature, takes part (default: width / 320 pixels).
    ``init`` is an optional (a, b, c) guess, e.g. from monocular depth.  With
    ``search`` the global search runs as well and the hypothesis that explains
    the most pixels wins, so a wrong guess cannot hurt.
    """
    mask = np.asarray(mask).astype(np.uint8)
    margin = max(2, round(images.shape[1] / 320)) if margin is None else margin
    if margin:
        mask = cv2.dilate(mask, np.ones((2 * margin + 1,) * 2, np.uint8))
    if mask.sum() < 200:
        raise ValueError('Plate region is too small for a stereo plane fit')
    region = _Region(mask, max_pixels)
    seeds = [] if init is None else [region.from_pixel_params(np.asarray(init, float))]
    if search or not seeds:
        coarse = _Region(mask, 800, seed=1)
        seeds += [region.from_pixel_params(coarse.to_pixel_params(p)) for p in _coarse_search(
            coarse, images.levels[0], d_range, max(images.scales[0], 1.), max_slope)]
    if not seeds:
        raise ValueError('No stereo hypothesis inside the disparity range')
    candidates = []
    for seed in seeds:
        p = seed
        for level in images.levels:
            p, scale, residual = _refine(region.x, region.y, region.basis, p, level)
        # Same footing for every hypothesis: finest scale, fixed residual threshold.
        inlier = np.isfinite(residual) & (np.abs(residual) < .5)
        candidates.append((float(np.mean(~inlier)) + scale, p, scale, inlier))
    candidates.sort(key=lambda c: c[0])
    _, p, scale, inlier = candidates[0]
    # Split-half repeat: refit two interleaved halves of the region.  Their
    # disagreement is an internal, truth-free measure of how well the data pins
    # the plane down (it does not see errors common to both halves).
    block = max(8, int(round(np.sqrt(len(region.x)) / 6)))
    parity = ((region.x // block + region.y // block) % 2).astype(bool)
    halves = []
    for chosen in (parity, ~parity):
        q = p
        for level in images.levels[-2:]:
            q, _, _ = _refine(region.x[chosen], region.y[chosen], region.basis[chosen], q, level)
        halves.append(region.to_pixel_params(q))
    return {'params': region.to_pixel_params(p), 'halves': halves, 'residual_scale': float(scale),
            'inlier_fraction': float(inlier.mean()), 'pixels': int(len(region.x)),
            'hypotheses': [{'params': region.to_pixel_params(c[1]).tolist(), 'cost': float(c[0])} for c in candidates]}


def disparity_plane_from_depth(depth, mask, focal, baseline):
    """Initial (a, b, c) from any roughly metric depth map on the same pixel grid (e.g. DA3)."""
    ys, xs = np.nonzero(np.asarray(mask, bool) & np.isfinite(depth) & (depth > 0))
    if len(xs) < 50:
        return None
    d = focal * baseline / depth[ys, xs]
    basis = np.stack((xs, ys, np.ones(len(xs))), axis=1).astype(float)
    p = np.linalg.lstsq(basis, d, rcond=None)[0]
    for _ in range(5):
        w = np.sqrt(_tukey(d - basis @ p, 1e-3)[0])
        p = np.linalg.lstsq(basis * w[:, None], d * w, rcond=None)[0]
    return p


def _rescale_params(params, factor):
    """Disparity plane of an image reduced by ``factor`` -> plane on the original pixel grid."""
    a, b, c = params
    return np.array([a, b, factor * c + (a + b) * (1 - factor) / 2])


# ------------------------------------------------------------------ geometry

def plane_from_disparity(params, intrinsics, baseline):
    """(a, b, c) -> unit normal n and offset h > 0 with n . X = h in the left camera frame (metres)."""
    a, b, c = params
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    normal = np.array([a * fx, b * fy, a * cx + b * cy + c])
    scale = np.linalg.norm(normal)
    return normal / scale, float(fx * baseline / scale)


def disparity_from_plane(normal, offset, intrinsics, baseline):
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    scale = fx * baseline / offset
    a, b = normal[0] * scale / fx, normal[1] * scale / fy
    return np.array([a, b, normal[2] * scale - a * cx - b * cy])


def intersect_planes(plane_a, plane_b, min_angle_deg=15):
    """Line shared by two planes: a point on it, its unit direction and the angle between the normals."""
    (na, ha), (nb, hb) = plane_a, plane_b
    direction = np.cross(na, nb)
    sine = np.linalg.norm(direction)
    if sine < np.sin(np.radians(min_angle_deg)):
        raise ValueError('Plates are nearly parallel; their intersection is ill-conditioned')
    direction /= sine
    point = np.linalg.solve(np.stack((na, nb, direction)), np.array([ha, hb, 0.]))
    return point, direction, float(np.degrees(np.arccos(np.clip(na @ nb, -1, 1))))


def backproject(xy, disparity, intrinsics, baseline):
    z = intrinsics[0, 0] * baseline / np.asarray(disparity)
    return (np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(intrinsics).T) * z[:, None]


def project(points, intrinsics):
    q = points @ intrinsics.T
    return q[:, :2] / q[:, 2:]


def _convex_hull(mask):
    hull = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(mask.astype(np.uint8))), 1)
    return hull.astype(bool)


def _extent_along(mask, start, stop, bands):
    """Range of a plate along an image line, read from the pixels next to the line.

    The narrowest band is the most exact one, but shadows and highlights beside
    a joint are routinely missing from a segmentation.  A band is trusted only
    if it spans nearly as much of the line as the widest one does.
    """
    tangent = (stop - start) / np.linalg.norm(stop - start)
    ys, xs = np.nonzero(mask)
    s = (xs - start[0]) * tangent[0] + (ys - start[1]) * tangent[1]
    w = np.abs((ys - start[1]) * tangent[0] - (xs - start[0]) * tangent[1])
    ranges = [(float(s[w <= band].min()), float(s[w <= band].max())) if (w <= band).sum() >= 10 else None
              for band in bands]
    if ranges[-1] is None:
        return None
    widest = ranges[-1][1] - ranges[-1][0]
    return next(r for r in ranges if r is not None and r[1] - r[0] >= .9 * widest)


# ------------------------------------------------------------------- driver

def _fit_plates(images, regions, d_range, inits, options):
    """Coarse-to-fine over image resolution: search on a reduced copy, polish on the full image."""
    factor = 2 if max(images.shape) > 1600 else 1
    if factor > 1:
        small = [cv2.resize(r.astype(np.uint8), (images.shape[1] // factor, images.shape[0] // factor),
                            interpolation=cv2.INTER_NEAREST).astype(bool) for r in regions]
        scaled = [None if q is None else _rescale_params(q, 1 / factor) for q in inits]
        coarse = _fit_plates(images.reduced(factor), small, tuple(d / factor for d in d_range), scaled, options)
        return [fit_disparity_plane(images, r, d_range, init=_rescale_params(f['params'], factor), search=False,
                                    **options) for r, f in zip(regions, coarse)]
    return [fit_disparity_plane(images, r, d_range, init=q, **options) for r, q in zip(regions, inits)]


def locate_seam(left, right, intrinsics, baseline, mask_a, mask_b, depth_range=(.25, 4.), init_depth=None,
                spacing=.002, end_margin=0., neighbourhood=None, **fit_options):
    """Two plate masks on the left image -> metric 3D seam polyline in the left camera frame.

    Each plate is fitted where it adjoins the other one (``neighbourhood`` pixels;
    by default the reach of the smaller plate): a seam is a local feature and
    segmentations leak far away from it.  ``status`` says whether a seam was
    found; ``checks`` carries truth-free quality indicators.
    """
    intrinsics = np.asarray(intrinsics, float)
    images = left if isinstance(left, StereoImages) else StereoImages(left, right)
    height, width = images.shape
    focal = float(intrinsics[0, 0])
    d_range = (focal * baseline / depth_range[1], focal * baseline / depth_range[0])
    masks = [clean_mask(m) for m in (mask_a, mask_b)]
    result = {'status': 'ok', 'xyz': np.zeros((0, 3)), 'xy': np.zeros((0, 2)), 'checks': {}}
    if min(m.sum() for m in masks) < 200:
        result['status'] = 'plate_mask_too_small'
        return result
    reach = [cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5) for m in masks]
    if neighbourhood is None:
        neighbourhood = max(24., min(float(np.percentile(reach[1 - k][masks[k]], 95)) for k in (0, 1)))
    regions = [masks[k] & (reach[1 - k] <= neighbourhood) for k in (0, 1)]
    # A shadow or a conservative segmentation keeps the masks somewhat apart; if the gap is
    # comparable with the plates themselves they are two unrelated objects.
    gap = float(reach[1][masks[0]].min())
    if gap > max(24., .5 * neighbourhood) or min(r.sum() for r in regions) < 200:
        result['status'] = 'plates_are_not_adjacent'
        return result
    inits = [None if init_depth is None else disparity_plane_from_depth(init_depth, r, focal, baseline)
             for r in regions]
    fits = _fit_plates(images, regions, d_range, inits, fit_options)
    planes = [plane_from_disparity(f['params'], intrinsics, baseline) for f in fits]
    result.update(fits=fits, planes=planes, neighbourhood_px=float(neighbourhood))
    try:
        point, direction, _ = intersect_planes(*planes)
    except ValueError:
        result['status'] = 'plates_nearly_parallel'
        return result

    # Finite extent: where both (convex) plates lie beside the projected intersection.
    span = 4 * max(height, width) / focal * point[2]
    ends = project(np.stack((point - span * direction, point + span * direction)), intrinsics)
    if ends[0, 0] > ends[1, 0]:
        ends, direction = ends[::-1], -direction
    bands = (max(3, width // 160), max(8, width // 40), neighbourhood)
    extents = [_extent_along(_convex_hull(m), ends[0], ends[1], bands) for m in masks]
    if any(e is None for e in extents) or min(e[1] for e in extents) - max(e[0] for e in extents) < 10:
        result['status'] = 'plates_do_not_meet_at_intersection'
        return result
    tangent = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
    segment = ends[0] + np.outer([max(e[0] for e in extents), min(e[1] for e in extents)], tangent)
    ends_xyz = backproject(segment, np.c_[segment, np.ones(2)] @ fits[0]['params'], intrinsics, baseline)
    ends_xyz = point + np.outer((ends_xyz - point) @ direction, direction)   # exactly on the line
    length = float(np.linalg.norm(ends_xyz[1] - ends_xyz[0]))
    if length <= 2 * end_margin:
        result['status'] = 'seam_shorter_than_end_margins'
        return result
    s = np.linspace(end_margin, length - end_margin, max(2, int(np.ceil((length - 2 * end_margin) / spacing)) + 1))
    xyz = ends_xyz[0] + s[:, None] * direction
    # Fillet torch axis: bisector of the two visible faces, perpendicular to the seam.
    facing = [-n for n, _ in planes]
    bisector = facing[0] + facing[1]
    bisector -= direction * (bisector @ direction)
    # Split-half repeatability: how far the seam moves when each plate is refitted from half its pixels.
    spread = []
    for half_a in fits[0]['halves']:
        for half_b in fits[1]['halves']:
            try:
                p, d, _ = intersect_planes(plane_from_disparity(half_a, intrinsics, baseline),
                                           plane_from_disparity(half_b, intrinsics, baseline))
            except ValueError:
                spread.append(np.inf)
                continue
            offset = ends_xyz - p
            spread.append(float(np.linalg.norm(offset - np.outer(offset @ d, d), axis=1).max()))
    border = max(4, width // 100)
    mean_z = float(xyz[:, 2].mean())
    result.update(
        xyz=xyz, xy=project(xyz, intrinsics), xy_right=project(xyz - np.array([baseline, 0, 0]), intrinsics),
        joint_ends_xyz=ends_xyz, length_m=length, direction=direction, image_segment=segment,
        # An end at the image border is where the view stops, not where the joint does.
        end_is_open=[bool(x < border or x > width - 1 - border or y < border or y > height - 1 - border)
                     for x, y in segment],
        torch_axis=bisector / np.linalg.norm(bisector), face_normals=facing,
        opening_deg=float(np.degrees(np.arccos(np.clip(facing[0] @ facing[1], -1, 1)))),
        checks={'plate_inlier_fraction': [f['inlier_fraction'] for f in fits],
                'plate_residual': [f['residual_scale'] for f in fits],
                'split_half_spread_mm': float(np.max(spread) / 2 * 1000),
                'depth_per_disparity_px_mm': float(mean_z ** 2 / (focal * baseline) * 1000)})
    return result


def fuse_seams(segments, tolerance=.005):
    """Combine seam segments of several views, given in one common frame, into one segment.

    The line is the least-squares axis of the segments; its extent is their
    union, because single views under-cover a joint (image border, shadow).
    A view is left out when it lies much further from the consensus than the
    others do (and further than ``tolerance``, which should reflect the depth
    resolution of the views).  Returns end points, direction, per-view distances and usage.
    """
    segments = [np.asarray(s, float) for s in segments]

    def deviations(point, direction):
        return np.array([np.linalg.norm((s - point) - np.outer((s - point) @ direction, direction), axis=1).max()
                         for s in segments])

    # Consensus first: the view whose own line the others agree with best decides who takes part.
    # (A least-squares line through everything would be pulled towards a stray view and hide it.)
    by_view = [deviations(s[0], (s[1] - s[0]) / np.linalg.norm(s[1] - s[0])) for s in segments]
    reference = by_view[int(np.argmin([np.median(d) for d in by_view]))]
    used = reference <= max(tolerance, 2.5 * np.median(reference))
    if used.sum() < 2:
        used = np.ones(len(segments), bool)
    samples = np.concatenate([a + np.linspace(0, 1, 50)[:, None] * (b - a) for (a, b), u in zip(segments, used) if u])
    centre = samples.mean(0)
    direction = np.linalg.svd(samples - centre)[2][0]
    deviation = deviations(centre, direction)
    if direction @ (segments[0][1] - segments[0][0]) < 0:
        direction = -direction
    along = np.array([(s - centre) @ direction for s, u in zip(segments, used) if u])
    return {'ends': centre + np.outer([along.min(), along.max()], direction), 'direction': direction,
            'length_m': float(along.max() - along.min()), 'view_deviation_m': deviation.tolist(),
            'view_used': used.tolist()}
