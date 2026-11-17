"""Metric seam localisation from a calibrated, rectified stereo pair.

A plate face is a plane, and a plane seen by a rectified stereo pair has a
disparity that is affine in the pixel coordinates.  Each plate therefore needs
only three numbers, estimated from *all* of its pixels by direct left/right
image alignment.  The seam is where two such planes meet.

Segmentation (SAM3 or anything else) only has to say roughly where each plate
is, and an optional depth prior is at most a starting guess.  Metric scale comes
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
    """Exhaustive correlation search over offset and slant on one blurred scale (nearest-pixel lookups)."""
    left, right, _ = level
    height, width = right.shape
    target = _sample(left, region.x, region.y)
    target = target - target.mean()
    rows = np.rint(region.y).astype(np.intp)[:, None]
    offsets = np.arange(d_range[0], d_range[1] + step, step)
    counts = [int(np.ceil(slope * half / step)) for slope, half in zip(max_slope, region.half)]
    found = []
    for i in range(-counts[0], counts[0] + 1):
        for j in range(-counts[1], counts[1] + 1):
            tilt = region.basis[:, 0] * i * step + region.basis[:, 1] * j * step
            disparity = tilt[:, None] + offsets[None]
            columns = np.rint(region.x[:, None] - disparity).astype(np.intp)
            ok = (columns >= 0) & (columns < width)
            usable = (ok.mean(0) > .6) & (disparity.min(0) >= d_range[0]) & (disparity.max(0) <= d_range[1])
            if not usable.any():
                continue
            columns, ok, chosen = columns[:, usable], ok[:, usable], offsets[usable]
            value = np.where(ok, right[rows, np.clip(columns, 0, width - 1)], 0.)
            value = np.where(ok, value - value.sum(0) / np.maximum(ok.sum(0), 1), 0.)
            score = (value * target[:, None]).sum(0) / (
                np.linalg.norm(value, axis=0) * np.sqrt((ok * target[:, None] ** 2).sum(0)) + 1e-9)
            found.extend((float(s), i * step, j * step, float(o)) for s, o in zip(score, chosen))
    found.sort(reverse=True)
    picked = []
    for item in found:
        if all(max(abs(item[k] - q[k]) for k in (1, 2, 3)) > 2 * step for q in picked):
            picked.append(item)
        if len(picked) == keep:
            break
    return [np.array(q[1:]) for q in picked]


def _depth_aliases(region, p, level, d_range, keep=3):
    """Other distances of the same plane orientation that also align the images.

    Scaling a disparity plane keeps the surface normal and moves the surface
    along the viewing rays.  Texture that repeats along the baseline (machining
    marks, weld ripples, gratings) aligns again at such scaled copies.
    """
    left, right, _ = level
    step = max(1, len(region.x) // 3000)
    x, y, disparity = region.x[::step], region.y[::step], (region.basis @ p)[::step]
    target = _sample(left, x, y)
    factors = np.arange(d_range[0] / disparity.min(), d_range[1] / disparity.max(), .4 / disparity.mean())
    if len(factors) < 3:
        return []
    cost = np.full(len(factors), np.inf)
    for index, factor in enumerate(factors):
        residual = _sample(right, x - factor * disparity, y) - target
        ok = np.isfinite(residual)
        if ok.mean() > .4:
            cost[index] = np.minimum(np.abs(residual[ok]), 1.).mean()
    minima = [i for i in range(1, len(factors) - 1) if np.isfinite(cost[i]) and cost[i] <= cost[i - 1]
              and cost[i] < cost[i + 1] and abs(factors[i] - 1) * disparity.mean() > 1.5]
    return [factors[i] * p for i in sorted(minima, key=lambda i: cost[i])[:keep]]


class _Hypotheses:
    """Locally optimal disparity planes of one region, all judged on the same footing."""

    def __init__(self, images, mask, d_range, margin=None, max_pixels=30000):
        mask = np.asarray(mask).astype(np.uint8)
        margin = max(2, round(images.shape[1] / 320)) if margin is None else margin
        if margin:
            mask = cv2.dilate(mask, np.ones((2 * margin + 1,) * 2, np.uint8))
        if mask.sum() < 200:
            raise ValueError('Plate region is too small for a stereo plane fit')
        self.images, self.mask, self.d_range = images, mask, d_range
        self.region = _Region(mask, max_pixels)
        self.coarse = _Region(mask, 800, seed=1)
        self.fine = len(images.levels) - 2
        self.found = []

    def polish(self, start, p):
        """Refine normalised parameters from scale ``start`` down; keep the result if it is new and valid."""
        region = self.region
        if any(np.abs(region.basis @ (p - c['p'])).max() < .25 for c in self.found):
            return
        for level in self.images.levels[start:]:
            p, scale, residual = _refine(region.x, region.y, region.basis, p, level)
        disparity = region.basis @ p
        valid = np.isfinite(residual)
        if (not np.isfinite(scale) or valid.mean() < .4 or disparity.min() < self.d_range[0]
                or disparity.max() > self.d_range[1]):
            return
        if any(np.abs(region.basis @ (p - c['p'])).max() < .25 for c in self.found):
            return
        # Finest scale, fixed residual threshold, pixels both eyes see.
        inlier = float(np.mean(np.abs(residual[valid]) < .5))
        self.found.append({'p': p, 'params': region.to_pixel_params(p), 'cost': 1 - inlier + float(scale),
                           'scale': float(scale), 'inlier': inlier})
        self.found.sort(key=lambda c: c['cost'])

    def search(self, max_slope=(.15, .3), search_levels=2):
        # Several scales propose hypotheses.  Mirror-like reflections and smooth shading are
        # not attached to the surface and can own the coarsest scale; repetitive texture
        # aliases the fine ones.
        images = self.images
        for index in range(min(search_levels, len(images.levels))):
            found = _coarse_search(self.coarse, images.levels[index], self.d_range, max(images.scales[index], 1.),
                                   max_slope)
            for q in found:
                self.polish(index, self.region.from_pixel_params(self.coarse.to_pixel_params(q)))
        if self.found:
            for alias in _depth_aliases(self.region, self.found[0]['p'], images.levels[self.fine], self.d_range):
                self.polish(self.fine, alias)

    def hinge(self, other, zone, reach, keep=2):
        """Hypotheses that meet the neighbouring plate's plane ``other`` (pixel parameters) at the joint.

        Two plates of a joint share a line that projects into ``zone``.  Given one
        plate, the other is the same disparity plane folded about a line there: two
        unknowns instead of three, cheap to scan densely on a fine scale.  This finds
        plates whose coarse appearance is owned by reflections of their neighbour.
        """
        if len(zone) < 20:
            return
        centre = np.median(zone[:, :2], axis=0)
        axis = np.linalg.svd(zone[:, :2] - zone[:, :2].mean(0), full_matrices=False)[2][0]
        scale = self.fine - 1
        region, (left, right, _) = _Region(self.mask, 400, seed=2), self.images.levels[scale]
        offset = np.stack((region.x - centre[0], region.y - centre[1]), axis=1)
        base = other[0] * region.x + other[1] * region.y + other[2]
        target = _sample(left, region.x, region.y)[:, None]
        step = self.images.scales[scale] / max(np.linalg.norm(offset, axis=1).max(), 1.)
        folds = np.arange(-.6, .6 + step, step)
        rows = np.broadcast_to(region.y[:, None], (len(base), len(folds)))
        # The zone also holds occlusion edges, so its long axis only roughly follows the joint.
        trials = [(angle, shift) for angle in np.radians((-20, -10, 0, 10, 20)) for shift in np.linspace(-reach, reach, 5)]
        cost = np.full((len(trials), len(folds)), np.inf)
        for index, (angle, shift) in enumerate(trials):
            normal = np.array([-axis[1] * np.cos(angle) - axis[0] * np.sin(angle),
                               axis[0] * np.cos(angle) - axis[1] * np.sin(angle)])
            disparity = base[:, None] + (offset @ normal - shift)[:, None] * folds[None]
            residual = _sample(right, region.x[:, None] - disparity, rows) - target
            ok = np.isfinite(residual)
            usable = (ok.mean(0) > .4) & (disparity.min(0) >= self.d_range[0]) & (disparity.max(0) <= self.d_range[1])
            mean = np.where(ok, np.minimum(np.abs(residual), 1.), 0.).sum(0) / np.maximum(ok.sum(0), 1)
            cost[index, usable] = mean[usable]
        best = cost.min(0)
        minima = [j for j in range(1, len(folds) - 1) if np.isfinite(best[j]) and best[j] <= best[j - 1]
                  and best[j] < best[j + 1] and abs(folds[j]) > 2 * step]
        for j in sorted(minima, key=lambda j: best[j])[:keep]:
            angle, shift = trials[int(np.argmin(cost[:, j]))]
            normal = np.array([-axis[1] * np.cos(angle) - axis[0] * np.sin(angle),
                               axis[0] * np.cos(angle) - axis[1] * np.sin(angle)])
            q = np.asarray(other, float) + folds[j] * np.array([normal[0], normal[1], -(normal @ centre) - shift])
            self.polish(scale, self.region.from_pixel_params(q))


def _plane_candidates(images, mask, d_range, init=None, search=True, margin=None, max_pixels=30000, **options):
    """Hypotheses of one region: from a guess (polished on the fine scales only) and/or the global search."""
    found = _Hypotheses(images, mask, d_range, margin, max_pixels)
    if init is not None:
        found.polish(found.fine, found.region.from_pixel_params(np.asarray(init, float)))
    if search or not found.found:
        found.search(**options)
    return found


def _finish_fit(images, region, chosen, candidates, ambiguity):
    """Result record of one plate; adds the split-half repeat of the chosen hypothesis."""
    # Split-half repeat: refit two interleaved halves of the region.  Their
    # disagreement is an internal, truth-free measure of how well the data pins
    # the plane down (it does not see errors common to both halves).
    block = max(8, int(round(np.sqrt(len(region.x)) / 6)))
    parity = ((region.x // block + region.y // block) % 2).astype(bool)
    halves = []
    for half in (parity, ~parity):
        q = chosen['p']
        for level in images.levels[-2:]:
            q, _, _ = _refine(region.x[half], region.y[half], region.basis[half], q, level)
        halves.append(region.to_pixel_params(q))
    return {'params': chosen['params'], 'halves': halves, 'residual_scale': chosen['scale'],
            'inlier_fraction': chosen['inlier'], 'pixels': int(len(region.x)), 'ambiguity': float(ambiguity),
            'hypotheses': [{'params': c['params'].tolist(), 'cost': float(c['cost'])} for c in candidates]}


def fit_disparity_plane(images, mask, d_range, init=None, prior_tolerance=.75, **options):
    """Fit ``d(u, v) = a*u + b*v + c`` (left -> right disparity, px) to one planar region.

    ``margin`` grows the region so that the plate's own outline, usually its
    strongest stereo feature, takes part (default: width / 320 pixels).
    ``init`` is an optional (a, b, c) guess, e.g. from monocular depth.  With
    ``search`` the global search runs as well and the hypothesis that explains
    the images best wins, so a wrong guess cannot hurt; among hypotheses that
    explain them about equally well the one nearest the guess is taken.

    ``ambiguity`` in the result is the cost of the winner relative to the best
    hypothesis at a different depth (0: no rival, 1: a rival fits as well).
    """
    found = _plane_candidates(images, mask, d_range, init=init, **options)
    region, candidates = found.region, found.found
    if not candidates:
        raise ValueError('No stereo hypothesis inside the disparity range')
    chosen = candidates[0]
    if init is not None:
        prior = region.from_pixel_params(np.asarray(init, float))
        close = [c for c in candidates if c['cost'] * prior_tolerance <= chosen['cost']]
        chosen = min(close, key=lambda c: np.abs(region.basis @ (c['p'] - prior)).mean())
    rivals = [c['cost'] for c in candidates if np.abs(region.basis @ (c['p'] - chosen['p'])).max() > 1.5]
    return _finish_fit(images, region, chosen, candidates, min(chosen['cost'] / min(rivals), 1.) if rivals else 0.)


def disparity_plane_from_depth(depth, mask, focal, baseline):
    """Initial (a, b, c) from any roughly metric depth map on the same pixel grid."""
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

def _choose_pair(sets, zone, tolerance, expected, prior_tolerance):
    """Pick one hypothesis per plate, preferring planes that meet where the masks meet.

    ``zone`` holds pixels (x, y, 1) between the two plates.  Hypotheses of the two
    plates at mismatched depths intersect elsewhere in the image; among pairs
    that explain the images about equally well, one that meets in the zone wins.
    (The zone is only a hint: rough masks also touch along occlusion edges.)
    What remains can still be a common wrong depth for both; ``expected`` (seam
    disparity, px) decides among near-ties.  Returns the pair, its ambiguity
    (cost relative to the best pair at another depth) and whether the
    expectation was needed to decide.
    """
    pairs = []
    for first in sets[0]:
        for second in sets[1]:
            delta = first['params'] - second['params']
            # Disparity difference where the masks meet: none for planes that cross there
            # (or coincide, which the caller reports as nearly parallel plates).
            apart = float(np.median(np.abs(zone @ delta))) if len(zone) else 0.
            meets = apart <= max(.5, tolerance * np.hypot(delta[0], delta[1]))
            depth = float(np.median(zone @ (first['params'] + second['params'])) / 2) if len(zone) else np.nan
            pairs.append((first['cost'] + second['cost'], depth, first, second, meets))
    pairs.sort(key=lambda q: q[0])
    near = [q for q in pairs if q[0] * prior_tolerance <= pairs[0][0]]
    pool = [q for q in near if q[4]] or near
    cheapest = chosen = pool[0]
    if expected is not None and len(zone):
        chosen = min(pool, key=lambda q: abs(q[1] - expected))
    rivals = [q[0] for q in pairs if (q[4] or not chosen[4]) and abs(q[1] - chosen[1]) > max(1.5, .02 * chosen[1])]
    ambiguity = min(chosen[0] / min(rivals), 1.) if rivals else 0.
    return (chosen[2], chosen[3]), ambiguity, chosen is not cheapest


def _fit_plates(images, regions, d_range, inits, options, zone, tolerance, expected, prior_tolerance=.75):
    """Both plates, coarse-to-fine over image resolution: search on a reduced copy, polish on the full image."""
    factor = 2 if max(images.shape) > 1600 else 1
    if factor > 1:
        small = [cv2.resize(r.astype(np.uint8), (images.shape[1] // factor, images.shape[0] // factor),
                            interpolation=cv2.INTER_NEAREST).astype(bool) for r in regions]
        scaled = [None if q is None else _rescale_params(q, 1 / factor) for q in inits]
        reduced = np.c_[(zone[:, :2] - (factor - 1) / 2) / factor, np.ones(len(zone))]
        coarse, info = _fit_plates(images.reduced(factor), small, tuple(d / factor for d in d_range), scaled, options,
                                   reduced, tolerance / factor, None if expected is None else expected / factor,
                                   prior_tolerance)
        fits = []
        polish_options = {**options, 'search': False}
        for region, found in zip(regions, coarse):
            fit = fit_disparity_plane(images, region, d_range, init=_rescale_params(found['params'], factor),
                                      **polish_options)
            fit['ambiguity'] = found['ambiguity']
            fits.append(fit)
        return fits, info
    found = [_plane_candidates(images, r, d_range, init=q, **options) for r, q in zip(regions, inits)]
    if not all(f.found for f in found):
        raise ValueError('No stereo hypothesis inside the disparity range')
    # Each plate proposes where its neighbour should be (see _Hypotheses.hinge).
    leading = [[c['params'] for c in f.found[:2]] for f in found]
    for k in (0, 1) if options.get('search', True) else ():
        for params in leading[1 - k]:
            found[k].hinge(params, zone, tolerance)
    sampled, sets = [f.region for f in found], [f.found for f in found]
    pair, ambiguity, by_prior = _choose_pair(sets, zone, tolerance, expected, prior_tolerance)
    fits = [_finish_fit(images, region, chosen, found, ambiguity)
            for region, chosen, found in zip(sampled, pair, sets)]
    return fits, {'ambiguity': float(ambiguity), 'resolved_by_prior': bool(by_prior)}


def locate_seam(left, right, intrinsics, baseline, mask_a, mask_b, depth_range=(.25, 4.), init_depth=None,
                spacing=.002, end_margin=0., neighbourhood=None, expected_distance=None, max_ambiguity=.85,
                exclude=None, init_planes=None, **fit_options):
    """Two plate masks on the left image -> metric 3D seam polyline in the left camera frame.

    Each plate is fitted where it adjoins the other one (``neighbourhood`` pixels;
    by default the reach of the smaller plate): a seam is a local feature and
    segmentations leak far away from it.  ``status`` says whether a seam was
    found; ``checks`` carries truth-free quality indicators.

    Texture that repeats along the baseline lets the images align at several
    depths.  Such a result is refused (``ambiguous_stereo_match``) unless a rough
    distance to the joint is known: ``expected_distance`` in metres (from an
    earlier overview, the planned stand-off, ...) or a depth map ``init_depth``.
    It only has to tell the rival depths apart, typically 10 % or more.
    ``exclude`` marks left-image pixels that belong to neither plate face.
    ``init_planes`` (two disparity planes from an earlier call) skips the search and only polishes them.
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
    # Where the plates meet in the image: between the masks (a weld bead may lie there), a little
    # into each.  Occlusion boundaries, where one plate hides the other, come along; they are a
    # minority and the contact test is a median.
    reach_of_gap = max(gap / 2 + width / 200, width / 40)
    contact = (reach[0] <= reach_of_gap) & (reach[1] <= reach_of_gap)
    rows, columns = np.nonzero(contact)
    zone = np.c_[columns, rows, np.ones(len(rows))][::max(1, len(rows) // 2000)].astype(float)
    if exclude is not None:
        regions = [r & ~np.asarray(exclude, bool) for r in regions]
        if min(r.sum() for r in regions) < 200:
            result['status'] = 'plates_are_not_adjacent'
            return result
    inits, expected = [None, None], None
    if init_depth is not None and np.ndim(init_depth) == 2:
        inits = [disparity_plane_from_depth(init_depth, r, focal, baseline) for r in regions]
        known = init_depth[contact]
        known = known[np.isfinite(known) & (known > 0)]
        expected = focal * baseline / float(np.median(known)) if len(known) else None
    if init_planes is not None:
        inits = [np.asarray(q, float) for q in init_planes]
        fit_options = {**fit_options, 'search': False}
        expected = float(np.median(zone @ (inits[0] + inits[1])) / 2) if len(zone) else None
    if expected_distance is not None:
        # Distance along the viewing ray -> depth along the optical axis at the contact zone.
        rays = np.c_[zone[:, :2], np.ones(len(zone))] @ np.linalg.inv(intrinsics).T
        cosine = float(np.median(1 / np.linalg.norm(rays, axis=1))) if len(zone) else 1.
        expected = focal * baseline / (expected_distance * cosine)
    try:
        fits, selection = _fit_plates(images, regions, d_range, inits, fit_options, zone,
                                      reach_of_gap + max(4., neighbourhood / 8), expected)
    except ValueError:
        result['status'] = 'no_stereo_match_in_depth_range'
        return result
    planes = [plane_from_disparity(f['params'], intrinsics, baseline) for f in fits]
    result.update(fits=fits, planes=planes, neighbourhood_px=float(neighbourhood),
                  checks={'depth_ambiguity': selection['ambiguity'],
                          'depth_resolved_by_prior': selection['resolved_by_prior'] or (
                              expected is not None and selection['ambiguity'] > max_ambiguity)})
    if selection['ambiguity'] > max_ambiguity and expected is None:
        result['status'] = 'ambiguous_stereo_match'
        return result
    try:
        point, direction, _ = intersect_planes(*planes)
    except ValueError:
        result['status'] = 'plates_nearly_parallel'
        return result

    # Finite extent: where both (convex) plates lie beside the projected intersection.
    span = 4 * max(height, width) / focal * point[2]
    reach = [-span, span]
    if abs(direction[2]) > 1e-9:       # stay in front of the camera: a joint seen end-on runs past it
        limit = (.5 * depth_range[0] - point[2]) / direction[2]
        reach = [max(reach[0], limit), reach[1]] if direction[2] > 0 else [reach[0], min(reach[1], limit)]
    ends = project(point + np.outer(reach, direction), intrinsics)
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
    if (ends_xyz[1] - ends_xyz[0]) @ direction < 0:
        direction = -direction
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
        checks={**result['checks'], 'plate_inlier_fraction': [f['inlier_fraction'] for f in fits],
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
