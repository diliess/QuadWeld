"""Weld toes of a joint that already carries a bead, from one calibrated stereo pair.

A deposited bead (fish-scale fillet, root pass, ...) is a rippled, curved body:
it has no plane to fit and it hides the corner the two plates used to form.
What the next pass needs are the two *toes*, the curves where the bead surface
runs out into each plate.  A toe lies on its plate, and plates are what
``stereo_seam`` measures well.  So:

1. fit the two plate planes beside the bead (``stereo_seam.locate_seam``); their
   intersection is the hidden root line and gives the joint a coordinate frame;
2. resample the images on each plate plane, in metres along the joint (s) and
   away from the root (t).  In this rectified strip the toe is a curve t(s)
   that separates bead from bare plate;
3. trace that curve (appearance of bead versus plate, dynamic programming along
   the joint, sub-pixel edge), independently in the left and in the right image;
4. a point of the plate plane looks the same from both eyes, so the two traces
   agree where the boundary really lies on the plate and disagree where it is
   a silhouette of the bead in front of it.  Agreeing points are lifted to 3D
   exactly: they are points of a known plane.

No depth is ever estimated on the bead itself.
"""
import warnings

import cv2
import numpy as np

from . import stereo_seam as ss


def _strip(image, origin, direction, across, s, t, intrinsics, shift=0.):
    """Sample ``image`` on the plane patch origin + s*direction + t*across -> (len(t), len(s), channels)."""
    points = origin + s[None, :, None] * direction + t[:, None, None] * across
    xy = ss.project(points.reshape(-1, 3) - np.array([shift, 0., 0.]), intrinsics)
    xy = xy.reshape(len(t), len(s), 2).astype(np.float32)
    values = cv2.remap(image, xy[..., 0], xy[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=np.nan)
    return values.reshape(len(t), len(s), -1), xy


def _running_median(values, window):
    """Median of each column's neighbourhood along the joint (rows: samples, columns: joint positions)."""
    count = values.shape[1]
    centres = np.arange(0, count, max(1, window // 4))
    coarse = np.stack([np.nanmedian(values[:, max(0, c - window):c + window + 1], axis=(0, 1)) for c in centres])
    return np.stack([np.interp(np.arange(count), centres, coarse[:, k]) for k in range(coarse.shape[1])], axis=1)


def _trace(strip, t, step, min_leg, max_leg, smooth, window, toe_level, stiffness=.6):
    """Toe t(s) between bead (small t) and plate (large t) in one rectified strip.

    Returns t(s) in metres, whether a toe was found in each column, and the
    bead/plate contrast in units of their spread.
    """
    rows, columns, _ = strip.shape
    known = np.isfinite(strip).all(2)
    filled = np.where(known[..., None], strip, 0.)
    sigma = max(smooth / step, .5)
    weight = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=.5)
    blurred = cv2.GaussianBlur(filled, (0, 0), sigmaX=sigma, sigmaY=.5).reshape(rows, columns, -1)
    with np.errstate(invalid='ignore'):
        blurred = blurred / np.maximum(weight, 1e-3)[..., None]
    blurred[weight < .5] = np.nan
    # Reference appearance along the joint.  The root line is covered by the bead, so the strip
    # around t = 0 shows bead; beyond the largest leg it shows bare plate.
    bead_rows = (t >= -.25 * min_leg) & (t <= .6 * min_leg)
    plate_rows = t >= max_leg
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        bead = _running_median(blurred[bead_rows], window)
        plate = _running_median(blurred[plate_rows], window)
        axis = bead - plate
        norm = np.maximum((axis ** 2).sum(1), 1e-12)
        score = ((blurred - plate[None]) * axis[None]).sum(2) / norm[None]          # 1: bead, 0: plate
        # How far apart bead and plate are, in units of what each of them varies by itself
        # (ripples, texture) before any smoothing: below a few units the two cannot be told apart.
        raw = ((np.where(known[..., None], strip, np.nan) - plate[None]) * axis[None]).sum(2) / norm[None]
        spread = sum(np.sqrt(np.maximum(_smooth_along(np.nan_to_num(np.nanvar(raw[rows_], axis=0)),
                                                      np.isfinite(raw[rows_]).any(0), window), 0.))
                     for rows_ in (bead_rows, plate_rows))
    contrast = np.nan_to_num(1 / np.maximum(spread, 1e-6))
    score = np.nan_to_num(score, nan=.5)
    clipped = np.clip(score, 0., 1.)
    # 1. Robust body of the bead: cost of placing the boundary before row r (bead above, plate from it on).
    above = np.cumsum((1 - clipped) ** 2, axis=0)
    below = np.cumsum((clipped ** 2)[::-1], axis=0)[::-1]
    cost = np.vstack((np.zeros((1, columns)), above[:-1])) + below
    cost[~((t >= .5 * min_leg) & (t <= max_leg))] = np.inf
    reach = 3                                         # rows the boundary may move per column
    total = np.empty_like(cost)
    back = np.zeros(cost.shape, np.int8)
    total[:, 0] = cost[:, 0]
    for column in range(1, columns):
        previous = total[:, column - 1]
        options = np.full((2 * reach + 1, rows), np.inf)
        for index, move in enumerate(range(-reach, reach + 1)):
            source = np.roll(previous, move)
            if move > 0:
                source[:move] = np.inf
            elif move < 0:
                source[move:] = np.inf
            options[index] = source + stiffness * abs(move)
        choice = options.argmin(0)
        total[:, column] = cost[:, column] + options[choice, np.arange(rows)]
        back[:, column] = choice - reach
    path = np.empty(columns, np.intp)
    path[-1] = int(np.argmin(total[:, -1]))
    for column in range(columns - 1, 0, -1):
        path[column - 1] = path[column] - back[path[column], column]
    # 2. The toe is the foot of the bead's flank, not the middle of the brightness step: the flank
    # is shaded and often ends in a dark crease.  Walk outwards from the body to where the score
    # has come down to the level it settles at (crease or plate), within ``toe_level`` of the step.
    span = max(2, int(round(.003 / step)))
    index = np.arange(columns)
    offsets = np.arange(0, span + 1)
    window_rows = np.clip(path[None] - 1 + offsets[:, None], 0, rows - 1)
    profile = score[window_rows, index[None]]                      # (span + 1, columns), outwards
    foot = profile.min(0)
    level = foot + toe_level * (1 - foot)
    under = profile <= level[None]
    first = under.argmax(0)                                         # first row at or below the level
    found = under.any(0) & (first > 0) & (profile[0] > level)
    inner = profile[np.maximum(first - 1, 0), index]
    outer = profile[first, index]
    fraction = (inner - level) / np.maximum(inner - outer, 1e-9)
    position = t[np.clip(path - 1, 0, rows - 1)] + (np.maximum(first - 1, 0) + np.clip(fraction, 0, 1)) * step
    return np.where(found, position, t[path]), found, contrast


def _smooth_along(values, valid, sigma):
    weight = cv2.GaussianBlur(valid.astype(np.float64)[None], (0, 0), sigmaX=sigma, sigmaY=.1)[0]
    total = cv2.GaussianBlur(np.where(valid, values, 0.)[None], (0, 0), sigmaX=sigma, sigmaY=.1)[0]
    return total / np.maximum(weight, 1e-6)


def trace_toes(left, right, intrinsics, baseline, seam, min_leg=.003, max_leg=.02, spacing=.001, smooth=.0015,
               agreement=.0006, min_contrast=3., toe_level=.1, max_footprint=.0016):
    """Toes of the bead on both plates, given the plate planes and root line from ``locate_seam``.

    ``min_leg`` / ``max_leg`` bound the distance of a toe from the root line (m).
    ``smooth`` is the length along the joint over which ripples are averaged.
    ``toe_level``: the toe is placed where the bead's flank has faded to this
    fraction of the bead/plate contrast above the level it settles at.
    ``max_footprint``: a toe is not reported where one pixel covers more than this
    many metres across it (plate seen at a grazing angle, or from too far).
    Returns one record per plate with a 3D polyline sampled every ``spacing``
    metres; ``valid`` marks points confirmed by both eyes within ``agreement``.
    """
    intrinsics = np.asarray(intrinsics, float)
    images = [np.asarray(im, np.float32) / 255 if np.asarray(im).dtype == np.uint8 else np.asarray(im, np.float32)
              for im in (left, right)]
    height, width = images[0].shape[:2]
    ends, direction = np.asarray(seam['joint_ends_xyz'], float), np.asarray(seam['direction'], float)
    length = float(np.linalg.norm(ends[1] - ends[0]))
    step = .5 * float(ends[:, 2].mean()) / float(intrinsics[0, 0])
    s = np.arange(0., length + step / 2, step)
    t = np.arange(-.5 * min_leg, max_leg + max(.008, .5 * max_leg) + step / 2, step)
    window = max(8, int(round(.01 / step)))
    facing = [np.asarray(f, float) for f in seam['face_normals']]
    out_s = np.arange(0., length + spacing / 2, spacing)
    toes = []
    for k in (0, 1):
        normal = seam['planes'][k][0]
        across = facing[1 - k] - (facing[1 - k] @ normal) * normal
        across /= np.linalg.norm(across)
        traces = []
        for image, shift in zip(images, (0., baseline)):
            strip, xy = _strip(image, ends[0], direction, across, s, t, intrinsics, shift)
            position, found, contrast = _trace(strip, t, step, min_leg, max_leg, smooth, window, toe_level)
            seen = np.isfinite(strip).all(2).mean(0) > .9
            traces.append((position, contrast, found, seen))
        (tl, cl, hl, sl), (tr, cr, hr, sr) = traces
        both = sl & sr
        valid = both & (np.abs(tl - tr) <= agreement) & (np.minimum(cl, cr) >= min_contrast) & hl & hr
        valid &= (np.minimum(tl, tr) > .5 * min_leg + step) & (np.maximum(tl, tr) < max_leg - step)
        mean = np.where(both, (tl + tr) / 2, np.where(sl, tl, tr))
        sigma = max(.5 * spacing / step, .5)
        if valid.any():      # between and beyond confirmed stretches: carried over from the nearest ones
            leg = np.interp(out_s, s[valid], _smooth_along(mean, valid, sigma)[valid])
        else:
            leg = np.interp(out_s, s, mean)
        support = np.interp(out_s, s, _smooth_along(valid.astype(float), np.ones(len(s), bool), sigma))
        xyz = ends[0] + out_s[:, None] * direction + leg[:, None] * across
        footprint = .001 / np.maximum(np.linalg.norm(
            ss.project(xyz + .001 * across, intrinsics) - ss.project(xyz, intrinsics), axis=1), 1e-9)
        ok = (support > .7) & (footprint <= max_footprint)
        toes.append({
            'plate': k, 'xyz': xyz, 'valid': ok, 'leg_m': leg, 's_m': out_s,
            'xy': ss.project(xyz, intrinsics), 'across': across,
            'xy_right': ss.project(xyz - np.array([baseline, 0., 0.]), intrinsics),
            'coverage': float(ok.mean()), 'across_footprint_m': float(np.median(footprint)),
            'left_right_difference_m': float(np.median(np.abs(tl - tr)[both])) if both.any() else np.nan,
            'contrast': float(np.median(np.minimum(cl, cr)[both])) if both.any() else 0.,
            'trace': {'s_m': s, 'left_m': tl, 'right_m': tr, 'valid': valid}})
    return toes


def polyline_segments(toe, min_points=5):
    """Valid runs of a toe as separate 3D polylines."""
    valid = np.asarray(toe['valid'], bool)
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(int)))
    return [toe['xyz'][a:b] for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_points]


def locate_bead_toes(left, right, intrinsics, baseline, mask_a, mask_b, min_leg=.003, max_leg=.02, spacing=.001,
                     refit=True, clearance=1.5, **seam_options):
    """Two plate masks on the left image -> root line and both weld toes (3D polylines, left camera frame).

    The masks may be rough and may run into the bead: with ``refit`` the planes
    are fitted a second time without the band the first pass found to be bead.
    ``clearance`` (in bead legs) is how far beyond each toe the plates are also left out.
    Result: ``status``; ``seam`` (the ``locate_seam`` record: planes, hidden root
    line, checks); ``toes`` (one record per mask, see ``trace_toes``).
    """
    intrinsics = np.asarray(intrinsics, float)
    stereo = ss.StereoImages(left, right)
    seam = ss.locate_seam(stereo, None, intrinsics, baseline, mask_a, mask_b, **seam_options)
    result = {'status': seam['status'], 'seam': seam, 'toes': []}
    if seam['status'] != 'ok':
        return result
    toes = trace_toes(left, right, intrinsics, baseline, seam, min_leg, max_leg, spacing)
    if refit and all(toe['valid'].any() for toe in toes):
        # Everything between the two toes is bead, not plate.  A glossy plate also mirrors the bead
        # and is shaded by it over a width comparable with the bead's size; that band is not
        # attached to the surface and drags the plane (most when the baseline crosses the joint).
        footprint = float(seam['xyz'][:, 2].mean() / intrinsics[0, 0])
        outline = [toe['xyz'] + (max(clearance * float(np.median(toe['leg_m'][toe['valid']])), .01) + 12 * footprint)
                   * toe['across'] for toe in toes]
        polygon = np.rint(np.concatenate((ss.project(outline[0], intrinsics),
                                          ss.project(outline[1], intrinsics)[::-1]))).astype(np.int32)
        bead = np.zeros(stereo.shape, np.uint8)
        cv2.fillPoly(bead, [polygon], 1)
        again = ss.locate_seam(stereo, None, intrinsics, baseline, mask_a, mask_b, exclude=bead.astype(bool),
                               init_planes=[f['params'] for f in seam['fits']],
                               **{k: v for k, v in seam_options.items() if k not in ('init_depth', 'expected_distance')})
        if again['status'] == 'ok':
            again['checks']['depth_ambiguity'] = seam['checks']['depth_ambiguity']
            again['checks']['depth_resolved_by_prior'] = seam['checks']['depth_resolved_by_prior']
            result['seam'], result['bead_mask'] = again, bead.astype(bool)
            toes = trace_toes(left, right, intrinsics, baseline, again, min_leg, max_leg, spacing)
    result['toes'] = toes
    if not any(toe['valid'].any() for toe in toes):
        result['status'] = 'no_bead_toe_found'
    return result


def pick_plate_pair(masks, scores=None, min_area=.005):
    """Choose the two plates of the joint in view from raw segmentation candidates (e.g. SAM text prompts).

    The pair sharing the longest border wins, counted with more weight near the
    image centre, where the camera is aimed, and weighted by the detection scores.  Returns the two indices (larger
    mask first) or ``None``.
    """
    masks = [ss.clean_mask(m) for m in masks]
    if not masks:
        return None
    height, width = masks[0].shape
    scores = np.ones(len(masks)) if scores is None else np.asarray(scores, float)
    areas = np.array([m.sum() for m in masks], float)
    keep = []
    for index in np.argsort(-scores):
        if areas[index] < min_area * height * width:
            continue
        # A plate face has room for a plane fit; thin strips are beads, edges and gaps.
        if cv2.distanceTransform(masks[index].astype(np.uint8), cv2.DIST_L2, 3).max() < width / 25:
            continue
        if all((masks[index] & masks[j]).sum() < .8 * min(areas[index], areas[j]) for j in keep):
            keep.append(int(index))
    radius = max(3, width // 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    grown = {j: cv2.dilate(masks[j].astype(np.uint8), kernel).astype(bool) for j in keep}
    ys, xs = np.mgrid[:height, :width]
    centrality = np.exp(-(((xs - width / 2) / (.5 * width)) ** 2 + ((ys - height / 2) / (.5 * height)) ** 2))
    best, best_value = None, 0.
    for a in keep:
        for b in keep:
            if a >= b or (masks[a] & masks[b]).sum() > .2 * min(areas[a], areas[b]):
                continue
            border = float(centrality[grown[a] & grown[b]].sum()) / (2 * radius)
            value = border * scores[a] * scores[b]
            if border > 20. and value > best_value:
                best, best_value = (a, b) if areas[a] >= areas[b] else (b, a), value
    return best


def fuse_toes(views, spacing=.001, min_views=2, tolerance=.005):
    """Toes seen from several camera poses, all given in one common frame -> one polyline per plate.

    ``views``: records with ``root`` (the two joint ends, shape (2, 3)), ``normals``
    (the two plate normals) and ``toes`` (per plate ``xyz`` and ``valid``), in the
    common frame.  Plates are matched across views by their normals.  Along the
    fused root line every ``spacing`` metres the median over the views is taken;
    ``views`` in the result counts how many contributed.
    """
    axis = ss.fuse_seams([v['root'] for v in views], tolerance)
    origin, direction = axis['ends'][0], axis['direction']
    count = int(np.ceil(axis['length_m'] / spacing)) + 1
    reference = np.asarray(views[0]['normals'], float)
    fused = []
    for plate in (0, 1):
        stack = np.full((len(views), count, 3), np.nan)
        for index, view in enumerate(views):
            if not axis['view_used'][index]:
                continue
            normals = np.asarray(view['normals'], float)
            own = int(np.argmax(normals @ reference[plate]))
            toe = view['toes'][own]
            points = np.asarray(toe['xyz'], float)[np.asarray(toe['valid'], bool)]
            if not len(points):
                continue
            bins = np.rint((points - origin) @ direction / spacing).astype(int)
            inside = (bins >= 0) & (bins < count)
            total = np.zeros((count, 3))
            number = np.zeros(count)
            np.add.at(total, bins[inside], points[inside])
            np.add.at(number, bins[inside], 1)
            stack[index, number > 0] = total[number > 0] / number[number > 0, None]
        seen = np.isfinite(stack[..., 0]).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            xyz = np.nanmedian(stack, axis=0)
            spread = np.nanmax(np.linalg.norm(stack - xyz[None], axis=2), axis=0)
        ok = seen >= min_views
        fused.append({'xyz': xyz, 'valid': ok, 'views': seen, 'spread_m': np.where(ok, spread, np.nan),
                      's_m': np.arange(count) * spacing})
    return {'root': axis, 'toes': fused}
