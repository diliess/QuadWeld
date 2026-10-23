"""Local host-surface estimates for a bead edge, in DA3 camera coordinates.

Inverse optical-Z is linear in image coordinates on a plane. A quadratic
model additionally permits a gently curved neighbouring bead. Both are
extrapolations from samples OUTSIDE the current bead and require held-out
depth agreement; neither establishes metrology accuracy.
"""

import cv2
import numpy as np
from scipy.ndimage import median_filter


def continuous_support(points, supported):
    """Flag abrupt 3D steps between otherwise supported neighbouring samples."""
    supported = np.asarray(supported, dtype=bool).copy()
    if len(points) < 3:
        return supported
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    eligible = supported[:-1] & supported[1:]
    if not eligible.any():
        return supported
    typical = float(np.median(steps[eligible]))
    local = median_filter(np.where(eligible, steps, typical), size=11, mode="nearest")
    threshold = np.maximum(5 * local, .003 * np.median(points[:, 2]))
    jumps = eligible & (steps > threshold)
    supported[:-1] &= ~jumps
    supported[1:] &= ~jumps
    return supported


def _sample(array, xy):
    return cv2.remap(array.astype(np.float32), xy[..., 0].astype(np.float32),
                     xy[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _robust_fit(design, target):
    if len(target) < 15 or np.linalg.cond(design) > 1e4:
        return None
    weights = np.ones(len(target))
    for _ in range(4):
        root = np.sqrt(weights)
        coefficients = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)[0]
        residual = target - design @ coefficients
        scale = max(1e-7, float(np.median(np.abs(residual))) * 1.4826)
        weights = np.minimum(1., 1.5 * scale / np.maximum(np.abs(residual), 1e-10))
    return coefficients


def local_host_surface(curve, centers, across, seed, sign, depth, intrinsics, normals, quality_mask):
    """Return per-point validated local plane/quadratic estimates, with finite fallbacks.

    Kind codes: 0 = unconfirmed raw depth, 2 = local plane, 3 = local quadratic.
    Code 1 is reserved for a mother-plane estimate chosen by the caller.
    """
    offsets = seed[None] + sign * np.array([3., 4., 5., 6., 8.])[:, None]
    xy = centers[None] + offsets[..., None] * across[None]
    z_samples = _sample(depth, xy)
    n_samples = _sample(normals, xy)
    eligible = ((xy[..., 0] >= 2) & (xy[..., 0] < depth.shape[1]-2)
                & (xy[..., 1] >= 2) & (xy[..., 1] < depth.shape[0]-2)
                & (_sample(quality_mask, xy) >= .99))
    observed_z = _sample(depth, curve[None])[0]
    chosen_z = observed_z.copy()
    kinds = np.zeros(len(curve), dtype=np.uint8)
    fit_rmse = np.zeros(len(curve))
    cv_error = np.zeros(len(curve))
    for index, point in enumerate(curve):
        first, last = max(0, index-7), min(len(curve), index+8)
        sample_xy = xy[:, first:last].reshape(-1, 2)
        sample_z = z_samples[:, first:last].ravel()
        sample_normals = n_samples[:, first:last].reshape(-1, 3)
        reference = np.median(n_samples[:, index], axis=0)
        reference /= max(np.linalg.norm(reference), 1e-8)
        valid = eligible[:, first:last].ravel() & (sample_normals @ reference > np.cos(np.deg2rad(45)))
        if valid.sum() < 30:
            continue
        tangent = np.array([across[index, 1], -across[index, 0]])
        relative = sample_xy-point
        u, v = relative @ across[index] / 8., relative @ tangent / 12.
        design = np.column_stack((np.ones(len(u)), u, v, u*u, u*v, v*v))[valid]
        target = 1. / sample_z[valid]
        row, column = np.indices((len(offsets), last-first))
        holdout = ((row+column) % 4 == 0).ravel()[valid]
        if holdout.sum() < 6 or (~holdout).sum() < 20:
            continue
        linear = _robust_fit(design[~holdout, :3], target[~holdout])
        quadratic = _robust_fit(design[~holdout], target[~holdout])
        if linear is None:
            continue

        def validation_error(coefficients):
            predicted = design[holdout, :len(coefficients)] @ coefficients
            if (predicted <= 0).any():
                return np.inf
            return float(np.median(np.abs(1./predicted - 1./target[holdout])))

        linear_error = validation_error(linear)
        quadratic_error = validation_error(quadratic) if quadratic is not None else np.inf
        # Prefer extra curvature only when held-out samples justify it.
        curved = quadratic_error < .7 * linear_error and linear_error > observed_z[index] * .0002
        columns = 6 if curved else 3
        error = quadratic_error if curved else linear_error
        coefficients = _robust_fit(design[:, :columns], target)
        if coefficients is None or coefficients[0] <= 0:
            continue
        fitted_z = 1./coefficients[0]
        predictions = design[:, :columns] @ coefficients
        if (predictions <= 0).any():
            continue
        rmse = float(np.sqrt(np.mean((1./predictions-1./target)**2)))
        if (error > observed_z[index] * .004 or rmse > observed_z[index] * .006
                or abs(fitted_z-observed_z[index])/observed_z[index] >= .03):
            continue
        chosen_z[index] = fitted_z
        kinds[index] = 3 if curved else 2
        fit_rmse[index], cv_error[index] = rmse, error
    rays = np.column_stack((curve, np.ones(len(curve)))) @ np.linalg.inv(intrinsics).T
    return {"xyz": rays * chosen_z[:, None], "valid": kinds > 0, "kind": kinds,
            "fit_rmse_estimated_m": fit_rmse, "validation_error_estimated_m": cv_error}
