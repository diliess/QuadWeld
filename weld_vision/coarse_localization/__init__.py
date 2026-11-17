"""Far-view stereo localization, without simulator, robot or filesystem inputs.

``locate`` consumes rectified images, two plate masks and calibration/pose.
``measure_overview`` adapts the existing capture/SAM directory format.
"""
import numpy as np

from .. import stereo_seam
from .._localization import read_capture, transform_points, validate_measurement

__all__ = ['locate', 'measure_overview']


def locate(left, right, intrinsics, baseline, mask_a, mask_b, world_from_left, *,
           ident=None, depth_range=(1.4, 2.7), expected_distance=None,
           range_origin_world=None, require_t_joint=False):
    """Return status, coarse world root/normals, range and stereo diagnostics.

    K uses integer pixel centres. Images/masks must already be rectified; poses
    map that rectified left optical frame to a shared world frame, in metres.
    The optional T-joint check assumes a Z-up world. It is enabled by the
    existing A25/A26 file adapter, not imposed on other joint geometries.
    """
    world = validate_measurement(left, right, intrinsics, baseline, mask_a, mask_b, world_from_left)
    options = {} if expected_distance is None else {'expected_distance': expected_distance}
    seam = stereo_seam.locate_seam(left, right, intrinsics, baseline, mask_a, mask_b,
                                  depth_range=depth_range, **options)
    result = {'id': ident, 'status': seam['status'], 'checks': seam['checks'], 'seam': seam}
    if seam['status'] != 'ok':
        return result
    ends = transform_points(seam['joint_ends_xyz'], world)
    normals = np.asarray(seam['face_normals']) @ world[:3, :3].T
    vertical = np.abs(normals[:, 2])
    if require_t_joint and (vertical.max() < .85 or vertical.min() > .35
                            or not .2 <= seam['length_m'] <= 2.5):
        result['status'] = 'unexpected_joint_geometry'
        return result
    origin = world[:3, 3] if range_origin_world is None else np.asarray(range_origin_world, float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('range_origin_world must be a finite 3D point in metres')
    result.update(root_world_m=ends, face_normals_world=normals,
                  estimated_range_m=float(np.linalg.norm(ends.mean(0) - origin)))
    return result


def measure_overview(run, view):
    """Read a saved overview pair/SAM candidates and preserve the legacy record."""
    ident = view['id']
    captured = read_capture(run, 'overview', ident)
    if captured is None:
        raise ValueError(f'No adjacent plate candidates in {ident}')
    *measurement, world = captured
    midpoint = np.asarray(view['world_from_optical_midpoint'])[:3, 3]
    result = locate(*measurement, world, ident=ident, range_origin_world=midpoint, require_t_joint=True)
    print('BEAD_OVERVIEW', view['station'], result['seam']['status'], result['checks'], flush=True)
    if result['status'] == 'unexpected_joint_geometry':
        raise ValueError(f'{ident}: coarse joint is inconsistent with a T joint')
    if result['status'] != 'ok':
        raise ValueError(f'{ident}: {result["status"]}')
    return {'id': ident, 'root_world_m': result['root_world_m'].tolist(),
            'face_normals_world': result['face_normals_world'].tolist(),
            'estimated_range_m': result['estimated_range_m'], 'checks': result['checks']}
