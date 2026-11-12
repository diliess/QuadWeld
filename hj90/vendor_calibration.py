"""Vendor rectified RGB calibration, kept separate from simulated stereo optics."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_RGB_CALIBRATION = Path(__file__).resolve().parents[1] / 'calibration/hj90_rgb_vendor.json'


def load_rgb_calibration(path=DEFAULT_RGB_CALIBRATION):
    """Interpret SDK T_optical_camera as RGB-from-left; convert only translation mm to m.

    VZNL_RGBConfig.h documents the source frame as the left-eye 3D frame.
    This record is for one RGB sensor, never two measured stereo intrinsics.
    The absent rectification map is unnecessary for an ideal rectified renderer,
    but prevents applying this model directly to raw distorted hardware images.
    """
    path = Path(path)
    raw = path.read_bytes()
    record = json.loads(raw)
    if (record.get('status') != 'calibrated' or record.get('length_unit') != 'mm'
            or record.get('optical_frame') != 'rgb_optical'
            or record.get('point_frame') != 'vizum_camera'):
        raise ValueError('Expected calibrated vendor RGB / left-eye coordinates in mm')
    size = np.asarray(record['resolution'])
    k = np.asarray(record['camera_matrix'], dtype=float)
    transform = np.asarray(record['T_optical_camera'], dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size < 1) or np.any(size != size.astype(int)):
        raise ValueError('Invalid vendor RGB resolution')
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0):
        raise ValueError('Invalid vendor RGB pinhole intrinsics')
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])):
        raise ValueError('Invalid vendor RGB homogeneous transform')
    rotation = transform[:3, :3]
    if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
        raise ValueError('Vendor RGB rotation must be proper orthonormal')
    if np.asarray(record['distortion']).shape != (5,) or np.any(np.asarray(record['distortion']) != 0):
        raise ValueError('Only the supplied rectified zero-distortion RGB raster is supported')
    transform[:3, 3] *= .001
    # Vendor/OpenCV integer pixel centres -> the edge-origin coordinates used by
    # USD and QuadWeld (pixel index + 0.5). Keep both conventions explicit.
    edge_k = k.copy()
    edge_k[:2, 2] += .5
    rectification = record.get('rectification', {})
    map_path = path.parent / rectification.get('path', '__missing_rectification_map__')
    map_supplied = map_path.is_file() and hashlib.sha256(map_path.read_bytes()).hexdigest() == rectification.get('sha256')
    return {
        'id': record['id'], 'device_serial': record['vendor_raw']['device_serial'],
        'image_size_wh': size.astype(int).tolist(), 'native_image_size_wh': size.astype(int).tolist(),
        'intrinsics_opencv': k.tolist(), 'intrinsics': edge_k.tolist(),
        'distortion_coefficients': record['distortion'], 'T_rgb_from_left_m': transform.tolist(),
        'source_file': path.name, 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'sdk_api': 'VZNL_GetRGBCalibParam',
        'vendor_pixel_convention': 'OpenCV integer pixel centres; board reprojection still required',
        'pixel_convention': 'edge origin; pixel centres at index + 0.5',
        'rectification_map_supplied': bool(map_supplied),
        'scope': f"rectified RGB calibration for serial {record['vendor_raw']['device_serial']}; not a stereo or laser calibration",
        'device_model_binding': 'applied from supplied materials; serial-to-HJ90 model identity not independently verified',
    }


def resize_rgb_calibration(calibration, size=None):
    """Resize the full raster, including its off-centre principal point; no cropping."""
    result = copy.deepcopy(calibration)
    if size is None:
        return result
    size = np.asarray(size)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size < 1) or np.any(size != size.astype(int)):
        raise ValueError('Expected a positive integer RGB width and height')
    scale = size / calibration['image_size_wh']
    k = np.asarray(calibration['intrinsics']).copy()
    k[:2] *= scale[:, None]
    result['image_size_wh'] = size.astype(int).tolist()
    result['intrinsics'] = k.tolist()
    opencv_k = k.copy()
    opencv_k[:2, 2] -= .5
    result['intrinsics_opencv'] = opencv_k.tolist()
    return result


def rgb_pose_in_midpoint(calibration, baseline_m):
    """Midpoint-from-RGB: midpoint-from-left @ inverse(RGB-from-left)."""
    if not np.isfinite(baseline_m) or baseline_m <= 0:
        raise ValueError('Expected positive stereo baseline in metres')
    midpoint_from_left = np.eye(4)
    midpoint_from_left[0, 3] = -baseline_m / 2
    return midpoint_from_left @ np.linalg.inv(np.asarray(calibration['T_rgb_from_left_m']))
