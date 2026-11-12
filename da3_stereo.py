"""Calibrated two-view DA3 adapter; geometry/calibration only, no depth truth input."""

from pathlib import Path
import hashlib
import json
import types

import numpy as np


def camera_centres(extrinsics):
    e = np.asarray(extrinsics, dtype=np.float64)
    if e.shape not in ((2, 3, 4), (2, 4, 4)) or not np.isfinite(e).all():
        raise ValueError('Expected two finite OpenCV world-to-camera transforms')
    r = e[:, :3, :3]
    if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-4) or not np.allclose(
            np.linalg.det(r), 1., atol=1e-4):
        raise ValueError('Camera rotations must be proper orthonormal matrices')
    if e.shape[1] == 4 and not np.allclose(e[:, 3], [0, 0, 0, 1]):
        raise ValueError('Invalid homogeneous extrinsics')
    return -np.einsum('nij,nj->ni', r.transpose(0, 2, 1), e[:, :3, 3])


def baseline_scale(predicted_extrinsics, calibrated_extrinsics):
    known = float(np.linalg.norm(np.diff(camera_centres(calibrated_extrinsics), axis=0)))
    predicted = float(np.linalg.norm(np.diff(camera_centres(predicted_extrinsics), axis=0)))
    if min(known, predicted) <= 1e-6:
        raise ValueError('Collapsed stereo baseline; cannot recover metric scale')
    return known/predicted, {'calibrated_baseline_m': known, 'predicted_baseline_before_alignment': predicted,
                             'depth_multiplier': known/predicted,
                             'method': 'known_baseline / predicted_baseline; no truth depth fitting'}


def load_pair(path):
    """Validate RGB hashes and the camera inputs without loading evaluation files."""
    from PIL import Image
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text())
    if data.get('schema') != 'quadweld-stereo-v1' or len(data.get('views', [])) != 2:
        raise ValueError('Expected quadweld-stereo-v1 with exactly two synchronized views')
    images, k, e, sizes = [], [], [], []
    if data.get('synchronized') is not True:
        raise ValueError('Stereo images must be synchronized')
    if data.get('length_unit') != 'm':
        raise ValueError('Stereo extrinsics and baseline must use metres (length_unit=m)')
    if [view.get('id') for view in data['views']] != ['left','right']:
        raise ValueError('Stereo views must be ordered left, right')
    for view in data['views']:
        image = (path.parent/view['image']).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        if hashlib.sha256(image.read_bytes()).hexdigest() != view['image_sha256']:
            raise ValueError('RGB image hash differs from stereo calibration record')
        with Image.open(image) as im:
            if list(im.size) != view['image_size_wh']:
                raise ValueError('Image size and intrinsics calibration disagree')
            sizes.append(list(im.size))
        if np.any(np.asarray(view['distortion_coefficients']) != 0):
            raise ValueError('Undistort stereo images and update intrinsics before inference')
        matrix = np.asarray(view['intrinsics'], dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0:
            raise ValueError('Invalid stereo intrinsics')
        if not np.allclose(matrix[2], [0, 0, 1]):
            raise ValueError('Invalid homogeneous intrinsic matrix')
        images.append(str(image))
        k.append(matrix)
        e.append(view['extrinsics_w2c'])
    if sizes[0] != sizes[1]:
        raise ValueError('This stereo adapter requires equal input image sizes')
    e = np.asarray(e, dtype=np.float64)
    centres = camera_centres(e)
    baseline = float(np.linalg.norm(centres[1]-centres[0]))
    if baseline <= 1e-6 or not np.isclose(baseline, data['baseline_m'], atol=1e-6):
        raise ValueError('Stereo baseline and extrinsics disagree')
    return data, images, np.asarray(k, dtype=np.float32), e.astype(np.float32)


def install_two_view_alignment(model):
    """Instance-local replacement for the rank-deficient two-centre Sim(3) fit.

    DA3's native preprocessing and pose-conditioned forward pass are unchanged.
    No vendor code or weights are changed. Full rotation alignment is unnecessary
    for per-view axial depths when calibrated camera poses are already supplied.
    """
    original = model._align_to_input_extrinsics_intrinsics

    def align(instance, extrinsics, intrinsics, prediction, align_to_input_ext_scale=True,
              ransac_view_thresh=10):
        if extrinsics is None or len(extrinsics) != 2:
            return original(extrinsics, intrinsics, prediction, align_to_input_ext_scale, ransac_view_thresh)
        if not align_to_input_ext_scale or intrinsics is None:
            raise ValueError('Calibrated stereo requires intrinsics and input-baseline scale alignment')
        calibrated = extrinsics.detach().cpu().numpy()
        processed_k = intrinsics.detach().cpu().numpy()
        factor, record = baseline_scale(prediction.extrinsics, calibrated)
        instance.stereo_alignment_record = record
        instance.stereo_raw = {'depth_before_baseline_m': prediction.depth.copy(),
                               'predicted_extrinsics_before_alignment': prediction.extrinsics.copy(),
                               'predicted_intrinsics_before_alignment': prediction.intrinsics.copy()}
        prediction.depth = prediction.depth*factor
        prediction.extrinsics = calibrated[:, :3, :].copy()
        prediction.intrinsics = processed_k.copy()
        # Metric provenance is the known baseline, independent of the checkpoint flag.
        prediction.is_metric = 1
        return prediction

    model._align_to_input_extrinsics_intrinsics = types.MethodType(align, model)
    return model


def stream_transformer_blocks(model):
    """Keep transformer blocks on CPU between calls, retaining original dtypes.

    This reduces VRAM without quantizing weights or changing arithmetic. The
    current Nested and Any-view implementations call each block sequentially.
    """
    import torch
    branches = [model.model.da3, model.model.da3_metric] if hasattr(model.model, 'da3_metric') else [model.model]
    detached = []
    for branch in branches:
        encoder = branch.backbone.pretrained
        blocks = encoder.blocks
        detached.append((encoder, blocks))
        encoder.blocks = torch.nn.ModuleList()
    model.to('cuda')
    model.device = torch.device('cuda')
    hooks = []
    for encoder, blocks in detached:
        encoder.blocks = blocks
        for block in blocks:
            def before(module, inputs):
                module.to('cuda')
            def after(module, inputs, output):
                module.to('cpu')
            hooks.extend([block.register_forward_pre_hook(before), block.register_forward_hook(after)])
    model._streaming_hooks = hooks
    return model
