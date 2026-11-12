"""Camera-only grouping and scale recovery for joint multi-pose DA3 inference."""

import types

import numpy as np

from da3_stereo import load_pair


def camera_centres(extrinsics):
    e = np.asarray(extrinsics, dtype=np.float64)
    if e.ndim != 3 or e.shape[0] < 2 or e.shape[1:] not in ((3, 4), (4, 4)) or not np.isfinite(e).all():
        raise ValueError('Expected at least two finite world-to-camera transforms')
    rotation = e[:, :3, :3]
    if not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(rotation), 1., atol=1e-4):
        raise ValueError('Camera rotations must be proper orthonormal matrices')
    if e.shape[1] == 4 and not np.allclose(e[:, 3], [0, 0, 0, 1]):
        raise ValueError('Invalid homogeneous camera transform')
    return -np.einsum('nij,nj->ni', rotation.transpose(0, 2, 1), e[:, :3, 3])


def camera_distance_scale(predicted_extrinsics, calibrated_extrinsics):
    """Fit one scale to known inter-camera distances; no scene depths required.

    Distances are rotation/translation invariant, including collinear capture
    trajectories for which a full Sim(3) rotation fit is not uniquely defined.
    """
    predicted = camera_centres(predicted_extrinsics)
    known = camera_centres(calibrated_extrinsics)
    if predicted.shape != known.shape:
        raise ValueError('Predicted and calibrated view counts differ')
    a, b = np.triu_indices(len(known), 1)
    known_distances = np.linalg.norm(known[a] - known[b], axis=1)
    predicted_distances = np.linalg.norm(predicted[a] - predicted[b], axis=1)
    if max(known_distances) <= 1e-6 or max(predicted_distances) <= 1e-6:
        raise ValueError('Collapsed camera trajectory')
    factor = float(predicted_distances @ known_distances / (predicted_distances @ predicted_distances))
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('Invalid camera-only depth multiplier')
    return factor, {
        'method': 'Least squares scalar on calibrated inter-camera distances; no scene/depth truth fitting',
        'view_pairs': np.stack((a, b), axis=1).tolist(),
        'calibrated_camera_distances_m': known_distances.tolist(),
        'predicted_camera_distances_before_alignment': predicted_distances.tolist(),
        'depth_multiplier': factor,
        'camera_distance_residual_rmse_m': float(np.sqrt(np.mean((factor * predicted_distances - known_distances) ** 2))),
    }


def group_station_views(source, plan, station):
    """Assemble one static workpiece's images using the authored camera poses.

    Stereo input.json poses restart at the left eye for each capture. Rebuild a
    common frame from each planned optical midpoint and the physical baseline.
    No evaluation files, rendered depth, or weld annotations are read here.
    """
    records, images, intrinsics, world_poses = [], [], [], []
    for view in plan['views']:
        if view['station'] != station:
            continue
        folder = source / 'captures' / view['id']
        pair, pair_images, pair_k, _ = load_pair(folder / 'input.json')
        midpoint = np.asarray(view['world_from_optical_midpoint'], dtype=np.float64)
        for i, side in enumerate(('left', 'right')):
            offset = np.eye(4)
            offset[0, 3] = (-.5 if i == 0 else .5) * pair['baseline_m']
            world_poses.append(midpoint @ offset)
            images.append(pair_images[i])
            intrinsics.append(pair_k[i])
            records.append({**pair['views'][i], 'id': view['id'] + '_' + side,
                            'case': view['id'], 'view': side, 'station': station,
                            'patch': view['patch'], 'image': pair_images[i]})
    if len(records) < 4:
        raise ValueError('Joint inference needs multiple stereo camera poses')
    world_poses = np.asarray(world_poses)
    # Use the first eye as the common origin to avoid large world translations.
    extrinsics = np.linalg.inv(world_poses) @ world_poses[0]
    camera_centres(extrinsics)
    return records, images, np.asarray(intrinsics, dtype=np.float32), extrinsics.astype(np.float32), world_poses


def install_multiview_alignment(model):
    """Retain native pose-conditioned attention; replace only metric alignment."""
    def align(instance, extrinsics, intrinsics, prediction, align_to_input_ext_scale=True,
              ransac_view_thresh=10):
        if extrinsics is None or intrinsics is None or not align_to_input_ext_scale:
            raise ValueError('Joint calibrated inference requires camera inputs and scale alignment')
        calibrated = extrinsics.detach().cpu().numpy()
        processed_k = intrinsics.detach().cpu().numpy()
        factor, record = camera_distance_scale(prediction.extrinsics, calibrated)
        instance.multiview_alignment_record = record
        instance.multiview_raw = {
            'depth_before_camera_alignment': prediction.depth.copy(),
            'extrinsics_before_camera_alignment': prediction.extrinsics.copy(),
            'intrinsics_before_camera_alignment': prediction.intrinsics.copy(),
        }
        prediction.depth = prediction.depth * factor
        prediction.extrinsics = calibrated[:, :3, :].copy()
        prediction.intrinsics = processed_k.copy()
        prediction.is_metric = 1
        return prediction

    model._align_to_input_extrinsics_intrinsics = types.MethodType(align, model)
    return model
