"""Model-independent geometry and reference metrics for stereo evaluation."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()



def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')



def backproject(xy, z, k):
    import numpy as np
    return (np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(k).T)*np.asarray(z)[:, None]



def project(xyz, k):
    q = xyz @ k.T
    return q[:, :2]/q[:, 2:]



def sample_depth(depth, xy):
    from scipy.ndimage import map_coordinates
    return map_coordinates(depth, xy[:, ::-1].T, order=1, mode='constant', cval=float('nan'))



def runs(mask):
    import numpy as np
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)) if b-a >= 2]



def segment_distances(points, polylines):
    """Distance to finite continuous segments, without pose/scale fitting."""
    import numpy as np
    best = np.full(len(points), np.inf)
    nearest = np.full_like(points, np.nan, dtype=float)
    for curve in polylines:
        for a, b in zip(curve[:-1], curve[1:]):
            direction = b-a
            length2 = float(direction @ direction)
            if length2 < 1e-16:
                continue
            t = np.clip((points-a) @ direction/length2, 0, 1)
            candidate = a+t[:, None]*direction
            distance = np.linalg.norm(points-candidate, axis=1)
            update = distance < best
            best[update], nearest[update] = distance[update], candidate[update]
    return best, nearest



def error_stats(values):
    import numpy as np
    values = np.asarray(values)
    if not len(values):
        return None
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite error samples')
    return {'samples': len(values), 'mean_mm': float(values.mean()*1000),
            'rmse_mm': float(np.sqrt(np.mean(values**2))*1000),
            'p95_mm': float(np.percentile(values, 95)*1000), 'max_mm': float(values.max()*1000)}



def per_seam_metrics(curves, valid_parts, depth, intrinsics):
    """Evaluation-only closest-reference association; every prediction is kept."""
    import numpy as np
    points = np.concatenate(valid_parts) if valid_parts else np.zeros((0, 3))
    references = [[c['xyz'][a:b] for a, b in runs(c['in_frame'])] for c in curves]
    distances = np.stack([segment_distances(points, parts)[0] for parts in references], axis=1)
    assigned = distances.argmin(axis=1) if len(points) else np.zeros(0, dtype=int)
    records = []
    for i, (curve, parts) in enumerate(zip(curves, references)):
        target = np.concatenate(parts) if parts else np.zeros((0, 3))
        reverse = segment_distances(target, valid_parts)[0]
        errors = distances[assigned == i, i]
        errors = errors[np.isfinite(errors)]
        diagnostic = np.linalg.norm(backproject(curve['xy'][curve['in_frame']],
            sample_depth(depth, curve['xy'][curve['in_frame']]), intrinsics)-curve['xyz'][curve['in_frame']], axis=1)
        coverage10 = float(np.mean(reverse <= .01)) if len(target) else None
        records.append({'seam': curve['name'], 'nominal_length_m': float(curve['arc'][-1]),
                        'in_frame_samples': len(target), 'depth_visible_samples': int(curve['visible'].sum()),
                        'assigned_prediction_samples': len(errors), 'position_error': error_stats(errors),
                        'coverage_at_5mm': float(np.mean(reverse <= .005)) if len(target) else None,
                        'coverage_at_10mm': coverage10,
                        'coverage_at_20mm': float(np.mean(reverse <= .02)) if len(target) else None,
                        'coverage_at_50percent_threshold_passed': coverage10 >= .5 if coverage10 is not None else None,
                        'known_2d_diagnostic': error_stats(diagnostic),
                        'in_frame_depth_range_m': [float(target[:, 2].min()), float(target[:, 2].max())] if len(target) else None})
    return records



def ground_truth(source, record, k_index, shape, reference_mode='usd_reference'):
    """Independent evaluator: authored nominal seam curves, sampled every 1 mm."""
    import numpy as np
    from pxr import Usd, UsdGeom
    capture_dir = source/'captures'/record['case']
    capture = json.loads((capture_dir/'evaluation/capture.json').read_text())
    if digest(ROOT/'assets/scenes/industrial-welding/industrial_welding.usdc') != capture['source_usdc_sha256']:
        raise ValueError('Scene changed since RGB capture')
    calibration = json.loads((capture_dir/'input.json').read_text())
    i = 0 if record['view'] == 'left' else 1
    raw_k = np.array(calibration['views'][i]['intrinsics'])
    raw_k[:2, 2] -= .5
    pose = np.array(capture['world_from_camera_optical'][i])
    inverse = np.linalg.inv(pose)
    with np.load(capture_dir/'evaluation/truth.npz', allow_pickle=False) as data:
        truth_depth = data[record['view']+'_z_m'].copy()
    stage = Usd.Stage.Open(capture['usd'])
    cache = UsdGeom.XformCache()
    curves, controls = [], []
    if reference_mode == 'a04_three_contacts':
        from tests.a04_reference import reference_from_stage
        if capture['station'] != 'A04':
            raise ValueError('Three-contact reference applies to A04 only')
        controls = [(r['name'], r['control_world_m'], [2], 'evaluation-only CAD intersection')
                    for r in reference_from_stage(stage)]
    elif reference_mode == 'usd_reference':
        root = stage.GetPrimAtPath('/World/Fixtures/'+capture['station']+'/WeldSeams')
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.BasisCurves):
                continue
            curve = UsdGeom.BasisCurves(prim)
            if curve.GetTypeAttr().Get() != 'linear':
                raise ValueError('Evaluator expects authored linear seam curves')
            transform = np.array(cache.GetLocalToWorldTransform(prim)).T
            vertices = np.array(curve.GetPointsAttr().Get(), dtype=float)
            vertices = (np.c_[vertices, np.ones(len(vertices))] @ transform.T)[:, :3]
            controls.append((prim.GetName(), vertices, curve.GetCurveVertexCountsAttr().Get(),
                             UsdGeom.Imageable(prim).ComputeVisibility()))
    else:
        raise ValueError('Unknown seam reference mode')
    for name, vertices, counts, visibility in controls:
        offset = 0
        for part, count in enumerate(counts):
            control = vertices[offset:offset+count]
            offset += count
            lengths = np.linalg.norm(np.diff(control, axis=0), axis=1)
            arc_control = np.r_[0., np.cumsum(lengths)]
            arc = np.linspace(0, arc_control[-1], int(np.ceil(arc_control[-1]/.001))+1)
            world = np.stack([np.interp(arc, arc_control, control[:, j]) for j in range(3)], axis=1)
            camera = (np.c_[world, np.ones(len(world))] @ inverse.T)[:, :3]
            raw_xy = project(camera, raw_k)
            xy = project(camera, k_index)
            pixels = np.floor(raw_xy+.5).astype(int)
            in_frame = ((camera[:, 2] > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < truth_depth.shape[1]) &
                       (pixels[:, 1] >= 0) & (pixels[:, 1] < truth_depth.shape[0]) &
                       (xy[:, 0] >= 0) & (xy[:, 0] <= shape[1]-1) & (xy[:, 1] >= 0) & (xy[:, 1] <= shape[0]-1))
            sampled = np.full(len(camera), np.nan)
            sampled[in_frame] = truth_depth[pixels[in_frame, 1], pixels[in_frame, 0]]
            # These curves are hidden reference roots, not rendered bead tubes.
            # In-frame roots remain evaluation targets even if below a groove
            # lip. Separately flag agreement with the nearest rendered surface.
            tolerance = .003
            visible = in_frame & np.isfinite(sampled) & (np.abs(sampled-camera[:, 2]) <= tolerance)
            curves.append({'name': name+f'_{part}', 'world': world, 'xyz': camera,
                           'xy': xy, 'raw_xy': raw_xy, 'visible': visible, 'in_frame': in_frame,
                           'render_z_m': sampled, 'arc': arc,
                           'reference_curve_visibility': visibility,
                           'visibility_tolerance_m': tolerance})
    if not curves:
        raise ValueError('No independent nominal seam truth found')
    return curves, pose
