"""Locate seams using saved calibrated Nested depth; evaluate against USD curves."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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


def prepare(source, output, cases, truth_mode='a04_three_contacts'):
    (output/'inputs').mkdir(parents=True)
    records = []
    for case in cases:
        for side in ('left', 'right'):
            name = case+'_'+side
            image = output/'inputs'/(name+'.png')
            shutil.copyfile(source/'captures'/case/(side+'.png'), image)
            depth = source/'predictions'/case/'stereo_calibrated'/(side+'_depth.npz')
            records.append({'id': name, 'case': case, 'view': side, 'image': str(image),
                            'depth': str(depth), 'image_sha256': digest(image), 'depth_sha256': digest(depth)})
    plan = {'source_run': str(source), 'records': records, 'joint_profile': 'nonparallel_corner',
            'truth_mode': truth_mode,
            'visualization_depth_range_m': [.3, 1.2] if truth_mode=='a04_three_contacts' else [.3, 2.5],
            'scope': 'RGB semantics and calibrated Nested depth predict seams; USD seam coordinates reserved for evaluation'}
    write_json(output/'plan.json', plan)
    return plan


def reuse_semantics(cache_path, records, destination):
    """Select cached SAM outputs by RGB hash, independently of station names."""
    import numpy as np
    arrays = {}
    with np.load(cache_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache['metadata_json']))
        if metadata['threshold'] != .25:
            raise ValueError('Expected the unchanged SAM threshold 0.25')
        arrays['metadata_json'] = cache['metadata_json']
        hashes = {str(cache[key]): key.split('_')[0] for key in cache.files if key.endswith('_image_sha256')}
        for index, record in enumerate(records):
            image_hash = digest(record['image'])
            if image_hash not in hashes:
                raise ValueError('Semantic cache has no matching RGB: '+record['id'])
            original = hashes[image_hash]
            for key in ('image_sha256','plate_masks','plate_scores','weld_masks','weld_scores'):
                arrays[f'{index}_{key}'] = cache[f'{original}_{key}']
    np.savez_compressed(destination, **arrays)


def predict(plan, output):
    """This stage never opens a USD, renderer truth, or a seam annotation."""
    import numpy as np
    from vision.weld_detector import detect
    from auto_weld import serialize_result

    (output/'predictions').mkdir(exist_ok=True)
    with np.load(output/'semantics.npz', allow_pickle=False) as semantics:
        for index, record in enumerate(plan['records']):
            image = Path(record['image'])
            if str(semantics[f'{index}_image_sha256']) != digest(image):
                raise ValueError('Semantic cache does not match RGB')
            with np.load(record['depth'], allow_pickle=False) as stored:
                data = dict(stored)
            if str(data['image_sha256']) != digest(image) or not bool(data['is_metric']):
                raise ValueError('Stereo depth does not match RGB')
            # USD/DA3 K measures centres at u+.5. Existing detector uses u.
            native_k = data['intrinsics'].copy()
            data['intrinsics'] = native_k.copy()
            data['intrinsics'][:2, 2] -= .5
            result = detect(data['depth_m'], data['intrinsics'], data['processed_rgb'],
                            plate_masks=semantics[f'{index}_plate_masks'],
                            weld_masks=semantics[f'{index}_weld_masks'],
                            weld_scores=semantics[f'{index}_weld_scores'], confidence=data.get('confidence'),
                            recover_missing_corners=True)
            arrays, detector_report, _ = serialize_result(index, image, data, result, 5)
            candidates = []
            for number, weld in enumerate(result['welds']):
                xy = weld['centerline']
                z = sample_depth(data['depth_m'], xy)
                xyz = backproject(xy, z, data['intrinsics'])
                valid = np.isfinite(xyz).all(axis=1) & (z > 0)
                candidates.append({'id': f'bead_{number}', 'kind': 'bead_centerline',
                                   'score': weld['score'], 'xy': xy, 'xyz': xyz, 'valid': valid})
            for number, joint in enumerate(result['joints']):
                for segment_index, segment in enumerate(joint['segments']):
                    if segment['branch'] != 'plate_intersection':
                        continue
                    a, b = segment['start_sample'], segment['stop_sample']
                    candidates.append({'id': f'joint_{number}_{segment_index}', 'kind': 'plate_intersection',
                                       'state': segment['state'], 'xy': joint['xy'][a:b],
                                       'xyz': joint['xyz_camera_estimated_m'][a:b],
                                       'valid': np.asarray(segment['geometry_supported'], dtype=bool)})
            descriptions = []
            for i, candidate in enumerate(candidates):
                for field in ('xy', 'xyz', 'valid'):
                    arrays[f'candidate_{i}_{field}'] = candidate[field]
                descriptions.append({k: v for k, v in candidate.items() if k not in ('xy', 'xyz', 'valid')})
            arrays.update(intrinsics_index=data['intrinsics'], intrinsics_native=native_k,
                          depth_m=data['depth_m'], processed_rgb=data['processed_rgb'])
            np.savez_compressed(output/'predictions'/(record['id']+'.npz'), **arrays)
            report = {'id': record['id'], 'image_sha256': digest(image), 'depth_sha256': digest(record['depth']),
                      'sam_plate_scores': semantics[f'{index}_plate_scores'].tolist(),
                      'sam_weld_scores': semantics[f'{index}_weld_scores'].tolist(),
                      'corner_recovery': result['corner_recovery'],
                      'candidates': descriptions, 'detector_report': detector_report,
                      'ground_truth_usage': 'None; prediction completed before evaluation opens any USD or truth file',
                      'centerline_method': 'Accepted SAM bead skeleton + bilinear Nested axial depth; roots use fitted plane intersections',
                      'toe_note': 'Toe boundaries are saved for inspection but are not scored against nominal seam centerlines'}
            write_json(output/'predictions'/(record['id']+'.json'), report)
            print('SEAM_PREDICTED', record['id'], 'beads', len(result['welds']), 'joints', len(result['joints']),
                  'paths', len(candidates), flush=True)


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
        from evaluation.a04_reference import reference_from_stage
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


def evaluate(plan, output):
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    source = Path(plan['source_run'])
    depth_lo, depth_hi = plan.get('visualization_depth_range_m', [.3, 1.2])
    (output/'evaluation').mkdir(exist_ok=True)
    rows, point_rows = [], []
    for record in plan['records']:
        name = record['id']
        report = json.loads((output/'predictions'/(name+'.json')).read_text())
        if digest(record['image']) != report['image_sha256'] or digest(record['depth']) != report['depth_sha256']:
            raise ValueError('Inputs changed after seam prediction')
        with np.load(output/'predictions'/(name+'.npz'), allow_pickle=False) as stored:
            data = dict(stored)
        curves, pose = ground_truth(source, record, data['intrinsics_index'], data['depth_m'].shape,
                                   plan.get('truth_mode', 'usd_reference'))
        truth_parts = [c['xyz'][a:b] for c in curves for a, b in runs(c['in_frame'])]
        truth_points = np.concatenate(truth_parts) if truth_parts else np.zeros((0, 3))
        candidates = []
        for i, item in enumerate(report['candidates']):
            candidates.append({**item, **{key: data[f'candidate_{i}_{key}'] for key in ('xy', 'xyz', 'valid')}})
        valid_parts = [p['xyz'][a:b] for p in candidates for a, b in runs(p['valid'])]
        valid_points = np.concatenate(valid_parts) if valid_parts else np.zeros((0, 3))
        row = {'id': name, 'automatic_paths': len(candidates), 'accepted_bead_paths': sum(p['kind']=='bead_centerline' for p in candidates),
               'corner_recovery': report.get('corner_recovery'),
               'valid_predicted_points': len(valid_points), 'in_frame_truth_samples': len(truth_points),
               'depth_visible_truth_samples': sum(int(c['visible'].sum()) for c in curves),
               'nominal_truth_samples': sum(len(c['xyz']) for c in curves),
               'automatic_position_error': None, 'truth_to_prediction_error': None,
               'prediction_precision_at_10mm': None, 'in_frame_truth_coverage_at_10mm': 0. if len(truth_points) else None,
               'in_frame_truth_coverage_at_20mm': 0. if len(truth_points) else None,
               'known_2d_depth_diagnostic': None, 'status': 'no_automatic_seam' if not len(valid_points) else 'predicted'}
        if len(truth_points) and len(valid_points):
            errors, nearest = segment_distances(valid_points, truth_parts)
            reverse, _ = segment_distances(truth_points, valid_parts)
            row.update(automatic_position_error=error_stats(errors), truth_to_prediction_error=error_stats(reverse),
                       prediction_precision_at_10mm=float(np.mean(errors <= .01)),
                       in_frame_truth_coverage_at_10mm=float(np.mean(reverse <= .01)),
                       in_frame_truth_coverage_at_20mm=float(np.mean(reverse <= .02)))
        diagnostic_errors = []
        evaluation_arrays = {}
        for i, curve in enumerate(curves):
            xy = curve['xy'][curve['in_frame']]
            estimated = backproject(xy, sample_depth(data['depth_m'], xy), data['intrinsics_index'])
            errors = np.linalg.norm(estimated-curve['xyz'][curve['in_frame']], axis=1)
            diagnostic_errors.extend(errors.tolist())
            for field in ('xyz', 'xy', 'visible', 'in_frame', 'render_z_m', 'arc'):
                evaluation_arrays[f'truth_{i}_{field}'] = curve[field]
            evaluation_arrays[f'known_2d_{i}_predicted_xyz'] = estimated
            evaluation_arrays[f'known_2d_{i}_error_m'] = errors
        row['known_2d_depth_diagnostic'] = error_stats(diagnostic_errors)
        row['seams'] = per_seam_metrics(curves, valid_parts, data['depth_m'], data['intrinsics_index'])
        row['seams_with_at_least_50percent_coverage_at_10mm'] = sum(
            item['coverage_at_50percent_threshold_passed'] is True for item in row['seams'])
        row['in_frame_truth_length_approx_m'] = sum(sum(c['arc'][b-1]-c['arc'][a] for a, b in runs(c['in_frame'])) for c in curves)
        row['path_metrics'] = []
        for i, candidate in enumerate(candidates):
            if len(truth_points):
                err, nearest = segment_distances(candidate['xyz'], truth_parts)
                evaluation_arrays[f'candidate_{i}_distance_to_truth_m'] = err
                valid = candidate['valid']
                row['path_metrics'].append({'id': candidate['id'], 'kind': candidate['kind'],
                    'valid_points': int(valid.sum()), 'position_error': error_stats(err[valid]),
                    'point_precision_at_10mm': float(np.mean(err[valid] <= .01)) if valid.any() else None})
                for j, (xy, xyz, valid, value) in enumerate(zip(candidate['xy'], candidate['xyz'], candidate['valid'], err)):
                    world = (pose @ np.r_[xyz, 1])[:3]
                    point_rows.append({'view': name, 'path': candidate['id'], 'kind': candidate['kind'], 'sample': j,
                                       'u_processed': float(xy[0]), 'v_processed': float(xy[1]),
                                       'x_camera_m': float(xyz[0]), 'y_camera_m': float(xyz[1]), 'z_camera_m': float(xyz[2]),
                                       'x_world_m': float(world[0]), 'y_world_m': float(world[1]), 'z_world_m': float(world[2]),
                                       'model_geometry_valid': bool(valid), 'distance_to_in_frame_truth_mm': float(value*1000)})
        np.savez_compressed(output/'evaluation'/(name+'.npz'), **evaluation_arrays)
        write_json(output/'evaluation'/(name+'.json'), row)
        rows.append(row)
        with Image.open(record['image']) as im:
            original = np.array(im.convert('RGB'))
        fig = plt.figure(figsize=(18, 11), layout='constrained')
        grid = fig.add_gridspec(2, 3)
        ax_original = fig.add_subplot(grid[0, 0])
        ax_depth = fig.add_subplot(grid[0, 1])
        ax_overlay = fig.add_subplot(grid[0, 2])
        ax_detail = fig.add_subplot(grid[1, 0])
        ax_3d = fig.add_subplot(grid[1, 1], projection='3d')
        ax_error = fig.add_subplot(grid[1, 2])
        ax_original.imshow(original)
        ax_original.set_title('Original RGB (640 x 1100)')
        im = ax_depth.imshow(data['depth_m'], cmap='turbo', vmin=depth_lo, vmax=depth_hi)
        ax_depth.set_title('Nested calibrated depth (m)\n504 x 294; known 90 mm baseline')
        fig.colorbar(im, ax=ax_depth, shrink=.8, label=f'Optical Z [m]; display range {depth_lo:g}-{depth_hi:g}')
        ax_overlay.imshow(original)
        ax_detail.imshow(data['processed_rgb'])
        scale = np.array([original.shape[1]/data['depth_m'].shape[1], original.shape[0]/data['depth_m'].shape[0]])
        for ci, curve in enumerate(curves):
            for a, b in runs(curve['in_frame']):
                xy = curve['xy'][a:b]
                raw = (xy+.5)*scale-.5
                ax_overlay.plot(*raw.T, '--', color='magenta', lw=1.6,
                                label=curve['name'] if a==runs(curve['in_frame'])[0][0] else None)
                ax_detail.plot(*xy.T, '--', color='magenta', lw=1.6)
                ax_3d.plot(*(curve['xyz'][a:b]*1000).T, color='magenta', lw=2)
            in_frame = curve['in_frame']
            errors = evaluation_arrays[f'known_2d_{ci}_error_m']
            ax_error.plot(curve['arc'][in_frame]*1000, errors*1000, lw=1.5, label=curve['name'])
        for candidate in candidates:
            color = 'lime' if candidate['kind']=='bead_centerline' else 'deepskyblue'
            raw = (candidate['xy']+.5)*scale-.5
            ax_overlay.plot(*raw.T, color=color, lw=1.3, label=candidate['id'])
            ax_detail.plot(*candidate['xy'].T, color=color, lw=1.4)
            for a, b in runs(candidate['valid']):
                ax_3d.plot(*(candidate['xyz'][a:b]*1000).T, color=color, lw=1.4)
            midpoint = len(raw)//2
            ax_overlay.annotate(candidate['id'], raw[midpoint], xytext=(8, 8), textcoords='offset points',
                                fontsize=7, color='black', bbox={'facecolor':'white', 'alpha':.8, 'pad':1})
        if not candidates:
            ax_overlay.text(.03, .04, 'NO AUTOMATIC SEAM', transform=ax_overlay.transAxes, color='red',
                            fontsize=12, bbox={'facecolor':'white', 'alpha':.9})
        ax_overlay.set_title('Seams on original RGB\nGreen: bead | Cyan: joint | Magenta dashed: truth')
        if ax_overlay.get_legend_handles_labels()[0]:
            ax_overlay.legend(fontsize=7, loc='upper right')
        visible_xy = [c['xy'][c['in_frame']] for c in curves if c['in_frame'].any()]
        if visible_xy:
            xy = np.concatenate(visible_xy)
            lo, hi = xy.min(0)-20, xy.max(0)+20
            ax_detail.set_xlim(max(-.5, lo[0]), min(data['depth_m'].shape[1]-.5, hi[0]))
            ax_detail.set_ylim(min(data['depth_m'].shape[0]-.5, hi[1]), max(-.5, lo[1]))
        ax_detail.set_title('Seam detail (crop for display only)')
        for ax in (ax_original, ax_depth, ax_overlay, ax_detail):
            ax.axis('off')
        ax_3d.set(xlabel='Camera X [mm]', ylabel='Camera Y [mm]', zlabel='Camera Z [mm]',
                  title='3D candidates and nominal truth\nNo alignment fitted to truth')
        plotted = [*truth_parts, *valid_parts]
        if plotted:
            all_xyz = np.concatenate(plotted)*1000
            low, high = all_xyz.min(0), all_xyz.max(0)
            centre = (low+high)/2
            half = np.maximum((high-low)*.55, 20.)
            for j, setter in enumerate((ax_3d.set_xlim, ax_3d.set_ylim, ax_3d.set_zlim)):
                setter(centre[j]-half[j], centre[j]+half[j])
        ax_3d.view_init(elev=22, azim=-65)
        ax_error.set(xlabel='Nominal seam arc length [mm]', ylabel='3D position error [mm]',
                     title='Diagnostic at KNOWN 2D nominal root\nMay be occluded; NOT automatic localization')
        ax_error.grid(alpha=.25)
        ax_error.legend(fontsize=8)
        metric = row['automatic_position_error']
        note = ('All candidates: mean %.2f mm | P95 %.2f mm | in-frame coverage@10mm %.1f%%'%(
                    metric['mean_mm'], metric['p95_mm'], row['in_frame_truth_coverage_at_10mm']*100)) if metric else 'No automatic valid seam: localization error is N/A'
        fig.suptitle(name+' | '+note, fontsize=14)
        fig.savefig(output/(name+'_seams.png'), dpi=130)
        plt.close(fig)
        # A compact version contains exactly the three requested image panels.
        overview, axes = plt.subplots(1, 3, figsize=(12, 8), layout='constrained')
        axes[0].imshow(original)
        axes[0].set_title('Original RGB')
        depth_im = axes[1].imshow(data['depth_m'], cmap='turbo', vmin=depth_lo, vmax=depth_hi)
        axes[1].set_title('Nested depth [m]')
        overview.colorbar(depth_im, ax=axes[1], shrink=.65, label='Optical Z [m]')
        axes[2].imshow(original)
        axes[2].set_title('Automatic joints / reference root')
        for curve in curves:
            for a, b in runs(curve['in_frame']):
                raw = (curve['xy'][a:b]+.5)*scale-.5
                axes[2].plot(*raw.T, '--', color='magenta', lw=1.6)
        for candidate in candidates:
            color = 'lime' if candidate['kind']=='bead_centerline' else 'deepskyblue'
            raw = (candidate['xy']+.5)*scale-.5
            axes[2].plot(*raw.T, color=color, lw=1.5)
            axes[2].annotate(candidate['id'], raw[len(raw)//2], xytext=(5, 5), textcoords='offset points',
                             fontsize=7, bbox={'facecolor':'white', 'alpha':.8, 'pad':1})
        if not candidates:
            axes[2].text(.04, .05, 'NO AUTOMATIC SEAM', transform=axes[2].transAxes, color='red',
                         fontsize=10, bbox={'facecolor':'white', 'alpha':.9})
        for ax in axes:
            ax.axis('off')
        overview.suptitle(name+'\nCyan: automatic joint | Magenta dashed: nominal GT (evaluation only)', fontsize=12)
        overview.savefig(output/(name+'_overview.png'), dpi=130)
        plt.close(overview)
        print('SEAM_EVALUATED', name, json.dumps(row), flush=True)
    summary = {'method': 'SAM3.1 + current calibrated Nested depth + existing geometric detector',
               'spatial_error_definition': 'Predicted valid points to nearest finite IN-FRAME nominal USD seam segment in 3D, no GT alignment',
               'coverage_definition': 'In-frame nominal roots, including occluded roots, sampled at <=1 mm intervals; within 10/20 mm of predicted segments',
               'known_2d_diagnostic': 'GT projected nominal root positions supplied ONLY to evaluator; includes occluded roots; not detection or pure visible-surface depth accuracy',
               'scale_provenance': 'Known 90 mm camera baseline; no scale, offset, rotation or ICP fitted to seam truth',
               'truth_mode': plan.get('truth_mode', 'usd_reference'),
               'ground_truth': ('All three finite inner-face CAD contacts are welds by user instruction'
                                if plan.get('truth_mode')=='a04_three_contacts' else 'Hidden USD nominal welding path'),
               'source_depth_model': json.loads((source/'predictions/model.json').read_text()),
               'sam_checkpoint_sha256': digest(ROOT/'thirdparty/sam3/assets/model/sam3.1_multiplex.pt'),
               'sam_cache_sha256': digest(output/'semantics.npz'),
               'source_hashes': {name:digest(ROOT/name) for name in ('scripts/benchmark_stereo_seams.py','vision/weld_detector.py','vision/skeleton.py','evaluation/a04_reference.py')},
               'limitations': ['Static synthetic views; ideal camera poses; not real weld localization accuracy',
                               'All automatic paths retained, including unrelated plate contacts; no GT-based candidate selection',
                               'Known-2D diagnostic separates depth error from misses and pixel localization',
                               'Overview seams may exceed 800 mm rated working distance; laser off; no robot control'], 'results': rows}
    write_json(output/'summary.json', summary)
    with (output/'seam_metrics.csv').open('w', newline='') as stream:
        fields = ['view','seam','nominal_length_m','in_frame_samples','depth_visible_samples',
                  'assigned_prediction_samples','mean_mm','rmse_mm','p95_mm','coverage_at_5mm',
                  'coverage_at_10mm','coverage_at_20mm','known_2d_diagnostic_mean_mm']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for seam in row['seams']:
                values = {key: seam[key] for key in fields if key in seam}
                values.update(view=row['id'], known_2d_diagnostic_mean_mm=(
                    seam['known_2d_diagnostic']['mean_mm'] if seam['known_2d_diagnostic'] else None))
                values.update({key: seam['position_error'][key] if seam['position_error'] else None
                               for key in ('mean_mm','rmse_mm','p95_mm')})
                writer.writerow(values)
    with (output/'seam_points.csv').open('w', newline='') as stream:
        fields = list(point_rows[0]) if point_rows else ['view','path','sample','distance_to_in_frame_truth_mm']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(point_rows)
    lines = ['# 当前 Nested 焊缝三维定位核对', '',
             '使用板面交线几何定位角接缝/T 接缝。SAM 选区未形成有效接缝时，启用独立的整图几何回退。',
             '使用现有标定双目深度与 SAM3；90 mm 基线校正保留。预测不读取 USD 焊缝或 RTX 真值。',
             '空间误差为候选有效点到画面内有限名义根部路径的三维距离；包括被遮挡的根部，同时报告覆盖率，未检出不记为零误差。', '',
             '| 视图 | 自动路径数 | 自动均距 mm | 自动 P95 mm | 画内根部覆盖@10mm | 已知二维根部投影诊断 mm |',
             '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        m, d = r['automatic_position_error'], r['known_2d_depth_diagnostic']
        lines.append('| %s | %d | %s | %s | %s | %s |'%(r['id'],r['automatic_paths'],
                      '%.2f'%m['mean_mm'] if m else 'N/A（未检出）', '%.2f'%m['p95_mm'] if m else 'N/A',
                      '%.1f%%'%(r['in_frame_truth_coverage_at_10mm']*100) if r['in_frame_truth_coverage_at_10mm'] is not None else 'N/A',
                      '%.2f'%d['mean_mm'] if d else 'N/A'))
    lines += ['', '## 焊缝逐项核对', '',
              ('V=竖直，B=背板与底板，R=侧板与底板；三条均为目标。' if plan.get('truth_mode')=='a04_three_contacts'
               else '本次按各工位隐藏的 USD 名义焊接路径评价。')+'关联只用于评价，不用于删选预测。', '',
              '| 视图 | 焊缝 | 关联预测点 | 均距 mm | P95 mm | 画内覆盖@10mm |',
              '|---|---|---:|---:|---:|---:|']
    for r in rows:
        for seam in r['seams']:
            m = seam['position_error']
            lines.append('| %s | %s | %d | %s | %s | %s |'%(r['id'],seam['seam'],seam['assigned_prediction_samples'],
                         '%.2f'%m['mean_mm'] if m else 'N/A', '%.2f'%m['p95_mm'] if m else 'N/A',
                         '%.1f%%'%(seam['coverage_at_10mm']*100) if seam['coverage_at_10mm'] is not None else '画外'))
    lines += ['', '## 角接缝回退记录', '',
              '| 视图 | SAM 区域拟合面数 | SAM 区域接缝数 | 触发整图回退 | 回退接缝数 |',
              '|---|---:|---:|---|---:|']
    for r in rows:
        audit = r['corner_recovery']
        if audit:
            lines.append('| %s | %d | %d | %s | %d |'%(r['id'],audit['semantic_plane_count'],
                         audit['semantic_joint_count'],'是' if audit['attempted'] else '否',audit['recovered_joint_count']))
    lines += ['', '## 自动候选逐条评价', '',
              '这里逐条与名义真值核对，不代表识别程序已自动选出目标；不根据真值删掉多余候选。', '',
              '| 视图 | 候选 | 有效点 | 均距 mm | P95 mm | 10 mm 内点占比 |',
              '|---|---|---:|---:|---:|---:|']
    for r in rows:
        for path in r['path_metrics']:
            m = path['position_error']
            if m:
                lines.append('| %s | %s | %d | %.2f | %.2f | %.1f%% |'%(r['id'],path['id'],path['valid_points'],
                              m['mean_mm'],m['p95_mm'],path['point_precision_at_10mm']*100))
    lines += ['', '绿色：自动焊道中心线；青色：自动板面交线候选；品红虚线：独立 USD 真值。',
              '品红线和右下角已知二维位置诊断只用于评价，不能当作自动识别结果。',
              '原始模型深度分辨率为 504×294。参考曲线默认隐藏，不出现在输入 RGB；白色条带是打磨区域，不是已完成焊道。',
              '已知二维投影诊断包含名义根部与可见表面的差异，不是纯表面深度误差。',
              '全候选均保留，包含可能的非焊接接触；完整误差、10/20 mm 覆盖率及点坐标见 summary.json / seam_points.csv。',
              ('本次三条真实 CAD 内角交线均按焊缝评价，包含完整有限接触长度；旧的单竖线指标不能直接比较。'
               if plan.get('truth_mode')=='a04_three_contacts' else '本次采用各工位原 USD 名义参考，未将其他接触线自动声明为工艺焊缝。'),
              '当前结果只适用于本次静态仿真视图，不能用于机器人焊接精度验收。', '', '## 图像', '']
    lines += [f'- [{r["id"]}：原图/深度/焊缝]({r["id"]}_overview.png) · [三维与误差详情]({r["id"]}_seams.png)' for r in rows]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=ROOT/'outputs/benchmarks/nested_stereo_20261110_01')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--sam-python', type=Path)
    p.add_argument('--semantic-cache', type=Path, help='Reuse SAM masks with matching RGB hashes; no GPU inference')
    p.add_argument('--cases', nargs='+', help='Explicit capture IDs; by default discover A04 captures')
    p.add_argument('--truth-mode', choices=('a04_three_contacts','usd_reference'), default='a04_three_contacts')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--evaluate-only', action='store_true')
    a = p.parse_args()
    output = (a.output_dir or ROOT/'outputs/benchmarks'/datetime.now().strftime('nested_seams_%Y%m%d_%H%M%S')).resolve()
    os.environ.setdefault('MPLCONFIGDIR', str(output/'mpl_cache'))
    if a.resume or a.evaluate_only:
        plan = json.loads((output/'plan.json').read_text())
    else:
        if output.exists() and any(output.iterdir()):
            p.error('Use an empty output directory, --resume, or --evaluate-only')
        source = a.source_run.resolve()
        cases = a.cases or [path.parent.name for path in sorted((source/'captures').glob('A04*/input.json'))]
        if not cases or (a.truth_mode=='a04_three_contacts' and any(not name.startswith('A04') for name in cases)):
            p.error('Three-contact reference requires A04; use --truth-mode usd_reference for other stations')
        if any(Path(name).name != name or not (source/'captures'/name/'input.json').is_file() for name in cases):
            p.error('Capture IDs must name existing input pairs inside source-run/captures')
        plan = prepare(source, output, cases, a.truth_mode)
    if not a.evaluate_only:
        if plan.get('truth_mode')=='a04_three_contacts' and any(not r['case'].startswith('A04') for r in plan['records']):
            p.error('Three-contact reference requires A04 captures')
        from main import find_python, run_process
        if a.semantic_cache and not (output/'semantics.npz').is_file():
            reuse_semantics(a.semantic_cache, plan['records'], output/'semantics.npz')
        if not (output/'semantics.npz').is_file():
            sam, error = find_python('sam', a.sam_python)
            if error:
                raise RuntimeError(error)
            run_process([sam, str(ROOT/'scripts/auto_weld.py'), *[r['image'] for r in plan['records']],
                         '--semantic-worker', str(output/'semantics.npz'), '--gpu', str(a.gpu)], output/'semantics.log')
        predict(plan, output)
    evaluate(plan, output)
    print('SEAM_BENCHMARK_COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
