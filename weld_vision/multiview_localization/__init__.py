"""Multi-view close stereo localization, world-frame fusion and file adapters."""
import json
import time

import numpy as np

from .. import bead_seam, stereo_seam
from .._localization import (SPACING, as_json, line_error, overview_overlay,
                             read_capture, transform_points, validate_measurement)

__all__ = ['locate_view', 'fuse_views', 'measure_close', 'measure', 'export_fusion']


def locate_view(left, right, intrinsics, baseline, mask_a, mask_b, world_from_left,
                coarse, *, ident=None, mode='bead', expected_distance=None,
                depth_range=(.20, 1.50)):
    """Measure one near view in the same world frame as the coarse result.

    mode='bead' measures the two toes of an existing bead; mode='root' measures
    a bare joint. The caller selects the task; this is not a bead classifier.
    """
    if mode not in ('bead', 'root'):
        raise ValueError('mode must be bead or root')
    world = validate_measurement(left, right, intrinsics, baseline, mask_a, mask_b, world_from_left)
    if coarse.get('status', 'ok') != 'ok':
        return {'id': ident, 'mode': mode, 'status': 'coarse_not_ready'}
    observed = np.asarray(coarse['root_world_m'], float)
    if observed.shape != (2, 3) or not np.isfinite(observed).all() or np.linalg.norm(observed[1]-observed[0]) < 1e-9:
        raise ValueError('coarse root must contain two distinct world endpoints')

    def detect(options):
        if mode == 'bead':
            return bead_seam.locate_bead_toes(left, right, intrinsics, baseline, mask_a, mask_b,
                                              depth_range=depth_range, **options)
        seam = stereo_seam.locate_seam(left, right, intrinsics, baseline, mask_a, mask_b,
                                      depth_range=depth_range, **options)
        return {'status': seam['status'], 'seam': seam, 'toes': []}

    result = detect({})
    if result['status'] == 'ambiguous_stereo_match' and expected_distance is not None:
        result = detect({'expected_distance': expected_distance})
    data = {'id': ident, 'mode': mode, 'status': result['status'], 'checks': result['seam']['checks']}
    if result['status'] != 'ok':
        return data
    seam = result['seam']
    ends = transform_points(seam['joint_ends_xyz'], world)
    a, b = observed
    direction = (b-a) / np.linalg.norm(b-a)
    deviation = float(line_error(ends, a, direction).max())
    alignment = abs((ends[1]-ends[0]) @ direction) / np.linalg.norm(ends[1]-ends[0])
    data['overview_line_deviation_m'] = deviation
    if deviation > .08 or alignment < np.cos(np.radians(10.)):
        data['status'] = 'inconsistent_with_overview'
        return data
    data.update(root=ends, normals=np.asarray([n for n, _ in seam['planes']]) @ world[:3, :3].T,
                toes=[{'xyz': transform_points(toe['xyz'], world), 'valid': toe['valid']} for toe in result['toes']],
                prior_used=seam['checks']['depth_resolved_by_prior'])
    return data


def fuse_views(views, minimum=2, *, spacing=SPACING, mode='bead'):
    """Fuse accepted observations; return native arrays, support and gap masks.

    All observations must use the same metric world frame. Invalid views are
    ignored; duplicate named observations are refused so they cannot supply
    false multiple-view support. No CAD or ground truth is consumed.
    """
    if minimum < 1 or spacing <= 0 or mode not in ('bead', 'root'):
        raise ValueError('Use positive minimum/spacing and mode bead or root')
    accepted = [v for v in views if v.get('status', 'ok') == 'ok']
    if any(v.get('mode', mode) != mode for v in accepted):
        raise ValueError('Do not mix root and bead observations in one fusion')
    identifiers = [v['id'] for v in accepted if v.get('id') is not None]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate view IDs cannot count as independent observations')
    if len(accepted) < minimum:
        return {'status': 'insufficient_views', 'available_views': len(accepted)}
    if mode == 'root':
        axis = stereo_seam.fuse_seams([v['root'] for v in accepted], tolerance=.005)
        status = 'ok' if np.count_nonzero(axis['view_used']) >= minimum else 'insufficient_views'
        return {'status': status, 'root': axis, 'toes': []}
    fused = bead_seam.fuse_toes(accepted, spacing=spacing, min_views=minimum)
    fused['status'] = 'ok' if all(t['valid'].sum() >= 2 for t in fused['toes']) else 'partial'
    return fused


def measure_close(run, view, overview, *, mode='bead'):
    """Saved-capture adapter; selection uses projected coarse plate geometry."""
    ident, distance = view['id'], view['distance_m']
    started = time.monotonic()
    data = {'id': ident, 'station': view['station'], 'view': view['view'], 'scan_index': view.get('scan_index'),
            'distance_m': distance}
    capture = read_capture(run, 'close', ident, overview)
    if capture is None:
        return {**data, 'status': 'no_plate_pair', 'seconds': time.monotonic()-started}
    *measurement, world = capture
    expected = float(np.linalg.norm(np.asarray(view['target_world_m']) - world[:3, 3]))
    result = locate_view(*measurement, world, overview, ident=ident, mode=mode, expected_distance=expected)
    data.update(result, seconds=time.monotonic()-started)
    return data


def export_fusion(output, station, views, minimum, *, mode="bead"):
    """Write a fused station record and metric CSVs from accepted observations."""
    views = [v for v in views if v.get('status', 'ok') == 'ok']
    fused = fuse_views(views, minimum=minimum, spacing=SPACING, mode=mode)
    folder = output / station
    folder.mkdir(parents=True, exist_ok=True)
    if 'root' not in fused:
        (folder / 'fused.json').write_text(json.dumps(fused, indent=2) + '\n')
        return fused
    root = fused['root']
    root_points = np.linspace(*root['ends'], max(2, int(np.ceil(root['length_m'] / SPACING)) + 1))
    np.savetxt(folder / 'root.csv', root_points, delimiter=',', fmt='%.6f',
               header='x_m,y_m,z_m', comments='')
    output_record = {'root_world_m': root['ends'].tolist(),
                     'views_used': [v['id'] for v, used in zip(views, root['view_used']) if used],
                     'root_length_m': root['length_m'], 'minimum_views_per_toe_sample': minimum, 'toes': {}}
    named_toes = []
    if mode == 'bead':
        reference = np.asarray(views[0]['normals'])
        base = int(np.argmax(np.abs(reference[:, 2])))
        named_toes = [('base', fused['toes'][base]), ('web', fused['toes'][1-base])]
    for name, toe in named_toes:
        points = np.where(toe['valid'][:, None], toe['xyz'], np.nan)
        table = np.column_stack((toe['s_m'], points, toe['valid'], toe['views'], toe['spread_m']))
        np.savetxt(folder / (name + '_toe.csv'), table, delimiter=',', fmt=['%.6f'] * 4 + ['%d', '%d', '%.6f'],
                   header='s_m,x_m,y_m,z_m,valid,views,max_deviation_m', comments='')
        runs = bead_seam.polyline_segments(toe, min_points=1)
        output_record['toes'][name] = {'segments_world_m': [run.tolist() for run in runs],
                                       'valid_samples': int(toe['valid'].sum())}
    output_record['status'] = fused['status']
    (folder / 'fused.json').write_text(json.dumps(output_record, indent=2, allow_nan=False) + '\n')
    return output_record



def measure(run, output, minimum=2, *, plan=None, station=None, mode="bead"):
    """Measure captures and export the world-frame root and observed toe segments."""
    if minimum < 1 or mode not in ('bead', 'root'):
        raise ValueError('Use a positive minimum and mode bead or root')
    if plan is None:
        plan = json.loads((run / 'close_plan.json').read_text())
    selected = [station] if station is not None else plan['overview']
    if any(name not in plan['overview'] for name in selected):
        raise ValueError(f'Scan plan is missing the requested station: {station}')
    captured = json.loads((run / 'close' / 'views.json').read_text())['views']
    if [v['id'] for v in captured] != [v['id'] for v in plan['views']]:
        raise ValueError('Close capture does not match the scan plan')
    output.mkdir(parents=True, exist_ok=True)
    views = []
    for view in captured:
        if view['station'] not in selected:
            continue
        record = measure_close(run, view, plan['overview'][view['station']], mode=mode)
        print('BEAD_CLOSE', record['id'], record['status'], f"{record['seconds']:.1f}s", flush=True)
        views.append(record)
    fused = {}
    for station in selected:
        eligible = [v for v in views if v['station'] == station and v['view'] == 'sweep' and v['status'] == 'ok']
        fused[station] = (export_fusion(output, station, eligible, minimum, mode=mode) if len(eligible) >= minimum
                          else {'status': 'insufficient_views'})
        if 'root_world_m' in fused[station]:
            overview_overlay(run, output / station, plan['overview'][station], fused[station])
    predictions = {'views': views, 'fused': fused}
    (output / 'predictions.json').write_text(json.dumps(predictions, default=as_json, allow_nan=False))
    return predictions
