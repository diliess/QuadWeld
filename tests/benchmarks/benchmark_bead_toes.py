"""Weld toes of the first-layer fillet beads from calibrated stereo, measured against the bead mesh.

The measurement sees the two images, the calibration and plate masks (SAM3.1 text
prompt candidates, or the renderer's labels as an ablation).  The bead mesh, the
labels and the camera poses' relation to the fixture are used for evaluation only.
Camera poses are used for multi-view fusion, as a robot would know them.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weld_vision import bead_seam, stereo_seam  # noqa: E402

SCENE = ROOT / 'assets/scenes/industrial-welding/industrial_welding_full.usda'


def truth_toes(stage, station, side):
    """World polylines from the bead mesh: toe on the base plate, toe on the web, root line."""
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath(f'/World/Fixtures/{station}/{station}_first_layer_bead' + ('_back' if side == 'back' else ''))
    points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
    matrix = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
    floor = points[:, 2].min()
    face = points[:, 1].max() if side == 'front' else points[:, 1].min()
    on_base, on_web = np.isclose(points[:, 2], floor, atol=1e-7), np.isclose(points[:, 1], face, atol=1e-7)
    curves = []
    for chosen in (on_base & ~on_web, on_web & ~on_base, on_base & on_web):
        local = points[chosen]
        local = local[np.argsort(local[:, 0])]
        curves.append((np.c_[local, np.ones(len(local))] @ matrix.T)[:, :3])
    return curves


def to_curve(points, curve):
    """Distance of each point to a polyline (m)."""
    a, b = curve[:-1], curve[1:]
    ab = b - a
    out = np.empty(len(points))
    for i, p in enumerate(points):
        u = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-18), 0, 1)
        out[i] = np.linalg.norm(a + u[:, None] * ab - p, axis=1).min()
    return out


def moved(points, matrix):
    return (np.c_[points, np.ones(len(points))] @ np.asarray(matrix, float).T)[:, :3]


def line_error(points, a, direction):
    offset = points - a
    return np.linalg.norm(offset - np.outer(offset @ direction, direction), axis=1)


def stats(errors):
    errors = np.asarray(errors) * 1000
    return {'mean_mm': float(errors.mean()), 'p95_mm': float(np.percentile(errors, 95)), 'max_mm': float(errors.max())}


def evaluate(result, pose, truth):
    """Errors of one view's toes (camera frame) against world truth.  Plate roles from the world vertical."""
    seam = result['seam']
    normals = [pose[:3, :3] @ n for n, _ in seam['planes']]
    base = int(np.argmax([abs(n[2]) for n in normals]))
    ends = moved(seam['joint_ends_xyz'], pose)
    direction = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
    along = (truth[2] - ends[0]) @ direction
    visible = (along >= 0) & (along <= np.linalg.norm(ends[1] - ends[0]))
    out = {'root_line': stats(line_error(truth[2][visible], ends[0], direction)) if visible.any() else None}
    for name, plate, curve in (('base_toe', base, truth[0]), ('web_toe', 1 - base, truth[1])):
        toe = result['toes'][plate]
        world = moved(toe['xyz'], pose)
        s = (world - ends[0]) @ direction
        truth_s = (curve - ends[0]) @ direction
        present = (s >= truth_s.min()) & (s <= truth_s.max())          # where a bead exists in the view
        ok = toe['valid'] & present
        record = {'coverage': float(ok.sum() / max(present.sum(), 1)), 'plate': plate,
                  'left_right_difference_mm': toe['left_right_difference_m'] * 1000,
                  'across_footprint_mm': toe['across_footprint_m'] * 1000}
        if ok.any():
            across = pose[:3, :3] @ toe['across']
            order = np.argsort(truth_s)
            leg_truth = np.interp(s[ok], truth_s[order], ((curve - ends[0]) @ across)[order])
            leg_error = (toe['leg_m'][ok] - leg_truth) * 1000
            record.update(stats(to_curve(world[ok], curve)), leg_bias_mm=float(leg_error.mean()),
                          leg_sd_mm=float(leg_error.std()), leg_mean_mm=float(toe['leg_m'][ok].mean() * 1000))
        out[name] = record
    return out


def overlay(left, intrinsics, result, pose, truth, path):
    """Full frame with the measured toes, and a magnified cut-out that also shows the mesh toes."""
    seam = result['seam']
    frame = left.copy()
    camera = np.linalg.inv(pose)

    def draw(image, points, colour, origin=(0, 0), factor=1, radius=1):
        xy = (stereo_seam.project(points, intrinsics) - origin + .5) * factor
        for x, y in xy:
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (int(x), int(y)), radius, colour, -1)

    draw(frame, seam['xyz'], (255, 220, 0))
    for toe, colour in zip(result['toes'], ((255, 0, 255), (0, 255, 255))):
        draw(frame, toe['xyz'][toe['valid']], colour)
    centre = stereo_seam.project(seam['xyz'][len(seam['xyz']) // 2][None], intrinsics)[0]
    size, factor = 110, 5
    x0 = int(np.clip(centre[0] - size / 2, 0, left.shape[1] - size))
    y0 = int(np.clip(centre[1] - size / 2, 0, left.shape[0] - size))
    cut = cv2.resize(left[y0:y0 + size, x0:x0 + size], None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    for curve in truth[:2]:
        draw(cut, moved(curve, camera), (0, 255, 0), (x0, y0), factor, 1)
    for toe, colour in zip(result['toes'], ((255, 0, 255), (0, 255, 255))):
        draw(cut, toe['xyz'][toe['valid']], colour, (x0, y0), factor, 2)
    cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (255, 255, 255), 1)
    height = max(frame.shape[0], cut.shape[0])
    canvas = np.zeros((height, frame.shape[1] + cut.shape[1] + 8, 3), np.uint8)
    canvas[:frame.shape[0], :frame.shape[1]] = frame
    canvas[:cut.shape[0], frame.shape[1] + 8:] = cut
    Image.fromarray(canvas).save(path, quality=92)


def report(output, results):
    """README.md (Chinese, like the other benchmark reports) and a contact sheet of the magnified cut-outs."""
    views, fused = results['views'], results['fused']
    done = [v for v in views if v['status'] == 'ok']
    lines = ['# 带焊道 T 形接头的焊趾（焊道与两侧板面的交线）双目测量', '',
             f"板件掩码：{'SAM3.1 文本提示候选 + 自动选板' if results['masks'] == 'sam' else '渲染标签（对照）'}；"
             f"第二遍平面拟合避开焊道：{'是' if results['refit_without_bead'] else '否'}。",
             '测量只读取左右目 RGB、双目标定和板件掩码；焊道网格、标签和夹具位置只在评价阶段打开。', '',
             '## 逐视图', '',
             '| 视图 | 状态 | 深度歧义 | 根线误差 mm | 底板焊趾 覆盖 / 均值 / P95 mm | 立板焊趾 覆盖 / 均值 / P95 mm | 左右目差 mm | 耗时 s |',
             '|---|---|---:|---:|---:|---:|---:|---:|']

    def toe(e):
        return f"{e['coverage']:.0%} / {e.get('mean_mm', float('nan')):.2f} / {e.get('p95_mm', float('nan')):.2f}"

    for v in views:
        ambiguity = v.get('checks', {}).get('depth_ambiguity', float('nan'))
        if v['status'] != 'ok':
            extra = v.get('with_distance_from_other_views')
            note = ''
            if extra and extra['status'] == 'ok':
                e = extra['errors']
                note = (f"（给定其余视图的根线距离 {extra['expected_distance_m']:.3f} m 后：根线 {e['root_line']['mean_mm']:.2f}，"
                        f"底板 {toe(e['base_toe'])}，立板 {toe(e['web_toe'])}）")
            lines.append(f"| {v['id']} | {v['status']}{note} | {ambiguity:.2f} | | | | | {v.get('seconds', 0):.1f} |")
            continue
        e = v['errors']
        lines.append(f"| {v['id']} | ok | {ambiguity:.2f} | {e['root_line']['mean_mm']:.2f} | {toe(e['base_toe'])} | {toe(e['web_toe'])} | "
                     f"{e['base_toe']['left_right_difference_mm']:.2f} / {e['web_toe']['left_right_difference_mm']:.2f} | {v['seconds']:.1f} |")
    if done:
        def collect(name, key):
            return [v['errors'][name][key] for v in done if key in v['errors'][name]]
        lines += ['', f"{len(done)} / {len(views)} 组输出结果，其余被拒绝（见状态列）。输出的视图中：",
                  f"- 根线误差均值的中位数 {np.median([v['errors']['root_line']['mean_mm'] for v in done]):.2f} mm，最大 {np.max([v['errors']['root_line']['mean_mm'] for v in done]):.2f} mm；"]
        for name, label in (('base_toe', '底板焊趾'), ('web_toe', '立板焊趾')):
            lines.append(f"- {label}：误差均值的中位数 {np.median(collect(name, 'mean_mm')):.2f} mm，最差视图 {np.max(collect(name, 'mean_mm')):.2f} mm，"
                         f"P95 的中位数 {np.median(collect(name, 'p95_mm')):.2f} mm，覆盖率中位数 {np.median(collect(name, 'coverage')):.0%}；")
    lines += ['', '误差：输出折线上每个有效点到焊道网格焊趾折线的三维距离。覆盖：视野内有焊道的长度中被判为有效并输出的比例。',
              '深度歧义、左右目差无需真值，可在实机上作为质量指示。', '',
              '## 多视角融合（世界坐标，每 1 mm 取各视图中位数，至少 2 个视图）', '',
              '| 接头 | 视图数 | 根线误差 mm | 底板焊趾 均值 / P95 / 最大 mm | 长度 m | 立板焊趾 均值 / P95 / 最大 mm | 长度 m |', '|---|---:|---:|---:|---:|---:|---:|']
    for f in fused:
        lines.append(f"| {f['station']} {f['side']} | {len(f['views'])} | {f['root_line']['mean_mm']:.2f} | "
                     + ' | '.join(f"{f[n]['mean_mm']:.2f} / {f[n]['p95_mm']:.2f} / {f[n]['max_mm']:.2f} | {f[n]['length_m']:.3f}（真值 {f[n]['truth_length_m']:.3f}）"
                                  for n in ('base_toe', 'web_toe')) + ' |')
    lines += ['', '融合折线保存在 `fused_<工位>_<侧>_<焊趾>.csv`（世界坐标，米）。', '',
              '## 图像', '', '`overlays/<视图>.jpg`：左为整幅左目图（黄：被焊道遮住的根线；品红 / 青：两条焊趾），'
              '右为白框处放大 5 倍，绿色细线是焊道网格的焊趾（仅评价用）。', '', '![overview](overview.jpg)', '']
    (output / 'README.md').write_text('\n'.join(lines))
    tiles = []
    for v in done[:24]:
        with Image.open(output / 'overlays' / (v['id'] + '.jpg')) as image:
            cut = np.array(image)[:550, -550:]
        cv2.putText(cut, v['id'], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        tiles.append(cut[::2, ::2])
    if tiles:
        columns = 6
        while len(tiles) % columns:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
        Image.fromarray(sheet).save(output / 'overview.jpg', quality=90)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--name', default='toes', help='Sub-directory of the run for the results')
    p.add_argument('--masks', choices=('sam', 'labels'), default='sam')
    p.add_argument('--no-refit', action='store_true')
    p.add_argument('--cases', nargs='*')
    a = p.parse_args()
    run = a.run_dir.resolve()
    output = run / a.name
    (output / 'overlays').mkdir(parents=True, exist_ok=True)
    from pxr import Usd
    stage = Usd.Stage.Open(str(SCENE))
    records, kept = [], {}
    for folder in sorted(q for q in (run / 'captures').iterdir() if q.is_dir()):
        if a.cases and folder.name not in a.cases:
            continue
        spec = json.loads((folder / 'input.json').read_text())
        capture = json.loads((folder / 'evaluation/capture.json').read_text())
        left, right = (np.array(Image.open(folder / f'{eye}.png').convert('RGB')) for eye in ('left', 'right'))
        intrinsics = np.array(spec['views'][0]['intrinsics'], float)
        intrinsics[:2, 2] -= .5                              # pixel-edge origin -> pixel-index origin
        baseline = spec['baseline_m']
        if a.masks == 'sam':
            data = np.load(run / 'sam' / (folder.name + '.npz'))
            pair = bead_seam.pick_plate_pair(data['masks'], data['scores'])
            masks = None if pair is None else [data['masks'][i] for i in pair]
        else:
            labels = np.load(folder / 'evaluation/truth.npz')['left_labels']
            masks = [labels == 2, labels == 3]
        pose = np.array(capture['world_from_left_optical'], float)
        truth = truth_toes(stage, capture['station'], capture['joint_side'])
        record = {'id': folder.name, 'station': capture['station'], 'side': capture['joint_side'],
                  'view': capture['view'], 'roll_deg': capture.get('roll_deg')}
        if masks is None:
            record['status'] = 'no_plate_pair'
            records.append(record)
            continue
        started = time.monotonic()
        result = bead_seam.locate_bead_toes(left, right, intrinsics, baseline, *masks, refit=not a.no_refit)
        record.update(status=result['status'], seconds=time.monotonic() - started,
                      checks=result['seam'].get('checks', {}))
        if result['status'] == 'ok':
            record['errors'] = evaluate(result, pose, truth)
            overlay(left, intrinsics, result, pose, truth, output / 'overlays' / (folder.name + '.jpg'))
        kept[folder.name] = (result, pose, truth, (left, right, intrinsics, baseline, masks))
        records.append(record)
        errors = record.get('errors')
        print(folder.name, record['status'], '%.1fs' % record['seconds'],
              'amb %.2f' % record['checks'].get('depth_ambiguity', np.nan),
              '' if not errors else 'root %.2f | base cov %.2f mean %.2f p95 %.2f | web cov %.2f mean %.2f p95 %.2f' % (
                  errors['root_line']['mean_mm'], errors['base_toe']['coverage'], errors['base_toe'].get('mean_mm', np.nan),
                  errors['base_toe'].get('p95_mm', np.nan), errors['web_toe']['coverage'],
                  errors['web_toe'].get('mean_mm', np.nan), errors['web_toe'].get('p95_mm', np.nan)), flush=True)

    # Several poses of one joint, fused in the world frame; views refused as ambiguous get a second
    # chance with the fused root line as their rough distance.
    fused_records = []
    for station, side in sorted({(r['station'], r['side']) for r in records}):
        group = [r for r in records if (r['station'], r['side']) == (station, side)]
        good = [r['id'] for r in group if r['status'] == 'ok']
        if len(good) < 2:
            continue

        def as_view(ident):
            result, pose = kept[ident][:2]
            return {'root': moved(result['seam']['joint_ends_xyz'], pose),
                    'normals': [pose[:3, :3] @ n for n, _ in result['seam']['planes']],
                    'toes': [{'xyz': moved(t['xyz'], pose), 'valid': t['valid']} for t in result['toes']]}

        views = [as_view(i) for i in good]
        truth = kept[good[0]][2]
        up = [abs(n[2]) for n in views[0]['normals']]
        tolerance = 1e-3 * float(np.mean([r['checks']['depth_per_disparity_px_mm'] for r in group if r['status'] == 'ok']))
        fused = bead_seam.fuse_toes(views, tolerance=tolerance)
        entry = {'station': station, 'side': side, 'views': good, 'views_used': fused['root']['view_used'],
                 'root_line': stats(line_error(truth[2], fused['root']['ends'][0], fused['root']['direction']))}
        for name, plate, curve in (('base_toe', int(np.argmax(up)), truth[0]), ('web_toe', int(np.argmin(up)), truth[1])):
            toe = fused['toes'][plate]
            ok = toe['valid']
            entry[name] = {**stats(to_curve(toe['xyz'][ok], curve)),
                           'length_m': float(ok.sum() * 1e-3), 'truth_length_m': float(np.linalg.norm(curve[-1] - curve[0])),
                           'median_views': float(np.median(toe['views'][ok])),
                           'spread_p95_mm': float(np.nanpercentile(toe['spread_m'][ok], 95) * 1000)}
            np.savetxt(output / f'fused_{station}_{side}_{name}.csv', toe['xyz'][ok], delimiter=',',
                       header='x_m,y_m,z_m (world)', comments='')
        fused_records.append(entry)
        print('FUSED', station, side, len(good), 'views | root %.2f | base %.2f / %.2f | web %.2f / %.2f (mean / p95 mm)' % (
            entry['root_line']['mean_mm'], entry['base_toe']['mean_mm'], entry['base_toe']['p95_mm'],
            entry['web_toe']['mean_mm'], entry['web_toe']['p95_mm']), flush=True)
        for r in group:
            if r['status'] != 'ambiguous_stereo_match':
                continue
            _, pose, truth, (left, right, intrinsics, baseline, masks) = kept[r['id']]
            samples = fused['root']['ends'][0] + np.linspace(0, 1, 200)[:, None] * (fused['root']['ends'][1] - fused['root']['ends'][0])
            local = moved(samples, np.linalg.inv(pose))
            xy = stereo_seam.project(local, intrinsics)
            seen = (local[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < left.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < left.shape[0])
            if not seen.any():
                continue
            distance = float(np.median(np.linalg.norm(local[seen], axis=1)))
            result = bead_seam.locate_bead_toes(left, right, intrinsics, baseline, *masks, refit=not a.no_refit,
                                                expected_distance=distance)
            r['with_distance_from_other_views'] = {'expected_distance_m': distance, 'status': result['status']}
            if result['status'] == 'ok':
                r['with_distance_from_other_views']['errors'] = errors = evaluate(result, pose, truth)
                overlay(left, intrinsics, result, pose, truth, output / 'overlays' / (r['id'] + '_with_distance.jpg'))
                print('RESOLVED', r['id'], 'root %.2f | base %.2f | web %.2f' % (
                    errors['root_line']['mean_mm'], errors['base_toe'].get('mean_mm', np.nan),
                    errors['web_toe'].get('mean_mm', np.nan)), flush=True)
    results = json.loads(json.dumps({'masks': a.masks, 'refit_without_bead': not a.no_refit, 'views': records,
                                     'fused': fused_records}, default=float))
    (output / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
    report(output, results)
    print('BEAD_TOE_BENCHMARK_COMPLETE', output)


if __name__ == '__main__':
    main()
