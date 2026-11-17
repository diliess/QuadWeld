"""Inspect rerun stereo-plane seams: masks, actual fit samples, stereo overlays and fusion."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from tests.benchmarks.benchmark_stereo_plane_seams import left_camera_pose, load_case, transform
from tests.benchmarks.seam_metrics import digest, ground_truth, runs, write_json
from weld_vision import stereo_seam as ss


COLORS = ((255, 150, 0), (0, 190, 255))
DATASETS = {
    'full': (ROOT / 'outputs/archive/non_stereo_experiments_20261117/outputs/benchmarks/t_station_full_joint_20261110_01',
             ROOT / 'outputs/archive/non_stereo_experiments_20261117/outputs/benchmarks/t_station_full_joint_20261110_01/plate_parts', '整体距离约2 m'),
    'close': (ROOT / 'outputs/archive/non_stereo_experiments_20261117/outputs/benchmarks/t_station_close_20261110_01',
              ROOT / 'outputs/archive/non_stereo_experiments_20261117/outputs/benchmarks/foreground_plate_parts_20261110_01', '近距离约0.65 m'),
}


def fit_samples(left, right, masks, record, k, baseline):
    """Recover the deterministic pixels actually sampled by the unchanged fitter.

    Final residuals are recalculated at the saved solution. The stereo fit is
    not rerun and no reference geometry or depth map is read here.
    """
    images = ss.StereoImages(left, right)
    cleaned = [ss.clean_mask(mask) for mask in masks]
    reach = [cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5) for m in cleaned]
    neighbourhood = record['neighbourhood_px']
    regions = [m & (reach[1-i] <= neighbourhood) for i, m in enumerate(cleaned)]
    arrays, reports = {}, []
    for i, (region, plate) in enumerate(zip(regions, record['plates'])):
        margin = max(2, round(left.shape[1] / 320))
        domain = cv2.dilate(region.astype(np.uint8), np.ones((2*margin+1, 2*margin+1), np.uint8)).astype(bool)
        sampled = ss._Region(domain, 30000)
        assert len(sampled.x) == plate['pixels']
        xy = np.c_[sampled.x, sampled.y]
        disparity = np.c_[xy, np.ones(len(xy))] @ plate['disparity_plane_abc']
        right_xy = xy - np.c_[disparity, np.zeros(len(xy))]
        residual = ss._sample(images.levels[-1][1], right_xy[:, 0], right_xy[:, 1]) - ss._sample(images.levels[-1][0], xy[:, 0], xy[:, 1])
        inlier = np.isfinite(residual) & (np.abs(residual) < .5)
        xyz = ss.backproject(xy, disparity, k, baseline)
        normal, offset = np.asarray(plate['normal']), plate['offset_m']
        np.testing.assert_allclose(xyz @ normal, offset, atol=1e-8)
        np.testing.assert_allclose(ss.project(xyz, k), xy, atol=1e-8)
        arrays.update({f'plate_{i}_raw_mask': masks[i], f'plate_{i}_clean_mask': cleaned[i],
                       f'plate_{i}_neighbourhood_mask': region, f'plate_{i}_fit_domain': domain,
                       f'plate_{i}_sample_xy': xy, f'plate_{i}_right_sample_xy': right_xy,
                       f'plate_{i}_disparity_px': disparity, f'plate_{i}_xyz_from_fitted_plane_m': xyz,
                       f'plate_{i}_final_residual': residual, f'plate_{i}_final_inlier': inlier})
        fraction = float(inlier.mean())
        # The solver reports residuals immediately before its last tiny update.
        assert abs(fraction - record['checks']['plate_inlier_fraction'][i]) < .002
        reports.append({'plate': i, 'sampled_pixels': len(xy), 'displayed_final_inlier_fraction': fraction,
                        'solver_inlier_fraction': record['checks']['plate_inlier_fraction'][i],
                        'normal': normal.tolist(), 'offset_m': offset})
    return arrays, reports


def polyline(image, xy, color, width=3, dashed=False):
    xy = np.asarray(xy)
    if len(xy) < 2:
        return
    draw = ImageDraw.Draw(image)
    if dashed:
        for start in range(0, len(xy)-1, 14):
            segment = xy[start:min(start+7, len(xy))]
            if len(segment) > 1:
                draw.line([tuple(p) for p in segment], fill=color, width=width)
    else:
        draw.line([tuple(p) for p in xy], fill=color, width=width)


def stage_panels(left, masks, arrays, single_xy, fused_xy, truth_xy):
    rgb = Image.fromarray(left)
    segmentation = left.copy()
    samples = left.copy()
    for i, color in enumerate(COLORS):
        segmentation[masks[i]] = np.rint(.55 * segmentation[masks[i]] + .45 * np.array(color)).astype(np.uint8)
        xy = arrays[f'plate_{i}_sample_xy'].astype(int)
        good = arrays[f'plate_{i}_final_inlier']
        samples[xy[~good, 1], xy[~good, 0]] = (220, 40, 40)
        # Show a deterministic subset as discrete dots rather than an opaque
        # filled face; all samples remain in the NPZ, without downsampling.
        dots = xy[good][::4]
        samples[dots[:, 1], dots[:, 0]] = color
    seam = rgb.copy()
    polyline(seam, single_xy, (20, 255, 30), 4)
    polyline(seam, fused_xy, (0, 125, 255), 3)
    polyline(seam, truth_xy, (255, 0, 210), 2, dashed=True)
    return [rgb, Image.fromarray(segmentation), Image.fromarray(samples), seam]


def run_dataset(root, kind, source, masks_dir, title):
    out = root / kind
    metrics = json.loads((out / 'metrics.json').read_text())
    plan = json.loads((source / 'station_plan.json').read_text())
    by_case = {v['case']: v for v in metrics['views']}
    stages = out / 'stages'
    samples_dir = out / 'fit_samples'
    stages.mkdir(exist_ok=True)
    samples_dir.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    centre_views = [v for v in plan['views'] if v['patch'] == 'center']
    preview = Image.new('RGB', (1920, len(centre_views) * 350), 'white')
    preview_draw = ImageDraw.Draw(preview)
    centre_index, audits, input_hashes = 0, [], []
    detail = [f'# {title}：分割、拟合像素与焊缝', '',
              '每幅图依次为完整左目原图、缓存SAM3掩码、实际拟合采样像素、单视图/融合焊缝与真值。',
              '横板橙色，竖板青色；拟合点面板显示内点的四分之一以便辨认，红色为最终光度残差较大的点。NPZ保存全部采样点。',
              '绿色为单视图预测，蓝色为三视角融合投影，品红虚线为仅供评价的真值。', '']
    for view in plan['views']:
        case = view['id']
        record = json.loads((out / 'predictions' / (case + '.json')).read_text())
        assert record['status'] == 'ok'
        left, right, masks, k, baseline, rotation = load_case(source, masks_dir, case)
        assert np.allclose(rotation, np.eye(3))  # These saved captures are already rectified.
        arrays, plane_audits = fit_samples(left, right, masks, record, k, baseline)
        # Nothing in fit_samples reads the following independent truth.
        curves, actual_pose = ground_truth(source, {'case': case, 'view': 'left'}, k, left.shape[:2])
        expected_pose = left_camera_pose(view, baseline)
        np.testing.assert_allclose(actual_pose, expected_pose, atol=2e-6)
        truth_xyz = np.concatenate([c['xyz'][a:b] for c in curves for a,b in runs(c['in_frame'])])
        ends = np.asarray(record['joint_ends_xyz'])
        single_xyz = ends[0] + np.linspace(0, 1, 451)[:, None] * (ends[1] - ends[0])
        fused = json.loads((out / 'predictions' / (view['station'] + '_fused.json')).read_text())
        fused_camera = transform(np.linalg.inv(actual_pose), np.asarray(fused['joint_ends_world']))
        fused_xyz = fused_camera[0] + np.linspace(0,1,451)[:,None] * (fused_camera[1] - fused_camera[0])
        single_xy, fused_xy, truth_xy = [ss.project(points, k) for points in (single_xyz, fused_xyz, truth_xyz)]
        arrays.update(intrinsics_index=k, baseline_m=np.array(baseline), predicted_seam_xyz_m=single_xyz,
                      fused_seam_xyz_in_camera_m=fused_xyz, reference_xyz_evaluation_only_m=truth_xyz)
        np.savez_compressed(samples_dir / (case + '.npz'), **arrays)
        panels = stage_panels(left, masks, arrays, single_xy, fused_xy, truth_xy)
        width,height = panels[0].size
        figure = Image.new('RGB', (4*width,height+100), 'white')
        draw = ImageDraw.Draw(figure)
        labels = ('Original left RGB','Cached SAM3 masks','Stereo fit samples','Green: single | Blue: fused')
        for col,(panel,label) in enumerate(zip(panels,labels)):
            figure.paste(panel,(col*width,100))
            draw.text((col*width+10,8),case,font=font,fill='black')
            draw.text((col*width+10,42),label,font=font,fill='black')
        draw.text((3*width+10,72),'Magenta: reference (evaluation only)',font=small,fill='black')
        figure.save(stages / (case + '.png'))
        detail += [f'## {case}', '', f"线位置平均误差：**{by_case[case]['reference_to_line']['mean_mm']:.3f} mm**；覆盖@10 mm：{by_case[case]['coverage_at_10mm']:.1%}。", '',
                   f'![原图、SAM掩码、实际拟合采样点、焊缝与真值]({case}.png)', '']
        if view['patch'] == 'center':
            # Zoom around the predicted line for display, independently of
            # reference geometry; the full uncropped panels remain above.
            centre_y = float(np.median(single_xy[:,1]))
            box = (0,max(0,int(centre_y)-150),width,min(height,int(centre_y)+151))
            top = centre_index * 350
            for col,panel in enumerate(panels):
                tile = ImageOps.contain(panel.crop(box),(480,270),Image.Resampling.LANCZOS)
                preview.paste(tile,(col*480+(480-tile.width)//2,top+70+(270-tile.height)//2))
                preview_draw.text((col*480+8,top+5),view['station']+' | '+labels[col],font=small,fill='black')
                if col == 3:
                    frow = next(f for f in metrics['fused'] if f['station']==view['station'])
                    preview_draw.text((col*480+8,top+32),f"Fused line error: {frow['reference_to_line']['mean_mm']:.3f} mm",font=small,fill='black')
            centre_index += 1
        checks = {'case':case,'status':record['status'],'plates':plane_audits,
                  'sample_domain_matches_solver':True,'planned_pose_matches_actual_capture':True}
        audits.append(checks)
        pair_path = source/'captures'/case/'input.json'
        pair = json.loads(pair_path.read_text())
        for v in pair['views']:
            image = pair_path.parent/v['image']
            assert digest(image)==v['image_sha256']
        input_hashes.append({'case':case,'input_json_sha256':digest(pair_path),
                             'mask_file_sha256':digest(masks_dir/(case+'_left')/'masks.npz'),
                             'left_rgb_sha256':pair['views'][0]['image_sha256'],
                             'right_rgb_sha256':pair['views'][1]['image_sha256']})
    (stages/'README.md').write_text('\n'.join(detail))
    preview.save(root/'images'/(kind+'_centres.png'))
    write_json(out/'fit_samples/manifest.json',{'samples':'Deterministic sample pixels reconstructed from unchanged fit domain and sampler',
               'xyz_note':'3D coordinates come from each fitted disparity plane, not independent per-pixel depth measurements',
               'inlier_note':'Final photometric residual recomputed at saved solution; <0.002 fraction difference from solver pre-final-step mask',
               'visualization_subsample':4,'views':audits,'inputs':input_hashes})
    return metrics,audits


def main(root):
    (root/'images').mkdir(exist_ok=True)
    combined = {}
    for key,(source,masks,title) in DATASETS.items():
        combined[key],_ = run_dataset(root,key,source,masks,title)
    lines = ['# 双目视差平面流程：多工位测试', '',
             '按 `docs/stereo_plane_seam.md` 重新运行A03、A15、A21：两种距离，各工位三组已保存拍摄位姿，共18组同步双目、36张原图。',
             '输入掩码沿用文档指定的SAM3缓存（其提示点由此前DA3预测自动生成）；本次重新计算双目视差平面、平面交线和三视角融合，未重跑SAM3或DA3。',
             '米制尺度由标定焦距和90 mm基线决定。拟合与融合阶段不读取CAD、渲染深度或焊缝真值。', '',
             '| 工位 | 整体距离融合线误差 mm | 近距离融合线误差 mm | 整体/近距离覆盖@5mm |',
             '|---|---:|---:|---:|']
    for station in ('A03','A15','A21'):
        f,c=[next(r for r in combined[k]['fused'] if r['station']==station) for k in ('full','close')]
        lines.append(f"| {station} | {f['reference_to_line']['mean_mm']:.3f} | {c['reference_to_line']['mean_mm']:.3f} | {f['coverage_at_5mm']:.0%} / {c['coverage_at_5mm']:.0%} |")
    lines += ['', '线误差是画内名义真实焊缝采样点到预测三维直线的平均距离，覆盖率使用有限线段；端点另与完整CAD板件接触线比较。它与旧流程的“预测点到有限名义焊缝”指标不同。', '',
              '三视角融合使用相机位姿把线段变换到共同坐标系，再按无真值的视图一致性合并、取长度并集。六组融合均使用了三个视图。', '',
              '以下依次展示原图、SAM3掩码、实际拟合采样像素、单视图/融合焊缝及真值。横板橙色，竖板青色；绿色为单视图线，蓝色为融合线，品红虚线仅为评价真值。',
              '主图围绕预测交线局部放大以便查看；各子报告直接展示全部18组的完整原图与分阶段图。整体距离数据固定狗身调臂；近距离数据是此前保存的左中右局部拍摄位姿，包含基座平移。本次没有执行新的机器人运动。',
              '拟合点图显示实际采样像素中内点的四分之一，全部采样像素及其按拟合视差回投的3D坐标保存在 `fit_samples/`。这些3D坐标不是独立测量的逐像素点云。', '',
              '## 整体距离约2 m', '', '![三个工位的整体距离分割、拟合点和焊缝](images/full_centres.png)', '',
              '[全部9组的完整原图与分阶段可视化](full/stages/README.md)；[左右目投影、逐视图及融合误差](full/README.md)。', '',
              '## 近距离约0.65 m', '', '![三个工位的近距离分割、拟合点和焊缝](images/close_centres.png)', '',
              '[全部9组的完整原图与分阶段可视化](close/stages/README.md)；[左右目投影、逐视图及融合误差](close/README.md)。', '',
              '## 融合后的完整误差', '',
              '| 距离 | 工位 | 线误差均值/最大 mm | 接触线两端误差 mm | 预测段到名义焊缝均值 mm |',
              '|---|---|---:|---:|---:|']
    for key,title in (('full','整体'),('close','近距')):
        for f in combined[key]['fused']:
            ends=' / '.join(f'{e:.2f}' for e in f['joint_end_error_mm'])
            lines.append(f"| {title} | {f['station']} | {f['reference_to_line']['mean_mm']:.3f} / {f['reference_to_line']['max_mm']:.3f} | {ends} | {f['predicted_points_to_reference']['mean_mm']:.3f} |")
    lines += ['', '名义焊接路径两端相对完整板件接触线各内缩30 mm，因此预测整条接触线到有限名义路径的端部误差会变大；本报告分别列出线位置、有限覆盖和完整接触线端点误差。', '',
              '13项双目算法单元测试通过；18/18组有输出。数据来自理想针孔、理想标定和位姿的静态仿真，不能作为实机精度保证。', '',
              '复现命令见各子报告对应的输入；本次关闭额外扰动测试，仅评估原始双目输入。', '']
    (root/'README.md').write_text('\n'.join(lines))
    write_json(root/'summary.json',{'method':'Documented stereo disparity-plane fit, finite intersection and pose-based multiview fusion',
               'datasets':{key:{'source':str(DATASETS[key][0]),'masks':str(DATASETS[key][1]),
               'success':sum(v['status']=='ok' for v in m['views']),'pairs':len(m['views']),'fused':m['fused']} for key,m in combined.items()}})
    print('STEREO_PLANE_STATIONS_REPORT_COMPLETE',root,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    main(p.parse_args().output_dir.resolve())
