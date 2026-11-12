"""End-to-end A04 arm-joint-angle stereo tests and all-three-weld reporting."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_stereo_seams import digest, error_stats, runs, segment_distances, write_json


def summarize(root):
    import csv
    import numpy as np
    from PIL import Image
    from pxr import Usd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from evaluation.a04_reference import reference_from_stage

    arm = json.loads((root/'arm_plan.json').read_text())
    results = json.loads((root/'seams/summary.json').read_text())
    by_id = {row['id']: row for row in results['results']}
    order = [pose['id']+'_'+side for pose in arm['views'] for side in ('left','right')]
    rows = [by_id[name] for name in order]
    pooled, seam_errors = [], {s['seam']: [] for s in rows[0]['seams']}
    for row in rows:
        name = row['id']
        with np.load(root/'seams/predictions'/(name+'.npz'), allow_pickle=False) as pred, np.load(
                root/'seams/evaluation'/(name+'.npz'), allow_pickle=False) as gt:
            report = json.loads((root/'seams/predictions'/(name+'.json')).read_text())
            parts = [pred[f'candidate_{i}_xyz'][a:b] for i in range(len(report['candidates']))
                     for a, b in runs(pred[f'candidate_{i}_valid'])]
            points = np.concatenate(parts) if parts else np.zeros((0, 3))
            distances = []
            for i, seam in enumerate(row['seams']):
                xyz, mask = gt[f'truth_{i}_xyz'], gt[f'truth_{i}_in_frame']
                refs = [xyz[a:b] for a, b in runs(mask)]
                distances.append(segment_distances(points, refs)[0])
            matrix = np.stack(distances, axis=1)
            if len(points):
                nearest = matrix.min(1)
                np.testing.assert_allclose(nearest.mean()*1000, row['automatic_position_error']['mean_mm'], atol=1e-7)
                pooled.extend(nearest.tolist())
                labels = matrix.argmin(1)
                for i, seam in enumerate(row['seams']):
                    values = matrix[labels==i, i]
                    if len(values):
                        np.testing.assert_allclose(values.mean()*1000, seam['position_error']['mean_mm'], atol=1e-7)
                        seam_errors[seam['seam']].extend(values.tolist())
    all_three_one_eye = sum(any(by_id[p['id']+'_'+side]['seams_with_at_least_50percent_coverage_at_10mm']==3
                               for side in ('left','right')) for p in arm['views'])
    all_three_both = sum(all(by_id[p['id']+'_'+side]['seams_with_at_least_50percent_coverage_at_10mm']==3
                            for side in ('left','right')) for p in arm['views'])
    aggregate = []
    for name, errors in seam_errors.items():
        records = [seam for row in rows for seam in row['seams'] if seam['seam']==name]
        samples = sum(r['in_frame_samples'] for r in records)
        aggregate.append({'seam': name, 'position_error': error_stats(errors),
                          'sample_weighted_coverage_at_10mm': sum((r['coverage_at_10mm'] or 0)*r['in_frame_samples'] for r in records)/samples,
                          'views_with_coverage_over_50percent_at_10mm': sum(r['coverage_at_50percent_threshold_passed'] is True for r in records)})
    summary = {'arm_pose_count': len(arm['views']), 'camera_view_count': len(rows),
               'all_three_welds_are_targets': True, 'pooled_position_error': error_stats(pooled),
               'point_precision_at_10mm': float(np.mean(np.asarray(pooled)<=.01)),
               'poses_all_three_covered_in_one_eye': all_three_one_eye, 'poses_all_three_covered_in_both_eyes': all_three_both,
               'coverage_success_definition': 'Each of the three in-frame welds has >=50% coverage within 10 mm',
               'per_seam': aggregate, 'views': rows, 'arm_plan': arm,
               'source_hashes': {'report_script': digest(Path(__file__)), **results['source_hashes']},
               'limits': ['Static FK poses and sampled collision checks, no dynamic controller tracking',
                          'Joint changes are 6/8/10 degrees about the same fixed base, not arbitrary viewpoints',
                          'Most seam points in these initial-arm views exceed the 800 mm rated working distance',
                          'Errors are for full finite CAD contacts; old single-vertical-reference results are not directly comparable',
                          'Known 90 mm baseline alignment retained; no truth-derived scale, shift or ICP']}
    write_json(root/'summary.json', summary)
    references = reference_from_stage(Usd.Stage.Open(arm['source_usd']))
    write_json(root/'three_weld_references.json', [{**r, 'control_world_m': r['control_world_m'].tolist(),
               'control_fixture_m': r['control_fixture_m'].tolist()} for r in references])
    with (root/'arm_poses.csv').open('w', newline='') as stream:
        fields = ['pose','description',*[f'joint{i}_deg' for i in range(1,7)],'camera_rotation_change_deg','camera_translation_change_mm']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for p in arm['views']:
            writer.writerow({'pose':p['id'],'description':p['description'],
                **{f'joint{i+1}_deg':value for i,value in enumerate(p['arm_q_deg'])},
                'camera_rotation_change_deg':p['optical_orientation_change_deg'],
                'camera_translation_change_mm':p['optical_translation_change_m']*1000})
    figure, axes = plt.subplots(3, 3, figsize=(15, 10), layout='constrained')
    for axis, p in zip(axes.flat, arm['views']):
        axis.imshow(Image.open(root/'captures'/p['id']/'robot_context.png'))
        axis.set_title(p['description'], fontsize=11); axis.axis('off')
    figure.suptitle('Actual arm joint poses | fixed B2 base | camera rigidly mounted on torch', fontsize=15)
    figure.savefig(root/'arm_pose_overview.png', dpi=130); plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(16, 11), layout='constrained', gridspec_kw={'width_ratios':[1.3,1]})
    x = np.arange(len(arm['views']))
    for side, colour in (('left','tab:blue'),('right','tab:orange')):
        errors = [by_id[p['id']+'_'+side]['automatic_position_error'] for p in arm['views']]
        values = [r['mean_mm'] if r else np.nan for r in errors]
        axes[0].plot(x, values, 'o-', color=colour, label=side)
    axes[0].set_xticks(x, [p['description'] for p in arm['views']], rotation=50, ha='right')
    axes[0].set(ylabel='Mean distance to any of 3 CAD welds [mm]', title='Current calibrated Nested + seam geometry')
    axes[0].grid(alpha=.3); axes[0].legend()
    coverage = np.array([[s['coverage_at_10mm'] if s['coverage_at_10mm'] is not None else np.nan for s in row['seams']] for row in rows])*100
    im = axes[1].imshow(coverage, vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
    axes[1].set_xticks(range(3), ['Vertical','Back/base','Return/base'])
    axes[1].set_yticks(range(len(rows)), [r['id'].replace('A04_arm_','').replace('A04_initial_arm','initial') for r in rows], fontsize=8)
    axes[1].set_title('Per-weld in-frame coverage within 10 mm [%]')
    for i in range(len(rows)):
        for j in range(3):
            value = coverage[i,j]
            axes[1].text(j,i,'N/A' if not np.isfinite(value) else f'{value:.0f}',ha='center',va='center',fontsize=8,
                         color='white' if value>60 else 'black')
    figure.colorbar(im,ax=axes[1],shrink=.6)
    figure.suptitle('9 arm poses / 18 camera views | all three contacts are welds | no truth fitting', fontsize=14)
    figure.savefig(root/'angle_errors_and_coverage.png',dpi=130);plt.close(figure)
    selected = ['A04_initial_arm','A04_arm_j1_p08','A04_arm_j4_m10']
    figure, axes = plt.subplots(3, 2, figsize=(16, 13), layout='constrained', gridspec_kw={'width_ratios':[1,2]})
    for row, name in enumerate(selected):
        axes[row,0].imshow(Image.open(root/'captures'/name/'robot_context.png'))
        axes[row,0].set_title(name+' | arm pose')
        axes[row,1].imshow(Image.open(root/'seams'/(name+'_left_overview.png')))
        axes[row,1].set_title('Left RGB / calibrated depth / three welds')
        axes[row,0].axis('off');axes[row,1].axis('off')
    figure.suptitle('Fixed display examples (initial, J1 +8 deg, J4 -10 deg), not selected by error',fontsize=14)
    figure.savefig(root/'image_examples.png',dpi=130);plt.close(figure)
    lines = ['# A04 机械臂多角度与三条焊缝测试', '',
             '改变实际机械臂关节姿态，相机始终固定在焊枪上，B2 底座保持不变；没有使用游离相机。',
             '9 组姿态：初始、J1 ±8°、J2 ±6°、J4 ±10°、J5 ±6°，共 18 张同步左右目图像。',
             '三条交线全部作为焊缝：竖直 V、背板/底板 B、侧板/底板 R。底部两条不再算误检。',
             '真值由三块真实 CAD 内表面的完整有限接触线独立计算，不参与推理；旧的单竖线指标不能直接比较。', '',
             f'预测点汇总均距 **{summary["pooled_position_error"]["mean_mm"]:.2f} mm**，P95 **{summary["pooled_position_error"]["p95_mm"]:.2f} mm**，10 mm 内点占比 **{summary["point_precision_at_10mm"]:.1%}**。',
             f'以每条画内焊缝 10 mm 覆盖率≥50%为阈值，至少一目三条均达标 **{all_three_one_eye}/9** 组，两目均达标 **{all_three_both}/9** 组。', '',
             '[机械臂姿态总览](arm_pose_overview.png) · [角度误差与覆盖率](angle_errors_and_coverage.png) · [原图/深度/焊缝示例](image_examples.png)', '',
             '| 姿态 | 目 | 路径数 | 均距 mm | P95 mm | V 覆盖@10mm | B 覆盖@10mm | R 覆盖@10mm |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for p in arm['views']:
        for side in ('left','right'):
            row = by_id[p['id']+'_'+side]; m = row['automatic_position_error']
            cov = ['画外' if s['coverage_at_10mm'] is None else f'{s["coverage_at_10mm"]:.1%}' for s in row['seams']]
            lines.append('| %s | %s | %d | %s | %s | %s | %s | %s |'%(p['description'],side,row['automatic_paths'],
                         f'{m["mean_mm"]:.2f}' if m else '未检出',f'{m["p95_mm"]:.2f}' if m else 'N/A',*cov))
    lines += ['', '## 逐图查看', '', '青色为自动预测，品红虚线为三条独立真值；虚线不属于识别输出。', '']
    for p in arm['views']:
        name = p['id']
        lines.append(f'- {p["description"]}：[左目](seams/{name}_left_seams.png) / '
                     f'[右目](seams/{name}_right_seams.png) · [机械臂姿态 USD](captures/{name}/arm_pose.usda)')
    lines += ['', '## 范围与数据', '',
              '采集为关节正运动学静态位姿；限位、相机安装刚性及从初始关节插值的采样碰撞均核对，不是动力学跟踪测试。',
              '保留 90 mm 基线尺度校正；未按焊缝真值拟合尺度、偏移或旋转。大部分焊缝深度超过 800 mm 工作距离，不能代表相机额定精度。',
              '误差仅统计有输出的预测点，需同时看逐缝覆盖率；缺失焊缝不计为零误差。',
              '`arm_poses.csv` 为关节角与相机变化，`seams/seam_metrics.csv` 为54项逐缝/逐目指标，`summary.json` 为汇总。',
              '捕获时的关节及相机源代码快照位于 source_snapshot（若存在）；原始场景未写回。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--evaluate-only',action='store_true')
    a = p.parse_args()
    root = (a.output_dir or ROOT/'outputs/benchmarks'/datetime.now().strftime('a04_arm_angles_%Y%m%d_%H%M%S')).resolve()
    os.environ.setdefault('MPLCONFIGDIR',str(root/'mpl_cache'))
    from main import find_python, run_process
    if not a.evaluate_only:
        if root.exists() and any(root.iterdir()) and not a.resume:
            p.error('Use an empty output directory or --resume')
        root.mkdir(parents=True,exist_ok=True)
        if not (root/'arm_plan.json').is_file():
            isaac,error = find_python('isaac')
            if error:raise RuntimeError(error)
            run_process([isaac,str(ROOT/'scripts/capture_a04_arm_angles.py'),'--output-dir',str(root),'--gpu',str(a.gpu)],root/'capture.log')
        plan = json.loads((root/'arm_plan.json').read_text())
        pairs = [root/'captures'/r['id']/'input.json' for r in plan['views']]
        from da3_stereo import load_pair
        for pair in pairs:load_pair(pair)
        if not (root/'predictions/completed.json').is_file():
            da3,error = find_python('da3')
            if error:raise RuntimeError(error)
            run_process([da3,str(ROOT/'scripts/predict_depth_stereo.py'),*map(str,pairs),'--modes','stereo_calibrated',
                         '--stream-blocks','--gpu',str(a.gpu),'--output-dir',str(root/'predictions')],root/'nested.log')
        if not (root/'seams/summary.json').is_file():
            command = [sys.executable,str(ROOT/'scripts/benchmark_stereo_seams.py'),'--source-run',str(root),
                       '--output-dir',str(root/'seams'),'--gpu',str(a.gpu)]
            if (root/'seams/plan.json').is_file():command.append('--resume')
            run_process(command,root/'seams.log')
    summarize(root)
    print('A04_ARM_ANGLE_BENCHMARK_COMPLETE',root,flush=True)


if __name__=='__main__':
    main()
