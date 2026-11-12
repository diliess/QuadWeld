"""Capture/evaluate whole-workpiece arm views for A15, A03 and A21."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_stereo_seams import digest, write_json


def summarize(root):
    import numpy as np
    from PIL import Image
    from pxr import Usd
    from scipy.ndimage import binary_erosion
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from benchmark_da3 import workpiece_faces
    from evaluate_depth_stereo import metrics, truth_at_processed_pixels
    from da3_stereo import load_pair

    plan = json.loads((root/'station_plan.json').read_text())
    clear = bool(plan.get('require_no_torch_pixels'))
    previous = (json.loads((Path(plan['baseline_run'])/'summary.json').read_text()) if clear else None)
    previous_rows = {(r['station'],r['view']):r for r in previous['views']} if previous else {}
    seam_summary = json.loads((root/'seams/summary.json').read_text())
    seams = {r['id']:r for r in seam_summary['results']}
    rows = []
    (root/'evaluation').mkdir(exist_ok=True)
    for view in plan['views']:
        case, station = view['id'], view['station']
        folder = root/'captures'/case
        _, _, raw_k, _ = load_pair(folder/'input.json')
        capture = json.loads((folder/'evaluation/capture.json').read_text())
        if digest(ROOT/'assets/scenes/industrial-welding/industrial_welding.usdc') != capture['source_usdc_sha256']:
            raise ValueError('Scene changed since capture')
        stage = Usd.Stage.Open(capture['usd'])
        with np.load(folder/'evaluation/truth.npz', allow_pickle=False) as gt:
            for i, side in enumerate(('left','right')):
                with np.load(root/'predictions'/case/'stereo_calibrated'/(side+'_depth.npz'), allow_pickle=False) as data:
                    pred, k, rgb = data['depth_m'], data['intrinsics'].copy(), data['processed_rgb']
                    depth = truth_at_processed_pixels(gt[side+'_z_m'], raw_k[i], k, pred.shape)
                    index_k = k.copy(); index_k[:2, 2] -= .5
                    surface, labels = workpiece_faces(stage, {'station': station,
                        'workpiece_meshes': [station+'_base', station+'_web'],
                        'world_from_camera_optical': capture['world_from_camera_optical'][i]}, index_k, pred.shape)
                    finite = np.isfinite(depth) & (depth > 0)
                    delta = np.full(pred.shape, np.inf)
                    np.subtract(surface, depth, out=delta, where=finite & np.isfinite(surface))
                    visible = finite & (labels > 0) & (np.abs(delta) < .005)
                    interior = np.zeros(pred.shape, bool)
                    for label in np.unique(labels[visible]):
                        interior |= binary_erosion(visible & (labels==label), structure=np.ones((7,7)))
                    if not interior.any(): raise ValueError('No workpiece interior: '+case+'/'+side)
                    rated = interior & (depth >= .3) & (depth <= .8)
                    value = metrics(pred[interior], depth[interior])
                    row = {'station':station, 'view':side, 'depth_workpiece_interior': value,
                           'torch_pixels':capture.get('torch_pixels',{}).get(side),
                           'rated_workpiece_pixels': int(rated.sum()),
                           'truth_workpiece_depth_range_m': [float(depth[interior].min()), float(depth[interior].max())],
                           'seam': seams[case+'_'+side]}
                    if clear:
                        with np.load(Path(plan['baseline_run'])/'evaluation'/(case+'_'+side+'_depth.npz')) as old:
                            common=old['workpiece_interior']&interior
                            if not common.any():raise ValueError('No common evaluation pixels')
                            np.testing.assert_allclose(depth[common],old['truth_z_m'][common],atol=2e-5)
                            row['common_workpiece_pixels_comparison']={
                                'pixels':int(common.sum()),
                                'old':metrics(old['depth_m'][common],old['truth_z_m'][common]),
                                'new':metrics(pred[common],depth[common])}
                    rows.append(row)
                    np.savez_compressed(root/'evaluation'/(case+'_'+side+'_depth.npz'),
                        truth_z_m=depth, workpiece_interior=interior, processed_rgb=rgb, depth_m=pred)
                    fig, axes = plt.subplots(1, 4, figsize=(13,8), layout='constrained')
                    axes[0].imshow(rgb); axes[0].set_title('RGB')
                    for axis, array, title in ((axes[1],depth,'Independent RTX Z'),(axes[2],pred,'Nested stereo Z')):
                        im = axis.imshow(array, cmap='turbo', vmin=.3, vmax=2.5); axis.set_title(title)
                    error = np.where(interior,np.abs(pred-depth)*1000,np.nan)
                    ei = axes[3].imshow(error, cmap='magma', vmin=0, vmax=200)
                    axes[3].set_title('Workpiece interior error [mm]')
                    fig.colorbar(im,ax=axes[1:3],shrink=.6,label='Axial Z [m]')
                    fig.colorbar(ei,ax=axes[3],shrink=.6,label='Error [mm], display clipped at 200')
                    for axis in axes:axis.axis('off')
                    fig.suptitle(f'{case} / {side} | interior MAE {value["mae_m"]*1000:.2f} mm | no truth fitting')
                    fig.savefig(root/(case+'_'+side+'_depth_error.png'),dpi=130);plt.close(fig)
    summary = {'stations':[v['station'] for v in plan['views']], 'views':rows, 'station_plan':plan,
               'no_torch_in_either_eye':all(r['torch_pixels']==0 for r in rows) if clear else None,
               'depth_mask_definition':'Visible CAD base/web surfaces agree with RTX within 5 mm; per-face 3-pixel erosion on processed grid',
               'seam_reference':'Original hidden 840 mm front root path at each T joint; no claim for back-side contacts',
               'scale_provenance':'Known 90 mm baseline only; no fitting to renderer or seam truth',
               'source_hashes': {'report_script':digest(Path(__file__)),**seam_summary['source_hashes']},
               'limitations':['Static geometric placements, no walking or arm control executed',
                              'Whole-workpiece views are beyond 800 mm rated range',
                              'CAD visibility selected poses before model inference, not by prediction error',
                              'All automatic candidates retained; missing seams are not assigned zero error']}
    write_json(root/'summary.json',summary)
    if clear and not summary['no_torch_in_either_eye']:
        raise ValueError('Missing or failed torch exclusion audit')
    fig,axes=plt.subplots(len(plan['views']),2,figsize=(16,13),layout='constrained',gridspec_kw={'width_ratios':[1,2]})
    for j,view in enumerate(plan['views']):
        case=view['id']
        axes[j,0].imshow(Image.open(root/'captures'/case/'robot_context.png'))
        axes[j,0].set_title(view['station']+' | actual base and arm pose')
        axes[j,1].imshow(Image.open(root/'seams'/(case+'_left_overview.png')))
        axes[j,1].set_title('Left RGB / Nested depth / automatic seam + evaluation reference')
        for axis in axes[j]:axis.axis('off')
    fig.suptitle('A15 / A03 / A21: complete-workpiece initial views | rigid arm-mounted stereo',fontsize=14)
    fig.savefig(root/'overview_results.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(len(plan['views']),3,figsize=(14,14),layout='constrained',gridspec_kw={'width_ratios':[1.5,1,1]})
    for j,view in enumerate(plan['views']):
        for i,(file,title) in enumerate((('robot_context.png','Actual robot pose'),('left.png','Left RGB'),('right.png','Right RGB'))):
            axes[j,i].imshow(Image.open(root/'captures'/view['id']/file));axes[j,i].axis('off')
            axes[j,i].set_title(view['station']+' | '+title)
    fig.suptitle('Whole T workpiece framing | original camera parameters and 90 mm baseline',fontsize=14)
    fig.savefig(root/'initial_views.png',dpi=130);plt.close(fig)
    if clear:
        figure,axes=plt.subplots(len(plan['views']),3,figsize=(12,14),layout='constrained')
        for j,view in enumerate(plan['views']):
            case=view['id']
            for i,(folder,side,title) in enumerate(((Path(plan['baseline_run']),'left','Previous left'),
                     (root,'left','New left: 0 torch pixels'),(root,'right','New right: 0 torch pixels'))):
                axes[j,i].imshow(Image.open(folder/'captures'/case/(side+'.png')))
                axes[j,i].set_title(view['station']+' | '+title);axes[j,i].axis('off')
        figure.suptitle('Physical 50 mm bracket extension | original K and optical world poses | full RGB')
        figure.savefig(root/'torch_removal_comparison.png',dpi=130);plt.close(figure)
    lines=['# A15 / A03 / A21：整体初始视角测试','',
           '通过机器人站位与真实机械臂关节姿态选择整体视角，相机始终刚性固定在焊枪上。每工位一组初始拍摄姿态、同步左右目，共3组6张 RGB。',
           '按固定 CAD 投影留边、前侧板面射线可见率、关节限位、静态碰撞与足部支撑选择；选择时没有运行模型或查看定位误差。',
           ('本次相机支架沿双目基线方向外移50 mm，外壳、光学双目、安装座及包络碰撞体同步平移，连接板同步延长。机器人反向平移补偿，保持上一轮的相机世界位姿和内参；机械臂关节角也相同。'
            if clear else '腹板和底板全部顶点均在两目画内并留4%边距；板面局部仍可能受焊枪或工件自身遮挡。')+
           '不是标准 home 关节角，也没有执行机器人行走/机械臂运动轨迹。','',
           '[初始视角与机械臂](initial_views.png) · [原图/深度/焊缝](overview_results.png)','',
           '| 工位 | 目 | 工件表面深度 MAE mm | 自动路径数 | 焊缝点均距 mm | 焊缝 P95 mm | 焊缝覆盖@10mm |',
           '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        seam=row['seam'];m=seam['automatic_position_error']
        lines.append('| %s | %s | %.2f | %d | %s | %s | %.1f%% |'%(row['station'],row['view'],
                     row['depth_workpiece_interior']['mae_m']*1000,seam['automatic_paths'],
                     f'{m["mean_mm"]:.2f}' if m else 'N/A（未检出）',f'{m["p95_mm"]:.2f}' if m else 'N/A',
                     seam['in_frame_truth_coverage_at_10mm']*100))
    if clear:
        lines+=['','## 焊枪排除与前后对比','',
                '六张原始RGB均为640×1100，未裁剪、未遮罩、未隐藏焊枪。每目367167个可见焊枪三角形均被视锥分离面排除；RTX语义标注再次验证六目焊枪像素均为0。',
                '另用原左目安装关系对同一个可见焊枪拍摄正对照，标注必须检出焊枪像素，排除标注失效造成的假零。掩码与正对照仅供评价，不送入Nested或SAM。',
                '相机内参、90 mm双目基线、相机世界位姿、机械臂关节角、场景和模型规则均与上一轮一致；相机相对焊枪的安装平移已改变，外参记录在station_plan.json。',
                '这是仿真支架方案；支架质量和刚度未知，未做实机安装或动态控制验证。',
                '[原图前后对比](torch_removal_comparison.png)','',
                '下表深度对比使用前后两轮可见工件内部掩码的交集，保持评价像素相同；主表仍报告新图全部可见工件内部。',
                '| 工位 | 目 | 原焊缝均距 mm | 新焊缝均距 mm | 相同像素原深度 MAE mm | 相同像素新深度 MAE mm |',
                '|---|---|---:|---:|---:|---:|']
        for row in rows:
            old=previous_rows[(row['station'],row['view'])]
            old_m,new_m=old['seam']['automatic_position_error'],row['seam']['automatic_position_error']
            lines.append('| %s | %s | %s | %s | %.2f | %.2f |'%(row['station'],row['view'],
                f'{old_m["mean_mm"]:.2f}' if old_m else 'N/A',f'{new_m["mean_mm"]:.2f}' if new_m else 'N/A',
                row['common_workpiece_pixels_comparison']['old']['mae_m']*1000,
                row['common_workpiece_pixels_comparison']['new']['mae_m']*1000))
    lines+=['','## 视角与评价范围','',
            '三者均为 T 接头；使用每个工位原 USD 隐藏的前侧根部参考路径（840 mm），与 A04 三面内角的三条接触线不同。未把背侧隐藏接触线当作本次前侧目标。',
            '深度误差仅评价独立 RTX 真值可见且作3像素边界腐蚀后的底板/腹板内部；焊缝空间误差为全部有效候选点到画内有限名义线的三维距离。两者不能混用，所有额外候选均保留。',
            'Nested 输入为双目 RGB + K/E，保留90 mm已知基线恢复尺度；未按真值拟合尺度或位姿。SAM阈值0.25与接缝几何流程不变，激光关闭。',
            '整体视角全部工件内部均超过800 mm额定距离，300–800 mm子集为空；不能据此声明相机额定精度。','',
            '| 工位 | 腹板根部高度 | 相对正面方位角 | 俯角 | 水平拍摄距离 | 前侧板面可见率 L/R（CAD采样） |',
            '|---|---:|---:|---:|---:|---:|']
    for view in plan['views']:
        height={'A15':.7,'A03':1.15,'A21':.25}[view['station']]
        vis=view['front_surface_ray_visibility_both_eyes']
        lines.append('| %s | %.0f mm | %+d° | %d° | %.2f m | %.1f%% / %.1f%% |'%(view['station'],height*1000,
            view['azimuth_deg'],view['elevation_deg'],view['horizontal_range_m'],vis[0]*100,vis[1]*100))
    lines+=['','## 分工位查看','']
    for view in plan['views']:
        case=view['id']
        lines.append(f'- **{view["station"]}**：[左目焊缝](seams/{case}_left_seams.png) / [右目焊缝](seams/{case}_right_seams.png) · '
                     f'[左目深度误差]({case}_left_depth_error.png) / [右目深度误差]({case}_right_depth_error.png) · [场景 USD](captures/{case}/arm_pose.usda)')
    lines+=['','`station_plan.json` 记录关节角、站位、候选选择及静态检查；`summary.json`、`seams/seam_metrics.csv` 保存完整指标。',
            '输入图像未裁剪；报告中的细节局部图只作展示，不作为推理输入。场景源文件与用户打开的 GUI 没有改动。','',
            ('复跑：先用`.venv/bin/python scripts/prepare_unobstructed_overviews.py --output-dir outputs/benchmarks/t_clear_new`生成新安装与姿态，再用`.venv/bin/python scripts/benchmark_station_overviews.py --resume --gpu 0 --output-dir outputs/benchmarks/t_clear_new`采集并评价。'
             if clear else '复跑：`.venv/bin/python scripts/benchmark_station_overviews.py --gpu 0`。'),
            '仅重新汇总：`.venv/bin/python scripts/benchmark_station_overviews.py --evaluate-only --output-dir '+str(root.relative_to(ROOT))+'`。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    for row in rows:
        print(row['station'],row['view'],'depth_MAE_mm',round(row['depth_workpiece_interior']['mae_m']*1000,2),
              'seam',row['seam']['automatic_position_error'],'coverage',row['seam']['in_frame_truth_coverage_at_10mm'],flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--evaluate-only',action='store_true')
    a=p.parse_args()
    root=(a.output_dir or ROOT/'outputs/benchmarks'/datetime.now(ZoneInfo('Asia/Taipei')).strftime('t_station_overviews_%Y%m%d_%H%M%S')).resolve()
    os.environ.setdefault('MPLCONFIGDIR',str(root/'mpl_cache'))
    if not a.evaluate_only:
        from main import find_python,run_process
        if root.exists() and any(root.iterdir()) and not a.resume:p.error('Use an empty directory or --resume')
        root.mkdir(parents=True,exist_ok=True)
        if not (root/'station_plan.json').is_file():
            run_process([sys.executable,str(ROOT/'scripts/capture_station_overviews.py'),'--plan-only','--resume','--output-dir',str(root)],root/'plan.log')
        if not (root/'capture_status.json').is_file():
            isaac,error=find_python('isaac')
            if error:raise RuntimeError(error)
            run_process([isaac,str(ROOT/'scripts/capture_station_overviews.py'),'--resume','--output-dir',str(root),'--gpu',str(a.gpu)],root/'capture.log')
        plan=json.loads((root/'station_plan.json').read_text())
        if not (root/'predictions/completed.json').is_file():
            da3,error=find_python('da3')
            if error:raise RuntimeError(error)
            pairs=[root/'captures'/v['id']/'input.json' for v in plan['views']]
            run_process([da3,str(ROOT/'scripts/predict_depth_stereo.py'),*map(str,pairs),'--modes','stereo_calibrated',
                         '--stream-blocks','--output-dir',str(root/'predictions'),'--gpu',str(a.gpu)],root/'nested.log')
        if not (root/'seams/summary.json').is_file():
            command=[sys.executable,str(ROOT/'scripts/benchmark_stereo_seams.py'),'--source-run',str(root),'--output-dir',str(root/'seams'),
                     '--cases',*[v['id'] for v in plan['views']],'--truth-mode','usd_reference','--gpu',str(a.gpu)]
            if (root/'seams/plan.json').is_file():command.append('--resume')
            run_process(command,root/'seams.log')
    summarize(root)
    print('STATION_OVERVIEW_BENCHMARK_COMPLETE',root,flush=True)


if __name__=='__main__':main()
