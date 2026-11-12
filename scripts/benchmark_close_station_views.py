"""Rated-distance stereo depth and full-seam coverage from overlapping arm views."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark_stereo_seams import digest,write_json,runs,segment_distances,error_stats


def summarize(root):
    import numpy as np
    from PIL import Image
    from pxr import Usd
    from scipy.ndimage import binary_erosion
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from benchmark_da3 import workpiece_faces
    from evaluate_depth_stereo import metrics,truth_at_processed_pixels
    from da3_stereo import load_pair

    plan=json.loads((root/'station_plan.json').read_text())
    seam_summary=json.loads((root/'seams/summary.json').read_text())
    by_id={r['id']:r for r in seam_summary['results']}
    rows=[];union={};station_errors={};coverage_tracks={}
    (root/'evaluation').mkdir(exist_ok=True)
    for view in plan['views']:
        case,station=view['id'],view['station'];folder=root/'captures'/case
        _,_,raw_k,_=load_pair(folder/'input.json')
        capture=json.loads((folder/'evaluation/capture.json').read_text())
        if digest(ROOT/'assets/scenes/industrial-welding/industrial_welding.usdc')!=capture['source_usdc_sha256']:
            raise ValueError('Source changed')
        stage=Usd.Stage.Open(capture['usd'])
        with np.load(folder/'evaluation/truth.npz') as gt:
            for i,side in enumerate(('left','right')):
                name=case+'_'+side
                with np.load(root/'predictions'/case/'stereo_calibrated'/(side+'_depth.npz')) as data:
                    depth=data['depth_m'];k=data['intrinsics'].copy();rgb=data['processed_rgb']
                    truth=truth_at_processed_pixels(gt[side+'_z_m'],raw_k[i],k,depth.shape)
                ik=k.copy();ik[:2,2]-=.5
                cad,labels=workpiece_faces(stage,{'station':station,'workpiece_meshes':[station+'_base',station+'_web'],
                    'world_from_camera_optical':capture['world_from_camera_optical'][i]},ik,depth.shape)
                finite=np.isfinite(truth)&(truth>0)
                delta=np.full(truth.shape,np.inf)
                np.subtract(cad,truth,out=delta,where=finite&np.isfinite(cad))
                visible=finite&(labels>0)&(np.abs(delta)<.005)
                interior=np.zeros(truth.shape,bool)
                for label in np.unique(labels[visible]):
                    interior|=binary_erosion(visible&(labels==label),structure=np.ones((7,7)))
                rated=interior&(truth>=.3)&(truth<=.8)
                if not rated.any():raise ValueError('No rated workpiece pixels: '+name)
                seam=by_id[name]
                row={'id':name,'station':station,'patch':view['patch'],'view':side,
                    'depth_rated':metrics(depth[rated],truth[rated]),
                    'depth_workpiece_all':metrics(depth[interior],truth[interior]),
                    'workpiece_rated_fraction':float(rated.sum()/max(1,interior.sum())),
                    'torch_pixels':capture['torch_pixels'][side],
                    'cad_depth_scale_ratio':capture['cad_depth_scale_ratio'][side],'seam':seam}
                np.savez_compressed(root/'evaluation'/(name+'_depth.npz'),depth_m=depth,truth_z_m=truth,
                    workpiece_interior=interior,rated_workpiece_interior=rated,processed_rgb=rgb,intrinsics=k)
                if view['patch']=='center':
                    figure,axes=plt.subplots(1,4,figsize=(13,8),layout='constrained')
                    axes[0].imshow(rgb);axes[0].set_title('RGB')
                    for axis,array,title in ((axes[1],truth,'Independent RTX Z'),(axes[2],depth,'Nested stereo Z')):
                        im=axis.imshow(array,cmap='turbo',vmin=.3,vmax=.9);axis.set_title(title)
                    err=np.where(rated,np.abs(depth-truth)*1000,np.nan)
                    ei=axes[3].imshow(err,cmap='magma',vmin=0,vmax=50);axes[3].set_title('Rated workpiece error [mm]')
                    figure.colorbar(im,ax=axes[1:3],shrink=.6,label='Optical Z [m]')
                    figure.colorbar(ei,ax=axes[3],shrink=.6,label='Error [mm]; clipped at 50')
                    for axis in axes:axis.axis('off')
                    figure.suptitle(name+f' | rated MAE {row["depth_rated"]["mae_m"]*1000:.2f} mm | no GT fitting')
                    figure.savefig(root/(name+'_depth_error.png'),dpi=130);plt.close(figure)
                with np.load(root/'seams/predictions'/(name+'.npz')) as predicted,np.load(root/'seams/evaluation'/(name+'.npz')) as reference:
                    report=json.loads((root/'seams/predictions'/(name+'.json')).read_text())
                    parts=[predicted[f'candidate_{j}_xyz'][a:b] for j in range(len(report['candidates']))
                           for a,b in runs(predicted[f'candidate_{j}_valid'])]
                    xyz=reference['truth_0_xyz'];arc=reference['truth_0_arc'];mask=reference['truth_0_in_frame']
                    distance=segment_distances(xyz,parts)[0]
                    refs=[xyz[a:b] for a,b in runs(mask)]
                    error=segment_distances(np.concatenate(parts) if parts else np.zeros((0,3)),refs)[0]
                    if len(error):
                        np.testing.assert_allclose(error.mean()*1000,seam['automatic_position_error']['mean_mm'],atol=1e-7)
                    station_errors.setdefault(station,[]).extend(error.tolist())
                    row['in_frame_seam_euclidean_range_m']=[float(np.linalg.norm(xyz[mask],axis=1).min()),float(np.linalg.norm(xyz[mask],axis=1).max())]
                    row['in_frame_seam_z_range_m']=[float(xyz[mask,2].min()),float(xyz[mask,2].max())]
                    if not (.3<=row['in_frame_seam_euclidean_range_m'][0] and row['in_frame_seam_euclidean_range_m'][1]<=.8):
                        raise ValueError('In-frame seam outside rated working distance')
                    if station not in union:
                        union[station]={'arc':arc.copy(),'seen':np.zeros(len(arc),bool),'covered':np.zeros(len(arc),bool)}
                    else:np.testing.assert_allclose(union[station]['arc'],arc,atol=1e-9)
                    covered=mask&(distance<=.01)
                    union[station]['seen']|=mask;union[station]['covered']|=covered
                    coverage_tracks.setdefault(station,[]).append((view['patch']+' / '+side,mask.copy(),covered.copy()))
                rows.append(row)
    groups=[]
    for station,data in union.items():
        group_rows=[r for r in rows if r['station']==station]
        count=sum(r['depth_rated']['pixels'] for r in group_rows)
        groups.append({'station':station,'capture_pairs':len(group_rows)//2,
            'position_error':error_stats(station_errors[station]),
            'rated_depth_mae_mm':sum(r['depth_rated']['mae_m']*r['depth_rated']['pixels'] for r in group_rows)/count*1000,
            'full_nominal_seam_length_m':float(data['arc'][-1]),
            'union_in_frame_fraction':float(data['seen'].mean()),
            'union_coverage_at_10mm':float(data['covered'].mean()),
            'all_torch_pixels':sum(r['torch_pixels'] for r in group_rows)})
        np.savez_compressed(root/'evaluation'/(station+'_coverage_union.npz'),**data)
    summary={'capture_scope':'rated_close_patches','station_plan':plan,'views':rows,'stations':groups,
        'source_hashes':{'report_script':digest(Path(__file__)),**seam_summary['source_hashes']},
        'evaluation_scope':'All predicted valid points scored; only physically in-frame nominal roots used per view. Full-seam union uses each view independently, no predicted-geometry fitting or fusion.',
        'depth_mask':'Visible per-face CAD/RTX agreement within 5 mm; 3-pixel erosion; primary subset has independent RTX optical Z 300–800 mm',
        'scale_provenance':'Known 90 mm baseline only, unchanged Nested/SAM/geometry; no truth-derived correction',
        'limitations':['Static synthetic arm/base poses with ideal FK/calibration; no physical robot/controller executed',
            'Some incidental far-side CAD pixels can exceed rated range; full and rated workpiece depth metrics both retained',
            'Partial views at different poses have different pixels; far/near results are not a same-pixel ablation',
            'No multi-view depth fusion or ICP; union is an independent evaluation of per-view predictions']}
    if any(g['all_torch_pixels'] for g in groups):raise ValueError('Torch visible')
    write_json(root/'summary.json',summary)
    centres=[v for v in plan['views'] if v['patch']=='center']
    figure,axes=plt.subplots(3,2,figsize=(16,13),layout='constrained',gridspec_kw={'width_ratios':[1,2]})
    for j,v in enumerate(centres):
        axes[j,0].imshow(Image.open(root/'captures'/v['id']/'robot_context.png'));axes[j,0].set_title(v['station']+' | actual close arm/base pose')
        axes[j,1].imshow(Image.open(root/'seams'/(v['id']+'_left_overview.png')));axes[j,1].set_title('Centre patch | RGB / depth / seams')
        for axis in axes[j]:axis.axis('off')
    figure.suptitle('650 mm working distance | original 90 mm stereo | unchanged 50 mm offset mount | 0 torch pixels')
    figure.savefig(root/'close_results.png',dpi=130);plt.close(figure)
    figure,axes=plt.subplots(3,3,figsize=(13,15),layout='constrained')
    for j,station in enumerate(union):
        for i,v in enumerate(v for v in plan['views'] if v['station']==station):
            axes[j,i].imshow(Image.open(root/'captures'/v['id']/'left.png'))
            axes[j,i].set_title(station+' / '+v['patch']+' | centre 650 mm');axes[j,i].axis('off')
    figure.suptitle('Three overlapping physical arm views per station | full uncropped left RGB')
    figure.savefig(root/'overlapping_views.png',dpi=130);plt.close(figure)
    figure,axes=plt.subplots(3,1,figsize=(14,10),layout='constrained')
    for axis,(station,data) in zip(axes,union.items()):
        tracks=coverage_tracks[station]
        for j,(label,seen,covered) in enumerate(tracks):
            axis.plot(data['arc'][seen]*1000,np.full(seen.sum(),j),color='lightgray',lw=8)
            axis.plot(data['arc'][covered]*1000,np.full(covered.sum(),j),'.',color='tab:blue',ms=3)
        j=len(tracks);axis.plot(data['arc']*1000,np.full(len(data['arc']),j),color='lightgray',lw=8)
        axis.plot(data['arc'][data['covered']]*1000,np.full(data['covered'].sum(),j),'.',color='green',ms=3)
        axis.set_yticks(range(j+1),[r[0] for r in tracks]+['union']);axis.set_xlim(0,840)
        axis.set_title(station+f' | in-frame union {data["seen"].mean():.1%} | coverage within 10 mm {data["covered"].mean():.1%}')
        axis.set_xlabel('Nominal seam arc [mm]');axis.grid(axis='x',alpha=.25)
    figure.suptitle('Gray: physically in frame | blue: predicted within 10 mm | green: union of independent views')
    figure.savefig(root/'full_seam_coverage.png',dpi=130);plt.close(figure)
    write_report(root,summary)
    print(json.dumps(groups,ensure_ascii=False),flush=True)
    return summary


def range_filtered_seam_metrics(root,plan):
    """Report subset chosen by predicted range, independently of truth errors."""
    import numpy as np

    rows=[];station_errors={};station_totals={};unions={}
    for view in plan['views']:
        station=view['station']
        for side in ('left','right'):
            name=view['id']+'_'+side
            report=json.loads((root/'seams/predictions'/(name+'.json')).read_text())
            with np.load(root/'seams/predictions'/(name+'.npz')) as predicted,np.load(root/'seams/evaluation'/(name+'.npz')) as reference:
                parts=[predicted[f'candidate_{j}_xyz'][a:b] for j in range(len(report['candidates']))
                       for a,b in runs(predicted[f'candidate_{j}_valid'])]
                points=np.concatenate(parts) if parts else np.zeros((0,3))
                distance=np.linalg.norm(points,axis=1)
                retained=(distance>=.3)&(distance<=.8)
                retained_parts=[]
                for part in parts:
                    distance=np.linalg.norm(part,axis=1)
                    retained_parts.extend(part[a:b] for a,b in runs((distance>=.3)&(distance<=.8)))
                xyz=reference['truth_0_xyz'];arc=reference['truth_0_arc'];mask=reference['truth_0_in_frame']
                refs=[xyz[a:b] for a,b in runs(mask)]
                errors=segment_distances(points[retained],refs)[0]
                covered=mask&(segment_distances(xyz,retained_parts)[0]<=.01)
                if station not in unions:
                    unions[station]={'arc':arc.copy(),'seen':np.zeros(len(arc),bool),'covered':np.zeros(len(arc),bool)}
                np.testing.assert_allclose(unions[station]['arc'],arc,atol=1e-9)
                unions[station]['seen']|=mask;unions[station]['covered']|=covered
                station_errors.setdefault(station,[]).extend(errors.tolist())
                station_totals[station]=station_totals.get(station,0)+len(points)
                rows.append({'id':name,'station':station,'patch':view['patch'],'view':side,
                    'original_points':len(points),'retained_points':int(retained.sum()),
                    'excluded_points':int((~retained).sum()),'position_error':error_stats(errors),
                    'in_frame_truth_coverage_at_10mm':float(covered[mask].mean()) if mask.any() else None})
    groups=[]
    for station,errors in station_errors.items():
        groups.append({'station':station,'original_points':station_totals[station],
            'retained_points':len(errors),'excluded_points':station_totals[station]-len(errors),
            'position_error':error_stats(errors),
            'union_in_frame_fraction':float(unions[station]['seen'].mean()),
            'union_coverage_at_10mm':float(unions[station]['covered'].mean())})
    result={'selection_rule':'Keep original valid predicted points with camera-frame Euclidean distance in [0.3, 0.8] metres; selection does not use truth or localization error.',
        'scope':'Range-filtered candidates; remaining in-range false positives are not independently classified. Coverage uses retained contiguous segments without bridging excluded points.',
        'source_summary_sha256':digest(root/'summary.json'),'report_script_sha256':digest(Path(__file__)),
        'views':rows,'stations':groups}
    write_json(root/'seam_range_filtered_metrics.json',result)
    return result


def write_report(root,summary):
    plan=summary['station_plan'];rows=summary['views'];groups=summary['stations']
    filtered=range_filtered_seam_metrics(root,plan)
    filtered_views={row['id']:row for row in filtered['views']}
    filtered_stations={group['station']:group for group in filtered['stations']}
    lines=['# A15 / A03 / A21：650 mm近距离复测','',
        '每工位左、中、右三个重叠机械臂拍摄姿态，共9组同步双目、18张完整RGB。相机仍用已验证的50 mm偏置安装、原内参及90 mm基线，逐目验证焊枪像素为0。',
        '根据CAD几何和静态IK/碰撞选择姿态后才推理；保持既有Nested、SAM和接缝几何规则，不按真值调整尺度或位置。','',
        '## 原图、深度与焊缝总览','',
        '![原图、深度与焊缝总览](close_results.png)','',
        '### 左、中、右三个重叠视角','',
        '![各工位左、中、右三个重叠视角](overlapping_views.png)','',
        '### 整缝覆盖','',
        '![各工位整缝覆盖情况](full_seam_coverage.png)','',
        '## 焊缝定位误差：剔除工作范围外候选','',
        '按模型预测点到当前镜头的欧氏距离，保留300–800 mm内的候选点，剔除范围外候选。筛选依据不使用真实焊缝距离或定位误差；范围内的候选全部计入，仍可能含有误检。',
        '本表报告范围筛选后的三维点到真实画内焊缝线段的距离。保留/剔除数量按采样点计，不代表独立焊缝条数。原始全候选统计保留在summary.json；本次筛选结果保存在seam_range_filtered_metrics.json。','',
        '| 工位 | 工件深度MAE mm | 筛选后焊缝均距 mm | 筛选后P95 mm | 保留点数 | 剔除点数 | 整缝覆盖@10mm |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        selected=filtered_stations[g['station']];m=selected['position_error']
        lines.append('| %s | %.2f | %s | %s | %d | %d | %.1f%% |'%(g['station'],g['rated_depth_mae_mm'],
            f'{m["mean_mm"]:.2f}' if m else 'N/A',f'{m["p95_mm"]:.2f}' if m else 'N/A',
            selected['retained_points'],selected['excluded_points'],selected['union_coverage_at_10mm']*100))
    lines+=['','## 各视图','',
        '| 工位 | 局部 | 目 | 画内焊缝光轴范围 mm | 画内焊缝欧氏距离 mm | 额定深度MAE mm | 筛选后焊缝均距 mm | 剔除点数 | 画内覆盖@10mm |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        selected=filtered_views[row['id']];m=selected['position_error'];z=row['in_frame_seam_z_range_m'];d=row['in_frame_seam_euclidean_range_m']
        lines.append('| %s | %s | %s | %.0f–%.0f | %.0f–%.0f | %.2f | %s | %d | %.1f%% |'%(row['station'],row['patch'],row['view'],
            z[0]*1000,z[1]*1000,d[0]*1000,d[1]*1000,row['depth_rated']['mae_m']*1000,
            f'{m["mean_mm"]:.2f}' if m else 'N/A',selected['excluded_points'],selected['in_frame_truth_coverage_at_10mm']*100))
    lines+=['','## 范围与方法','',
        '650 mm是镜头到局部目标根部的距离；逐目另外核对画内焊缝真实欧氏距离均在300–800 mm内。一次只覆盖局部，三组交叠视野合起来核对完整840 mm前侧名义线，不将画外部分算漏检。',
        '主深度指标仅使用独立RTX真值300–800 mm内的可见底板/腹板内部，逐面腐蚀3像素。summary.json另保留全部可见工件指标和额定像素比例；零散远侧像素不隐瞒。',
        '整缝覆盖为范围筛选后各视图独立评价的逻辑并集；未作多视图深度融合或ICP。无保留候选时定位误差记N/A。与远距结果视角和像素不同，不能称为相同像素消融。',
        '静态检查包括关节限位、0.5 mm穿透阈值、四足接地与质心支撑；没有执行机器人行走或关节运动。当前仍为理想标定的仿真结果，支架质量和刚度未知。',
        '每个captures/*/arm_pose.usda可用Isaac Sim打开；依赖前一轮安装场景。原USDC及用户GUI没有写回。','',
        '## 全部原图、深度和焊缝','',
        '以下图片保留原始预测及其图内统计，包含已剔除的范围外候选；筛选后的误差和点数以上表为准。','']
    for v in plan['views']:
        name=v['id']
        lines += [f'### {name}','',
            f'[姿态USD](captures/{name}/arm_pose.usda)','',
            '**左目：原图、深度和焊缝**','',
            f'![{name} 左目原图、深度和焊缝](seams/{name}_left_seams.png)','',
            '**右目：原图、深度和焊缝**','',
            f'![{name} 右目原图、深度和焊缝](seams/{name}_right_seams.png)','']
    lines+=['','复跑：`.venv/bin/python scripts/benchmark_close_station_views.py --gpu 0`。',
        '仅汇总：`.venv/bin/python scripts/benchmark_close_station_views.py --evaluate-only --output-dir '+str(root.relative_to(ROOT))+'`。',
        '仅更新报告及范围筛选统计：`.venv/bin/python scripts/benchmark_close_station_views.py --report-only --output-dir '+str(root.relative_to(ROOT))+'`。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--report-only',action='store_true',help='Reuse saved predictions and summary; update range-filtered metrics and Markdown only')
    a=p.parse_args()
    root=(a.output_dir or ROOT/'outputs/benchmarks'/datetime.now(ZoneInfo('Asia/Taipei')).strftime('t_station_close_%Y%m%d_%H%M%S')).resolve()
    os.environ.setdefault('MPLCONFIGDIR',str(root/'mpl_cache'))
    if a.report_only:
        write_report(root,json.loads((root/'summary.json').read_text()))
        print('CLOSE_STATION_REPORT_COMPLETE',root,flush=True)
        return
    if not a.evaluate_only:
        from main import find_python,run_process
        if root.exists() and any(root.iterdir()) and not a.resume:p.error('Use a new directory or --resume')
        root.mkdir(parents=True,exist_ok=True)
        if not (root/'station_plan.json').is_file():
            # Planner requires an empty directory; its stdout stays in the parent log.
            from prepare_close_station_views import prepare
            from argparse import Namespace
            prepare(Namespace(baseline_run=ROOT/'outputs/benchmarks/t_station_unobstructed_20261110_01',output_dir=root,distance_m=.65))
        plan=json.loads((root/'station_plan.json').read_text())
        if not (root/'capture_status.json').is_file():
            isaac,error=find_python('isaac')
            if error:raise RuntimeError(error)
            run_process([isaac,str(ROOT/'scripts/capture_station_overviews.py'),'--resume','--output-dir',str(root),'--gpu',str(a.gpu)],root/'capture.log')
        if not (root/'predictions/completed.json').is_file():
            da3,error=find_python('da3')
            if error:raise RuntimeError(error)
            run_process([da3,str(ROOT/'scripts/predict_depth_stereo.py'),*[str(root/'captures'/v['id']/'input.json') for v in plan['views']],
                '--modes','stereo_calibrated','--stream-blocks','--output-dir',str(root/'predictions'),'--gpu',str(a.gpu)],root/'nested.log')
        if not (root/'seams/summary.json').is_file():
            if not (root/'seams/plan.json').is_file():
                from benchmark_stereo_seams import prepare
                seam_plan=prepare(root,root/'seams',[v['id'] for v in plan['views']],'usd_reference')
                seam_plan['visualization_depth_range_m']=[.3,.9];write_json(root/'seams/plan.json',seam_plan)
            run_process([sys.executable,str(ROOT/'scripts/benchmark_stereo_seams.py'),'--resume',
                '--output-dir',str(root/'seams'),'--gpu',str(a.gpu)],root/'seams.log')
    summarize(root)
    print('CLOSE_STATION_BENCHMARK_COMPLETE',root,flush=True)


if __name__=='__main__':main()
