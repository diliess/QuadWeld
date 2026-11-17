"""Auditable, illustrated record of the fresh staged stereo experiment."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from tests.benchmarks.benchmark_staged_regions import ROOT, REGIONS, SCENE, digest, read_json, transform, write_json
from tests.benchmarks.benchmark_candidate_regions import line_metrics
from tests.benchmarks.seam_metrics import error_stats, segment_distances
from weld_vision.stereo_seam import project

COLORS = ((255, 156, 25), (0, 195, 255))
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def image_text(image, text, xy=(12, 8), size=22, color='white'):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, size)
    box = draw.textbbox(xy, text, font=font)
    draw.rectangle((box[0]-5, box[1]-4, box[2]+5, box[3]+4), fill='#14283c')
    draw.text(xy, text, fill=color, font=font)


def blend(rgb, mask, color, alpha=.4):
    out = np.array(rgb).copy()
    out[mask] = np.rint((1-alpha)*out[mask] + alpha*np.asarray(color)).astype(np.uint8)
    return Image.fromarray(out)


def draw_curve(image, points, pose, k, color, width=3, dashed=False):
    local = transform(np.asarray(points), np.linalg.inv(pose))
    if len(local) < 2 or not np.isfinite(local).all() or (local[:, 2] <= 0).any():
        return
    xy = project(local, k)
    draw = ImageDraw.Draw(image)
    if dashed:
        # Parameterized dashes are visible even on a two-endpoint reference.
        for a, b in zip(xy[:-1], xy[1:]):
            steps = max(1, int(np.linalg.norm(b-a)/12))
            for j in range(0, steps, 2):
                p, q = a+(b-a)*j/steps, a+(b-a)*min(j+1, steps)/steps
                draw.line([tuple(p), tuple(q)], fill=color, width=width)
    else:
        draw.line([tuple(p) for p in xy], fill=color, width=width)


def montage(panels, labels, path, tile_width=None):
    if tile_width:
        panels = [p.resize((tile_width, round(p.height*tile_width/p.width))) for p in panels]
    width, height = panels[0].size
    figure = Image.new('RGB', (width*len(panels), height+75), '#f2f5f9')
    for i, (panel, label) in enumerate(zip(panels, labels)):
        figure.paste(panel, (i*width, 75))
        ImageDraw.Draw(figure).text((i*width+10, 14), label, font=ImageFont.truetype(FONT, 20), fill='#14283c')
    figure.save(path)


def plot_setup():
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/quadweld-staged-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def render_view(run, phase, view, folder, ref):
    source = run / phase
    capture = source / 'captures' / view['id']
    result = source / 'results' / view['id']
    spec = read_json(capture / 'input.json')
    pose = np.asarray(spec['world_from_left_optical'])
    k = np.asarray(spec['views'][0]['intrinsics'], float)
    k[:2, 2] -= .5
    left, right = (Image.open(capture / (s + '.png')).convert('RGB') for s in ('left', 'right'))
    folder.mkdir(parents=True, exist_ok=True)
    montage([left, right], ['原始左目 RGB', '原始右目 RGB'], folder / 'stereo.jpg')
    selected, fit_panel = left.copy(), left.copy()
    with np.load(source / 'sam' / (view['id'] + '.npz')) as data:
        masks, scores, prompts, texts = data['masks'], data['scores'], data['prompts'], data['prompt_texts']
    tiles = []
    for index, (mask, score, prompt) in enumerate(zip(masks, scores, prompts)):
        tile = blend(left, mask, COLORS[index % 2]).resize((256, 440))
        picked = index in (view['selected_pair'] or [])
        image_text(tile, f'#{index}  {score:.3f}' + ('  已选' if picked else ''), size=15)
        image_text(tile, str(texts[prompt]), xy=(10, 40), size=14)
        tiles.append(tile)
    candidate_grid = Image.new('RGB', (256*max(1,min(4,len(tiles))), 440*max(1,int(np.ceil(len(tiles)/4)))), '#e9eef4')
    for index, tile in enumerate(tiles):
        candidate_grid.paste(tile, ((index % 4)*256, (index // 4)*440))
    if not tiles:
        image_text(candidate_grid, '无 SAM 候选', size=20)
    candidate_grid.save(folder / 'sam_candidates.jpg')
    if view['selected_pair']:
        for index, chosen in enumerate(view['selected_pair']):
            selected = blend(selected, masks[chosen], COLORS[index])
    arrays = None
    if (result / 'fit_samples.npz').exists():
        with np.load(result / 'fit_samples.npz') as data:
            arrays = {key: data[key] for key in data.files}
        raw = np.array(fit_panel)
        for i, color in enumerate(COLORS):
            xy = arrays[f'plate_{i}_sample_xy'].astype(int)
            good = arrays[f'plate_{i}_inlier']
            bad = xy[~good][::4]
            dots = xy[good][::4]
            raw[bad[:, 1], bad[:, 0]] = (240, 55, 55)
            raw[dots[:, 1], dots[:, 0]] = color
        fit_panel = Image.fromarray(raw)
        if 'bead_exclusion_mask' in arrays:
            excluded = blend(left, arrays['bead_exclusion_mask'], (235, 40, 190), alpha=.65)
            image_text(excluded, '二次拟合排除焊道、反光和阴影带', size=20)
            excluded.save(folder / 'bead_exclusion.jpg')
        plot_planes(view, arrays, folder)
    prediction = left.copy()
    if view['root_status'] == 'ok':
        draw_curve(prediction, view['root_world_m'], pose, k, (30, 245, 70), 3)
        for toe, color in zip(view.get('toes', []), ((255, 45, 215), (0, 245, 245))):
            for curve in toe['segments_world_m']:
                draw_curve(prediction, curve, pose, k, color, 3)
    prediction.save(folder / 'prediction.jpg')
    comparison = prediction.copy()
    draw_curve(comparison, ref['root_world_m'], pose, k, (255, 220, 0), 2, dashed=True)
    for toe in ref.get('toes_world_m', []):
        draw_curve(comparison, toe, pose, k, (255, 220, 0), 2, dashed=True)
    comparison.save(folder / 'truth_comparison.jpg')
    montage([left, right, selected, fit_panel, prediction],
            ['1 原始左目', '2 原始右目', '3 SAM3.1 自动选板', '4 实际拟合采样点', '5 本视角预测（无真值）'],
            folder / 'stages.jpg')
    lines = [f"# {view['id']}", '', f"状态：**{view['status']}**；根线：**{view['root_status']}**；测量耗时 {view['seconds']:.2f} s。", '',
             f"候选 {view['candidate_count']} 个；所选 {view['selected_pair']}；选择方式：{view['selection_method']}。", '',
             f"初次状态：{view.get('initial_status', '无候选')}；距离先验尝试：{view.get('range_prior_attempted', False)}；焊道排除后二次拟合：{view.get('bead_refit_used', False)}。", '',
             '![完整检测步骤](stages.jpg)', '', '橙色/青色为两个板面；拟合点图显示全部内点的 1/4，红色为光度残差较大的采样点。完整点集在 NPZ 中。绿色为根线，品红/亮青为焊趾。', '',
             '## SAM3.1 原始候选与自动选板', '', '![候选](sam_candidates.jpg)', '',
             '## 板面拟合', '', '| 板面 | 实际采样点 | 最终内点比例（全采样） | 鲁棒光度残差尺度 | 相机系 n | h（m） |',
             '|---|---:|---:|---:|---|---:|']
    for p in view.get('plane_audit', []):
        normal = ', '.join(f'{v:.6f}' for v in p['normal_camera'])
        lines.append(f"| {p['plate']} | {p['sample_count']} | {p['final_inlier_fraction_all_samples']:.1%} | {p['residual_scale']:.5f} | ({normal}) | {p['offset_camera_m']:.6f} |")
    lines += ['', '板面方程为 n·X=h。三维点由拟合视差平面重建，展示拟合结果，并非独立深度真值；光度内点阈值为归一化残差绝对值 <0.5。', '']
    if arrays is not None:
        lines += ['![板面与根线三维图](planes_3d.png)', '', '![视差与光度残差](fit_diagnostics.png)', '']
    if view.get('bead_refit_used'):
        lines += ['![焊道排除带](bead_exclusion.jpg)', '']
    if view.get('toes'):
        lines += ['## 焊趾提取', '', '| 板面编号 | 本视角有效采样比例 | 连续段数（≥5 点） | 连续有效长度 m |', '|---|---:|---:|---:|']
        for toe in view['toes']:
            length=sum(np.linalg.norm(np.diff(np.asarray(curve),axis=0),axis=1).sum() for curve in toe['segments_world_m'])
            lines.append(f"| {toe['plate']} | {toe['coverage']:.1%} | {len(toe['segments_world_m'])} | {length:.4f} |")
        lines += ['', '比例以本视角算法根线采样为分母；全长 CAD 覆盖在工位融合报告另行统计。左右一致性、焊脚距离、条带对比度等原始结果保存在每侧 NPZ 中。', '']
    lines += ['## 独立真值对比', '', '黄色虚线仅供评价；不参与选板、拟合、位姿规划或融合。', '',
              '![CAD 对比](truth_comparison.jpg)', '', '## 原始数据', '',
              f'- [双目输入与标定]({os.path.relpath(capture / "input.json", folder)})',
              f'- [SAM 全部候选 NPZ]({os.path.relpath(source / "sam" / (view["id"]+".npz"), folder)})',
              f'- [预测、板面参数、世界坐标及质量检查]({os.path.relpath(result / "record.json", folder)})']
    if arrays is not None:
        lines.append(f'- [实际拟合像素、左右对应点、视差、残差和重建点 NPZ]({os.path.relpath(result / "fit_samples.npz", folder)})')
    for toe in view.get('toes',[]):
        lines.append(f"- [板面 {toe['plate']} 焊趾的采样、有效标记与质量量]({os.path.relpath(result / ('toe_plate'+str(toe['plate'])+'.npz'),folder)})")
    lines += ['', '质量检查：', '', '```json', json.dumps(view.get('checks', {}), ensure_ascii=False, indent=2), '```', '']
    (folder / 'README.md').write_text('\n'.join(lines))
    return prediction


def plot_planes(view, arrays, folder):
    plt = plot_setup()
    fig = plt.figure(figsize=(8, 6), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    all_points = []
    for i, color in enumerate(('#ff9c19', '#00bfff')):
        xyz = arrays[f'plate_{i}_xyz_from_fitted_plane_m'][::50]
        ax.scatter(*xyz.T, s=1, color=color, alpha=.65, label=f'fitted plate {i}')
        all_points.append(xyz)
        centre = xyz.mean(0)
        normal = np.asarray(view['plane_audit'][i]['normal_camera'])
        ax.quiver(*centre, *(-normal*.08), color=color, arrow_length_ratio=.2)
    if view['root_status'] == 'ok':
        root = np.asarray(view['seam']['joint_ends_xyz'])
        ax.plot(*root.T, color='green', lw=3, label='plane intersection')
    xyz = np.concatenate(all_points)
    span = np.maximum(np.ptp(xyz, axis=0), .05)
    ax.set_box_aspect(span)
    ax.set(xlabel='camera X / m', ylabel='camera Y / m', zlabel='camera Z / m', title=view['id']+' | fitted geometry')
    ax.legend(fontsize=8)
    fig.savefig(folder / 'planes_3d.png', dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 10), layout='constrained')
    for i in (0, 1):
        xy = arrays[f'plate_{i}_sample_xy'][::5]
        for j, (key, title, cmap, limits) in enumerate((('disparity_px', 'Fitted disparity / px', 'viridis', {}),
                                                      ('photometric_residual', 'Final photometric residual', 'coolwarm', {'vmin':-.5, 'vmax':.5}))):
            value = arrays[f'plate_{i}_{key}'][::5]
            artist = axes[i, j].scatter(*xy.T, c=value, s=1, cmap=cmap, **limits)
            axes[i, j].invert_yaxis()
            axes[i, j].set_title(f'Plate {i} | '+title)
            axes[i, j].set_aspect('equal')
            fig.colorbar(artist, ax=axes[i, j], shrink=.7)
    fig.savefig(folder / 'fit_diagnostics.png', dpi=100)
    plt.close(fig)


def sample_curve(curve, spacing=.001):
    curve = np.asarray(curve, float)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance)>1e-12]
    distance, curve = distance[keep], curve[keep]
    samples = np.linspace(0., distance[-1], max(2,int(np.ceil(distance[-1]/spacing))+1))
    return np.stack([np.interp(samples,distance,curve[:,i]) for i in range(3)],axis=1)


def metrics_root(ends, ref, physical=True):
    ends, ref = np.asarray(ends), np.asarray(ref)
    metrics = line_metrics(ends, ref)
    pred, truth = sample_curve(ends), sample_curve(ref)
    metrics['predicted_to_finite_reference'] = error_stats(segment_distances(pred, [ref])[0])
    metrics['reference_to_finite_prediction'] = error_stats(segment_distances(truth, [ends])[0])
    metrics['reference_length_m'] = float(np.linalg.norm(ref[1]-ref[0]))
    metrics['predicted_length_m'] = float(np.linalg.norm(ends[1]-ends[0]))
    direction = (ref[1]-ref[0])/metrics['reference_length_m']
    positions = (ends-ref[0]) @ direction
    metrics['overextension_before_m'] = max(0., -float(positions.min()))
    metrics['overextension_after_m'] = max(0., float(positions.max())-metrics['reference_length_m'])
    metrics['uncovered_start_m'] = max(0., float(positions.min()))
    metrics['uncovered_end_m'] = max(0., metrics['reference_length_m']-float(positions.max()))
    if physical:
        ordered = ends[::-1] if np.linalg.norm(ends[0]-ref[1]) < np.linalg.norm(ends[0]-ref[0]) else ends
        metrics['endpoint_error_mm'] = (np.linalg.norm(ordered-ref,axis=1)*1000).tolist()
    return metrics


def metrics_toe(segments, reference):
    reference = np.asarray(reference)
    samples = sample_curve(reference)
    curves = [np.asarray(c) for c in segments if len(c)>=2]
    output = {'segments': len(curves), 'valid_length_m': float(sum(np.linalg.norm(np.diff(c,axis=0),axis=1).sum() for c in curves)),
              'reference_length_m': float(np.linalg.norm(np.diff(reference,axis=0),axis=1).sum())}
    reverse = segment_distances(samples, curves)[0]
    output['coverage_at_5mm'] = float(np.mean(reverse<=.005))
    output['coverage_at_10mm'] = float(np.mean(reverse<=.010))
    uncovered = reverse>.005
    changes = np.diff(np.r_[False,uncovered,False].astype(int))
    intervals = [(int(a),int(b)) for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]
    output['uncovered_intervals_sample_indices'] = intervals
    step = output['reference_length_m']/max(len(samples)-1,1)
    output['uncovered_intervals_s_m'] = [[a*step,min(b*step,output['reference_length_m'])] for a,b in intervals]
    output['largest_uncovered_run_m_approx'] = max((b-a for a,b in intervals),default=0)*step
    if curves:
        points = np.concatenate([sample_curve(c) for c in curves])
        output['prediction_to_reference'] = error_stats(segment_distances(points,[reference])[0])
    return output


def selection_diagnostic(run, phase, view, ref):
    capture = run / phase / 'captures' / view['id'] / 'evaluation'
    labels = read_json(capture / 'label_maps.json')['left']
    with np.load(capture / 'truth.npz') as data:
        ids = data['left_part_ids']
    truth = [np.isin(ids,[int(key) for key,value in labels.items() if name.lower() in str(value).lower()]) for name in ref['expected_parts']]
    with np.load(run / phase / 'sam' / (view['id']+'.npz')) as data:
        masks = data['masks']
    result = {'expected_parts':ref['expected_parts'], 'evaluation_only':True, 'selected_masks':[]}
    for index in view['selected_pair'] or []:
        iou = [float((masks[index]&m).sum()/max((masks[index]|m).sum(),1)) for m in truth]
        result['selected_masks'].append({'candidate':index,'iou':iou,'best_part':ref['expected_parts'][int(np.argmax(iou))]})
    return result


def draw_plan(run, observation, plans, folder):
    phase = observation.get('phase', 'overview')
    capture = run / phase / 'captures' / observation['id']
    spec = read_json(capture / 'input.json')
    image = Image.open(capture / 'left.png').convert('RGB')
    pose = np.asarray(spec['world_from_left_optical'])
    k = np.asarray(spec['views'][0]['intrinsics'], float)
    k[:2, 2] -= .5
    if observation['root_status'] != 'ok':
        image_text(image, observation['root_status'])
        image.save(folder / 'coarse_localization.jpg')
        return
    line = np.asarray(observation['root_world_m'])
    draw_curve(image, line, pose, k, (25, 240, 60), 4)
    for i, plan in enumerate(plans):
        target = np.asarray(plan['target_world_m'])
        xy = project(transform(target[None], np.linalg.inv(pose)), k)[0]
        x, y = map(float, xy)
        draw = ImageDraw.Draw(image)
        draw.ellipse((x-7, y-7, x+7, y+7), fill=('#ff9933','#00ccff','#d95cff')[i])
        image_text(image, f'V{i+1}', xy=(int(x)+9,int(y)-20), size=18)
    image_text(image, '绿色：双目粗定位；V1–V3：近拍目标', size=20)
    image.save(folder / 'coarse_localization.jpg')
    plt = plot_setup()
    fig = plt.figure(figsize=(9, 7), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    origin = line.mean(0)
    local = line-origin
    ax.plot(*local.T, color='green', lw=4, label='measured coarse root')
    normal = np.asarray(observation['face_normals_world'])
    for i,n in enumerate(normal):
        ax.quiver(0,0,0,*(n*.2),color=('#ff9933','#00bbdd')[i],label=f'observed normal {i}')
    for i, plan in enumerate(plans):
        pose = np.asarray(plan['world_from_optical_midpoint'])
        eye, target = pose[:3,3]-origin, np.asarray(plan['target_world_m'])-origin
        color = ('#ff9933','#00bbdd','#bd44e8')[i]
        ax.scatter(*eye,color=color,s=60,label=f'V{i+1}: {plan["fraction"]:.0%}, tilt {plan["along_axis_tilt_deg"]:+.0f} deg')
        ax.plot(*np.asarray([eye,target]).T,color=color,alpha=.7)
        ax.quiver(*eye,*(pose[:3,2]*.15),color=color)
    ax.set(xlabel='world X - origin / m',ylabel='world Y - origin / m',zlabel='world Z - origin / m',
           title=observation['station']+' | three views from measured coarse geometry')
    ax.set_box_aspect((1.1,1.,.8))
    ax.legend(fontsize=8)
    fig.savefig(folder / 'camera_plan_3d.png',dpi=130)
    plt.close(fig)


def fusion_image(run, region, observation, fused, ref, folder, data_dir):
    phase = observation.get('phase','overview')
    capture = run / phase / 'captures' / observation['id']
    spec = read_json(capture / 'input.json')
    pose = np.asarray(spec['world_from_left_optical'])
    k = np.asarray(spec['views'][0]['intrinsics'],float)
    k[:2,2] -= .5
    image = Image.open(capture / 'left.png').convert('RGB')
    draw_curve(image,ref['root_world_m'],pose,k,(255,210,0),2,dashed=True)
    if fused['status']=='ok':
        draw_curve(image,fused['root_world_m'],pose,k,(30,245,70),3)
        for name,color in (('base',(255,45,215)),('web',(0,245,245))):
            for curve in fused.get('toes',{}).get(name,{}).get('segments_world_m',[]):
                draw_curve(image,curve,pose,k,color,3)
    for toe in ref.get('toes_world_m',[]):
        draw_curve(image,toe,pose,k,(255,210,0),2,dashed=True)
    image_text(image,region+' | 绿：融合根线；黄：独立真值',size=20)
    image.save(folder / 'fusion_overview.jpg')
    plt = plot_setup()
    fig = plt.figure(figsize=(8,6),layout='constrained')
    ax = fig.add_subplot(111,projection='3d')
    origin = np.mean(ref['root_world_m'],axis=0)
    ax.plot(*(np.asarray(ref['root_world_m'])-origin).T,color='#e7a800',lw=4,ls='--',label='CAD root (evaluation)')
    if fused['status']=='ok':
        ax.plot(*(np.asarray(fused['root_world_m'])-origin).T,color='green',lw=2,label='fused root')
        for name,color in (('base','#ed28c5'),('web','#00adcc')):
            for curve in fused.get('toes',{}).get(name,{}).get('segments_world_m',[]):
                ax.plot(*(np.asarray(curve)-origin).T,color=color,lw=2)
        for toe in ref.get('toes_world_m',[]):
            ax.plot(*(np.asarray(toe)-origin).T,color='#e7a800',lw=1,ls='--')
    ax.set(xlabel='world X - origin / m',ylabel='world Y - origin / m',zlabel='world Z - origin / m',title=region+' | fused trajectories')
    ax.legend(fontsize=8)
    fig.savefig(folder/'fusion_3d.png',dpi=120)
    plt.close(fig)
    if fused.get('toes'):
        fig,axes=plt.subplots(2,1,figsize=(11,5),layout='constrained',sharex=True)
        for ax,name in zip(axes,('base','web')):
            with np.load(data_dir/(name+'_toe.npz')) as data:
                s,valid,views=data['s_m'],data['valid'],data['views']
            ax.step(s,views,where='mid',color='#476986',label='observing views')
            ax.scatter(s[valid],views[valid],s=5,color='green',label='valid (>=2 views)')
            ax.scatter(s[~valid],views[~valid],s=4,color='red',label='invalid / gap')
            ax.axhline(2,color='gray',ls='--',lw=1)
            ax.set(title=name+' toe',ylabel='view count',ylim=(-.1,3.3))
            ax.legend(fontsize=8,loc='lower right')
        axes[-1].set_xlabel('distance along fused toe-root axis / m')
        fig.savefig(folder/'toe_support_gaps.png',dpi=120)
        plt.close(fig)
    return image


def provenance(run):
    snapshot=run/'source_snapshot'
    sources=[ROOT/'main.py',ROOT/'pyproject.toml']
    sources+=list((ROOT/'calibration').glob('*.json'))
    sources+=list((ROOT/'environments').glob('*.yml'))+list((ROOT/'environments').glob('*.txt'))
    for name in ('weld_vision','weld_scene','model_runtime','tests'):
        sources+=list((ROOT/name).rglob('*.py'))
    hashes={}
    for source in sources:
        dest=snapshot/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        hashes[str(source.relative_to(ROOT))]=digest(source)
    for args,name in ((['git','status','--short'],'git_status.txt'),(['git','diff','--no-ext-diff'],'git_diff.patch')):
        result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,check=True)
        (snapshot/name).write_text(result.stdout)
    versions={}
    for package in ('numpy','opencv-python','scipy','isaacsim','Pillow','matplotlib'):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]='not registered in measurement environment'
    from model_runtime.runtime import SAM_CHECKPOINT
    write_json(run/'provenance.json',{'created':'2026-11-17','scene_usdc_sha256':digest(SCENE.with_name('industrial_welding.usdc')),
               'source_sha256':hashes,'sam_checkpoint':str(SAM_CHECKPOINT),'sam_checkpoint_sha256':digest(SAM_CHECKPOINT),
               'python':platform.python_version(),'measurement_packages':versions,'capture_gpu':'NVIDIA GeForce RTX 5090',
               'sam_environment':read_json(run/'logs/sam_environment.json'),
               'sam_prompts':['metal panel','metal plate','steel plate'],'sam_threshold':.25,
               'known_task_modes':{r:('bead' if r.endswith('_back') else 'root') for r in REGIONS},
               'coordinate_convention':'world metres; original optical (right, down, forward); index-centre K = capture K minus 0.5 principal point',
               'truth_in_prediction':False,'fresh_captures':True,'arm_motion_tested':False})


def report(run):
    references=read_json(run/'evaluation/reference.json')
    assert references['source_usdc_sha256']==digest(SCENE.with_name('industrial_welding.usdc'))
    refs=references['regions']
    output=run/'report'
    output.mkdir(exist_ok=True)
    phases={}
    for phase in ('overview','coarse-retry','close'):
        path=run/phase/'results/predictions.json'
        if path.exists():
            phases[phase]=read_json(path)['views']
            for view in phases[phase]: view['phase']=phase
    selection=read_json(run/'close_plan.json')
    fused=read_json(run/'close/results/fusion.json')
    first_fused=dict(fused)
    retry_plan = None
    retry_path = run/'bead-retry/close/results/predictions.json'
    if retry_path.exists():
        phases['bead-retry/close']=read_json(retry_path)['views']
        for view in phases['bead-retry/close']: view['phase']='bead-retry/close'
        retry_plan=read_json(run/'bead-retry/close_plan.json')
        retry_fused=read_json(run/'bead-retry/close/results/fusion.json')
        for region in retry_plan['overview']: fused[region]=retry_fused[region]
    write_json(run/'evaluation/final_fusion.json',fused)
    summary,diagnostics,thumbnails={},[],[]
    for region in REGIONS:
        folder=output/region
        folder.mkdir(exist_ok=True)
        attempts=[v for p,vs in phases.items() if p in ('overview','coarse-retry') for v in vs if v['station']==region]
        first_closes=[v for v in phases['close'] if v['station']==region]
        retried = retry_plan is not None and region in retry_plan['overview']
        close_phase='bead-retry/close' if retried else 'close'
        closes=[v for v in phases[close_phase] if v['station']==region]
        observation=selection['overview'][region]
        plans=[v for v in (retry_plan if retried else selection)['views'] if v['station']==region]
        data_dir=run/close_phase/'results'/region
        ref=refs[region]
        row={'mode':ref['mode'],'initial_status':attempts[0]['root_status'],'coarse_status':observation['root_status'],
             'coarse_attempts':len(attempts),'close_root_outputs':sum(v['root_status']=='ok' for v in closes),
             'close_task_outputs':sum(v['status']=='ok' for v in closes),'close_total':len(closes),
             'coarse_consistency_accepted':sum(v.get('coarse_consistency',{}).get('accepted',False) for v in closes),
             'fusion':fused[region], 'fusion_directory':str(data_dir.relative_to(run)),
             'initial_close_outputs':len(first_closes),'close_view_law':'existing inspection_pose, elevation 35 deg' if retried else 'visible normal bisector + along-axis tilt',
             'views':[]}
        if retried:
            row['initial_toes']={name:metrics_toe(first_fused[region].get('toes',{}).get(name,{}).get('segments_world_m',[]),curve)
                                 for name,curve in zip(('base','web'),ref.get('toes_world_m',[]))}
        near_images=[]
        for phase,views in phases.items():
            for view in views:
                if view['station']!=region: continue
                rendered=render_view(run,phase,view,folder/view['id'],ref)
                diagnostic=selection_diagnostic(run,phase,view,ref)
                diagnostics.append({'id':view['id'],'phase':phase,**diagnostic})
                metric={'id':view['id'],'phase':phase,'root_status':view['root_status'],'task_status':view['status'],
                        'mask_diagnostic':diagnostic,'seconds':view['seconds']}
                if view['root_status']=='ok':
                    metric['root']=metrics_root(view['root_world_m'],ref['root_world_m'],ref['reference_scope']=='physical contact')
                row['views'].append(metric)
                if phase==close_phase:
                    near_images.append(Image.open(run/phase/'captures'/view['id']/'left.png').convert('RGB'))
        if near_images:
            montage(near_images,[f"V{i+1}：粗定位 {p['fraction']:.0%}，倾角 {p['along_axis_tilt_deg']:+.0f}°" for i,p in enumerate(plans)],folder/'three_views.jpg')
        draw_plan(run,observation,plans,folder)
        if fused[region]['status']=='ok':
            row['root']=metrics_root(fused[region]['root_world_m'],ref['root_world_m'],ref['reference_scope']=='physical contact')
            row['toes']={name:metrics_toe(fused[region].get('toes',{}).get(name,{}).get('segments_world_m',[]),curve)
                         for name,curve in zip(('base','web'),ref.get('toes_world_m',[]))}
        thumb=fusion_image(run,region,observation,fused[region],ref,folder,data_dir)
        thumbnails.append((region,thumb))
        summary[region]=row
        region_report(run,folder,region,row,attempts,closes,observation,plans,ref,first_closes if retried else [])
        print('REPORTED',region,flush=True)
    write_json(run/'evaluation/metrics.json',{'regions':summary,'truth_used_for_prediction':False,
               'line_error_scope':'whole reference to infinite fitted line; finite segment/endpoint/coverage metrics separate'})
    write_json(run/'evaluation/selection_diagnostics.json',diagnostics)
    top_report(run,output,summary,phases)
    grid=Image.new('RGB',(4*320,2*600),'#eff4f8')
    for index,(region,im) in enumerate(thumbnails):
        im=im.resize((320,550))
        x,y=(index%4)*320,(index//4)*600
        grid.paste(im,(x,y+40))
        ImageDraw.Draw(grid).text((x+10,y+5),region,font=ImageFont.truetype(FONT,22),fill='#14283c')
    grid.save(output/'summary.jpg')
    provenance(run)
    print('STAGED_REPORT',run/'README.md',flush=True)


def region_report(run,folder,region,row,attempts,closes,observation,plans,ref,first_closes):
    initial=attempts[0]
    lines=[f'# {region}：从初始视角到融合轨迹', '',
           f"任务模式：**{'已有焊道的双侧焊趾' if row['mode']=='bead' else '两板接触根线'}**。模式来自工位任务配置；当前算法没有自动判断有无焊道。", '',
           '## 1 初始视角与输入', '',
           f"初始名义工作距离 2 m，状态 **{initial['root_status']}**。完整左右目、标定、世界位姿和图像哈希均保留。", '',
           f"![初始左右目]({initial['id']}/stereo.jpg)", '',
           f"[初始分割、拟合及状态详情]({initial['id']}/README.md)", '']
    if len(attempts)>1:
        lines += ['### 粗定位重试', '',
                  '初始失败后沿原光轴前移 1 m；重试位姿由原拍摄方向和已知初始工作距离产生。保留失败图像，没有用 CAD 替代测量。', '']
        for attempt in attempts[1:]:
            lines += [f"状态：**{attempt['root_status']}**。", '',
                      f"![重试左右目]({attempt['id']}/stereo.jpg)", '',
                      f"[重试的 SAM、板面拟合与质量检查]({attempt['id']}/README.md)", '']
    lines += ['## 2 粗定位结果与三个近拍视角', '',
              '绿色为本轮双目板面交线；标记点是沿测得根线 25%、50%、75% 的近拍目标。相机离目标 0.65 m，常规沿接缝倾角为 −10°、0°、+10°，绕光轴 −35°；竖直接缝避免相机向下偏移，实际倾角见下表。', '',
              '位姿计算使用测得的两板可见法向与根线，未读取 CAD。', '',
              '![粗定位示意](coarse_localization.jpg)', '', '![位姿规划三维图](camera_plan_3d.png)', '',
              f"[近拍计划与粗定位输入哈希]({os.path.relpath(run/('bead-retry/close_plan.json' if first_closes else 'close_plan.json'),folder)})", '',
              '| 视角 | 根线上目标比例 | 沿接缝倾角 | 目标世界坐标（m） | 相机中点世界坐标（m） |',
              '|---|---:|---:|---|---|']
    for p in plans:
        target=', '.join(f'{v:.5f}' for v in p['target_world_m'])
        eye=', '.join(f'{v:.5f}' for v in np.asarray(p['world_from_optical_midpoint'])[:3,3])
        lines.append(f"| V{p['view_index']+1} | {p['fraction']:.0%} | {p['along_axis_tilt_deg']:+.0f}° | {target} | {eye} |")
    if region == 'A04':
        lines += ['', '首个近拍计划的向下相机偏移造成 V1 右目图像不可用。改为向上偏移后重新采集；计划只按测得根线与重力方向调整，未修正根线端点。', '',
                  f"[初始近拍计划]({os.path.relpath(run/'close_plan_initial.json',folder)}) · [失败记录]({os.path.relpath(run/'close/capture_failure_initial.json',folder)}) · [保留的部分左目]({os.path.relpath(run/'close/failed_captures/A04_view0_attempt1/left.png',folder)})", '']
    if closes:
        lines += ['', '下面为实际重拍的三个左目全图，左右目原图均在各视角详情中。', '', '![三个实际近拍视角](three_views.jpg)', '']
    if first_closes:
        lines += ['### 观察方向补充对照', '',
                  '首轮使用两板可见法向的角平分线（本区域约 45°），底板焊趾多视角有效覆盖很低。为这两个焊道区域另拍三个视角，采用当前主流程已有 inspection_pose：相对底板 35°、绕光轴 −35°、0.65 m 工作距离，目标仍为原粗定位根线的 25%/50%/75%。', '',
                  '下文主结果为 35° 补拍；首轮全部原图、分割、拟合和结果保留，可逐项对照。', '',
                  f"[补拍位姿计划]({os.path.relpath(run/'bead-retry/close_plan.json',folder)})", '']
        for view in first_closes:
            lines.append(f"- [首轮 {view['id']} 的完整流程]({view['id']}/README.md)")
        lines += ['']
    lines += ['## 3 每个近拍视角的检测流程', '',
              '每幅流程图依次展示：原始左右目 → SAM3.1 自动选板 → 实际拟合采样点 → 根线/焊趾结果。板面三维图、视差图、残差图和参数表见逐视角详情。', '']
    for view in closes:
        consistent=view.get('coarse_consistency',{}).get('accepted',False)
        lines += [f"### V{view['view_index']+1} · {view['id']}", '',
                  f"根线状态 **{view['root_status']}**；任务状态 **{view['status']}**；与粗定位一致性通过：{consistent}。", '',
                  f"![该视角流程]({view['id']}/stages.jpg)", '',
                  f"[SAM 原始候选、板面拟合、三维结果、独立真值与数据文件]({view['id']}/README.md)", '']
    lines += ['## 4 世界坐标融合与独立评价', '',
              f"融合状态：**{row['fusion']['status']}**；使用视角：{row['fusion'].get('views_used',[])}。", '',
              '预测由左目光学坐标按拍摄外参变换到世界坐标，然后使用 5 mm 共识容差融合。与粗定位偏差 >80 mm 或方向差 >10° 的视角不参与融合；上述门限均不使用真值。', '',
              '![融合与真值投影](fusion_overview.jpg)', '', '![世界系轨迹](fusion_3d.png)', '']
    metrics=row.get('root')
    if metrics:
        errors=metrics['line_error']
        lines += [f"融合根线位置误差：均值 **{errors['mean_mm']:.3f} mm**，P95 **{errors['p95_mm']:.3f} mm**，最大 **{errors['max_mm']:.3f} mm**。", '',
                  f"方向误差 {metrics['direction_error_deg']:.4f}°；有限段覆盖@5 mm **{metrics['coverage_at_5mm']:.1%}**。", '',
                  f"真值长度 {metrics['reference_length_m']:.4f} m，预测长度 {metrics['predicted_length_m']:.4f} m。", '',
                  f"沿真值起点/终点的外延分别为 {metrics['overextension_before_m']*1000:.1f} / {metrics['overextension_after_m']*1000:.1f} mm；未覆盖起点/终点分别为 {metrics['uncovered_start_m']*1000:.1f} / {metrics['uncovered_end_m']*1000:.1f} mm。", '']
        if 'endpoint_error_mm' in metrics:
            lines += ['两物理端点误差：**'+' / '.join(f'{v:.1f}' for v in metrics['endpoint_error_mm'])+' mm**。', '']
        else:
            lines += ['B01 的真值是预选 1 m 窗口，窗口端点不是物理接头端点；外延仅表示超出该评价窗口。', '']
    if row.get('toes'):
        lines += ['| 焊趾 | 到 CAD 焊趾均值 / P95 / 最大 mm | 有效长度 m | 全长覆盖@5 mm | 连续有效段数 | 最大未覆盖段约 m |',
                  '|---|---:|---:|---:|---:|---:|']
        for name,toe in row['toes'].items():
            error=toe.get('prediction_to_reference')
            text=' / '.join(f"{error[k]:.3f}" for k in ('mean_mm','p95_mm','max_mm')) if error else '无有效输出'
            lines.append(f"| {name} | {text} | {toe['valid_length_m']:.3f} | {toe['coverage_at_5mm']:.1%} | {toe['segments']} | {toe['largest_uncovered_run_m_approx']:.3f} |")
        lines += ['', '焊趾融合要求同一位置至少两个视角支持。无效段保留为缺口，CSV 含 valid、views、spread 列；没有跨缺口连线。', '',
                  '![焊趾视角支持数与缺口](toe_support_gaps.png)', '']
        if row.get('initial_toes'):
            lines += ['### 两轮观察方向对照', '', '| 焊趾 | 首轮角平分线有效长度 m / 覆盖@5mm | 主流程 35° 有效长度 m / 覆盖@5mm |', '|---|---:|---:|']
            for name,toe in row['toes'].items():
                before=row['initial_toes'][name]
                lines.append(f"| {name} | {before['valid_length_m']:.3f} / {before['coverage_at_5mm']:.1%} | {toe['valid_length_m']:.3f} / {toe['coverage_at_5mm']:.1%} |")
            lines += ['']
    lines += ['线位置误差按整条 CAD 根线到预测无限直线统计；有限段覆盖、双向有限段误差、端点、外延和断段另列。真值只在预测完成后读取，不做位置/尺度配准。', '',
              '## 5 逐视角评价', '', '| 视角 | 线位置均值 mm | 最大端点误差 mm | 覆盖@5 mm |', '|---|---:|---:|---:|']
    for v in row['views']:
        m=v.get('root')
        text=f"{max(m['endpoint_error_mm']):.1f}" if m and 'endpoint_error_mm' in m else '—'
        error=f"{m['line_error']['mean_mm']:.3f}" if m else '失败'
        cov=f"{m['coverage_at_5mm']:.1%}" if m else '—'
        lines.append(f"| {v['id']} | {error} | {text} | {cov} |")
    lines += ['', f"[融合世界坐标 CSV / 焊趾 NPZ]({os.path.relpath(run/row['fusion_directory'],folder)})", '',
              f"[完整评价 JSON]({os.path.relpath(run/'evaluation/metrics.json',folder)})", '',
              '自由静态相机、理想仿真标定；本轮没有检验机械臂可达性、碰撞和焊枪姿态执行。', '']
    (folder/'README.md').write_text('\n'.join(lines))


def top_report(run,output,summary,phases):
    total=sum(len(vs) for vs in phases.values())
    lines=['# 八个区域的分阶段双目焊缝识别（2026-11-17）', '',
           f'本轮重新采集 **{total} 组双目 / {total*2} 张 RGB**：8 组初始视角、2 组粗定位重试、24 组首轮近拍、6 组背面焊道观察方向补充对照。没有复用旧实验的图像或 SAM 结果。', '',
           '覆盖 A01、A03、A23、A06、A04、B01 的两板根线，以及 A25/A26 背面已有焊道的双侧焊趾。SAM3.1 阈值 0.25，640×1100 像素、90 mm 基线、激光关闭。', '',
           '流程：初始双目 → 输入/标定检查 → SAM 自动选板 → 双目板面拟合 → 世界系粗根线/法向 → 三个近拍位姿 → 重新采集与 SAM → 板面拟合与根线/焊趾 → 外参变换 → 一致性检查 → 多视角融合 → 独立 CAD 评价。', '',
           '初始 2 m 远景中 A04 拟合失败、B01 无分割候选；沿原光轴前移至名义 1 m 后，二者均完成粗定位。全部失败记录保留。', '',
           '初始瞄准使用已知场景区域；粗定位、近拍规划、检测与融合不读取 CAD 或渲染深度/标签。每个工位的近拍目标为测得根线的 25%、50%、75%，工作距离 0.65 m，常规沿接缝倾角 −10°/0°/+10°。A04 竖直接缝首个向下偏移位姿右目不可用，改为向上偏移后完成采集；失败部分图像与前后计划均保留。', '',
           '![八个区域融合结果](report/summary.jpg)', '',
           '| 区域（点击查看完整流程） | 粗定位尝试 | 近拍任务成功/总数 | 融合线均值 / 最大 mm | 最大物理端点误差 mm | 根线覆盖@5mm |',
           '|---|---:|---:|---:|---:|---:|']
    for region,row in summary.items():
        m=row.get('root')
        err=f"{m['line_error']['mean_mm']:.3f} / {m['line_error']['max_mm']:.3f}" if m else '未融合'
        ends=f"{max(m['endpoint_error_mm']):.1f}" if m and 'endpoint_error_mm' in m else '局部窗口'
        cov=f"{m['coverage_at_5mm']:.1%}" if m else '—'
        lines.append(f"| [{region}](report/{region}/README.md) | {row['coarse_attempts']} | {row['close_task_outputs']}/{row['close_total']} | {err} | {ends} | {cov} |")
    lines += ['', '## 焊趾结果', '', '| 区域 | 焊趾 | 均值 / P95 mm | 有效长度 m | 全长覆盖@5mm | 有效段数 |', '|---|---|---:|---:|---:|---:|']
    for region,row in summary.items():
        for name,toe in row.get('toes',{}).items():
            err=toe.get('prediction_to_reference')
            text=f"{err['mean_mm']:.3f} / {err['p95_mm']:.3f}" if err else '无有效输出'
            lines.append(f"| {region} | {name} | {text} | {toe['valid_length_m']:.3f} | {toe['coverage_at_5mm']:.1%} | {toe['segments']} |")
    lines += ['', '线位置精度、完整轨迹长度和焊趾覆盖需要分别检查。根线融合保留多个视角有限段的范围，不能自动消除单视图的端点外延。焊趾至少要求两个视角支持，无效段不补齐。', '',
              '## 检查是否遗漏的步骤', '',
              '| 步骤 | 本轮记录 | 尚需工作 |', '|---|---|---|',
              '| 输入与传感器条件 | 原始左右图、尺寸/哈希、同步静态状态、K、畸变、90 mm 基线、单位、外参正交性检查 | 实机时间同步、标定误差与漂移验证 |',
              '| 极线与像素约定 | 理想针孔已校正；核心 K 主点减 0.5，保存原标定 | 实机校正残差/重投影误差测量 |',
              '| 粗定位与重试 | 初始视角、全部 SAM 候选、板面对/交线、A04/B01 的失败与前移重试 | 初始寻找工位仍使用已知区域 |',
              '| 三视角规划 | 源粗定位 JSON 哈希、目标点、位姿、倾角、实际近拍左右图 | 机器人逆解、可达性、碰撞、遮挡预检 |',
              '| 分割与板面拟合 | SAM3.1 原始候选/分数、自动选板、实际采样点、光度残差、三维板面及参数；焊道排除后二次拟合 | 更复杂反光/纹理条件的鲁棒性 |',
              '| 检测任务选择 | 无焊道工位输出根线；已有焊道工位输出两条焊趾 | 自动判断是否已有焊道尚未实现 |',
              '| 坐标变换与质量门限 | 相机到世界外参、求解状态、距离消歧、粗定位一致性、融合共识/拒绝结果 | 实机手眼标定与机械臂定位误差 |',
              '| 有限轨迹与覆盖 | 物理端点、长度、外延、未覆盖范围、焊趾断段与视角支持数；B01 仅 1 m 窗口 | 端点外延约束及自适应补拍，不能只验收线误差 |',
              '| 独立评价与复现 | CAD 源哈希、仅评价使用的深度/标签、无真值配准、逐视角指标、CSV、NPZ、日志、源码快照 | 不构成实机精度/焊接执行验收 |', '',
              '## 数据索引', '',
              '- [全部评价指标](evaluation/metrics.json)',
              '- [选板独立诊断（仅评价，不回馈预测）](evaluation/selection_diagnostics.json)',
              '- [粗定位最终选择](coarse_selection.json)',
              '- [近拍计划与粗定位来源](close_plan.json)',
              '- [近拍逐视角预测](close/results/predictions.json)',
              '- [最终融合状态与轨迹（背面焊道采用 35° 补拍）](evaluation/final_fusion.json)',
              '- [首轮融合结果（完整保留）](close/results/fusion.json)',
              '- [背面焊道 35° 补拍计划](bead-retry/close_plan.json)',
              '- [日志](logs/)', '- [源码快照与工作区状态](source_snapshot/)',
              '- [版本、模型与源文件哈希](provenance.json)',
              '- [历史非当前双目实验归档](../../archive/non_stereo_experiments_20261117/README.md)', '',
              '每个 report/<区域>/README.md 按初始视角、粗定位、三视角、逐视角分割/板面拟合、融合和误差组织。真实渲染图像与绘图分开保存。', '',
              '## 复现', '', '从项目根目录运行；渲染/SAM 需要可用的 NVIDIA 驱动。请换一个全新 RUN 路径；采集脚本会核对并续接位姿一致的完整输入，拒绝覆盖位姿不同的输入。', '', '```bash',
              'RUN=outputs/benchmarks/20261118_01__stereo__multi8__staged',
              '.venv/bin/python -m tests.benchmarks.benchmark_staged_regions capture-overview --run-dir "$RUN"',
              '.venv-sam/bin/python -m model_runtime.sam_adapter "$RUN"/overview/captures/*/input.json --output-dir "$RUN"/overview/sam --gpu 0 --threshold 0.25',
              '.venv/bin/python -m tests.benchmarks.benchmark_staged_regions measure-overview --run-dir "$RUN"',
              '# 初始粗定位有失败时重拍；无失败可跳过这三步',
              '.venv/bin/python -m tests.benchmarks.benchmark_staged_regions capture-coarse-retry --run-dir "$RUN"',
              '.venv-sam/bin/python -m model_runtime.sam_adapter "$RUN"/coarse-retry/captures/*/input.json --output-dir "$RUN"/coarse-retry/sam --gpu 0 --threshold 0.25',
              '.venv/bin/python -m tests.benchmarks.benchmark_staged_regions measure-coarse-retry --run-dir "$RUN"',
              '.venv/bin/python -m tests.benchmarks.benchmark_staged_regions capture-close --run-dir "$RUN"',
              '.venv-sam/bin/python -m model_runtime.sam_adapter "$RUN"/close/captures/*/input.json --output-dir "$RUN"/close/sam --gpu 0 --threshold 0.25',
              '.venv/bin/python -m tests.benchmarks.benchmark_staged_regions measure-close --run-dir "$RUN"',
              '# 背面焊道采用当前主流程 35° 观察方向补充对照',
              '.venv/bin/python -m tests.benchmarks.benchmark_staged_regions plan-bead-retry --run-dir "$RUN"',
              '.venv/bin/python -m tests.benchmarks.benchmark_staged_regions capture-close --run-dir "$RUN/bead-retry"',
              '.venv-sam/bin/python -m model_runtime.sam_adapter "$RUN"/bead-retry/close/captures/*/input.json --output-dir "$RUN/bead-retry/close/sam" --gpu 0 --threshold 0.25',
              '.venv/bin/python -m tests.benchmarks.benchmark_staged_regions measure-close --run-dir "$RUN/bead-retry"',
              '.venv/bin/python -m tests.benchmarks.benchmark_staged_regions report --run-dir "$RUN"', '```', '',
              '静态自由双目相机；没有运行机械臂路径、控制或焊接。仿真世界外参为已知值，所有误差直接在世界系计算，未使用真值对齐预测。', '']
    (run/'README.md').write_text('\n'.join(lines))
