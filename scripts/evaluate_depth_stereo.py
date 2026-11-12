"""Compare saved Nested stereo depths with separately captured RTX axial depth."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def metrics(pred,true):
    import numpy as np
    if len(true)==0:
        return None
    error=np.asarray(pred,dtype=float)-true
    return {'pixels':len(true),'mae_m':float(np.mean(np.abs(error))),
            'rmse_m':float(np.sqrt(np.mean(error**2))),'bias_m':float(np.mean(error)),
            'p95_m':float(np.percentile(np.abs(error),95)),
            'abs_rel_percent':float(np.mean(np.abs(error)/true)*100),
            'truth_mean_m':float(np.mean(true)),'prediction_mean_m':float(np.mean(pred))}


def truth_at_processed_pixels(raw_depth,k_raw,k_processed,shape):
    """Sample each processed pixel's central optical ray using the actual K transform.

    Pixel centres use index+0.5 in the USD renderer. K_processed is returned by
    DA3's own resize/crop processor, not an independently guessed resize matrix.
    """
    import numpy as np
    yy,xx=np.indices(shape)
    rays=np.stack((xx+.5,yy+.5,np.ones(shape)),axis=-1) @ np.linalg.inv(k_processed).T
    raw=rays @ k_raw.T
    # Matrix inversion may put an exact pixel boundary a few ulps below an
    # integer. Keep the same containing-pixel convention at those boundaries.
    x=np.floor(raw[...,0]/raw[...,2]+1e-7).astype(int)
    y=np.floor(raw[...,1]/raw[...,2]+1e-7).astype(int)
    valid=(x>=0)&(x<raw_depth.shape[1])&(y>=0)&(y<raw_depth.shape[0])
    truth=np.full(shape,np.nan,dtype=np.float32)
    truth[valid]=raw_depth[y[valid],x[valid]]
    return truth


def evaluate(output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_erosion
    from pxr import Usd
    from benchmark_da3 import STATIONS,workpiece_faces
    from da3_stereo import load_pair
    output=Path(output).resolve()
    predictions=output/'predictions'
    result_dir=output/'evaluation'
    result_dir.mkdir(exist_ok=True)
    modes=['mono','stereo_uncalibrated','stereo_calibrated']
    if set(json.loads((predictions/'model.json').read_text())['modes'])!=set(modes):
        raise ValueError('Comparison requires all three inference modes')
    rows=[]
    pictures=[]
    for capture_input in sorted((output/'captures').glob('*/input.json')):
        case=capture_input.parent.name
        calibration,images,k_orig,_=load_pair(capture_input)
        capture=json.loads((capture_input.parent/'evaluation/capture.json').read_text())
        source=ROOT/'assets/scenes/industrial-welding/industrial_welding.usdc'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=capture['source_usdc_sha256']:
            raise ValueError('Workpiece scene changed since capture')
        stage=Usd.Stage.Open(capture['usd'])
        with np.load(capture_input.parent/'evaluation/truth.npz',allow_pickle=False) as gt:
            truth_original={side:gt[side+'_z_m'].copy() for side in ('left','right')}
        for i,side in enumerate(('left','right')):
            # A common evaluation grid uses calibrated processed K for every mode.
            calibrated_file=predictions/case/'stereo_calibrated'/(side+'_depth.npz')
            with np.load(calibrated_file,allow_pickle=False) as data:
                target_k=data['intrinsics'].astype(float)
                shape=data['depth_m'].shape
                rgb=data['processed_rgb'].copy()
            truth=truth_at_processed_pixels(truth_original[side],k_orig[i],target_k,shape)
            face_k=target_k.copy()
            face_k[:2,2]-=.5
            geom,labels=workpiece_faces(stage,{'station':capture['station'],
                'workpiece_meshes':STATIONS[capture['station']][1],
                'world_from_camera_optical':capture['world_from_camera_optical'][i]},face_k,shape)
            finite=np.isfinite(truth)&(truth>0)
            delta=np.full(shape,np.inf)
            np.subtract(geom,truth,out=delta,where=np.isfinite(geom)&finite)
            visible=finite&(labels>0)&(np.abs(delta)<.005)
            interior=np.zeros(shape,bool)
            for label in np.unique(labels[visible]):
                interior|=binary_erosion(visible&(labels==label),structure=np.ones((7,7)))
            if not interior.any():
                raise ValueError(f'No visible workpiece interior: {case}/{side}')
            rated=interior&(truth>=.3)&(truth<=.8)
            masks={'workpiece_interior':interior,'workpiece_rated_0p3_0p8':rated,'whole_scene_finite':finite}
            with open(result_dir/(case+'_'+side+'_mask.json'),'w') as f:
                json.dump({'interior_pixels':int(interior.sum()),'rated_pixels':int(rated.sum()),
                           'boundary_erosion_processed_pixels':3,'occlusion_truth_tolerance_m':.005},f,indent=2)
            arrays={'truth_z_m':truth,'workpiece_interior':interior,'workpiece_rated':rated,'intrinsics':target_k,
                    'face_labels':labels,'processed_rgb':rgb}
            for mode in modes:
                path=predictions/case/mode/(side if mode=='mono' else '')/(side+'_depth.npz')
                with np.load(path,allow_pickle=False) as stored:
                    pred=stored['depth_m'].copy()
                    if pred.shape!=shape:
                        raise ValueError('Prediction and common evaluation grid differ')
                    if not np.array_equal(stored['processed_rgb'],rgb):
                        raise ValueError('Inference modes used different processed image grids')
                    if str(stored['image_sha256'])!=calibration['views'][i]['image_sha256']:
                        raise ValueError('Prediction does not match captured RGB')
                    if not np.isfinite(pred).all() or not (pred>0).all():
                        raise ValueError('Invalid predicted depth; no selective valid-pixel filtering')
                arrays[mode+'_depth_m']=pred
                for region,mask in masks.items():
                    value=metrics(pred[mask],truth[mask])
                    if value is not None:
                        rows.append({'case':case,'view':side,'mode':mode,'region':region,**value})
            np.savez_compressed(result_dir/(case+'_'+side+'.npz'),**arrays)
            pictures.append((case,side,arrays))
    if not rows:
        raise ValueError('No saved stereo captures found')
    summary={'scope':'Nested axial depth in metres; same RGB and common truth-defined mask for all modes; no truth-derived scale/offset fit',
             'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'depth_alignment':'Known physical baseline / model-predicted baseline only; ground truth excluded from prediction',
             'rows':rows,'model':json.loads((predictions/'model.json').read_text()),
             'calibration_audit':{},'limitations':['Two static presets; not a multi-scene accuracy certification',
                'Ground truth sampled at processed pixel central rays; 3-pixel workpiece-boundary erosion',
                'Laser disabled; this tests RGB stereo camera conditioning, not laser-depth fusion',
                'No SAM3, weld localization or robot control executed']}
    for name in sorted({r['case'] for r in rows}):
        report=json.loads((predictions/name/'stereo_calibrated/report.json').read_text())
        if not report['camera_parameters_used_in_forward'] or not report['camera_encoder_inputs']:
            raise ValueError('Camera conditioning was not used')
        summary['calibration_audit'][name]=report['alignment']
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (output/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # One figure per view keeps numerical ranges comparable and preserves image aspect.
    for case,side,data in pictures:
        fig,axes=plt.subplots(1,6,figsize=(16,8),layout='constrained')
        axes[0].imshow(data['processed_rgb'])
        axes[0].set_title(case+' / '+side+' RGB')
        cmap=plt.colormaps['turbo'].copy()
        cmap.set_bad('#222222')
        lo,hi=.3,1.2
        axes[1].imshow(data['truth_z_m'],cmap=cmap,vmin=lo,vmax=hi)
        axes[1].set_title('RTX truth')
        for j,mode in enumerate(modes,2):
            im=axes[j].imshow(data[mode+'_depth_m'],cmap=cmap,vmin=lo,vmax=hi)
            val=metrics(data[mode+'_depth_m'][data['workpiece_interior']],data['truth_z_m'][data['workpiece_interior']])
            label={'mono':'Single RGB','stereo_uncalibrated':'Stereo RGB','stereo_calibrated':'Stereo + K,R,t'}[mode]
            axes[j].set_title(label+ ('\nMAE %.1f mm'%(val['mae_m']*1000) if val else '\nNo visible workpiece'))
        error=np.where(data['workpiece_interior'],np.abs(data['stereo_calibrated_depth_m']-data['truth_z_m']),np.nan)
        ei=axes[5].imshow(error,cmap='magma',vmin=0,vmax=.1)
        axes[5].set_title('Calibrated error\nworkpiece interior')
        for ax in axes:
            ax.axis('off')
        fig.colorbar(im,ax=axes[1:5],shrink=.45,label='Axial depth [m]; shared range .3–1.2')
        fig.colorbar(ei,ax=axes[5],shrink=.45,label='Absolute error [m]; display clipped at .1')
        fig.suptitle('Nested stereo evaluation: no ground-truth scale/offset correction',fontsize=14)
        fig.savefig(output/(case+'_'+side+'_comparison.png'),dpi=130)
        plt.close(fig)
    text=['# Nested 双目与标定深度核对','',summary['scope'],'',
          '| 工位 | 目 | 模式 | 工件像素 | MAE mm | RMSE mm | P95 mm | 偏差 mm |',
          '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['region']=='workpiece_interior':
            text.append('| %s | %s | %s | %d | %.2f | %.2f | %.2f | %.2f |'%(row['case'],row['view'],row['mode'],
              row['pixels'],row['mae_m']*1000,row['rmse_m']*1000,row['p95_m']*1000,row['bias_m']*1000))
    text+=['','## 数据范围','', '同一目、同一掩码对照单目、无标定双目和有标定双目。标定基线为 90 mm。',
           '使用模型预处理返回的内参匹配评价网格；不根据真值拟合尺度或偏移。',
           '完整指标（含 300–800 mm 工件子集与全画面）见 metrics.csv 和 summary.json。',
           '评价是这两个静态场景的光轴深度误差，不是焊缝三维定位误差或实机测量精度。',
           '本次激光关闭；A04 正常初始臂姿的工件大部分超过 800 mm。',
           '工件内部的误差不能推广到机器人自身、边缘和其他场景表面。',
           '', '## 标定双目全画面误差', '',
           '| 工位 | 目 | 有限深度像素 | MAE mm | RMSE mm |',
           '|---|---|---:|---:|---:|']
    for row in rows:
        if row['region']=='whole_scene_finite' and row['mode']=='stereo_calibrated':
            text.append('| %s | %s | %d | %.2f | %.2f |'%(row['case'],row['view'],row['pixels'],
                        row['mae_m']*1000,row['rmse_m']*1000))
    text+=['', '## 对照图', '']
    for case,side,_ in pictures:
        name=case+'_'+side+'_comparison.png'
        text.append(f'- [{case} / {side}]({name})')
    (output/'README.md').write_text('\n'.join(text)+'\n')
    for row in rows:
        if row['region']=='workpiece_interior':
            print(row['case'],row['view'],row['mode'],'pixels',row['pixels'],'MAE mm',round(row['mae_m']*1000,2))
    print('STEREO_EVALUATION_COMPLETE',output)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    a=p.parse_args()
    os.environ.setdefault('MPLCONFIGDIR',str(a.directory/'mpl_cache'))
    evaluate(a.directory)
