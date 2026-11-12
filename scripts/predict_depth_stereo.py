"""Run Nested DA3 on synchronized stereo RGB with camera parameters at inference."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from da3_stereo import load_pair, install_two_view_alignment


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def infer(model, images, intrinsics, extrinsics, mode, args, output, view_records):
    import numpy as np
    import torch
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    model.stereo_alignment_record = None
    model.stereo_raw = None
    camera_encoder_inputs = []
    def audit(module, inputs):
        camera_encoder_inputs.append({'normalized_extrinsics_w2c':inputs[0].detach().cpu().numpy().tolist(),
                                     'processed_intrinsics':inputs[1].detach().cpu().numpy().tolist(),
                                     'processed_hw':list(inputs[2])})
    hook = model.model.da3.cam_enc.register_forward_pre_hook(audit)
    started=time.monotonic()
    try:
        kwargs={'process_res':args.process_res,'process_res_method':'upper_bound_resize'}
        if mode=='stereo_calibrated':
            kwargs.update(intrinsics=intrinsics.copy(),extrinsics=extrinsics.copy(),align_to_input_ext_scale=True)
        prediction=model.inference(images,**kwargs)
    finally:
        hook.remove()
    torch.cuda.synchronize()
    elapsed=time.monotonic()-started
    expected_conditioned=mode=='stereo_calibrated'
    if bool(camera_encoder_inputs)!=expected_conditioned:
        raise RuntimeError('Camera conditioning audit failed')
    if not bool(prediction.is_metric):
        raise RuntimeError('Nested did not produce metric output')
    output.mkdir(parents=True,exist_ok=True)
    report={'mode':mode,'images':images,'image_sha256':[digest(p) for p in images],
            'seconds':elapsed,'peak_gpu_memory_mib':torch.cuda.max_memory_allocated()/2**20,
            'camera_parameters_used_in_forward':expected_conditioned,
            'camera_encoder_inputs':camera_encoder_inputs,'alignment':model.stereo_alignment_record,
            'nested_metric_scale_factor':prediction.scale_factor,'views':[],
            'truth_usage':'No renderer depth, target distance or ground-truth scale passed to inference'}
    if model.stereo_raw is not None:
        np.savez_compressed(output/'before_baseline_alignment.npz',**model.stereo_raw)
    for i,view in enumerate(view_records):
        depth=np.asarray(prediction.depth[i],dtype=np.float32)
        rgb=np.asarray(prediction.processed_images[i],dtype=np.uint8)
        if depth.shape!=rgb.shape[:2] or not np.isfinite(depth).all() or not (depth>0).all():
            raise RuntimeError('Invalid predicted depth')
        name=view['id']
        arrays={'depth_m':depth,'processed_rgb':rgb,'intrinsics':np.asarray(prediction.intrinsics[i]),
                'extrinsics_w2c':np.asarray(prediction.extrinsics[i]),'is_metric':np.array(True),
                'original_hw':np.array(view['image_size_wh'][::-1]),'image_sha256':np.array(digest(images[i]))}
        if prediction.conf is not None:
            arrays['confidence']=np.asarray(prediction.conf[i])
        if prediction.sky is not None:
            arrays['sky']=np.asarray(prediction.sky[i])
        np.savez_compressed(output/(name+'_depth.npz'),**arrays)
        Image.fromarray(rgb).save(output/(name+'_rgb.png'))
        lo,hi=np.percentile(depth,[2,98])
        plt.imsave(output/(name+'_depth.png'),depth,cmap='turbo',vmin=lo,vmax=max(hi,lo+1e-6))
        report['views'].append({'id':name,'processed_hw':list(depth.shape),
                                'depth_min_m':float(depth.min()),'depth_median_m':float(np.median(depth)),
                                'depth_max_m':float(depth.max()),'depth':name+'_depth.npz'})
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NESTED_STEREO_INFERRED',output,'seconds',round(elapsed,2),'alignment',report['alignment'],flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stereo_input',type=Path,nargs='+',help='Capture input.json (RGB and calibration only)')
    p.add_argument('--model-dir',type=Path,default=ROOT/'thirdparty/depth-anything-3/weights/DA3NESTED-GIANT-LARGE-1.1')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--process-res',type=int,default=504)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--stream-blocks',action='store_true',help='CPU offload transformer blocks without dtype/weight changes')
    p.add_argument('--modes',nargs='+',choices=('mono','stereo_uncalibrated','stereo_calibrated'),
                   default=['mono','stereo_uncalibrated','stereo_calibrated'])
    a=p.parse_args()
    if a.process_res<28 or a.gpu<0:
        p.error('Invalid GPU or processing resolution')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Use a new empty output directory to avoid mixing different runs')
    inputs=[load_pair(path) for path in a.stereo_input]
    if len({path.parent.name for path in a.stereo_input})!=len(a.stereo_input):
        p.error('Each input pair needs a unique parent directory name')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ.setdefault('MPLCONFIGDIR',str(a.output_dir/'mpl_cache'))
    import torch
    from predict_depth import load_model
    a.output_dir.mkdir(parents=True,exist_ok=True)
    model,info=load_model(a.model_dir,stream_blocks=a.stream_blocks)
    if not hasattr(model.model,'da3_metric'):
        raise ValueError('This entry point requires the Nested model')
    install_two_view_alignment(model)
    metadata={'model_dir':str(a.model_dir.resolve()),'checkpoint_sha256':digest(a.model_dir/'model.safetensors'),
              'config_sha256':digest(a.model_dir/'config.json'),
              'source_commit':subprocess.check_output(['git','-C',str(ROOT/'thirdparty/depth-anything-3'),'rev-parse','HEAD'],text=True).strip(),
              'adapter_sha256':digest(ROOT/'da3_stereo.py'),'predictor_sha256':digest(Path(__file__)),
              'loader_sha256':digest(ROOT/'scripts/predict_depth.py'),
              'depth_anything_3':importlib.metadata.version('depth-anything-3'),
              'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
              'weight_dtype':str(next(model.parameters()).dtype),'stream_blocks':a.stream_blocks,
              'process_res':a.process_res,'random_seed_per_inference':0,'modes':a.modes,**info}
    (a.output_dir/'model.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for input_path,(data,images,k,e) in zip(a.stereo_input,inputs):
        case=a.output_dir/input_path.parent.name
        for mode in a.modes:
            if mode=='mono':
                for i in range(2):
                    infer(model,[images[i]],None,None,mode,a,case/mode/data['views'][i]['id'],[data['views'][i]])
            else:
                infer(model,images,k,e,mode,a,case/mode,data['views'])
    (a.output_dir/'completed.json').write_text(json.dumps({'completed':True,'pairs':len(inputs),'modes':a.modes},indent=2)+'\n')


if __name__=='__main__':
    main()
