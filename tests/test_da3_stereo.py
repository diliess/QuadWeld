"""Stereo scale recovery and input integrity without a checkpoint or scene truth."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image

from da3_stereo import baseline_scale,camera_centres,install_two_view_alignment,load_pair


def pair(baseline):
    e=np.repeat(np.eye(4)[None],2,axis=0)
    e[1,0,3]=-baseline
    return e


class StereoGeometryTests(unittest.TestCase):
    def test_baseline_recovers_scale_including_rotated_reference(self):
        known=pair(.09)
        predicted=pair(.3)
        scale,record=baseline_scale(predicted,known)
        self.assertAlmostEqual(scale,.3)
        # Camera centres must be computed with -R.T @ t, not translation norm difference.
        angle=.7
        rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        world=np.eye(4)
        world[:3,:3]=rotation
        world[:3,3]=[.2,.1,1]
        transformed=known @ world
        self.assertAlmostEqual(baseline_scale(transformed,known)[0],1.)
        self.assertIn('no truth',record['method'])
        with self.assertRaisesRegex(ValueError,'Collapsed'):
            baseline_scale(pair(0),known)

    def test_instance_adapter_preserves_raw_output_and_applies_baseline_only(self):
        class CpuTensor:
            def __init__(self,value):self.value=value
            def detach(self):return self
            def cpu(self):return self
            def numpy(self):return self.value
            def __len__(self):return len(self.value)
        def fallback(*args):return 'fallback'
        model=SimpleNamespace(_align_to_input_extrinsics_intrinsics=fallback)
        install_two_view_alignment(model)
        raw=np.ones((2,4,5),np.float32)*2
        k=np.repeat(np.eye(3)[None],2,axis=0)
        p=SimpleNamespace(depth=raw.copy(),extrinsics=pair(.18)[:,:3].copy(),intrinsics=k.copy(),is_metric=0)
        result=model._align_to_input_extrinsics_intrinsics(CpuTensor(pair(.09)),CpuTensor(k),p)
        np.testing.assert_array_equal(result.depth,np.ones_like(raw))
        np.testing.assert_array_equal(model.stereo_raw['depth_before_baseline_m'],raw)
        np.testing.assert_array_equal(result.extrinsics,pair(.09)[:,:3])
        self.assertTrue(result.is_metric)
        self.assertEqual(model._align_to_input_extrinsics_intrinsics(None,None,p),'fallback')

    def test_synchronized_pair_hashes_dimensions_and_distortion_are_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            views=[]
            for i,side in enumerate(('left','right')):
                file=folder/(side+'.png')
                Image.fromarray(np.full((12,10,3),i*20,np.uint8)).save(file)
                views.append({'id':side,'image':file.name,'image_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                              'image_size_wh':[10,12],'intrinsics':[[20,0,5],[0,20,6],[0,0,1]],
                              'extrinsics_w2c':pair(.09)[i].tolist(),'distortion_coefficients':[0,0,0,0,0]})
            data={'schema':'quadweld-stereo-v1','synchronized':True,'length_unit':'m','baseline_m':.09,'views':views}
            path=folder/'input.json'
            def write():path.write_text(json.dumps(data))
            write()
            _,images,k,e=load_pair(path)
            self.assertEqual(len(images),2)
            self.assertEqual(k.shape,(2,3,3))
            data['length_unit']='mm'
            write()
            with self.assertRaisesRegex(ValueError,'metres'):load_pair(path)
            data['length_unit']='m'
            data['views'].reverse()
            write()
            with self.assertRaisesRegex(ValueError,'ordered'):load_pair(path)
            data['views'].reverse()
            data['views'][1]['distortion_coefficients'][0]=.1
            write()
            with self.assertRaisesRegex(ValueError,'Undistort'):load_pair(path)
            data['views'][1]['distortion_coefficients'][0]=0
            data['synchronized']=False
            write()
            with self.assertRaisesRegex(ValueError,'synchronized'):load_pair(path)
            data['synchronized']=True
            data['baseline_m']=.1
            write()
            with self.assertRaisesRegex(ValueError,'baseline'):load_pair(path)
            data['baseline_m']=.09
            write()
            Image.fromarray(np.zeros((12,10,3),np.uint8)).save(folder/'right.png')
            with self.assertRaisesRegex(ValueError,'hash'):load_pair(path)

    def test_evaluation_samples_central_ray_under_actual_intrinsic_resize(self):
        from scripts.evaluate_depth_stereo import truth_at_processed_pixels
        raw=np.arange(8*6,dtype=np.float32).reshape(8,6)+1
        k=np.array([[10,0,3],[0,10,4],[0,0,1.]])
        small=k.copy()
        small[:2]*=.5
        sampled=truth_at_processed_pixels(raw,k,small,(4,3))
        np.testing.assert_array_equal(sampled,raw[1::2,1::2])


if __name__=='__main__':
    unittest.main()
