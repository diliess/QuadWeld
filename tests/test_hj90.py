"""Functional optics checks: correspondence failures must never become filled depth."""

import unittest

import numpy as np

from weld_scene.optics import load_calibration, reconstruct, stripe_centres, triangulate, rasterize


class StereoTests(unittest.TestCase):
    def setUp(self):
        self.c = load_calibration()
        self.k = np.asarray(self.c['intrinsics'])

    def test_three_working_distances_and_invalid_disparity(self):
        for z in (.3, .5, .8):
            d = self.k[0,0]*self.c['baseline_m']/z
            xyz, valid = triangulate(np.full(1100,320+d/2), np.full(1100,320-d/2), self.c, 0.)
            self.assertTrue(valid.all())
            np.testing.assert_allclose(xyz[:,2],z)
            np.testing.assert_allclose(xyz[:,0],0,atol=1e-12)
        for l,r in ((320,320),(300,320),(np.nan,320),(330,320)):
            xyz,valid=triangulate(np.full(1100,l),np.full(1100,r),self.c)
            self.assertFalse(valid.any())
            self.assertTrue(np.isnan(xyz).all())

    def test_blue_on_off_subpixel_stereo_without_truth(self):
        w,h=self.c['image_size_wh']
        x=np.arange(w)+.5
        images=[]
        z=.6
        d=self.k[0,0]*self.c['baseline_m']/z
        off=np.full((h,w,3),40,np.uint8)
        for u in (320+d/2,320-d/2):
            on=off.copy()
            on[...,2]+=np.rint(120*np.exp(-.5*((x-u)/1.5)**2)).astype(np.uint8)
            images.append(on)
        xyz,uv,_=reconstruct(images[0],off,images[1],off,self.c,0.)
        self.assertEqual(len(xyz),h)
        np.testing.assert_allclose(xyz[:,2],z,atol=.0003)
        np.testing.assert_allclose(xyz[:,0],0,atol=.0001)
        no_laser,_,_=reconstruct(off,off,off,off,self.c,0.)
        self.assertEqual(len(no_laser),0)
        occluded=images[1].copy()
        occluded[400:600]=off[400:600]
        xyz,_,_=reconstruct(images[0],off,occluded,off,self.c,0.)
        self.assertEqual(len(xyz),h-200)

    def test_ambiguous_reflections_and_wrong_laser_plane_rejected(self):
        off=np.zeros((4,640,3),np.uint8)
        on=off.copy()
        on[:,300,2]=100
        on[:,350,2]=100
        u,_,_=stripe_centres(on,off,self.c['detector'])
        self.assertTrue(np.isnan(u).all())
        xyz,valid=triangulate(np.full(4,390.),np.full(4,275.),self.c,0.)
        self.assertFalse(valid.any())

    def test_rasterizer_keeps_occlusion_holes(self):
        depth=rasterize(np.array([[0,0,.6]]),self.c)
        self.assertEqual(np.isfinite(depth).sum(),1)
        self.assertAlmostEqual(float(np.nanmin(depth)),.6,places=6)


class USDOpticsTests(unittest.TestCase):
    def test_camera_baseline_and_projector_remain_on_torch(self):
        from pathlib import Path
        import tempfile
        from pxr import Gf, Usd, UsdGeom
        from weld_scene.usd import attach, laser_texture, SOURCE_CAMERA
        c=load_calibration()
        with tempfile.TemporaryDirectory() as directory:
            stage=Usd.Stage.CreateInMemory()
            parent=UsdGeom.Xform.Define(stage,'/World/Robot/e05l_welding_torch')
            parent.AddTranslateOp().Set(Gf.Vec3d(1,2,3))
            parent.AddRotateZOp().Set(32.)
            source=UsdGeom.Camera.Define(stage,SOURCE_CAMERA)
            source.AddTranslateOp().Set(Gf.Vec3d(.03,-.02,.1))
            cameras,laser,rig=attach(stage,c,laser_texture(Path(directory)/'stripe.png',c))
            cache=UsdGeom.XformCache()
            left=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(cameras['left'])).ExtractTranslation()
            right=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(cameras['right'])).ExtractTranslation()
            self.assertAlmostEqual((right-left).GetLength(),.09)
            self.assertEqual(str(rig.GetPath().GetParentPath()),str(parent.GetPath()))
            self.assertTrue(laser.GetPrim().GetAttribute('isProjector').Get())
            self.assertAlmostEqual(laser.GetPrim().GetAttribute('hj90:wavelengthNm').Get(),450.)

    def test_portable_full_scene_contains_sensor_and_only_local_assets(self):
        import hashlib,json
        from pathlib import Path
        from pxr import Usd,UsdGeom,UsdUtils
        bundle=Path(__file__).resolve().parents[1]/'assets/scenes/industrial-welding'
        manifest=json.loads((bundle/'manifest.json').read_text())
        stage=Usd.Stage.Open(str(bundle/'industrial_welding_full.usda'))
        for side,path in manifest['full_scene']['active_stereo_laser']['cameras'].items():
            self.assertTrue(stage.GetPrimAtPath(path).IsA(UsdGeom.Camera))
        layers,assets,missing=UsdUtils.ComputeAllDependencies(str(bundle/'industrial_welding_full.usda'))
        self.assertFalse(missing)
        for path in [*(layer.realPath for layer in layers),*assets]:
            self.assertTrue(Path(path).resolve().is_relative_to(bundle))
        for entry in manifest['files']:
            self.assertEqual(hashlib.sha256((bundle/entry['path']).read_bytes()).hexdigest(),entry['sha256'])


class AcquisitionTests(unittest.TestCase):
    def test_line_rate_and_full_frame_timing_limits(self):
        import contextlib,io
        from weld_scene.active_scan import parse_args
        for args in (['--lines','0'],['--lines','3000','--duration','.4'],['--duration','.01'],
                     ['--mode','dynamic','--angle-deg','90'],['--target-speed','1']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parse_args(args)
        args,c=parse_args(['--mode','dynamic','--lines','50','--duration','.01'])
        self.assertEqual(args.lines/args.duration,5000)
        self.assertEqual(args.output_dir.parent.name, 'validation')
        self.assertRegex(args.output_dir.name, r'^\d{8}_\d{6}_\d{6}__active_stereo__original__dynamic_scan$')


if __name__=='__main__':
    unittest.main()
