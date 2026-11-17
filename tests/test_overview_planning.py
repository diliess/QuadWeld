"""Whole-workpiece framing must account for both physical stereo eyes."""

import unittest
import numpy as np
from pxr import Usd

from weld_scene.overview_planning import framing, look_at, t_geometry


class OverviewFramingTests(unittest.TestCase):
    def test_look_at_uses_right_handed_opencv_axes(self):
        eye=np.array([2.,-3.,1.7]); target=np.array([0.,0.,.7])
        pose=look_at(eye,target)
        rotation=pose[:3,:3]
        np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation),1.)
        np.testing.assert_allclose(rotation[:,2],(target-eye)/np.linalg.norm(target-eye))
        self.assertLess(rotation[2,1],0.)

    def test_one_eye_crop_is_not_mistaken_for_full_stereo_framing(self):
        k=np.array([[100.,0,50],[0,100.,50],[0,0,1.]])
        # Midpoint projects to x=2, inside the image; right-eye baseline shift crops it.
        vertices=np.array([[-.48,0,1],[0,0,1]])
        full,_=framing(np.eye(4),vertices,k,[100,100],baseline=0,margin=0)
        stereo,_=framing(np.eye(4),vertices,k,[100,100],baseline=.09,margin=0)
        self.assertEqual(full,1.)
        self.assertEqual(stereo,.5)

    def test_three_stations_have_distinct_physical_heights(self):
        stage=Usd.Stage.Open('assets/scenes/industrial-welding/industrial_welding.usdc')
        for station,height in (('A15',.55),('A03',.49),('A21',.31)):
            with self.subTest(station=station):
                fixture,bounds,vertices,surfaces=t_geometry(stage,station)
                self.assertAlmostEqual(bounds['base'][1,2],height,places=6)
                self.assertAlmostEqual(bounds['web'][0,2],height,places=6)
                self.assertAlmostEqual(np.ptp(bounds['base'][:,0]),1.1,places=6)
                self.assertEqual(fixture.shape,(4,4))
                self.assertEqual(vertices.shape[1],3)
                self.assertGreater(len(surfaces),30)


if __name__=='__main__':unittest.main()
