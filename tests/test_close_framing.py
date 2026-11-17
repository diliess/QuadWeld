"""Near views must cover the whole seam together while staying in range."""

import unittest
import numpy as np

from weld_scene.optics import load_calibration
from weld_scene.overview_planning import look_at,transform_points


class CloseFramingTests(unittest.TestCase):
    def test_overlapping_650mm_stereo_patches_cover_full_seam_in_rated_range(self):
        calibration=load_calibration();k=np.asarray(calibration['intrinsics'])
        seam=np.c_[np.linspace(-.42,.42,841),np.zeros(841),np.zeros(841)]
        seen=np.zeros(841,bool);angle=np.radians(35)
        for centre in (-.3,0.,.3):
            target=np.array([centre,0.,0.])
            pose=look_at(target+.65*np.array([0.,-np.cos(angle),np.sin(angle)]),target)
            for sign in (-1,1):
                eye=pose.copy();eye[:3,3]+=sign*.045*pose[:3,0]
                xyz=transform_points(np.linalg.inv(eye),seam)
                p=xyz@k.T;xy=p[:,:2]/p[:,2:]
                mask=(xy[:,0]>=0)&(xy[:,0]<640)&(xy[:,1]>=0)&(xy[:,1]<1100)&(xyz[:,2]>0)
                self.assertLess(mask.sum(),len(seam))
                np.testing.assert_allclose(xyz[mask,2],.65,atol=1e-12)
                distances=np.linalg.norm(xyz[mask],axis=1)
                self.assertGreaterEqual(distances.min(),.3)
                self.assertLessEqual(distances.max(),.8)
                seen|=mask
        self.assertTrue(seen.all())


if __name__=='__main__':unittest.main()
