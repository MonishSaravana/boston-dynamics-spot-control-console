from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.io import savemat

from scope.ipo import IpoSource


class IpoTests(unittest.TestCase):
    def test_explicit_calibration_and_missing_depth_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xyz = np.array([[-.2,.1,1.],[.3,-.2,1.5],[.2,.3,2.],[-.3,-.2,1.4]])
            uv = xyz[:,:2]/xyz[:,2,None]*[520.,522.]+[315.,258.]
            savemat(root/"objLocGroundTruth.mat",{"obj2DLoc":uv,"obj3DLoc":xyz})
            for i in (1,2):
                Image.fromarray(np.zeros((480,640,3),np.uint8)).save(root/f"{i:05d}.png")
            Image.fromarray(np.full((480,640),1000,np.uint16)).save(root/"00001d.png")
            source = IpoSource(root,root)
            batches = list(source)
            self.assertEqual(len(batches),1)
            self.assertAlmostEqual(source.K.fx,520.)
            self.assertIsNone(source.metadata["acquisition_timestamp_s"])
            self.assertEqual(len(source.metadata["input_failures"]),1)
            self.assertEqual(batches[0][0].frame.depth_m[0,0],1.)
