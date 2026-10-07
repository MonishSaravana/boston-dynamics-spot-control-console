"""Synthetic semantic masks and known-pose object projection."""

from dataclasses import replace
import unittest

import numpy as np

from scope.evaluate import surface_distance
from scope.objects import project_object
from scope.synthetic import SyntheticRoom
from scope.truth import SyntheticTruthDetector, semantic_identity


class ObjectProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.room = SyntheticRoom(frames=10)
        cls.frames = list(cls.room)
        cls.detector = SyntheticTruthDetector(cls.room.boxes)

    def test_truth_masks_and_world_projection(self):
        frame = self.frames[4]
        detections = self.detector.detect(frame)
        self.assertEqual({d.truth_id for d in detections},
                         {"chair_a", "chair_b", "table", "backpack"})
        used = np.zeros(frame.depth_m.shape, dtype=bool)
        for detection in detections:
            self.assertFalse(np.any(used & detection.mask))
            used |= detection.mask
            projected = project_object(frame, detection)
            self.assertIsNotNone(projected)
            self.assertGreater(projected.support_pixels, 50)
            self.assertGreater(projected.valid_depth_fraction, .99)
            parts = tuple(part for part in self.room.boxes
                          if (identity := semantic_identity(part.name)) is not None
                          and identity[0] == detection.truth_id)
            self.assertLess(float(np.percentile(
                surface_distance(projected.points_world, parts), 99)), .001)

    def test_depth_holes_and_outliers_are_bounded(self):
        frame = self.frames[4]
        chair = next(d for d in self.detector.detect(frame) if d.truth_id == "chair_a")
        original = project_object(frame, chair)
        altered = frame.depth_m.copy()
        yy, xx = np.nonzero(chair.mask)
        altered[yy[::5], xx[::5]] = 0
        altered[yy[1::7], xx[1::7]] = 4.9
        result = project_object(replace(frame, depth_m=altered), chair)
        self.assertIsNotNone(result)
        self.assertLess(result.valid_depth_fraction, original.valid_depth_fraction)
        self.assertLess(float(np.linalg.norm(result.center - original.center)), .12)
        self.assertGreater(float(np.trace(result.covariance)),
                           float(np.trace(original.covariance)))


if __name__ == "__main__":
    unittest.main()
