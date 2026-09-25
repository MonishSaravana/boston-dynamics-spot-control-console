"""Offline safety checks. No robot connection or GUI launch."""

import unittest
from types import SimpleNamespace

import numpy as np

from spot_gesture import GestureGate, Observation, torso_depth


class GestureGateTests(unittest.TestCase):
    def hold(self, gate, label, start, end, distance=None):
        result = None
        for tick in range(round(start * 10), round(end * 10) + 1):
            result = gate.observe(Observation(label, distance), tick / 10)
        return result

    def test_neutral_required_and_approach_stops_at_margin(self):
        gate = GestureGate()
        self.assertEqual(self.hold(gate, '1', 0, .7, 3.0)[0], 0)
        self.hold(gate, 'neutral', .8, 1.4)
        self.assertEqual(self.hold(gate, '1', 1.5, 2.2, 3.0)[0], 1)
        self.assertEqual(gate.observe(Observation('1', 2.5), 2.3)[0], 0)
        self.assertEqual(gate.observe(Observation('1', 3.0), 2.4)[0], 0)

    def test_backup_capped_and_rearms(self):
        gate = GestureGate()
        self.hold(gate, 'neutral', 0, .6)
        self.assertEqual(self.hold(gate, '2', .7, 1.4)[0], -1)
        self.assertLessEqual(gate.backup_until - 1.4, .5 + 1e-9)
        self.assertEqual(gate.observe(Observation('2'), 1.9)[0], 0)
        self.assertEqual(self.hold(gate, '2', 2.0, 2.7)[0], 0)

    def test_unclear_and_gap_stop(self):
        gate = GestureGate()
        self.hold(gate, 'neutral', 0, .6)
        self.hold(gate, '1', .7, 1.4, 3.0)
        self.assertEqual(gate.observe(Observation('unclear'), 1.5)[0], 0)
        self.assertEqual(gate.observe(Observation('1', 3.0), 1.6)[0], 0)
        self.hold(gate, 'neutral', 1.7, 2.3)
        self.hold(gate, '1', 2.4, 3.1, 3.0)
        self.assertEqual(gate.observe(Observation('1', 3.0), 3.5)[0], 0)


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.full((100, 100), 3000, dtype=np.uint16)
        self.mask = np.ones((100, 100), dtype=np.float32)
        self.landmarks = [SimpleNamespace(x=.5, y=.5, visibility=.95, presence=.95)
                          for _ in range(33)]
        for i, x, y in [(11, .4, .3), (12, .6, .3), (23, .42, .7), (24, .58, .7)]:
            self.landmarks[i] = SimpleNamespace(x=x, y=y, visibility=.95, presence=.95)

    def test_reliable_torso_depth(self):
        self.assertAlmostEqual(torso_depth(self.depth, 1000, self.landmarks, self.mask), 3.0)

    def test_missing_depth_or_uncertain_person_rejected(self):
        self.assertIsNone(torso_depth(np.zeros_like(self.depth), 1000, self.landmarks, self.mask))
        self.assertIsNone(torso_depth(self.depth, 0, self.landmarks, self.mask))
        self.landmarks[11].visibility = .2
        self.assertIsNone(torso_depth(self.depth, 1000, self.landmarks, self.mask))

    def test_background_depth_is_not_used(self):
        self.assertIsNone(torso_depth(self.depth, 1000, self.landmarks,
                                      np.zeros_like(self.mask)))

    def test_nearest_person_pixel_controls_stop_distance(self):
        self.depth[15, 50] = 2200  # hand or other person pixel outside torso patch
        self.assertAlmostEqual(torso_depth(self.depth, 1000, self.landmarks,
                                           self.mask), 2.2)


if __name__ == '__main__':
    unittest.main()
