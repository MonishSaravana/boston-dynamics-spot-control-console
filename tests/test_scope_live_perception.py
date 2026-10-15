"""Live Spot perception fixes: OWLv2 box scale, upright frames, pose latency."""
from types import SimpleNamespace
import time
import unittest

import numpy as np

from scope.console_backend import SpotBackend, camera_box, _project_world
from scope.data import Intrinsics, RgbdFrame
from scope.human_pipeline import MAX_POSE_LATENCY_ALLOWANCE_S, HumanPipeline
from scope.human_synthetic import HumanScene
from scope.humans import lift_pose
from scope.live_interaction import SelectionSource, TargetCandidate
from scope.person_tracking import PersonTracker, fuse_views
from scope.pointing import estimate_pointing
from scope.runtime import ModuleConfig
from scope.spot_live_source import (
    CameraPolicy, SpotImage, pair_to_frame, rotate_frame, upright_quarter_turns)


def rigid(seed=3):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = q, rng.normal(size=3)
    return T


def world_grid(frame):
    K = frame.intrinsics
    v, u = np.mgrid[:K.height, :K.width]
    z = frame.depth_m
    optical = np.stack(((u-K.cx)/K.fx*z, (v-K.cy)/K.fy*z, z), axis=-1)
    T = frame.T_world_camera
    return optical @ T[:3, :3].T + T[:3, 3]


class OwlBoxScaleTests(unittest.TestCase):
    def test_non_square_frame_boxes_use_padded_square(self):
        import torch
        from transformers.models.owlv2.image_processing_owlv2 import Owlv2ImageProcessor
        from scope.open_vocab import OpenVocabularyDetector
        from scope.interaction import TextCommand
        from scope.semantic_query import SemanticQuery

        h, w = 120, 240  # OWLv2 pads bottom/right to 240x240
        # Normalized to the padded square: pixel box (60, 30, 120, 90), and one
        # box whose lower half falls in the padding below the frame.
        boxes = torch.tensor([[[90/w, 60/w, 60/w, 60/w],
                               [180/w, 120/w, 40/w, 80/w]]])
        outputs = SimpleNamespace(logits=torch.full((1, 2, 1), 5.), pred_boxes=boxes)
        image_processor = Owlv2ImageProcessor()

        class Processor:
            def __call__(self, **kw):
                return SimpleNamespace(to=lambda device: {})
            def post_process_object_detection(self, *args, **kw):
                return image_processor.post_process_object_detection(*args, **kw)

        detector = object.__new__(OpenVocabularyDetector)
        detector.torch, detector.backend, detector.device = torch, "owlv2", "cpu"
        detector.size, detector.threshold, detector.low_threshold = 512, .4, .15
        detector.processor, detector.model = Processor(), lambda **kw: outputs
        detector.metadata, detector.name = {}, "owlv2-test"
        frame = RgbdFrame(np.zeros((h, w, 3), np.uint8), np.full((h, w), np.nan, np.float32),
                          Intrinsics(w, h, 1., 1., (w-1)/2, (h-1)/2), np.eye(4), 1., 1.,
                          "TEST", "owl")
        result = detector.query(frame, SemanticQuery.from_command(TextCommand("chair")))
        found = sorted(c.box_xyxy for c in result.candidates)
        np.testing.assert_allclose(found[0], (60., 30., 120., 90.), atol=1e-3)
        np.testing.assert_allclose(found[1], (160., 80., 200., 120.), atol=1e-3)


class UprightFrameTests(unittest.TestCase):
    def frame(self):
        K = Intrinsics(8, 6, 5., 7., 3.2, 2.1)
        rng = np.random.default_rng(1)
        rgb = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)
        depth = rng.uniform(.5, 3., (6, 8)).astype(np.float32)
        return RgbdFrame(rgb, depth, K, rigid(), 1., 1., "SPOT_LIVE", "f")

    def test_quarter_turns_match_preview_rotation(self):
        self.assertEqual(upright_quarter_turns("frontleft_fisheye_image"), 3)
        self.assertEqual(upright_quarter_turns("frontright_fisheye_image"), 3)
        self.assertEqual(upright_quarter_turns("right_fisheye_image"), 2)
        self.assertEqual(upright_quarter_turns("left_fisheye_image"), 0)

    def test_rotation_keeps_pixels_depth_and_world_geometry_together(self):
        frame = self.frame()
        original = world_grid(frame)
        for k in range(4):
            turned = rotate_frame(frame, k)
            turned.validate()
            np.testing.assert_array_equal(turned.rgb, np.rot90(frame.rgb, k))
            np.testing.assert_allclose(world_grid(turned), np.rot90(original, k), atol=1e-9)
            # Every world point projects back onto the same rotated pixel.
            rows, cols = np.mgrid[:turned.intrinsics.height, :turned.intrinsics.width]
            for i, j in zip(rows.ravel()[::5], cols.ravel()[::5]):
                u, v = _project_world(turned, world_grid(turned)[i, j])
                self.assertAlmostEqual(u, j, places=6)
                self.assertAlmostEqual(v, i, places=6)

    def test_front_pair_reaches_detectors_upright_with_aligned_overlay(self):
        K = Intrinsics(64, 48, 50., 50., 31.5, 23.5)
        pixels = np.zeros((48, 64, 1), np.uint8)
        pixels[5, 40] = 255
        T = rigid(7)
        visual = SpotImage("frontleft_fisheye_image", pixels, 100., 10., 123., "camera",
                           T, "odom", K, None, 12., 2.)
        depth = SpotImage("frontleft_depth_in_visual_frame", np.full((48, 64, 1), 2000, np.uint16),
                          100., 10., 123., "camera", T, "odom", K, 1000., 12., 2.)
        frame = pair_to_frame(visual, depth, alignment_verified=True)
        self.assertEqual(frame.rgb.shape, (64, 48, 3))
        # Clockwise quarter turn: raw (row 5, col 40) moves to (row 40, col 42).
        self.assertEqual(tuple(np.argwhere(frame.rgb[:, :, 0] == 255)[0]), (40, 42))
        # The world point behind the raw pixel draws on the rotated pixel.
        ray = np.array([(40-K.cx)/K.fx*2., (5-K.cy)/K.fy*2., 2.])
        point = T[:3, :3] @ ray + T[:3, 3]
        u, v = _project_world(frame, point)
        self.assertAlmostEqual(u, 42., places=5)
        self.assertAlmostEqual(v, 40., places=5)
        target = TargetCandidate("chair", "chair", SelectionSource.QUERY, 1., 1., 100.,
                                 tuple(point-.01), tuple(point+.01), "odom")
        box = camera_box(target, frame)
        self.assertTrue(box[0] <= 42 <= box[2] and box[1] <= 40 <= box[3])

    def test_rgb_only_tick_sends_upright_frame_to_pipeline(self):
        K = Intrinsics(64, 48, 50., 50., 31.5, 23.5)
        image = SpotImage("frontleft_fisheye_image", np.zeros((48, 64, 1), np.uint8),
                          100., time.monotonic(), 123., "camera", None, "odom", K, None, 1., 1.)

        class Source:
            sources = {"frontleft_fisheye_image": None}
            policy = {"frontleft_fisheye_image": CameraPolicy()}
            latest = {"frontleft_fisheye_image": image}
            def configure(self, *a, **kw): pass
            def poll(self): pass
            def robot_pose(self): raise ValueError("no robot in test")
            def mark_processed(self, *a): pass

        class Pipeline:
            frames = []
            def tick(self, frame, now_s):
                self.frames.append(frame)
                return {"entities": []}
            def close(self): pass

        backend = SpotBackend(None, "frontleft_fisheye_image", source=Source())
        pipeline = Pipeline()
        backend._init_pipeline = lambda frame, geometry: setattr(backend, "pipeline", pipeline)
        try:
            deadline = time.monotonic()+2
            while not pipeline.frames and time.monotonic() < deadline:
                backend.tick()
                time.sleep(.01)
        finally:
            backend.close()
        self.assertEqual(pipeline.frames[0].rgb.shape, (64, 48, 3))
        self.assertEqual((pipeline.frames[0].intrinsics.width,
                          pipeline.frames[0].intrinsics.height), (48, 64))


class PoseLatencyTests(unittest.TestCase):
    def test_measured_latency_admits_older_joints(self):
        scene = HumanScene(frames=1, width=192)
        samples = scene.tick(0)
        poses = [lift_pose(o, s.frame) for s in samples for o in s.observations]
        track = PersonTracker().update(fuse_views(poses)[0], .45)[0]
        # Keypoint R-CNN on CPU: joints from t=0 reach pointing at t=0.45.
        self.assertEqual(estimate_pointing(track, .45).state, "ABSTAIN")
        self.assertNotEqual(estimate_pointing(track, .45, latency_allowance_s=.45).state,
                            "ABSTAIN")

    def test_realtime_pose_result_is_consumed_between_due_ticks(self):
        p = HumanPipeline(configs={"human_pose": ModuleConfig(max_hz=1)}, realtime=True)
        scene = HumanScene(width=96)
        try:
            p.step(scene.tick(0), 0.)
            deadline = time.monotonic()+2
            workers = [w for name, w in p.workers.items() if name.startswith("pose/")]
            while time.monotonic() < deadline and not all(w.completed for w in workers):
                time.sleep(.01)
            snapshot = p.step(scene.tick(3), .3)  # human_pose is not due at 1 Hz
        finally:
            p.close()
        self.assertTrue(snapshot["tracks"])
        self.assertAlmostEqual(snapshot["pointing_latency_allowance_s"], .3)
        self.assertTrue(any(h.state != "ABSTAIN" for h in snapshot["pointing"]))

    def test_latency_allowance_is_capped(self):
        p = HumanPipeline()
        p.pose_latencies_s.extend([2.]*5)
        self.assertEqual(p.pose_latency_allowance_s(), MAX_POSE_LATENCY_ALLOWANCE_S)
        p.close()


if __name__ == "__main__":
    unittest.main()
