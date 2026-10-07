from dataclasses import replace
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np

from scope.human_pipeline import HumanPipeline
from scope.human_synthetic import HumanScene
from scope.humans import JointState,lift_pose
from scope.person_tracking import PersonTracker,fuse_views
from scope.pointing import estimate_pointing,score_targets
from scope.runtime import ModuleConfig


def track_for(scenario="complete",cameras=("front","left")):
    scene = HumanScene(frames=1,width=192,scenario=scenario,cameras=cameras)
    samples = scene.tick(0)
    poses = [lift_pose(o,s.frame) for s in samples for o in s.observations]
    fused,unresolved = fuse_views(poses)
    tracks = PersonTracker().update(fused,0.)
    return samples,tracks,unresolved


class HumanTests(unittest.TestCase):
    def test_lifting_observed_coordinates_and_missing_depth(self):
        samples,tracks,_ = track_for()
        truth = samples[0].truth[0]["keypoints_world"]
        for n,j in tracks[0].pose.joints.items():
            if j.position is not None:
                self.assertLess(np.linalg.norm(j.position-truth[n]),.025)
                self.assertTrue(np.all(np.linalg.eigvalsh(j.covariance)>0))
        sample = samples[0]
        frame = replace(sample.frame,depth_m=np.zeros_like(sample.frame.depth_m))
        pose = lift_pose(sample.observations[0],frame)
        self.assertTrue(all(j.state==JointState.UNAVAILABLE for j in pose.joints.values()))
        self.assertTrue(all(j.position is None for j in pose.joints.values()))

    def test_complementary_views_recover_wrist_and_reduce_uncertainty(self):
        _,front,_ = track_for("complementary_views",("front",))
        self.assertEqual(estimate_pointing(front[0],0.).state,"ABSTAIN")
        _,fused,_ = track_for("complementary_views")
        self.assertNotEqual(estimate_pointing(fused[0],0.).state,"ABSTAIN")
        self.assertEqual(len(fused[0].pose.source_cameras),2)
        _,single,_ = track_for(cameras=("front",))
        _,complete,_ = track_for()
        self.assertGreater(estimate_pointing(single[0],0.).angular_uncertainty_rad,
                           estimate_pointing(complete[0],0.).angular_uncertainty_rad)

    def test_occlusion_noisy_depth_and_resting_arm(self):
        _,complete,_ = track_for()
        baseline = estimate_pointing(complete[0],0.)
        for case in ("wrist_missing","hand_outside","not_pointing"):
            _,tracks,_ = track_for(case)
            self.assertTrue(all(estimate_pointing(t,0.).state=="ABSTAIN" for t in tracks),case)
        _,elbow,_ = track_for("elbow_missing")
        broad = estimate_pointing(elbow[0],0.)
        self.assertEqual(broad.state,"DEGRADED")
        self.assertGreater(broad.angular_uncertainty_rad,baseline.angular_uncertainty_rad*2)
        for case in ("lower_body_outside","partial_frame","noisy_depth"):
            _,tracks,_ = track_for(case)
            self.assertGreater(len(tracks),0)
            for t in tracks:
                for j in t.pose.joints.values():
                    if j.position is not None:
                        self.assertTrue(np.isfinite(j.position).all())

    def test_two_people_overlap_and_unsynchronized_views(self):
        _,tracks,_ = track_for("two_people")
        self.assertEqual(len(tracks),2)
        _,tracks,unresolved = track_for("overlapping_people")
        self.assertGreater(len(unresolved),0)
        samples,_,_ = track_for()
        poses = [lift_pose(o,s.frame) for s in samples for o in s.observations]
        poses[1] = replace(poses[1],timestamp_s=1.)
        fused,_ = fuse_views(poses)
        self.assertEqual(len(fused),2)
        pipeline = HumanPipeline()
        result = pipeline.step(HumanScene(width=128,scenario="overlapping_people").tick(0),0.)
        self.assertFalse(result["pointing"])

    def test_temporal_id_hold_and_expiration(self):
        tracker = PersonTracker()
        scene = HumanScene(frames=12,width=128,scenario="brief_exit")
        ids = []
        for i,samples in enumerate(scene):
            fused,_ = fuse_views([lift_pose(o,s.frame) for s in samples for o in s.observations])
            tracks = tracker.update(fused,i/10)
            if tracks:
                ids.append(tracks[0].person_id)
            if 5<=i<9:
                self.assertTrue(all(estimate_pointing(t,i/10).state=="ABSTAIN" for t in tracks))
        self.assertEqual(len(set(ids)),1)
        tracker.expire(5.)
        self.assertFalse(tracker.tracks)

    def test_core_isolation_toggles_targets_and_memory(self):
        scene = HumanScene(frames=4,width=128)
        pipeline = HumanPipeline()
        first = pipeline.step(scene.tick(0),0.)
        scores = first["candidates"]["person_001"]["scores"]
        self.assertEqual(max(scores,key=scores.get),"chair_02")
        self.assertAlmostEqual(sum(scores.values()),1.)
        revision = pipeline.mapping.revision
        self.assertEqual(revision,2)
        pipeline.runtime.configure("human_pose",ModuleConfig(enabled=False))
        pipeline.runtime.configure("map_visualization",ModuleConfig(enabled=False))
        disabled = pipeline.step(scene.tick(1),.1)
        self.assertFalse(disabled["pointing"])
        self.assertGreater(pipeline.mapping.revision,revision)
        self.assertEqual(len(pipeline.entities.entities),4)
        pipeline.runtime.configure("human_pose",ModuleConfig())
        resumed = pipeline.step(scene.tick(2),.2)
        self.assertTrue(resumed["pointing"])
        pipeline.runtime.configure("camera/left",ModuleConfig(enabled=False))
        lost = pipeline.step(scene.tick(3),.3)
        self.assertEqual(lost["telemetry"]["human_pose"]["status"],"DEGRADED")
        self.assertEqual(lost["pointing"][0].state,"DEGRADED")
        with tempfile.TemporaryDirectory() as tmp:
            beliefs = pipeline.save_memory(Path(tmp))
            self.assertEqual(len(beliefs),4)
        pipeline.close()

    def test_pose_failure_does_not_stop_map_entities_memory(self):
        class Broken:
            name = "broken-test"
            def detect(self,*args):
                raise RuntimeError("pose unavailable")
        p = HumanPipeline(pose_detector=Broken())
        with self.assertLogs("scope.runtime",level="ERROR"):
            result = p.step(HumanScene(width=96).tick(0),0.)
        self.assertGreater(p.mapping.revision,0)
        self.assertEqual(len(p.entities.entities),4)
        self.assertEqual(result["telemetry"]["human_pose"]["status"],"FAILED")
        self.assertGreater(len(p.memory_frames),0)
        self.assertFalse(result["pointing"])

    def test_no_target_and_outside_cone_retain_unknown(self):
        for case in ("no_target","outside_cone"):
            p = HumanPipeline()
            r = p.step(HumanScene(width=128,scenario=case).tick(0),0.)
            self.assertGreater(r["candidates"]["person_001"]["scores"]["unknown"],.9)

    def test_nonblocking_pipeline_and_stale_entities(self):
        gate = threading.Event()
        entered = threading.Event()
        class SlowPose:
            name = "slow-test-model"
            def detect(self,*args):
                entered.set()
                gate.wait(2)
                return []
        p = HumanPipeline(pose_detector=SlowPose(),realtime=True)
        scene = HumanScene(width=96)
        try:
            p.step(scene.tick(0),0.)
            self.assertTrue(entered.wait(1))
            revision = p.mapping.revision
            for i in range(1,4):
                p.step(scene.tick(i),i/10)
            self.assertGreater(p.mapping.revision,revision)
            self.assertFalse(p.tracker.tracks)
        finally:
            gate.set()
            p.close()
        p = HumanPipeline()
        r = p.step(scene.tick(0),0.)
        hypothesis = replace(r["pointing"][0],timestamp_s=1.)
        scores = score_targets(hypothesis,list(p.entities.entities.values()))
        self.assertEqual(scores["scores"]["unknown"],1.)
        self.assertEqual(len(scores["stale_entities_excluded"]),4)

    def test_finger_cue_requires_quality_depth_and_has_broad_uncertainty(self):
        from scope.humans import HumanJoint3D
        _,tracks,_ = track_for()
        track = tracks[0]
        joints = dict(track.pose.joints)
        wrist = joints["right_wrist"]
        direction = np.array([1.,0.,0.])
        for n,offset in (("mcp",.02),("pip",.05),("tip",.10)):
            name = f"right_index_{n}"
            joints[name] = HumanJoint3D(name,wrist.position+direction*offset,np.eye(3)*.0001,
                1.,JointState.OBSERVED_3D,0.,("front",))
        track.pose = replace(track.pose,joints=joints,hand_quality={"right":.9})
        h = estimate_pointing(track,0.)
        self.assertEqual(h.state,"DEGRADED")
        self.assertTrue(np.allclose(h.direction,direction))
        self.assertGreater(h.angular_uncertainty_rad,np.deg2rad(10))
        joints["right_index_tip"] = replace(joints["right_index_tip"],position=None,covariance=None)
        h = estimate_pointing(track,0.)
        self.assertNotIn("index",h.reason)


@unittest.skipUnless(os.environ.get("SCOPE_HUMAN_EPISODE"),"Set SCOPE_HUMAN_EPISODE to recorded RGB-D")
class RecordedHumanTests(unittest.TestCase):
    def test_estimated_pose_pointing(self):
        from scope.pose_detector import TorchvisionPoseDetector
        from scope.storage import load_episode
        from scope.human_synthetic import HumanCameraSample
        from scope.mapping import MapConfig
        frames = load_episode(Path(os.environ["SCOPE_HUMAN_EPISODE"]))
        p = HumanPipeline(cameras=("front",),pose_detector=TorchvisionPoseDetector(),
                          configs={"semantic_detection":ModuleConfig(enabled=False),
                                   "memory":ModuleConfig(enabled=False)},
                          map_config=MapConfig.around_frames(frames))
        hypotheses = []
        for f in frames:
            r = p.step([HumanCameraSample("front",f,[],[])],f.timestamp_s)
            hypotheses.extend(r["pointing"])
        self.assertTrue(any(h.state!="ABSTAIN" for h in hypotheses))
        p.close()


@unittest.skipUnless(os.environ.get("SCOPE_IPO_DATASET"),"Set SCOPE_IPO_DATASET to extracted IPO dataset root")
class RecordedPointingTests(unittest.TestCase):
    def test_estimated_finger_direction_against_labeled_targets(self):
        from scope.ipo import benchmark_ipo
        root = Path(os.environ["SCOPE_IPO_DATASET"])
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_ipo(root/"00017",root/"00001",
                Path(os.environ.get("SCOPE_HAND_MODEL","models/gesture_recognizer.task")),Path(tmp))
        self.assertGreaterEqual(result["nonabstained"],5)
        self.assertLess(result["angular_error_mean_deg"],15.)
        self.assertGreaterEqual(result["top1_rate_all"],.5)
        self.assertGreater(result["abstention_rate"],0.)
