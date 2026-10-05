from dataclasses import replace
import time
import unittest

import numpy as np

from scope.data import Intrinsics,RgbdFrame
from scope.mapping import MapConfig
from scope.object_tracking import MaskTracker
from scope.perception_pipeline import PerceptionPipeline
from scope.runtime import ModuleConfig
from scope.semantic_query import ObjectCandidate,QueryResult,QueryState,SemanticQuery
from scope.interaction import TextCommand


def frame(timestamp=0.,shift=0,blank=False):
    import cv2
    rgb = np.zeros((96,128,3),np.uint8)
    rgb[20:70,30+shift:80+shift] = np.random.default_rng(4).integers(50,250,(50,50,3),dtype=np.uint8)
    if blank:rgb[:] = 0
    return RgbdFrame(rgb,np.full((96,128),2.,np.float32),Intrinsics(128,96,100,100,64,48),
        np.eye(4),timestamp,timestamp,"TEST",f"frame-{timestamp}")


def candidate():
    mask=np.zeros((96,128),bool);mask[20:70,30:80]=True
    return ObjectCandidate((30.,20.,80.,70.),.8,"box","test",mask)


class PerceptionTests(unittest.TestCase):
    def test_flow_tracks_then_rejects_disappearance_and_expires(self):
        tracker=MaskTracker(max_age_s=1)
        tracker.initialize(frame(),SemanticQuery.from_command(TextCommand("box")),[candidate()])
        t=tracker.update(frame(.1,shift=5))[0]
        self.assertEqual(t.state,QueryState.FOUND)
        self.assertAlmostEqual(t.candidate.box_xyxy[0],35,delta=1)
        self.assertEqual(t.verified_s,0.)
        self.assertIn("estimated mask",t.candidate.evidence["tracking"]["kind"])
        t=tracker.update(frame(.2,blank=True))[0]
        self.assertEqual(t.state,QueryState.TRACK_LOST)
        tracker.update(frame(2.))
        self.assertEqual(t.state,QueryState.STALE)

    def test_unique_rediscovery_keeps_track_identity(self):
        tracker=MaskTracker()
        query=SemanticQuery.from_command(TextCommand("box"))
        first=tracker.initialize(frame(),query,[candidate()])[0]
        second=tracker.initialize(frame(.1),query,[candidate()])[0]
        self.assertEqual(first.track_id,second.track_id)

    def pipeline(self,function,recorder=None):
        class Detector:
            name="test-query"
            metadata={"backend":"test","model":"test-query","device":"cpu"}
            def query(self,f,q):return function(f,q)
        return PerceptionPipeline(MapConfig((-3.,-3.,0.),(60,60,40)),Detector,
            recorder=recorder,configs={"mapping":ModuleConfig(enabled=False),"human_pose":ModuleConfig(enabled=False)})

    def test_slow_query_and_viewer_do_not_block_core(self):
        def query(f,q):
            time.sleep(.2)
            return QueryResult(q,QueryState.NOT_FOUND,"absent",[],f.timestamp_s,f.frame_id)
        class Recorder:
            def log(self,s):time.sleep(.2)
        p=self.pipeline(query,Recorder())
        try:
            p.request("power strip")
            start=time.perf_counter()
            for i in range(5):p.tick(frame(i*.02))
            self.assertLess(time.perf_counter()-start,.15)
            time.sleep(.25)
            s=p.tick(frame(.15))
            self.assertEqual(s["queries"][0]["state"],"NOT_FOUND")
            self.assertEqual(s["telemetry"]["open_vocabulary"]["device"],"cpu")
        finally:p.close()

    def test_superseded_requests_and_disabled_work_are_discarded(self):
        def query(f,q):
            time.sleep(.05)
            return QueryResult(q,QueryState.FOUND,"found",[candidate()],f.timestamp_s,f.frame_id)
        p=self.pipeline(query)
        try:
            p.request("box");p.tick(frame())
            p.request("chair")
            time.sleep(.08);p.tick(frame(.1))
            self.assertFalse(p.tracker.tracks)
            p.configure("open_vocabulary",ModuleConfig(enabled=False))
            time.sleep(.08);s=p.tick(frame(.2))
            self.assertFalse(p.tracker.tracks)
            self.assertEqual(s["telemetry"]["open_vocabulary"]["status"],"DISABLED")
        finally:p.close()

    def test_depth_entities_and_tracking_survive_query_disabled(self):
        p=self.pipeline(lambda f,q:QueryResult(q,QueryState.FOUND,"found",[candidate()],f.timestamp_s,f.frame_id))
        try:
            p.request("box");p.tick(frame());time.sleep(.05);p.tick(frame(.1))
            self.assertEqual(len(p.core.entities.entities),1)
            self.assertEqual(len(p.events),1)
            p.configure("open_vocabulary",ModuleConfig(enabled=False))
            s=p.tick(frame(.2,shift=5))
            self.assertEqual(s["tracks"][0]["state"],"FOUND")
            self.assertAlmostEqual(s["tracks"][0]["candidate"]["box_xyxy"][0],35,delta=1)
            self.assertEqual(len(s["entities"]),1)
        finally:p.close()

    def test_worker_failure_retains_explicit_reason(self):
        def fail(f,q):raise RuntimeError("detector unavailable")
        p=self.pipeline(fail)
        try:
            p.request("box");p.tick(frame());time.sleep(.05);s=p.tick(frame(.1))
            self.assertEqual(s["queries"][0]["state"],"FAILED")
            self.assertIn("detector unavailable",s["queries"][0]["reason"])
            self.assertEqual(s["telemetry"]["object_tracking"]["status"],"OK")
        finally:p.close()

    def test_background_memory_works_without_query_or_humans(self):
        from scope.objects import ObjectObservation2D
        class Background:
            name="common-test"
            metadata={"backend":"test","device":"cpu"}
            def detect(self,f):
                return [ObjectObservation2D(f.frame_id+":box",f.frame_id,candidate().mask,{"box":1.},.8,self.name)]
        p=PerceptionPipeline(MapConfig((-3.,-3.,0.),(60,60,40)),lambda:None,
            background_detector=Background(),configs={"mapping":ModuleConfig(enabled=False),
            "human_pose":ModuleConfig(enabled=False),"open_vocabulary":ModuleConfig(enabled=False)})
        try:
            p.tick(frame());time.sleep(.05);s=p.tick(frame(.1))
            self.assertEqual(len(s["entities"]),1)
            self.assertEqual(len(s["tracks"]),1)
            self.assertEqual(p.events[0]["kind"],"common_discovery")
            self.assertFalse(p.results)
        finally:p.close()


if __name__=="__main__":unittest.main()
