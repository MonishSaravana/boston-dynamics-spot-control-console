"""One local perception system, with bounded discovery and viewer workers."""

from collections import deque
import time

import numpy as np

from .backends import hardware_report
from .human_pipeline import HumanPipeline
from .human_synthetic import HumanCameraSample
from .interaction import TextCommand
from .object_tracking import MaskTracker
from .objects import ObjectObservation2D, project_object
from .runtime import Health, LatestWorker, ModuleConfig
from .semantic_query import QueryResult, QueryState, SemanticQuery


class PerceptionPipeline:
    def __init__(self, map_config, query_factory, segmenter_factory=None, *,
                 background_detector=None, pose_detector=None, recorder=None, configs=None):
        cfg = {"mapping": ModuleConfig(max_hz=5), "human_pose": ModuleConfig(enabled=bool(pose_detector),max_hz=3,stale_after_s=2),
               "pose/front": ModuleConfig(stale_after_s=2),
               "semantic_detection": ModuleConfig(enabled=False), "memory": ModuleConfig(enabled=False)}
        cfg.update(configs or {})
        self.core = HumanPipeline(("front",),cfg,pose_detector,realtime=True,map_config=map_config)
        from .entities import EntityStore
        self.core.entities=EntityStore(history_limit=512,entity_limit=128)
        self.runtime = self.core.runtime
        for name,config in {
            "open_vocabulary":ModuleConfig(stale_after_s=3),
            "segmentation":ModuleConfig(stale_after_s=3), "object_tracking":ModuleConfig(stale_after_s=3),
            "object_detector":ModuleConfig(enabled=bool(background_detector),max_hz=.5,stale_after_s=3),
            "query_projection":ModuleConfig(stale_after_s=3),"query_memory":ModuleConfig(),
            "viewer":ModuleConfig(enabled=bool(recorder),max_hz=5,stale_after_s=3),
            "serialization":ModuleConfig(), "resource_metrics":ModuleConfig(max_hz=1)}.items():
            self.runtime.configure(name,(configs or {}).get(name,config))
        self.query_factory,self.segmenter_factory = query_factory,segmenter_factory
        self.query_detector,self.segmenter = None,None
        self.background_detector,self.recorder = background_detector,recorder
        self.tracker = MaskTracker()
        self.workers = {}
        self.results,self.events,self.memory_evidence = {},[],{}
        self.pending_query = None
        self.generation,self.verification = 0,0
        self.last_query_submit_s = None
        self.latest_frame,self.last_core_frame,self.core_snapshot = None,None,None
        self.recent_frames = deque(maxlen=60)
        self.resources = {}
        self.last_core_ms = None

    def configure(self,name,config):
        self.runtime.configure(name,config)
        if name == "open_vocabulary" and config.enabled:
            self.last_query_submit_s = None

    def request(self, command):
        query = SemanticQuery.from_command(command if isinstance(command,TextCommand) else TextCommand(command))
        self.generation += 1
        self.pending_query = (self.generation,query)
        self.last_query_submit_s = None
        if self.latest_frame is not None:
            f = self.latest_frame
            self.results[query.normalized] = QueryResult(query,QueryState.PENDING,"Queued local visual query",[],f.timestamp_s,f.frame_id)
        return query

    def _query(self, payload):
        generation,query,frame,use_masks = payload
        if self.query_detector is None:
            self.query_detector = self.query_factory()
        result = self.query_detector.query(frame,query)
        if use_masks and self.segmenter_factory and result.state in (QueryState.FOUND,QueryState.AMBIGUOUS):
            try:
                if self.segmenter is None:
                    self.segmenter = self.segmenter_factory()
                result.timings_ms.update(self.segmenter.segment(frame.rgb,result.candidates))
            except Exception as exc:
                import logging
                logging.getLogger(__name__).exception("Query segmentation failed")
                result.metadata["segmentation_error"] = f"{type(exc).__name__}: {exc}"
        return result

    def _submit(self,name,payload,function):
        if name not in self.workers:
            self.workers[name] = LatestWorker(function)
        self.workers[name].submit(payload)

    def _poll(self,name,now_s):
        worker = self.workers.get(name)
        if worker is None:
            return None
        state = self.runtime.state(name)
        state.queue_depth,state.dropped_inputs = worker.queue_depth,worker.dropped
        state.in_flight = worker.in_flight
        completed = worker.poll()
        if completed is None:
            return None
        payload,value,error,duration = completed
        if error:
            state.failures += 1
            state.recent_failures.append({"timestamp_s":now_s,"reason":error})
            self.runtime.mark(name,Health.FAILED,error)
            return payload,None
        stamp = now_s if name=="resource_metrics" else payload.timestamp_s if hasattr(payload,"timestamp_s") else payload[2].timestamp_s if name=="open_vocabulary" else now_s
        model = self.query_detector if name=="open_vocabulary" else self.background_detector if name=="object_detector" else None
        metadata = getattr(model,"metadata",{}) if model is not None else {"backend":"rerun" if name=="viewer" else "host resources","device":"cpu"}
        self.runtime.observe(name,stamp,duration,output=value,
            source=getattr(model,"name",name),metadata=metadata)
        if not state.config.enabled:
            self.runtime.mark(name,Health.DISABLED,"Disabled; completed work discarded")
            return None
        return payload,value

    def _project(self,frame,query,candidates):
        self.verification += 1
        observations,projected,pairs = [],[],[]
        for i,candidate in enumerate(candidates):
            if candidate.mask is None:
                candidate.evidence["projection_unavailable"] = "No supported object mask"
                continue
            # Scores are evidence, not class probabilities. The identified label
            # is separate from its measured detector confidence.
            label = query.entity_label
            observation = ObjectObservation2D(f"{frame.frame_id}:query-{self.verification}-{i}",frame.frame_id,
                candidate.mask.copy(),{label:1.},candidate.score,candidate.model)
            projection = self.runtime.run("object_3d_projection",frame.timestamp_s,
                lambda:project_object(frame,observation),source="aligned mask depth")
            if projection is None:
                candidate.evidence["projection_unavailable"] = "Insufficient valid aligned depth"
                continue
            observations.append(observation); projected.append(projection);pairs.append(candidate)
        if projected:
            self.runtime.run("world_entities",frame.timestamp_s,lambda:self.core.entities.add_frame(projected))
            for c,p in zip(pairs,projected):
                c.entity_id = self.core.entities.observation_entity.get(p.observation_id)
                c.projection = {"state":"OBSERVED_DEPTH","center_m":p.center.tolist(),
                    "bounds_low_m":p.low.tolist(),"bounds_high_m":p.high.tolist(),
                    "covariance_m2":p.covariance.tolist(),"source_timestamp_s":frame.timestamp_s,
                    "valid_depth_fraction":p.valid_depth_fraction}
            if self.runtime.state("query_memory").config.enabled:
                self.runtime.run("query_memory",frame.timestamp_s,
                    lambda:self._remember(frame,observations,projected,pairs,"verified_query",query.raw_phrase))
        return projected

    def _remember(self,frame,observations,projected,candidates,kind,phrase=None):
        _,old_obs,old_proj = self.memory_evidence.get(frame.frame_id,(frame,{},{}))
        for c,o,p in zip(candidates,observations,projected):
            key = c.entity_id or (o.label,o.box_xyxy)
            old_obs[key],old_proj[key] = o,p
        self.memory_evidence[frame.frame_id] = (frame,old_obs,old_proj)
        # Bounded keyframes, not a growing in-memory video archive.
        if len(self.memory_evidence)>128:
            oldest=min(self.memory_evidence,key=lambda k:self.memory_evidence[k][0].timestamp_s)
            del self.memory_evidence[oldest]
        event={"kind":kind,"phrase":phrase,"frame_id":frame.frame_id,
            "source_timestamp_s":frame.timestamp_s,"candidates":[c.summary() for c in candidates]}
        self.events.append(event);self.events=self.events[-256:]
        return event

    def _accept_query(self,payload,result,now_s):
        generation,query,frame,_ = payload
        if generation != self.generation:
            self.runtime.state("open_vocabulary").dropped_inputs += 1
            return
        if result is None:
            result = QueryResult(query,QueryState.FAILED,self.runtime.state("open_vocabulary").reason,[],frame.timestamp_s,frame.frame_id)
        immutable = frame.source in ("RGB_FILE","MEASURED_NYU_REGISTERED_STILL","MEASURED_RGBD_FILE")
        if not immutable and now_s-frame.timestamp_s > self.runtime.state("open_vocabulary").config.stale_after_s:
            result.state,result.reason = QueryState.STALE,"Query completed after its observation age limit"
        self.results[query.normalized] = result
        result.metadata["result_consumed_monotonic_s"]=time.monotonic()
        while len(self.results)>32:
            del self.results[next(iter(self.results))]
        for name,duration in result.timings_ms.items():
            self.runtime.observe("query_"+name,frame.timestamp_s,duration,metadata=result.metadata)
        if "segmentation_error" in result.metadata:
            self.runtime.mark("segmentation",Health.FAILED,result.metadata["segmentation_error"])
        elif "segmentation" in result.timings_ms:
            self.runtime.observe("segmentation",frame.timestamp_s,result.timings_ms["segmentation"],
                metadata=getattr(self.segmenter,"metadata",{}))
        if result.state not in (QueryState.FOUND,QueryState.AMBIGUOUS):
            return
        self.runtime.run("query_projection",now_s,lambda:self._project(frame,query,result.candidates),input_s=frame.timestamp_s)
        self.tracker.initialize(frame,query,result.candidates)
        for recent in self.recent_frames:
            if recent.timestamp_s>frame.timestamp_s:
                self.tracker.update(recent)

    def tick(self, frame, now_s=None):
        start = time.perf_counter()
        now_s = frame.timestamp_s if now_s is None else now_s
        self.latest_frame = frame
        if self.last_core_frame != (frame.frame_id,frame.timestamp_s):
            self.recent_frames.append(frame)
            self.core_snapshot = self.core.step([HumanCameraSample("front",frame,[],[])],now_s)
            self.last_core_frame = (frame.frame_id,frame.timestamp_s)
            self.runtime.run("object_tracking",now_s,lambda:self.tracker.update(frame))
            self.runtime.run("tracked_depth",now_s,lambda:self._tracked_depth(frame))
        completed = self._poll("open_vocabulary",now_s)
        if completed:
            self._accept_query(*completed,now_s)
        if self.pending_query:
            generation,query = self.pending_query
            enabled = self.runtime.state("open_vocabulary").config.enabled
            # Refresh verified targets periodically; never run the query model
            # on every camera frame. Latest-only worker bounds the pending input.
            immutable = frame.source in ("RGB_FILE","MEASURED_NYU_REGISTERED_STILL","MEASURED_RGBD_FILE")
            if enabled and (self.last_query_submit_s is None or not immutable and now_s-self.last_query_submit_s>=1.5) and self.runtime.due("open_vocabulary",now_s):
                self._submit("open_vocabulary",(generation,query,frame,self.runtime.state("segmentation").config.enabled),self._query)
                self.last_query_submit_s = now_s
            elif not enabled and query.normalized in self.results:
                self.results[query.normalized].state = QueryState.DISABLED
                self.results[query.normalized].reason = "Heavy query detector disabled; existing tracks remain independent"
        if self.background_detector:
            if self.runtime.due("object_detector",now_s):
                self._submit("object_detector",frame,self.background_detector.detect)
            found = self._poll("object_detector",now_s)
            if found and found[1] is not None:
                used,objects = found
                from dataclasses import replace
                original_labels={o.observation_id:o.label for o in objects}
                objects=[replace(o,class_probabilities={
                    SemanticQuery.from_command(TextCommand(o.label)).entity_label:1.}) for o in objects]
                if now_s-used.timestamp_s <= self.runtime.state("object_detector").config.stale_after_s:
                    projections = self.runtime.run("background_projection",now_s,
                        lambda:[p for o in objects if (p:=project_object(used,o)) is not None],input_s=used.timestamp_s)
                    from .semantic_query import ObjectCandidate
                    candidates=[ObjectCandidate(tuple(o.box_xyxy),o.confidence,o.label,o.source,o.mask.copy()) for o in objects]
                    for o,c in zip(objects,candidates):c.evidence["closed_category"]=original_labels[o.observation_id]
                    if projections:
                        self.runtime.run("world_entities",now_s,lambda:self.core.entities.add_frame(projections),input_s=used.timestamp_s)
                        by_observation={o.observation_id:c for o,c in zip(objects,candidates)}
                        projected_candidates=[]
                        for projection in projections:
                            o=projection.detection
                            c=by_observation[o.observation_id]
                            c.entity_id=self.core.entities.observation_entity.get(o.observation_id)
                            c.projection={"state":"OBSERVED_DEPTH","center_m":projection.center.tolist(),
                                "source_timestamp_s":used.timestamp_s}
                            projected_candidates.append(c)
                        self.runtime.run("query_memory",now_s,lambda:self._remember(used,
                            [p.detection for p in projections],projections,projected_candidates,"common_discovery"))
                    self.tracker.initialize(used,SemanticQuery.from_command(TextCommand("background/common")),candidates)
                    for recent in self.recent_frames:
                        if recent.timestamp_s>used.timestamp_s:self.tracker.update(recent)
                else:
                    self.runtime.mark("object_detector",Health.STALE,"Background result too old")
        if self.runtime.due("resource_metrics",now_s):
            self._submit("resource_metrics",frame,lambda f:hardware_report())
        resources = self._poll("resource_metrics",now_s)
        if resources and resources[1] is not None:
            self.resources = resources[1]
        self.last_core_ms = (time.perf_counter()-start)*1000
        snapshot = self.snapshot(now_s)
        if self.recorder and self.runtime.due("viewer",now_s):
            snapshot["map"] = tuple(a.copy() for a in self.core.mapping.cloud())
            snapshot["frame"] = frame
            if self.core_snapshot:
                from .human_cli import summary
                snapshot["human_snapshot"] = summary(self.core_snapshot)
            self._submit("viewer",snapshot,self.recorder.log)
        self._poll("viewer",now_s)
        return snapshot

    def _tracked_depth(self,frame):
        """Current depth under a flow mask is estimated track geometry, not a new semantic verification."""
        if not np.isfinite(frame.depth_m).any():
            return
        for t in self.tracker.active():
            o=ObjectObservation2D(frame.frame_id+":"+t.track_id,frame.frame_id,t.candidate.mask,
                {t.candidate.detector_match:1.},t.candidate.score,"estimated optical-flow mask")
            p=project_object(frame,o)
            t.candidate.evidence["tracked_depth"] = ({"state":"ESTIMATED_MASK_MEASURED_DEPTH",
                "center_m":p.center.tolist(),"source_timestamp_s":frame.timestamp_s,
                "verified_source_s":t.verified_s,"valid_depth_fraction":p.valid_depth_fraction} if p is not None else
                {"state":"UNAVAILABLE","reason":"Insufficient aligned depth under tracked mask"})

    def snapshot(self,now_s):
        queries=[]
        for r in self.results.values():
            q=r.summary(now_s);q["detector_state"]=q["state"]
            related=[self.tracker.tracks[c.track_id] for c in r.candidates if c.track_id in self.tracker.tracks]
            if r.state in (QueryState.FOUND,QueryState.AMBIGUOUS) and related:
                if all(t.state==QueryState.TRACK_LOST for t in related):
                    q["state"],q["reason"]="TRACK_LOST","Target tracking lost; retained detector evidence is historical"
                elif all(t.state==QueryState.STALE for t in related):
                    q["state"],q["reason"]="STALE","Detector verification expired; no fresh object claim"
            queries.append(q)
        return {"timestamp_s":now_s,"queries":queries,
            "input_kind":"immutable still" if self.latest_frame and self.latest_frame.source in ("RGB_FILE","MEASURED_NYU_REGISTERED_STILL","MEASURED_RGBD_FILE") else "sequence/camera",
            "active_query":self.pending_query[1].normalized if self.pending_query else None,
            "observation_age_s":None if self.latest_frame is None else max(0.,now_s-self.latest_frame.timestamp_s),
            "tracks":[{"track_id":t.track_id,"queries":sorted(t.queries),"state":str(t.state),"reason":t.reason,
                "verified_age_s":max(0.,now_s-t.verified_s),"candidate":t.candidate.summary(),
                "mask":None if t.state!=QueryState.FOUND else t.candidate.mask.copy()} for t in self.tracker.tracks.values()],
            "entities":[e.summary() for e in self.core.entities.entities.values()],
            "telemetry":self.runtime.snapshot(now_s),"resources":dict(self.resources),
            "core_tick_ms":self.last_core_ms,"active_camera_count":1,"memory_events":len(self.events),
            "rerun_enabled":self.runtime.state("viewer").config.enabled}

    def close(self):
        done = [w.close(timeout_s=5.) for w in self.workers.values()]
        return self.core.close() and all(done)
