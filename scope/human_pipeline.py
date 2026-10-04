"""Independent offline/realtime stages sharing M1 frames and M2 entities."""

from dataclasses import replace
import time

import numpy as np

from .entities import EntityStore
from .humans import lift_pose
from .mapping import MapConfig, VoxelMap
from .objects import project_object
from .person_tracking import PersonTracker, fuse_views
from .pointing import estimate_pointing, score_targets
from .runtime import Health, LatestWorker, ModuleConfig, Runtime
from .truth import SyntheticTruthDetector
from .synthetic import room_boxes

MODULES = ("mapping", "semantic_detection", "human_pose", "lifting", "fusion", "tracking",
           "pointing", "intersection", "world_entities", "memory", "robot_visualization",
           "map_visualization", "rerun", "visualization", "decode")


class HumanPipeline:
    """Replay is deterministic for evaluation; realtime uses independent workers.

    No human stage is a dependency of mapping/entities/memory. Viewers consume
    snapshots and cannot mutate core state. Config changes take effect per tick.
    """
    def __init__(self, cameras=("front","left"), configs=None, pose_detector=None,
                 semantic_detector=None, realtime=False, map_config=None):
        defaults = {n:ModuleConfig() for n in MODULES}
        defaults["robot_visualization"] = ModuleConfig(enabled=False)
        defaults["decode"] = ModuleConfig(enabled=False)  # source-dependent instrumentation
        for c in cameras:
            defaults[f"camera/{c}"] = ModuleConfig()
            defaults[f"depth/{c}"] = ModuleConfig()
            defaults[f"pose/{c}"] = ModuleConfig()
            defaults[f"semantic/{c}"] = ModuleConfig()
        defaults.update(configs or {})
        self.runtime = Runtime(defaults)
        self.cameras,self.pose_detector = tuple(cameras),pose_detector
        self.semantic_detector = semantic_detector or SyntheticTruthDetector(room_boxes())
        self.mapping = VoxelMap(map_config or MapConfig.synthetic())
        self.entities = EntityStore()
        self.tracker = PersonTracker()
        self.realtime = realtime
        self.workers = {}
        self.memory_frames, self.memory_detections, self.memory_projections = [],[],[]
        self.memory_status = {"state":"COLLECTING", "frames":0,
                              "reason":"M3 imports completed immutable episodes"}
        self.snapshots = []

    def _infer(self, name, timestamp, payload, function, source):
        rt = self.runtime
        if not self.realtime:
            result = rt.run(name,timestamp,lambda:function(payload),source=source)
            return None if result is None else (payload,result)
        worker = self.workers.get(name)
        if worker is None:
            worker = LatestWorker(function)
            self.workers[name] = worker
        state = rt.state(name)
        if rt.due(name,timestamp):
            worker.submit(payload)
        state.queue_depth,state.dropped_inputs = worker.queue_depth,max(state.dropped_inputs,worker.dropped)
        completed = worker.poll()
        if completed is None:
            return None
        old_payload,value,error,duration = completed
        old_s = old_payload.frame.timestamp_s
        state.durations_ms.append(duration)
        state.source = source
        if error:
            state.failures += 1
            state.recent_failures.append({"timestamp_s":old_s,"reason":error})
            rt.mark(name,Health.FAILED,error)
            return None
        state.last_output,state.last_success_s = value,old_s
        state.input_timestamp_s = old_s
        state.success_times_s.append(old_s)
        state.successes += 1
        if timestamp-old_s>state.config.stale_after_s or not state.config.enabled:
            rt.mark(name,Health.STALE,"Inference result expired before consumption")
            return None
        rt.mark(name,Health.OK,"Fresh worker result")
        return old_payload,value

    def step(self,samples,timestamp_s):
        rt = self.runtime
        active,poses,detections = [],[],{}
        semantics_due = rt.due("semantic_detection",timestamp_s)
        humans_due = rt.due("human_pose",timestamp_s)
        for sample in samples:
            c = sample.camera_id
            frame = rt.run(f"camera/{c}",timestamp_s,lambda:sample.frame,source=sample.frame.source)
            if frame is None:
                continue
            depth = rt.run(f"depth/{c}",timestamp_s,lambda:frame.depth_m,source=frame.source)
            if depth is None:
                frame = replace(frame,depth_m=np.full_like(frame.depth_m,np.nan))
                sample = replace(sample,frame=frame)
            active.append(sample)
            rt.run("mapping",timestamp_s,lambda:self.mapping.integrate(frame),source=frame.source,
                   scheduled=True)
            if semantics_due:
                found = self._infer(f"semantic/{c}",timestamp_s,sample,
                    lambda x:self.semantic_detector.detect(x.frame),self.semantic_detector.name)
                if found is not None:
                    used,objects = found
                    projected = rt.run("entity_projection",timestamp_s,
                        lambda:[p for o in objects if (p:=project_object(used.frame,o)) is not None],
                        input_s=used.frame.timestamp_s,scheduled=False)
                    if projected is not None:
                        rt.run("world_entities",timestamp_s,lambda:self.entities.add_frame(projected),
                               input_s=used.frame.timestamp_s)
                        detections[c] = (objects,projected)
            if humans_due:
                function = (lambda x:self.pose_detector.detect(x.frame,x.camera_id)) if self.pose_detector else (
                    lambda x:x.observations)
                result = self._infer(f"pose/{c}",timestamp_s,sample,function,
                    self.pose_detector.name if self.pose_detector else "oracle-visible-joints")
                if result is not None:
                    used,observations = result
                    lifted = rt.run("lifting",timestamp_s,
                        lambda:[lift_pose(o,used.frame) for o in observations],
                        input_s=used.frame.timestamp_s,scheduled=False)
                    if lifted is not None:
                        poses.extend(lifted)
        present = {s.camera_id for s in samples}
        for c in self.cameras:
            if c not in present and rt.state(f"camera/{c}").config.enabled:
                rt.mark(f"camera/{c}",Health.UNAVAILABLE,"Camera input missing")
        # Aggregate health is derived from camera stages, not fabricated success.
        for aggregate,prefix,enabled in (("human_pose","pose",humans_due),
                                          ("semantic_detection","semantic",semantics_due)):
            if enabled:
                statuses = [rt.snapshot(timestamp_s)[f"{prefix}/{s.camera_id}"]["status"] for s in active]
                bad_camera = any(rt.snapshot(timestamp_s)[f"camera/{c}"]["status"]!="OK" for c in self.cameras)
                if "OK" in statuses:
                    state = rt.state(aggregate)
                    state.last_success_s = timestamp_s
                    rt.mark(aggregate,Health.DEGRADED if bad_camera or any(v!="OK" for v in statuses)
                            else Health.OK,"Some streams unavailable" if bad_camera else "Camera inference health")
                else:
                    rt.mark(aggregate,Health.FAILED if "FAILED" in statuses else Health.UNAVAILABLE,
                            "No fresh camera inference output")
        fused = rt.run("fusion",timestamp_s,lambda:fuse_views(poses)) if poses else None
        unresolved_views = []
        if fused is not None:
            fused,unresolved_views = fused
            if unresolved_views:
                rt.mark("fusion",Health.DEGRADED,"Person association unresolved across views")
        tracks = rt.run("tracking",timestamp_s,lambda:self.tracker.update(fused or [],timestamp_s))
        self.tracker.expire(timestamp_s)
        tracks = tracks if tracks is not None else list(self.tracker.tracks.values())
        hypotheses = rt.run("pointing",timestamp_s,
            lambda:[estimate_pointing(t,timestamp_s) for t in tracks]) if rt.state("human_pose").config.enabled else None
        hypotheses = hypotheses or []
        if not hypotheses and rt.state("pointing").config.enabled:
            rt.mark("pointing",Health.UNAVAILABLE,"No fresh person geometry")
        elif any(h.state!="OK" for h in hypotheses):
            rt.mark("pointing",Health.UNAVAILABLE if all(h.state=="ABSTAIN" for h in hypotheses)
                    else Health.DEGRADED,"; ".join(sorted(set(h.reason for h in hypotheses))))
        candidates = rt.run("intersection",timestamp_s,
            lambda:{h.person_track_id:score_targets(h,list(self.entities.entities.values()),
                self.mapping if rt.state("mapping").config.enabled else None) for h in hypotheses})
        # Memory collects primary-camera semantic evidence independently of humans.
        if active and rt.state("memory").config.enabled:
            primary = active[0]
            if primary.camera_id in detections:
                objects,projected = detections[primary.camera_id]
                def collect():
                    self.memory_frames.append(primary.frame)
                    self.memory_detections.append(objects)
                    self.memory_projections.append(projected)
                    self.memory_status["frames"] = len(self.memory_frames)
                    return dict(self.memory_status)
                rt.run("memory",timestamp_s,collect)
        snapshot = {"timestamp_s":timestamp_s,"samples":active,"tracks":tracks,
                    "pointing":hypotheses,"candidates":candidates or {},
                    "unresolved_views":unresolved_views,"unresolved_tracks":self.tracker.unresolved,
                    "memory":dict(self.memory_status),"telemetry":rt.snapshot(timestamp_s)}
        return snapshot

    def save_memory(self,output):
        """Use the unchanged M3 immutable-episode import/link path."""
        if not self.runtime.state("memory").config.enabled or not self.memory_frames:
            return None
        from .memory import process_episode
        from .memory_store import MapAlignment, MemoryStore, episode_snapshot
        from .semantic_pipeline import SemanticRun
        from .semantic_storage import save_semantic_run
        from .storage import save_episode,save_map
        store = EntityStore()
        mapping = VoxelMap(self.mapping.config)
        for frame,projected in zip(self.memory_frames,self.memory_projections):
            store.add_frame(projected)
            mapping.integrate(frame)
        run = SemanticRun(self.memory_frames,self.memory_detections,self.memory_projections,
                          store,[],self.semantic_detector.name,"entities")
        directory = output/"memory-episode"
        save_episode(directory,self.memory_frames,"human-demo-primary-camera")
        save_map(directory,mapping)
        save_semantic_run(directory,run)
        memory = MemoryStore(output/"memory.sqlite")
        try:
            memory.import_snapshot(episode_snapshot("human-demo",directory,run,
                MapAlignment("synthetic-room","known_shared")))
            process_episode(memory,"human-demo")
            beliefs = memory.beliefs()
            self.memory_status = {"state":"PERSISTED","global_entities":len(beliefs),
                                  "database":str(output/"memory.sqlite")}
            return beliefs
        finally:
            memory.close()

    def close(self):
        for worker in self.workers.values():
            worker.close()
