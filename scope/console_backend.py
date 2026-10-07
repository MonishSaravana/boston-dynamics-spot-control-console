"""Shared camera, mapping and target adapters for the Qt and browser consoles."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
import threading
import time

import numpy as np
from PIL import Image, ImageDraw

from .data import Intrinsics, RgbdFrame
from .live_interaction import (
    DryRunExecutor, InteractionError, InteractionSession, RobotPose,
    SelectionSource, VirtualExecutor, candidates_from_pointing,
    candidates_from_query, direct_candidate, _clear_disk)
from .mapping import MapConfig, VoxelMap
from .perception_pipeline import PerceptionPipeline
from .runtime import ModuleConfig
from .semantic_query import SemanticQuery
from .interaction import TextCommand
from .synthetic import SyntheticRoom, look_at, render, room_boxes


def _demo_map():
    mapping = VoxelMap(MapConfig((-3., -3., 0.), (60, 60, 20), .1))
    mapping.frees[:] = 1
    cfg = mapping.config
    for box in room_boxes():
        if box.name == "floor":
            continue
        lo = np.maximum(0, np.floor((np.asarray(box.low)-cfg.origin)/cfg.voxel_m).astype(int))
        hi = np.minimum(np.asarray(cfg.shape), np.ceil((np.asarray(box.high)-cfg.origin)/cfg.voxel_m).astype(int))
        mapping.hits[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
    mapping.revision = 1
    mapping.world_frame = "simulated-world"
    return mapping


def _demo_entities():
    definitions = (
        ("chair_a", "chair", (-1.31, -.13, 0.), (-.69, .48, 1.17)),
        ("chair_b", "chair", (.71, -.03, 0.), (1.33, .58, 1.17)),
        ("table", "table", (-.46, .54, 0.), (.56, 1.41, .85)),
        ("backpack", "backpack", (.80, .68, 0.), (1.21, 1.0, .49)))
    return [{"entity_id": key, "label": label, "bounds_low_m": list(low),
             "bounds_high_m": list(high), "confidence": 1.0, "last_seen_s": None}
            for key, label, low, high in definitions]


class DemoBackend:
    mode = "SIMULATED"
    world_frame = "simulated-world"

    def __init__(self):
        room = SyntheticRoom(frames=1, width=320, height=240)
        self.K, self.boxes = room.K, room.boxes
        self.mapping = _demo_map()
        self.entities = _demo_entities()
        self.executor = VirtualExecutor()
        self.robot = RobotPose(-1.75, -1.45, .25, time.monotonic(), self.world_frame)
        self.frame = None
        self._rendered_pose = None
        self.camera_status = {"synthetic-front": {"health": "OK", "acquired": True,
            "processed": True, "displayed": True, "max_hz": 5, "age_s": 0, "error": None}}
        self.timings = {}
        self.tick()

    def tick(self):
        now = time.monotonic()
        pose = (self.robot.x_m, self.robot.y_m, self.robot.yaw_rad)
        if self.frame is not None and pose == self._rendered_pose:
            self.frame = replace(self.frame, timestamp_s=now, depth_timestamp_s=now,
                                 frame_id=f"demo:{now:.3f}")
            self.robot = replace(self.robot, observed_host_s=now)
            return self.frame
        eye = np.array([self.robot.x_m, self.robot.y_m, 1.35])
        T = look_at(eye, np.array([0., .35, .83]))
        start = time.perf_counter()
        rgb, depth = render(self.boxes, self.K, T)
        self.timings["sensor_simulation_ms"] = (time.perf_counter()-start)*1000
        self.frame = RgbdFrame(rgb, depth, self.K, T, now, now,
                               "SIMULATED", f"demo:{now:.3f}")
        self._rendered_pose = pose
        self.robot = replace(self.robot, observed_host_s=now)
        return self.frame

    def query(self, phrase):
        query = SemanticQuery.from_command(TextCommand(phrase))
        label = query.entity_label.split()[-1]
        matches = [e for e in self.entities if e["label"] == label]
        now = time.monotonic()
        return [replace(direct_candidate(e, now, self.world_frame),
                        source=SelectionSource.QUERY,
                        evidence={"query": phrase, "kind": "simulated room truth"})
                for e in matches]

    def pointing(self):
        ranking = {"state": "DEGRADED", "scores": {"chair_a": .74, "chair_b": .08,
                   "unknown": .18}, "score_kind": "simulated pointing example"}
        return candidates_from_pointing(ranking, self.entities,
                                        time.monotonic(), self.world_frame)

    def direct(self, entity_id):
        entity = next(e for e in self.entities if e["entity_id"] == entity_id)
        return direct_candidate(entity, time.monotonic(), self.world_frame)

    def ready_for_go(self, candidate):
        return None

    def close(self):
        pass


class SpotBackend:
    mode = "SPOT_SENSORS_DRY_RUN"
    world_frame = "odom"

    def __init__(self, hostname, visual_source, depth_source=None, alignment_verified=False,
                 human_pose=False, supervised_go=False, source=None):
        from .spot_live_source import SpotReadOnlySource
        self.source = source if source is not None else SpotReadOnlySource.connect(hostname)
        if visual_source not in self.source.sources:
            raise ValueError("Visual source unavailable; run 'scope spot-sensors' first")
        if depth_source and depth_source not in self.source.sources:
            raise ValueError("Depth source unavailable; run 'scope spot-sensors' first")
        self.visual_source, self.depth_source = visual_source, depth_source
        for name in self.source.sources:
            needed = name == visual_source or name == depth_source
            self.source.configure(name, acquire=needed, process=needed,
                                  display=name == visual_source)
        self.source.alignment_verified = alignment_verified
        self.lease_keepalive = None
        self.map_lock = threading.RLock()
        if supervised_go:
            from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
            from bosdyn.client.robot_command import RobotCommandClient
            from .spot_navigation import AsyncSpotExecutor, SpotTrajectoryExecutor
            client = self.source.robot.ensure_client(RobotCommandClient.default_service_name)
            lease_client = self.source.robot.ensure_client(LeaseClient.default_service_name)
            self.lease_keepalive = LeaseKeepAlive(lease_client, must_acquire=True,
                                                  return_at_exit=True)
            self.executor = AsyncSpotExecutor(SpotTrajectoryExecutor(
                client, self.source.robot_pose, self._route_clear, enabled=True))
            self.mode = "SPOT_SUPERVISED"
        else:
            self.executor = DryRunExecutor()
            self.mode = "SPOT_SENSORS_DRY_RUN"
        self.frame = self.pipeline = self.snapshot = self.robot = None
        self.frame_receipts = {}
        self.entities = []
        self.mapping = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scope-sensors")
        self.pending = None
        self.error = None
        self.human_pose = human_pose
        self.timings = {}

    @property
    def camera_status(self):
        return self.source.status()

    def _read(self):
        self.source.poll()
        try:
            pose = self.source.robot_pose()
        except Exception as exc:
            pose = None
            self.error = f"Robot pose: {exc}"
        return pose

    def _init_pipeline(self, frame, has_geometry):
        from .open_vocab import VerifiedQueryDetector
        from .query_segmentation import GrabCutSegmenter
        config = MapConfig.around_frames([frame]) if has_geometry else MapConfig.synthetic()
        configs = {"mapping": ModuleConfig(enabled=has_geometry, max_hz=5),
                   "human_pose": ModuleConfig(enabled=self.human_pose, max_hz=3),
                   "object_detector": ModuleConfig(enabled=False),
                   "viewer": ModuleConfig(enabled=False),
                   "segmentation": ModuleConfig(enabled=True)}
        class LazyPose:
            name = "torchvision-keypointrcnn-resnet50-fpn-coco-v1"
            metadata = {"backend": "torchvision", "model": name, "device": "cpu"}
            model = None
            def detect(self, f, c):
                if self.model is None:
                    from .pose_detector import TorchvisionPoseDetector
                    self.model = TorchvisionPoseDetector()
                return self.model.detect(f, c)
        self.pipeline = PerceptionPipeline(config,
            lambda: VerifiedQueryDetector("auto", 512, .45),
            GrabCutSegmenter, pose_detector=LazyPose() if self.human_pose else None,
            configs=configs)
        self.mapping = self.pipeline.core.mapping
        self.mapping.world_frame = self.world_frame if has_geometry else None

    def tick(self):
        if self.pending is None:
            self.pending = self.pool.submit(self._read)
        if not self.pending.done():
            return self.frame
        try:
            self.robot = self.pending.result()
            self.error = None if self.robot else self.error
        except Exception as exc:
            self.error = f"Sensor acquisition: {type(exc).__name__}: {exc}"
        self.pending = self.pool.submit(self._read)
        visual = self.source.latest.get(self.visual_source)
        if visual is None or not self.source.policy[self.visual_source].process:
            return self.frame
        has_geometry = False
        if self.depth_source:
            try:
                frame = self.source.rgbd(self.visual_source, self.depth_source)
                has_geometry = True
            except Exception as exc:
                self.error = f"Geometry: {exc}"
        if not has_geometry:
            rgb = visual.pixels
            if rgb.ndim == 2 or rgb.shape[2] == 1:
                gray = rgb if rgb.ndim == 2 else rgb[:, :, 0]
                rgb = np.repeat(gray[:, :, None], 3, axis=2)
            elif rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]
            h, w = rgb.shape[:2]
            K = visual.intrinsics or Intrinsics(w, h, 1., 1., (w-1)/2, (h-1)/2)
            frame = RgbdFrame(rgb.copy(), np.full((h, w), np.nan, np.float32),
                K, np.eye(4), visual.acquisition_robot_s, visual.acquisition_robot_s,
                "SPOT_RGB_ONLY_NO_WORLD_GEOMETRY", f"{visual.source}:{visual.acquisition_robot_s:.6f}")
        if (self.pipeline is not None and self.frame is not None and
                frame.frame_id == self.frame.frame_id and frame.source == self.frame.source):
            return self.frame
        self.frame = frame
        self.frame_receipts[frame.frame_id] = visual.receive_host_s
        if len(self.frame_receipts) > 64:
            del self.frame_receipts[next(iter(self.frame_receipts))]
        if self.pipeline is None and self.depth_source and self.source.alignment_verified and not has_geometry:
            return frame  # Keep cameras visible; wait for safe geometry before sizing the map.
        if self.pipeline is None:
            self._init_pipeline(frame, has_geometry)
        start = time.perf_counter()
        with self.map_lock:
            self.snapshot = self.pipeline.tick(frame, frame.timestamp_s)
        self.source.mark_processed(self.visual_source, visual.acquisition_robot_s)
        if has_geometry and self.depth_source:
            depth = self.source.latest[self.depth_source]
            self.source.mark_processed(self.depth_source, depth.acquisition_robot_s)
        self.timings["perception_tick_ms"] = (time.perf_counter()-start)*1000
        self.entities = self.snapshot["entities"] if has_geometry else []
        return frame

    def query(self, phrase):
        if self.pipeline is None:
            raise ValueError("Wait for a verified RGB-D pair, or restart in camera/query-only mode")
        return self.pipeline.request(phrase)

    def query_candidates(self):
        if self.snapshot is None:
            return None
        query = next((q for q in self.snapshot["queries"]
                      if q["normalized_query"] == self.snapshot["active_query"]), None)
        if query is None or query["state"] == "PENDING":
            return None
        host_s = self.frame_receipts.get(query.get("frame_id"))
        if host_s is None:
            return None
        return candidates_from_query(query, self.entities, host_s,
                                     self.world_frame if self.source.alignment_verified else None)

    def tracked_candidate(self, candidate):
        if self.snapshot is None or self.frame is None:
            return None
        track = next((t for t in self.snapshot["tracks"]
                      if t["candidate"].get("entity_id") == candidate.entity_id), None)
        if track is None:
            return None
        if track["state"] != "FOUND":
            return replace(candidate, track_state=track["state"])
        item = track["candidate"]
        depth = item.get("evidence", {}).get("tracked_depth", {})
        original = item.get("projection") or {}
        if (depth.get("state") != "ESTIMATED_MASK_MEASURED_DEPTH" or
                depth.get("source_timestamp_s") != self.frame.timestamp_s or
                "center_m" not in original or candidate.bounds_low_m is None):
            return None
        shift = np.asarray(depth["center_m"])-np.asarray(original["center_m"])
        host_s = self.frame_receipts.get(self.frame.frame_id)
        if host_s is None:
            return None
        entity = next((e for e in self.entities if e["entity_id"] == candidate.entity_id), None)
        if entity is None:
            return None
        return replace(candidate, observed_host_s=host_s,
            source_timestamp_s=self.frame.timestamp_s,
            bounds_low_m=tuple(np.asarray(entity["bounds_low_m"])+shift),
            bounds_high_m=tuple(np.asarray(entity["bounds_high_m"])+shift),
            box_xyxy=tuple(item["box_xyxy"]), track_state="FOUND",
            evidence={**candidate.evidence, "tracked_depth": depth})

    def pointing(self):
        if self.pipeline is None or not self.pipeline.core_snapshot:
            return [], True
        receipt = self.frame_receipts.get(self.frame.frame_id) if self.frame else None
        if (receipt is None or time.monotonic()-receipt > .6 or
                self.pipeline.core_snapshot["timestamp_s"] != self.frame.timestamp_s):
            return [], True
        rankings = self.pipeline.core_snapshot.get("candidates", {})
        if not rankings:
            return [], True
        ranking = max(rankings.values(), key=lambda r: max(r.get("scores", {}).values(), default=0))
        return candidates_from_pointing(ranking, self.entities, receipt, self.world_frame,
                                        self.frame.timestamp_s if self.frame else None)

    def direct(self, entity_id):
        receipt = self.frame_receipts.get(self.frame.frame_id) if self.frame else None
        if receipt is None or time.monotonic()-receipt > .6:
            raise InteractionError("CAMERA STALE: cannot select an old entity")
        entity = next(e for e in self.entities if e["entity_id"] == entity_id)
        return direct_candidate(entity, receipt, self.world_frame,
                                self.frame.timestamp_s if self.frame else None)

    def ready_for_go(self, candidate):
        if self.pipeline is None or self.mapping is None or self.mapping.revision < 1:
            raise InteractionError("MAP UNAVAILABLE")
        if not self.depth_source:
            raise InteractionError("DEPTH UNAVAILABLE")
        self.source.rgbd(self.visual_source, self.depth_source)
        if self.robot is None:
            raise InteractionError("ROBOT STATE STALE")
        entity = next((e for e in self.entities if e["entity_id"] == candidate.entity_id), None)
        if entity is None or self.frame is None:
            raise InteractionError("TARGET LOST")
        track_fresh = (candidate.evidence.get("tracked_depth", {}).get("source_timestamp_s") ==
                       self.frame.timestamp_s and candidate.track_state == "FOUND")
        if not track_fresh and not 0 <= self.frame.timestamp_s-entity["last_seen_s"] <= 1.5:
            raise InteractionError("TARGET STALE")
        low = np.asarray(entity["bounds_low_m"])
        high = np.asarray(entity["bounds_high_m"])
        if (candidate.bounds_low_m is None or
                np.linalg.norm(low-np.asarray(candidate.bounds_low_m)) > .1 or
                np.linalg.norm(high-np.asarray(candidate.bounds_high_m)) > .1):
            raise InteractionError("TARGET MOVED: preview again")
        if self.snapshot:
            active = next((q for q in self.snapshot["queries"]
                           if q["normalized_query"] == self.snapshot["active_query"]), None)
            if active and active["state"] in ("TRACK_LOST", "STALE", "FAILED"):
                raise InteractionError("TARGET LOST or stale")

    def _route_clear(self, destination):
        try:
            self.source.rgbd(self.visual_source, self.depth_source)
            with self.map_lock:
                return (self.mapping is not None and
                    getattr(self.mapping, "world_frame", None) == destination.world_frame and
                    all(_clear_disk(self.mapping, x, y, .42+destination.clearance_m)
                        for x, y in destination.route_xy_m))
        except (ValueError, KeyError):
            return False

    def execution_status(self):
        return self.executor.status() if self.mode == "SPOT_SUPERVISED" else None

    def close(self):
        if self.mode == "SPOT_SUPERVISED":
            self.executor.close()
        self.pool.shutdown(wait=False, cancel_futures=True)
        if self.pipeline:
            self.pipeline.close()
        if self.lease_keepalive is not None:
            self.lease_keepalive.shutdown()


def world_picture(mapping, entities, robot, proposal, selected, size=420):
    image = Image.new("RGB", (size, size), "#070808")
    draw = ImageDraw.Draw(image)
    if mapping is None:
        draw.text((20, 20), "Map unavailable", fill="#9ba3a8")
        return image
    state = mapping.topdown()
    cfg = mapping.config
    sx = size/state.shape[0]
    sy = size/state.shape[1]
    # Same palette as the browser map: unknown, observed free, occupied.
    colors = {0: "#070808", 1: "#16191b", 2: "#3b4347"}
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            x, y = i*sx, size-(j+1)*sy
            draw.rectangle((x, y, x+sx+1, y+sy+1), fill=colors[int(state[i, j])])
    def pixel(x, y):
        return ((x-cfg.origin[0])/cfg.voxel_m*sx,
                size-(y-cfg.origin[1])/cfg.voxel_m*sy)
    for entity in entities:
        low, high = entity["bounds_low_m"], entity["bounds_high_m"]
        p1, p2 = pixel(low[0], low[1]), pixel(high[0], high[1])
        color = "#f0b44c" if selected and entity["entity_id"] == selected.entity_id else "#9ba3a8"
        draw.rectangle((p1[0], p2[1], p2[0], p1[1]), outline=color, width=3)
        draw.text((p1[0], p2[1]-15), entity["entity_id"], fill=color)
    if proposal:
        route = [pixel(x, y) for x, y in proposal.route_xy_m]
        draw.line(route, fill="#3ecf9b", width=3)
        x, y = pixel(proposal.x_m, proposal.y_m)
        draw.ellipse((x-7, y-7, x+7, y+7), fill="#3ecf9b")
    if robot:
        x, y = pixel(robot.x_m, robot.y_m)
        draw.ellipse((x-7, y-7, x+7, y+7), fill="#ffffff")
        draw.line((x, y, x+18*math.cos(robot.yaw_rad),
                   y-18*math.sin(robot.yaw_rad)), fill="#ffffff", width=3)
    return image


def camera_box(candidate, frame):
    if candidate.box_xyxy is not None:
        return candidate.box_xyxy
    if candidate.bounds_low_m is None or candidate.bounds_high_m is None:
        return None
    low, high = candidate.bounds_low_m, candidate.bounds_high_m
    corners = np.array([[x, y, z] for x in (low[0], high[0])
                        for y in (low[1], high[1]) for z in (low[2], high[2])])
    T = frame.T_world_camera
    optical = (corners-T[:3, 3]) @ T[:3, :3]
    valid = optical[:, 2] > .05
    if not valid.any():
        return None
    pts = optical[valid]
    K = frame.intrinsics
    u = K.fx*pts[:, 0]/pts[:, 2]+K.cx
    v = K.fy*pts[:, 1]/pts[:, 2]+K.cy
    return (float(np.clip(u.min(), 0, K.width-1)), float(np.clip(v.min(), 0, K.height-1)),
            float(np.clip(u.max(), 0, K.width-1)), float(np.clip(v.max(), 0, K.height-1)))


def _project_world(frame, xyz):
    point = (np.asarray(xyz)-frame.T_world_camera[:3, 3]) @ frame.T_world_camera[:3, :3]
    if point[2] <= .05:
        return None
    K = frame.intrinsics
    return (float(K.fx*point[0]/point[2]+K.cx),
            float(K.fy*point[1]/point[2]+K.cy))


def draw_perception_overlay(image, frame, backend):
    if backend.mode == "SIMULATED" or backend.snapshot is None:
        return
    draw = ImageDraw.Draw(image)
    for track in backend.snapshot.get("tracks", []):
        box = track["candidate"].get("box_xyxy")
        if box:
            color = "#5fe0d2" if track["state"] == "FOUND" else "#a0aab1"
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0], max(0, box[1]-12)),
                      f'{track["track_id"]} {track["state"]}', fill=color)
    if frame.source != "SPOT_LIVE" or backend.pipeline is None:
        return
    human = backend.pipeline.core_snapshot
    if not human:
        return
    for track in human.get("tracks", []):
        joints = track.pose.joints
        for side in ("left", "right"):
            for a, b in (("shoulder", "elbow"), ("elbow", "wrist"),
                         ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle")):
                first, second = joints.get(f"{side}_{a}"), joints.get(f"{side}_{b}")
                if first is None or second is None or first.position is None or second.position is None:
                    continue
                p, q = _project_world(frame, first.position), _project_world(frame, second.position)
                if p and q:
                    draw.line((p, q), fill="#75cfff", width=2)
                    draw.ellipse((p[0]-2, p[1]-2, p[0]+2, p[1]+2), fill="#75cfff")
    for hypothesis in human.get("pointing", []):
        if hypothesis.state == "ABSTAIN" or hypothesis.origin is None or hypothesis.direction is None:
            continue
        p = _project_world(frame, hypothesis.origin)
        q = _project_world(frame, hypothesis.origin+2.0*hypothesis.direction)
        if p and q:
            draw.line((p, q), fill="#ffe36b", width=3)
            draw.text(q, f"pointing {hypothesis.state}", fill="#ffe36b")
