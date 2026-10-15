"""Save what a live console session builds, on its own bounded worker.

The workspace thread only copies small samples and hands them over without
waiting. Rerun encoding, file writes and the final map/entity export happen on
the recorder thread, so a slow disk never delays perception, Stop or GO.
"""
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np

from .data import camera_to_world, project_depth
from .mapping import FREE, OCCUPIED, VoxelMap

MAX_BYTES = 1 << 30          # 1 GiB per session, then the recording finishes itself
WARN_FRACTION = .8
SAMPLE_HZ = 2.
MAP_EVERY_S = 10.
QUEUE_SAMPLES = 8
TRAJECTORY_POINTS = 4000
DEPTH_STRIDE = 6
RECORDING = "session.rrd"


def copy_map(mapping):
    """Detached copy taken on the thread that integrates the map."""
    if mapping is None:
        return None
    copy = VoxelMap(mapping.config)
    for name in ("hits", "frees", "point_sum", "color_sum"):
        setattr(copy, name, getattr(mapping, name).copy())
    copy.revision = mapping.revision
    copy.world_frame = getattr(mapping, "world_frame", None)
    return copy


def _session_folder(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for suffix in [""] + [f"-{i}" for i in range(2, 100)]:
        folder = root / f"live-{stamp}{suffix}"
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        # Camera captures stay out of Git even under a --runs-dir inside a checkout.
        (folder / ".gitignore").write_text("*\n")
        return folder
    raise ValueError("Could not create a new session folder")


def _human_bytes(value):
    return f"{value / 1e9:.2f} GB" if value >= 1e9 else f"{value / 1e6:.0f} MB"


class SessionRecorder:
    """One recording per start; states are recording → finishing → saved | failed."""

    def __init__(self, root, *, info, max_bytes=MAX_BYTES, sample_hz=SAMPLE_HZ,
                 map_every_s=MAP_EVERY_S):
        self.folder = _session_folder(root)
        self.info = dict(info)
        self.max_bytes, self.map_every_s = int(max_bytes), float(map_every_s)
        self.period_s = 1. / sample_hz
        self.started_host_s = time.monotonic()
        self.started_wall = datetime.now().astimezone()
        self.state = "recording"
        self.error = None
        self.reason = None
        self.samples = self.dropped = 0
        self.bytes = 0
        self.limit_reached = False
        self.last_sample_s = None
        self.last_map_s = None
        self.ended_host_s = None
        self.lock = threading.Lock()
        self.queue = queue.Queue(maxsize=QUEUE_SAMPLES)
        self.final = None
        self.finish_requested = threading.Event()
        self.thread = threading.Thread(target=self._run, name="scope-session-recorder", daemon=True)
        self.thread.start()

    @property
    def active(self):
        return self.state == "recording"

    def due(self, now_s):
        return (self.active and not self.limit_reached and
                (self.last_sample_s is None or now_s - self.last_sample_s >= self.period_s))

    def map_due(self, now_s):
        return self.last_map_s is None or now_s - self.last_map_s >= self.map_every_s

    def offer(self, sample):
        """Never waits. A full queue means the disk is behind; drop the sample."""
        if not self.active or self.limit_reached:
            return False
        self.last_sample_s = sample["host_s"]
        if "map_cloud" in sample:
            self.last_map_s = sample["host_s"]
        try:
            self.queue.put_nowait(sample)
            return True
        except queue.Full:
            with self.lock:
                self.dropped += 1
            return False

    def finish(self, mapping=None, entities=(), reason="operator"):
        """Hand over the final map copy; writing it happens on the recorder thread."""
        if self.state != "recording":
            return
        self.final = {"mapping": mapping, "entities": list(entities), "reason": reason}
        self.state = "finishing"
        self.finish_requested.set()

    def join(self, timeout_s=None):
        self.thread.join(timeout_s)
        return not self.thread.is_alive()

    def status(self, now_s=None):
        now_s = time.monotonic() if now_s is None else now_s
        with self.lock:
            elapsed = (self.ended_host_s or now_s) - self.started_host_s
            warning = None
            if self.limit_reached:
                warning = f"Size limit of {_human_bytes(self.max_bytes)} reached; recording finished"
            elif self.bytes >= WARN_FRACTION * self.max_bytes:
                warning = f"Recording is near its {_human_bytes(self.max_bytes)} limit"
            elif self.dropped:
                warning = f"{self.dropped} samples dropped while the disk caught up"
            return {"state": self.state, "folder": self.folder.name, "path": str(self.folder),
                    "elapsed_s": max(0., elapsed), "bytes": self.bytes,
                    "max_bytes": self.max_bytes, "samples": self.samples,
                    "dropped": self.dropped, "warning": warning, "error": self.error,
                    "reason": self.reason}

    # Recorder thread ---------------------------------------------------------

    def _run(self):
        stream = None
        try:
            import rerun as rr
            stream = self._open(rr)
            last_size_check = 0.
            while True:
                try:
                    sample = self.queue.get(timeout=.2)
                except queue.Empty:
                    if self.finish_requested.is_set():
                        break
                    continue
                if not self.limit_reached:
                    self._log(rr, stream, sample)
                    with self.lock:
                        self.samples += 1
                if time.monotonic() - last_size_check >= 1.:
                    last_size_check = time.monotonic()
                    stream.flush(timeout_sec=5.)
                    self._measure()
            self._finalize(rr, stream)
        except Exception as exc:
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "failed"
                self.ended_host_s = time.monotonic()
        finally:
            if stream is not None:
                try:
                    stream.flush(timeout_sec=10.)
                    stream.disconnect()
                except Exception:
                    pass
            self._measure()

    def _measure(self):
        path = self.folder / RECORDING
        size = path.stat().st_size if path.exists() else 0
        with self.lock:
            self.bytes = size
            if size >= self.max_bytes and not self.limit_reached:
                # The workspace thread notices and hands over the final map.
                self.limit_reached = True

    def _open(self, rr):
        import rerun.blueprint as rrb
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="Depth points · map · robot pose · entities",
                                  background=[17, 25, 35]),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="camera", name="Processed camera frame"),
                    rrb.Spatial2DView(origin="map/topdown",
                                      name="Occupied orange · Free teal · Unknown gray"),
                    rrb.TextLogView(origin="log", name="Session log")),
                column_shares=[2., 1.]),
            rrb.BlueprintPanel(expanded=False), rrb.SelectionPanel(expanded=False),
            rrb.TimePanel(expanded=True, timeline="elapsed"),
            auto_layout=False, auto_views=False)
        stream = rr.RecordingStream("SCOPE live session")
        stream.save(self.folder / RECORDING, default_blueprint=blueprint)
        stream.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        stream.log("world/axes", rr.TransformAxes3D(axis_length=.5), static=True)
        stream.log("log", rr.TextLog(
            f"{self.info.get('backend_mode')} session in world frame "
            f"{self.info.get('world_frame')}. {self.info.get('evidence_note', '')}"), static=True)
        self.trajectory = []
        return stream

    def _time(self, stream, host_s):
        stream.set_time("elapsed", duration=max(0., host_s - self.started_host_s))

    def _log(self, rr, stream, s):
        self._time(stream, s["host_s"])
        if s.get("jpeg"):
            stream.log("camera", rr.EncodedImage(contents=s["jpeg"], media_type="image/jpeg"))
        frame = s.get("frame")
        if frame is not None and s.get("geometry"):
            T = frame.T_world_camera
            K = frame.intrinsics
            stream.log("world/camera", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
            stream.log("world/camera", rr.Pinhole(
                width=K.width, height=K.height, focal_length=[K.fx, K.fy],
                principal_point=[K.cx, K.cy], camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=.3))
            points, colors, _ = project_depth(frame, stride=DEPTH_STRIDE, max_depth_m=5.)
            if len(points):
                stream.log("world/depth_points", rr.Points3D(
                    camera_to_world(points, T), colors=colors, radii=.012))
            else:
                stream.log("world/depth_points", rr.Clear(recursive=False))
        robot = s.get("robot")
        if robot is not None:
            x, y, yaw = robot["x_m"], robot["y_m"], robot["yaw_rad"]
            stream.log("world/robot", rr.Arrows3D(
                origins=[[x, y, .05]], vectors=[[.5 * math.cos(yaw), .5 * math.sin(yaw), 0.]],
                colors=[230, 232, 233], labels=["robot pose"]))
            self.trajectory.append((x, y, .05))
            if len(self.trajectory) > TRAJECTORY_POINTS:
                self.trajectory = self.trajectory[::2]
            stream.log("world/trajectory", rr.LineStrips3D([self.trajectory],
                                                           colors=[62, 207, 155], radii=.015))
        self._entities(rr, stream, s.get("entities", []))
        if s.get("topdown") is not None:
            stream.log("map/topdown", rr.Image(_topdown_image(s["topdown"])))
        if "map_cloud" in s:
            positions, colors = s["map_cloud"]
            stream.log("world/map", rr.Points3D(positions, colors=colors, radii=.03)
                       if len(positions) else rr.Clear(recursive=False))

    def _entities(self, rr, stream, entities):
        boxes = [e for e in entities if e.get("bounds_low_m") is not None
                 and e.get("bounds_high_m") is not None]
        if not boxes:
            stream.log("world/entities", rr.Clear(recursive=False))
            return
        low = np.asarray([e["bounds_low_m"] for e in boxes], dtype=float)
        high = np.asarray([e["bounds_high_m"] for e in boxes], dtype=float)
        stream.log("world/entities", rr.Boxes3D(
            centers=(low + high) / 2, sizes=high - low, colors=[240, 180, 76],
            labels=[f'{e["entity_id"]} · {e.get("label", "")}' for e in boxes]))

    def _finalize(self, rr, stream):
        final = self.final or {}
        mapping, entities = final.get("mapping"), final.get("entities", [])
        self._time(stream, time.monotonic())
        if mapping is not None:
            from .storage import save_map
            positions, colors, _ = mapping.cloud()
            stream.log("world/map", rr.Points3D(positions, colors=colors, radii=.03))
            stream.log("map/topdown", rr.Image(_topdown_image(mapping.topdown())))
            save_map(self.folder, mapping)
        self._entities(rr, stream, entities)
        stream.log("log", rr.TextLog(f"Recording finished: {final.get('reason', 'unknown')}"))
        (self.folder / "entities.json").write_text(json.dumps(
            {"world_frame": self.info.get("world_frame"), "entities": _json(entities)}, indent=2) + "\n")
        stream.flush(timeout_sec=10.)
        self._measure()
        ended = datetime.now().astimezone()
        with self.lock:
            self.reason = final.get("reason")
            self.ended_host_s = time.monotonic()
            manifest = {
                "format_version": 1, "kind": "scope-live-session",
                "started": self.started_wall.isoformat(timespec="seconds"),
                "ended": ended.isoformat(timespec="seconds"),
                "duration_s": round(self.ended_host_s - self.started_host_s, 2),
                "stopped_by": self.reason, **self.info,
                "samples": self.samples, "dropped_samples": self.dropped,
                "recording_bytes": self.bytes, "max_recording_bytes": self.max_bytes,
                "sample_hz": round(1. / self.period_s, 3),
                "files": {"recording": RECORDING,
                          "map": "map.npz" if mapping is not None else None,
                          "entities": "entities.json"},
                "map_revision": None if mapping is None else int(mapping.revision),
                "map_config": None if mapping is None else asdict(mapping.config),
            }
        (self.folder / "manifest.json").write_text(json.dumps(_json(manifest), indent=2) + "\n")
        with self.lock:
            self.state = "saved"


def _topdown_image(state):
    state = np.asarray(state, dtype=np.uint8).T[::-1]
    image = np.empty(state.shape + (3,), dtype=np.uint8)
    image[...] = [57, 64, 76]  # Same palette as the offline mapping recording
    image[state == FREE] = [65, 153, 166]
    image[state == OCCUPIED] = [235, 156, 74]
    return image


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
