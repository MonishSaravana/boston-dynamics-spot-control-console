"""Browser integration of existing console adapters; no new perception or planner.

A single worker owns perception and target mutations. HTTP reads cached views;
Stop and the movement-presence watchdog bypass expensive rendering/inference.
"""
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import asdict, replace
import io
import json
import math
from pathlib import Path
import queue
import sqlite3
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageDraw

from .console_backend import DemoBackend, SpotBackend, camera_box, draw_perception_overlay
from .live_interaction import DryRunExecutor, InteractionSession, RobotPose, VirtualExecutor
from .runtime import ModuleConfig

MODES = ('observe', 'dry_run', 'robot_control')


class GuardedDispatch:
    """Authorize only final dispatch, leaving route checks interruptible by Stop."""
    def __init__(self, workspace, executor, epoch, controller):
        self.workspace, self.executor = workspace, executor
        self.epoch, self.controller = epoch, controller

    def execute(self, destination):
        w = self.workspace
        # Same lock order as manual Stop/drive. A held-key request and GO cannot
        # both cross this boundary, and Stop cancels a GO still being planned.
        with w.control.command_lock, w.authority_lock:
            if self.epoch != w.epoch or w.mode == 'observe':
                raise ValueError('Stop or a mode change cancelled this GO request')
            if w.control.held or w.control.gesture_active or w.control.failed:
                raise ValueError('Release manual controls and disable gestures before GO')
            if w.control.disconnecting:
                raise ValueError('Disconnection cancelled this GO request')
            manual = w.control.session if w.mode == 'robot_control' else None
            if w.mode == 'robot_control' and not manual:
                raise ValueError('Fresh standing state is required before GO')
            with manual.lock if manual else nullcontext():
                if manual and (not w.control.connected or not w.control.armed or
                        not manual.armed or not manual.powered):
                    raise ValueError('Fresh standing state is required before GO')
                if manual and (manual.pending is not None or manual.active_action is not None or
                               manual.keys or manual.gesture_mode):
                    raise ValueError('Wait for the pending manual request before GO')
                result = self.executor.execute(destination)
                if w.backend.mode == 'SPOT_SUPERVISED':
                    with w.lock:
                        w.navigation_controller = self.controller
                        w.last_presence = time.monotonic()
                return result

    def stop(self):
        self.executor.stop()


def clean(value):
    """JSON-safe, bounded telemetry. Non-finite measurements are unavailable."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def encode(picture):
    out = io.BytesIO()
    picture.convert('RGB').save(out, 'JPEG', quality=82)
    return out.getvalue()


class RunLibrary:
    """Explicit local roots; never arbitrary host filesystem browsing."""
    def __init__(self, roots):
        self.roots = [Path(p).resolve() for p in roots]

    def _allowed(self, path):
        path = Path(path).resolve()
        if not any(path.is_relative_to(root) for root in self.roots):
            raise ValueError('Artifact is outside the configured run directories')
        return path

    def scan(self):
        result = []
        for index, root in enumerate(self.roots):
            if not root.is_dir():
                continue
            for path in sorted(root.rglob('*')):
                if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                    continue
                if path.suffix not in ('.rrd', '.json', '.sqlite', '.db', '.npz', '.ply'):
                    continue
                if any(word in path.name.lower() for word in ('credential', 'secret', 'token')):
                    continue
                stat = path.stat()
                result.append({'id': f'{index}/{path.relative_to(root).as_posix()}',
                    'name': path.name, 'run': path.parent.relative_to(root).as_posix(),
                    'kind': {'.rrd': 'Viewer recording', '.sqlite': 'Memory', '.db': 'Memory',
                             '.npz': 'Map / episode', '.ply': 'Mesh'}.get(path.suffix, 'Evidence / result'),
                    'bytes': stat.st_size, 'modified_s': stat.st_mtime})
        return sorted(result, key=lambda item: item['modified_s'], reverse=True)[:500]

    def resolve(self, key):
        index, relative = key.split('/', 1)
        if not index.isdigit() or int(index) >= len(self.roots):
            raise ValueError('Unknown run directory')
        path = self._allowed(self.roots[int(index)] / relative)
        if not path.is_file() or path.suffix not in ('.rrd', '.json', '.sqlite', '.db', '.npz', '.ply'):
            raise ValueError('Artifact unavailable')
        if any(word in path.name.lower() for word in ('credential', 'secret', 'token')):
            raise ValueError('Artifact unavailable')
        return path

    def inspect(self, key):
        path = self.resolve(key)
        if path.suffix == '.json':
            if path.stat().st_size > 2_000_000:
                raise ValueError('Result exceeds the 2 MB inspection limit; use the CLI')
            return clean(json.loads(path.read_text()))
        if path.suffix in ('.sqlite', '.db'):
            with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
                # SCOPE's immutable memory schema, read only and bounded.
                return {name: [json.loads(row[0]) for row in db.execute(
                    f'SELECT payload FROM {name} ORDER BY rowid DESC LIMIT 100')]
                    for name in ('episodes', 'beliefs', 'events')}
        if path.name == 'map.npz':
            from .storage import load_map
            mapping = load_map(path.parent)
            states = mapping.states()
            return {'config': asdict(mapping.config), 'revision': mapping.revision,
                    'unknown': int((states == 0).sum()), 'free': int((states == 1).sum()),
                    'occupied': int((states == 2).sum()),
                    'viewer_command': f'{sys.executable} -m scope view {path.parent}'}
        return {'path': str(path), 'bytes': path.stat().st_size,
                'note': 'Open Rerun for recordings; use the existing CLI to replay RGB-D or meshes.'}

    def open_viewer(self, key):
        path = self.resolve(key)
        if path.suffix != '.rrd':
            raise ValueError('Select a .rrd viewer recording')
        from .viewer import open_recording
        open_recording(path)
        return 'Opened the local Rerun recording'


class Workspace:
    def __init__(self, control, *, demo=False, runs_dirs=()):
        self.control = control
        self.lock = threading.RLock()
        self.authority_lock = threading.RLock()
        self.closed = threading.Event()
        self.invalidation = threading.Event()
        self.tasks = queue.Queue(maxsize=32)
        self.backend = None
        self.interaction = None
        self.mode = 'dry_run' if demo else 'observe'
        self.epoch = 0
        self.revision = 0
        self.version = 0
        self.images = {}
        self.cached = {}
        self.message = 'Starting synthetic room' if demo else 'Connect sensors or start the offline demo'
        self.query_key = None
        self.navigation_controller = None
        self.last_presence = 0.
        self.library = RunLibrary(runs_dirs)
        self.thread = threading.Thread(target=self._run, name='scope-workspace', daemon=True)
        self.thread.start()
        self.initialization = None
        if demo:
            self.initialization = self.submit('demo', {'enabled': True})

    def submit(self, name, data):
        future = Future()
        if self.control.disconnecting and name != 'disconnect':
            raise ValueError('Disconnection is in progress')
        if name == 'disconnect':
            # A disconnect follows Stop and supersedes queued work, even when
            # every normal request slot is occupied.
            while True:
                try:
                    _, _, _, cancelled = self.tasks.get_nowait()
                except queue.Empty:
                    break
                cancelled.set_exception(ValueError('Disconnected; pending request cancelled'))
        with self.lock:
            epoch = self.epoch
        try:
            self.tasks.put_nowait((name, data, epoch, future))
        except queue.Full:
            raise ValueError('Workspace busy; wait for the current request')
        return future

    def snapshot(self):
        with self.lock:
            return {**self.cached, 'mode': self.mode, 'epoch': self.epoch,
                    'revision': self.revision, 'version': self.version, 'message': self.message,
                    'available': self.backend is not None,
                    'navigation_active': self.navigation_controller is not None}

    def image(self, name):
        with self.lock:
            return self.images.get(name)

    def stop(self):
        # No perception/render lock. Advance epoch before any later dispatch.
        with self.authority_lock:
            with self.lock:
                self.epoch += 1
                self.navigation_controller = None
                backend = self.backend
                self.message = 'Stop requested; class E-stop remains separate'
            if backend is not None:
                try:
                    backend.executor.stop()
                except Exception as exc:
                    self.message = f'Stop request failed: {type(exc).__name__}; use the class E-stop'
        # Stop must also work with a full request queue. The worker consumes
        # this event before the next action, clearing any previous approval.
        self.invalidation.set()

    def presence(self, controller):
        with self.lock:
            if controller != self.navigation_controller:
                raise ValueError('Navigation is not active in this tab')
            self.last_presence = time.monotonic()

    def watchdog(self):
        with self.lock:
            expired = (self.navigation_controller is not None and
                       time.monotonic() - self.last_presence > .30)
        if expired:
            self.control.stop()

    def _replace(self, backend):
        old = self.backend
        with self.authority_lock:
            with self.lock:
                self.backend = backend
                self.interaction = InteractionSession(backend.executor) if backend else None
                self.navigation_controller = None
                self.epoch += 1
                self.revision += 1
                self.query_key = None
                self.invalidation.clear()
                self.cached = {}
                self.images = {}
                self.version += 1
        if old is not None:
            old.close()

    def _connect(self, data, epoch):
        if self.control.demo or self.backend is not None or self.control.session is not None:
            raise ValueError('Leave demo or disconnect before connecting')
        hostname, username, password = (data.get(k, '') for k in ('hostname', 'username', 'password'))
        if (not all(isinstance(v, str) and v for v in (hostname, username, password)) or
                len(hostname) > 255 or any(c.isspace() for c in hostname)):
            raise ValueError('Enter a valid robot address, username and password')
        mode = data.get('mode', 'observe')
        if mode not in MODES:
            raise ValueError('Unknown control mode')
        with self.control.lock:
            self.control.disconnecting = False
        self.mode = mode
        self.message = 'Connecting to Spot sensors…'
        from .spot_live_source import SpotReadOnlySource
        if mode == 'robot_control':
            if data.get('command_authority') is not True:
                raise ValueError('Robot control requires explicit command authority and the class E-stop')
            self.control.connect(hostname, username, password)
            deadline = time.monotonic() + 20
            while not self.control.connected and not self.control.failed:
                if self.closed.wait(.1) or time.monotonic() > deadline or self.epoch != epoch:
                    self.control.disconnect()
                    raise ValueError('Connection cancelled or timed out')
            if self.control.failed:
                self.control.disconnect()
                raise ValueError('Robot connection failed; check the connection status')
            from bosdyn.client.image import ImageClient
            session = self.control.session
            source = SpotReadOnlySource(session.robot.ensure_client(ImageClient.default_service_name),
                                        session.state_client, session.robot)
            # Acquisition/display policy lives in the shared sensor adapter;
            # the legacy gesture worker still owns its required front pair.
            with session.lock:
                session.sources = []
        else:
            source = SpotReadOnlySource.connect(hostname, username=username, password=password)
        if self.closed.is_set() or self.epoch != epoch:
            raise ValueError('Connection cancelled')
        visual = data.get('visual_source') or next((n for n in source.sources if 'fisheye_image' in n), '')
        backend = SpotBackend(hostname, visual, data.get('depth_source') or None,
            data.get('alignment_verified') is True, data.get('human_pose') is True, source=source)
        self._enable_executor(backend)
        self._replace(backend)
        self.message = 'Connected; one visual/depth pair is processed'

    def _enable_executor(self, backend):
        if self.mode == 'robot_control' and self.control.session is not None:
            from .spot_navigation import AsyncSpotExecutor, SpotTrajectoryExecutor
            backend.executor = AsyncSpotExecutor(SpotTrajectoryExecutor(
                self.control.session.client, backend.source.robot_pose,
                backend._route_clear, enabled=True))
            backend.mode = 'SPOT_SUPERVISED'

    def _action(self, name, data, epoch):
        if name == 'invalidate':
            if self.interaction:
                self.interaction.stop()
            return
        if name == 'disconnect':
            self._replace(None)
            self.message = 'Disconnected'
            return
        if epoch != self.epoch:
            raise ValueError('Workspace changed; refresh and try again')
        if name == 'demo':
            self._replace(DemoBackend() if data['enabled'] else None)
            self.mode = 'dry_run' if data['enabled'] else 'observe'
            self.message = 'Synthetic room; GO moves only the virtual robot' if data['enabled'] else 'Connect to Spot sensors'
            return
        if name == 'connect':
            return self._connect(data, epoch)
        if name == 'mode':
            mode = data.get('mode')
            if mode not in MODES:
                raise ValueError('Unknown control mode')
            if mode == 'robot_control' and (self.control.demo or self.control.session is None):
                raise ValueError('Robot control requires a new connection with command authority')
            self.control.stop()
            self.invalidation.clear()
            if self.interaction:
                self.interaction.stop()
            self.mode = mode
            if self.backend:
                if self.backend.mode == 'SPOT_SUPERVISED':
                    self.backend.executor.close()
                    self.backend.mode = 'SPOT_SENSORS_DRY_RUN'
                self.backend.executor = (VirtualExecutor() if self.control.demo and mode == 'dry_run'
                                         else DryRunExecutor())
                self._enable_executor(self.backend)
                self.interaction = InteractionSession(self.backend.executor)
            self.epoch += 1
            self.revision += 1
            return
        b, session = self.backend, self.interaction
        if b is None:
            raise ValueError('Connect sensors or start the offline demo first')
        if self.navigation_controller is not None:
            raise ValueError('Stop the current GO before changing targets, cameras or modules')
        if name in ('select', 'confirm', 'reject', 'preview', 'go'):
            if data.get('revision') != self.revision:
                raise ValueError('Target selection changed; inspect the current evidence first')
        if name == 'query':
            phrase = data.get('phrase', '').strip()
            if not phrase or len(phrase) > 160:
                raise ValueError('Enter an object phrase of 1–160 characters')
            result = b.query(phrase)
            session.offer(result if b.mode == 'SIMULATED' else [], ambiguous=b.mode == 'SIMULATED' and len(result) > 1)
            if b.mode != 'SIMULATED':
                session.message = 'PENDING · local visual query'
            self.query_key = None
        elif name == 'point':
            candidates, unresolved = b.pointing()
            session.offer(candidates, ambiguous=unresolved)
            if unresolved:
                session.message = 'Pointing unresolved; inspect evidence or type a target'
        elif name == 'select':
            entity_id = data.get('entity_id')
            if any(c.entity_id == entity_id for c in session.candidates):
                session.select(entity_id)
            else:
                session.offer([b.direct(entity_id)])
        elif name == 'confirm':
            session.confirm()
            if b.robot is None:
                raise ValueError('Target confirmed; waiting for measured robot pose before preview')
            self._mapping_enabled()
            session.propose(b.robot, b.mapping)
        elif name == 'reject':
            session.reject()
        elif name == 'preview':
            self._mapping_enabled()
            session.propose(b.robot, b.mapping)
        elif name == 'go':
            if self.mode == 'observe':
                raise ValueError('Observe has no movement authorization; choose Dry run or Robot control')
            controller = data.get('controller')
            if not isinstance(controller, str) or not 0 < len(controller) <= 80:
                raise ValueError('Invalid browser control session')
            executor = session.executor
            session.executor = GuardedDispatch(self, executor, epoch, controller)
            try:
                self._mapping_enabled()
                result = session.go(b.robot, b.mapping, b.ready_for_go)
                if b.mode == 'SIMULATED' and getattr(b.executor, 'pose', None):
                    x, y, yaw = b.executor.pose
                    b.robot = RobotPose(x, y, yaw, time.monotonic(), b.world_frame)
            finally:
                session.executor = executor
            self.message = result
        elif name == 'camera':
            if b.mode == 'SIMULATED':
                raise ValueError('Synthetic camera policy is fixed')
            b.source.configure(data.get('source'), acquire=data.get('acquire'),
                               display=data.get('display'), max_hz=data.get('max_hz'))
            session.invalidate('Camera settings changed; inspect fresh evidence')
            if self.control.session and self.control.gesture_active:
                self.control.stop()
        elif name == 'pair':
            if b.mode == 'SIMULATED':
                raise ValueError('Synthetic pair is fixed')
            visual, depth = data.get('visual_source'), data.get('depth_source') or None
            if visual not in b.source.sources or (depth and depth not in b.source.sources):
                raise ValueError('Choose available visual and depth sources')
            b.source.alignment_verified = data.get('alignment_verified') is True
            policies = {name: replace(policy) for name, policy in b.source.policy.items()}
            # Close previous workers before changing the selected processing pair.
            b.pool.shutdown(wait=True, cancel_futures=True)
            new = SpotBackend('', visual, depth, b.source.alignment_verified,
                              b.human_pose, source=b.source)
            for name, policy in policies.items():
                new.source.configure(name, acquire=policy.acquire or name in (visual, depth),
                    display=policy.display, process=name in (visual, depth), max_hz=policy.max_hz)
            self._enable_executor(new)
            self._replace(new)
            return
        elif name == 'module':
            if b.mode == 'SIMULATED' or b.pipeline is None:
                raise ValueError('Live module controls require a running perception pipeline')
            module = data.get('module')
            if module not in b.pipeline.runtime.modules:
                raise ValueError('Unknown module')
            previous = b.pipeline.runtime.state(module).config
            enabled = data.get('enabled', previous.enabled)
            if not isinstance(enabled, bool):
                raise ValueError('Module enabled must be boolean')
            if module == 'human_pose' and enabled and b.pipeline.core.pose_detector is None:
                raise ValueError('Enable human pose when connecting; its model is loaded on demand')
            b.pipeline.configure(module, replace(previous, enabled=enabled,
                max_hz=float(data.get('max_hz', previous.max_hz or 5))))
            session.invalidate('Module settings changed; inspect fresh evidence')
        else:
            raise ValueError('Unknown workspace action')
        self.revision += 1
        self.message = session.message

    def _mapping_enabled(self):
        b = self.backend
        if b.mode != 'SIMULATED' and (b.pipeline is None or
                not b.pipeline.runtime.state('mapping').config.enabled):
            raise ValueError('Mapping is disabled; no destination authorization')

    def _tick(self):
        b, session = self.backend, self.interaction
        if b is None:
            return
        b.tick()
        if b.mode == 'SIMULATED' and session.selected:
            session.refresh_observation(replace(session.selected, observed_host_s=time.monotonic(),
                                               source_timestamp_s=b.frame.timestamp_s))
        elif b.mode != 'SIMULATED':
            if session.selected:
                track = b.tracked_candidate(session.selected)
                if track:
                    session.refresh_observation(track)
            candidates = b.query_candidates()
            if candidates is not None:
                key = tuple((c.entity_id, c.source_timestamp_s, c.score) for c in candidates)
                if key != self.query_key:
                    self.query_key = key
                    if session.confirmed:
                        refreshed = next((c for c in candidates if c.entity_id == session.selected.entity_id), None)
                        if refreshed:
                            session.refresh_observation(refreshed)
                        else:
                            session.invalidate('Target lost or query stale')
                    else:
                        session.offer(candidates, ambiguous=len(candidates) > 1)
                        self.revision += 1
            execution = b.execution_status()
            if execution and execution != 'IDLE':
                self.message = execution
                if not execution.startswith('MOVING'):
                    self.navigation_controller = None
            if b.error:
                self.message = b.error
        self._publish()

    def _publish(self):
        b, session = self.backend, self.interaction
        now = time.monotonic()
        if self.mode == 'robot_control' and self.control.failed:
            session.invalidate('Robot connection failed; GO cancelled')
        c, p = session.selected, session.proposal
        fresh = c is not None and 0 <= now - c.observed_host_s <= 1.5
        def candidate(item):
            row = asdict(item)
            row['age_s'] = now - item.observed_host_s
            if item.bounds_low_m is not None:
                row['center_m'] = ((np.array(item.bounds_low_m) + item.bounds_high_m) / 2).tolist()
            return row
        mapping = b.mapping
        cameras = b.camera_status
        payload = {'simulated': b.mode == 'SIMULATED', 'backend_mode': b.mode,
            'world_frame': b.world_frame, 'robot': asdict(b.robot) if b.robot else None,
            'entities': b.entities, 'candidates': [candidate(item) for item in session.candidates],
            'selected': candidate(c) if c else None, 'confirmed': session.confirmed is not None,
            'destination': asdict(p) if p else None,
            'can_confirm': fresh and session.confirmed is None,
            'can_go': (fresh and p is not None and 0 <= now-p.proposed_host_s <= 10 and
                       self.mode != 'observe' and not self.control.failed),
            'cameras': cameras,
            'visual_source': getattr(b, 'visual_source', 'synthetic-front'),
            'depth_source': getattr(b, 'depth_source', 'synthetic-depth'),
            'alignment_verified': b.mode == 'SIMULATED' or b.source.alignment_verified,
            'human_pose': getattr(b, 'human_pose', False),
            'timings': {**b.timings, 'interaction': session.timings.snapshot()},
            'modules': b.snapshot.get('telemetry', {}) if b.mode != 'SIMULATED' and b.snapshot else {},
            'mapping': {'revision': mapping.revision, 'config': asdict(mapping.config),
                        'topdown': mapping.topdown().tolist()} if mapping else None}
        if b.mode != 'SIMULATED':
            payload['timings']['sensors'] = b.source.timings.snapshot()
        images = {}
        if b.frame is not None:
            displayed = b.mode == 'SIMULATED' or (b.source.policy[b.visual_source].display and
                                                 b.source.policy[b.visual_source].acquire)
            if displayed:
                picture = Image.fromarray(b.frame.rgb)
                draw_perception_overlay(picture, b.frame, b)
                box = camera_box(c, b.frame) if c else None
                if box:
                    draw = ImageDraw.Draw(picture)
                    draw.rectangle(box, outline='#edc76e', width=3)
                    draw.text((box[0]+4, box[1]+4), c.entity_id, fill='#edc76e')
                images['evidence'] = encode(picture)
        if b.mode != 'SIMULATED':
            manual_pictures = {}
            for name, visual in list(b.source.latest.items()):
                if b.source.policy[name].display and b.source.policy[name].acquire:
                    pixels = visual.pixels
                    if pixels.dtype == np.uint16:
                        depth = pixels[:, :, 0].astype(float) / (visual.depth_scale or 1000.)
                        pixels = np.uint8(np.clip(depth / 5., 0, 1) * 255)
                    elif pixels.ndim == 3 and pixels.shape[2] == 1:
                        pixels = pixels[:, :, 0]
                    picture = Image.fromarray(pixels)
                    if name in self.control.cameras:
                        from spot_control_gui import ROTATION
                        preview = picture.rotate(ROTATION.get(name, 0), expand=True)
                        preview.thumbnail((720, 480))
                        manual_pictures[name] = preview
                    picture.thumbnail((720, 480))
                    images[name] = encode(picture)
                    b.source.mark_displayed(name, visual.acquisition_robot_s)
            # Keep the existing approximate panorama worker and manual gesture
            # camera presentation. No second camera acquisition/fusion path.
            manual = self.control.session
            if manual is not None and manual_pictures:
                with manual.lock:
                    manual.latest_frames.update(manual_pictures)
                    manual.frame_times.update({name: b.source.latest[name].receive_host_s
                                               for name in manual_pictures})
                    auto_due = manual.auto_panorama and now - manual.last_panorama_request >= 2.
                if auto_due:
                    manual.request_panorama()
        with self.lock:
            self.cached = clean(payload)
            self.images = images
            self.version += 1

    def _run(self):
        while not self.closed.is_set():
            try:
                task = self.tasks.get(timeout=.2)
            except queue.Empty:
                task = None
            if self.invalidation.is_set():
                self.invalidation.clear()
                if self.interaction:
                    self.interaction.stop()
                    self.revision += 1
            if task:
                name, data, epoch, future = task
                try:
                    self._action(name, data, epoch)
                    self._tick()
                    future.set_result({'ok': True})
                    continue
                except Exception as exc:
                    self.message = f'{type(exc).__name__}: {exc}'
                    if name == 'connect' and self.control.session is not None:
                        self.control.disconnect()
                    future.set_exception(ValueError(str(exc)))
            try:
                self._tick()
            except Exception as exc:
                self.message = f'Degraded: {type(exc).__name__}: {exc}'
                if self.interaction:
                    self.interaction.invalidate(self.message)
                # Failure never permits continued supervised movement.
                if self.navigation_controller:
                    self.control.stop()
        if self.backend:
            self.backend.close()

    def close(self):
        self.stop()
        self.closed.set()
        self.thread.join(timeout=5)
