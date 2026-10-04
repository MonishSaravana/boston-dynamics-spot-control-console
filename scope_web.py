#!/usr/bin/env python3
"""Local-only browser console for SCOPE. Live commands still use SpotSession."""

import argparse
from functools import lru_cache
import importlib.util
import io
import json
import math
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageDraw

from spot_demo import make_demo_frames, make_demo_panorama


ROOT = Path(__file__).resolve().parent
WEB = ROOT / 'web'
CAMERAS = ('frontleft_fisheye_image', 'frontright_fisheye_image',
           'left_fisheye_image', 'right_fisheye_image', 'back_fisheye_image')
DRIVE_KEYS = frozenset('wasd')
DRIVE_TIMEOUT = 0.30
MAX_BODY = 4096


def jpeg_bytes(picture):
    output = io.BytesIO()
    picture.save(output, format='JPEG', quality=76)
    return output.getvalue()


class Event:
    def __init__(self, callback):
        self.callback = callback

    def emit(self, *args):
        self.callback(*args)


class WebSignals:
    """Match the signal surface used by SpotSession without a Qt event loop."""

    def __init__(self, app, generation):
        for name in ('status', 'connected', 'powered', 'armed', 'model_state', 'failed',
                     'frame', 'panorama', 'panorama_status', 'posture_status',
                     'gesture_info', 'gesture_off'):
            setattr(self, name, Event(
                lambda *args, name=name: app.receive(generation, name, *args)))


class WebControl:
    def __init__(self, demo=False):
        self.demo = demo
        self.lock = threading.RLock()
        self.command_lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        self.session = None
        self.connected = False
        self.powered = False
        self.armed = False
        self.failed = False
        self.generation = 0
        self.gesture_active = False
        self.gesture_available = False
        self.status = 'Offline demo • no robot connection' if demo else 'Enter Spot details to connect'
        self.posture_status = 'Adjust a request, then apply after Stand is confirmed.'
        self.gesture_status = 'OFF • manual keyboard control'
        self.panorama_status = 'Approximate front stitch; not calibrated'
        self.model_status = 'SIMULATED • URDF zero configuration, not measured robot state' if demo else 'Waiting for fresh joint telemetry'
        self.model_angles = None
        self.model_version = 0
        self.frames = {}
        self.frame_versions = {}
        self.panorama = None
        self.panorama_version = 0
        self.cameras = list(CAMERAS) if demo else []
        self.controller = None
        self.gesture_controller = None
        self.control_epoch = 0
        self.last_drive = 0.0
        self.last_presence = 0.0
        self.drive_seq = -1
        self.held = set()
        self.closed = threading.Event()
        if demo:
            self._load_demo()
        self.watchdog = threading.Thread(target=self._watch_drive, daemon=True)
        self.watchdog.start()

    def snapshot(self):
        with self.lock:
            return {
                'demo': self.demo, 'connected': self.connected, 'powered': self.powered,
                'armed': self.armed, 'failed': self.failed,
                'gesture_active': self.gesture_active,
                'gesture_available': self.gesture_available,
                'control_epoch': self.control_epoch,
                'status': self.status,
                'posture_status': self.posture_status, 'gesture_status': self.gesture_status,
                'panorama_status': self.panorama_status, 'model_status': self.model_status,
                'cameras': list(self.cameras), 'frame_versions': dict(self.frame_versions),
                'model_version': self.model_version, 'has_panorama': self.panorama is not None,
                'panorama_version': self.panorama_version,
                'e_stop_note': 'Keep the separate class E-stop available. STOP MOVEMENT only requests zero velocity.',
            }

    def receive(self, generation, name, *args):
        with self.lock:
            if generation != self.generation:
                return
        getattr(self, f'on_{name}')(*args)

    def _load_demo(self):
        self.frames = {name: jpeg_bytes(frame) for name, frame in make_demo_frames().items()}
        self.frame_versions = {name: 1 for name in self.frames}
        self.panorama = jpeg_bytes(make_demo_panorama())
        self.panorama_version += 1
        self.cameras = list(CAMERAS)
        try:
            from spot_model_view import JOINT_NAMES, SDK_URDF
            if SDK_URDF.is_file():
                self.model_angles = {name: 0.0 for name in JOINT_NAMES}
                self.model_status = 'SIMULATED • URDF zero configuration, not measured robot state'
            else:
                self.model_angles = None
                self.model_status = 'SDK reference mesh unavailable in this installation'
        except ImportError:
            self.model_angles = None
            self.model_status = 'SDK reference mesh unavailable in this environment'
        self.model_version += 1

    def set_demo(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError('Invalid demo setting')
        with self.lock:
            if self.session is not None:
                raise ValueError('Disconnect from Spot before changing mode')
            self.demo = enabled
            self.connected = self.powered = self.armed = self.failed = False
            self.gesture_active = False
            self.gesture_available = False
            self.held.clear()
            self.controller = None
            self.gesture_controller = None
            self.control_epoch += 1
            if enabled:
                self.status = 'Offline demo • no robot connection'
                self._load_demo()
            else:
                self.status = 'Enter Spot details to connect'
                self.frames.clear()
                self.frame_versions.clear()
                self.panorama = None
                self.cameras = []
                self.model_angles = None
                self.model_status = 'Waiting for fresh joint telemetry'
                self.model_version += 1

    def connect(self, hostname, username, password):
        with self.command_lock:
            return self._connect_locked(hostname, username, password)

    def _connect_locked(self, hostname, username, password):
        if self.demo:
            raise ValueError('Demo mode cannot connect to a robot')
        if not hostname or not username or not password or len(hostname) > 255:
            raise ValueError('Enter the robot address, username, and password')
        if any(char.isspace() for char in hostname):
            raise ValueError('Robot address must not contain spaces')
        with self.lock:
            if self.session is not None:
                raise ValueError('A connection is already active; disconnect first')
            # Import here so the offline demo can run without the robot SDK.
            from spot_control_gui import SpotSession
            self.generation += 1
            self.session = SpotSession(hostname, WebSignals(self, self.generation), username, password)
            self.status = 'Connecting to Spot…'
            self.failed = False
            self.session.start()

    def disconnect(self):
        with self.command_lock:
            return self._disconnect_locked()

    def _disconnect_locked(self):
        self.stop()
        with self.lock:
            session, self.session = self.session, None
            self.generation += 1
            self.connected = self.powered = self.armed = False
            self.gesture_active = False
            self.gesture_available = False
            self.cameras = [] if not self.demo else list(CAMERAS)
            self.status = 'Disconnected from Spot'
            self.model_angles = None
            self.model_version += 1
        if session is not None:
            session.close()

    def stop(self):
        with self.command_lock:
            return self._stop_locked()

    def _stop_locked(self):
        with self.lock:
            self.held.clear()
            self.controller = None
            self.drive_seq = -1
            self.gesture_active = False
            self.gesture_controller = None
            self.control_epoch += 1
            session = self.session
        if session is not None:
            session.set_gesture_mode(False)
            session.action('stop')

    def drive(self, controller, sequence, keys, released, epoch):
        with self.command_lock:
            return self._drive_locked(controller, sequence, keys, released, epoch)

    def _drive_locked(self, controller, sequence, keys, released, epoch):
        if not isinstance(controller, str) or len(controller) > 80 or not controller:
            raise ValueError('Invalid control session')
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise ValueError('Invalid drive sequence')
        if not isinstance(keys, list) or set(keys) - DRIVE_KEYS:
            raise ValueError('Invalid direction keys')
        now = time.monotonic()
        with self.lock:
            if epoch != self.control_epoch:
                raise ValueError('Control session changed; release keys and try again')
            if not self.session or not self.connected or not self.armed or self.failed or self.gesture_active:
                raise ValueError('Movement is locked until fresh standing state is confirmed')
            if self.controller not in (None, controller) and now - self.last_drive < DRIVE_TIMEOUT:
                raise ValueError('Another browser tab is controlling movement')
            if self.controller != controller:
                self.drive_seq = -1
            if sequence <= self.drive_seq:
                return
            self.controller = controller
            self.drive_seq = sequence
            self.last_drive = now
            self.held = set(keys)
            session = self.session
        # The SDK worker performs its own fresh readiness and final command checks.
        from PySide6.QtCore import Qt
        key_codes = {getattr(Qt, f'Key_{key.upper()}') for key in keys}
        session.set_keys(key_codes, released=bool(released))

    def presence(self, controller):
        with self.lock:
            if not self.gesture_active or controller != self.gesture_controller:
                raise ValueError('Gesture control is not active in this tab')
            self.last_presence = time.monotonic()

    def _watch_drive(self):
        while not self.closed.wait(0.05):
            with self.command_lock:
                with self.lock:
                    expired = bool(self.held) and time.monotonic() - self.last_drive > DRIVE_TIMEOUT
                    gesture_expired = (self.gesture_active and
                                       time.monotonic() - self.last_presence > DRIVE_TIMEOUT)
                    if expired:
                        self.held.clear()
                        self.controller = None
                        self.drive_seq = -1
                        self.control_epoch += 1
                        session = self.session
                    elif gesture_expired:
                        self.gesture_active = False
                        self.gesture_controller = None
                        self.control_epoch += 1
                        session = self.session
                    else:
                        session = None
                if session is not None:
                    if gesture_expired:
                        session.set_gesture_mode(False)
                    else:
                        session.set_keys(set(), released=True)
                    session.action('stop')

    def action(self, name, data):
        with self.command_lock:
            return self._action_locked(name, data)

    def _action_locked(self, name, data):
        if name == 'stop':
            self.stop()
            return
        with self.lock:
            if data.get('epoch') != self.control_epoch:
                raise ValueError('Control session changed; refresh before sending another request')
            session = self.session
            connected, powered, armed, failed = (
                self.connected, self.powered, self.armed, self.failed)
        if not session or not connected or failed:
            raise ValueError('Connect to Spot first')
        if name == 'power':
            if powered:
                raise ValueError('Spot is already powered on')
            session.action('power')
        elif name == 'stand':
            if not powered:
                raise ValueError('Power on before requesting Stand')
            if self.gesture_active:
                raise ValueError('Turn off gesture mode before requesting Stand')
            session.action('stand')
        elif name == 'speed':
            value = float(data.get('value', 0))
            if not math.isfinite(value) or not 0.05 <= value <= 0.35:
                raise ValueError('Speed must be 0.05–0.35 m/s')
            session.set_speed(value)
        elif name == 'posture':
            if not armed:
                raise ValueError('Stand must be confirmed before applying posture')
            values = [data.get(key) for key in ('height', 'roll', 'pitch')]
            if any(not isinstance(v, int) or isinstance(v, bool) for v in values):
                raise ValueError('Posture values must be whole numbers')
            if not session.apply_posture(*values):
                raise ValueError('Standing posture request was blocked')
        elif name == 'gesture':
            if not isinstance(data.get('enabled'), bool):
                raise ValueError('Invalid gesture mode setting')
            if data['enabled'] and not self.gesture_available:
                raise ValueError('Gesture mode needs aligned depth, MediaPipe, and local model files')
            if data['enabled'] and (not isinstance(data.get('controller'), str) or
                                    not 0 < len(data['controller']) <= 80):
                raise ValueError('Invalid control session')
            self.stop()
            with self.lock:
                self.gesture_active = data['enabled']
                self.gesture_controller = data.get('controller') if data['enabled'] else None
                self.last_presence = time.monotonic()
            session.set_gesture_mode(data['enabled'])
        elif name == 'stitch':
            session.request_panorama()
        elif name == 'auto_panorama':
            if not isinstance(data.get('enabled'), bool):
                raise ValueError('Invalid panorama setting')
            session.set_auto_panorama(data['enabled'])
        else:
            raise ValueError('Unknown action')

    def on_status(self, message):
        with self.lock:
            self.status = message

    def on_connected(self, cameras, powered):
        from spot_gesture import MODEL_DIR
        with self.lock:
            self.connected = True
            self.powered = powered
            self.cameras = list(cameras)
            session = self.session
            depth = bool(session and session.gesture_depth_source)
            models = (importlib.util.find_spec('mediapipe') is not None and
                      (MODEL_DIR / 'gesture_recognizer.task').is_file() and
                      (MODEL_DIR / 'pose_landmarker_lite.task').is_file())
            self.gesture_available = depth and models
            if not depth:
                self.gesture_status = 'OFF • aligned front depth unavailable'
            elif not models:
                self.gesture_status = 'OFF • MediaPipe or gesture model files missing'
            else:
                self.gesture_status = 'OFF • manual keyboard control'
        if session is not None:
            session.set_auto_panorama(True)

    def on_powered(self):
        with self.lock:
            self.powered = True

    def on_armed(self):
        with self.lock:
            self.armed = True

    def on_model_state(self, angles, message):
        with self.lock:
            self.model_angles = angles
            self.model_status = message
            self.model_version += 1

    def on_failed(self, message):
        with self.lock:
            self.failed = True
            self.armed = False
            self.held.clear()
            self.controller = None
            self.gesture_active = False
            self.gesture_controller = None
            self.gesture_status = 'OFF • connection problem'
            self.control_epoch += 1
            self.status = message

    def on_frame(self, pictures):
        encoded = {name: jpeg_bytes(picture) for name, picture in pictures.items()}
        with self.lock:
            self.frames.update(encoded)
            for name in encoded:
                self.frame_versions[name] = self.frame_versions.get(name, 0) + 1

    def on_panorama(self, picture):
        with self.lock:
            self.panorama = jpeg_bytes(picture)
            self.panorama_version += 1

    def on_panorama_status(self, message):
        with self.lock:
            self.panorama_status = message

    def on_posture_status(self, message):
        with self.lock:
            self.posture_status = message

    def on_gesture_info(self, message):
        with self.lock:
            self.gesture_status = message
            self.gesture_active = message.startswith('ON')

    def on_gesture_off(self, message):
        with self.lock:
            self.gesture_status = message
            self.gesture_active = False
            self.gesture_controller = None

    def close(self):
        self.closed.set()
        self.disconnect()


def model_png(angles, message):
    """Render the SDK reference geometry read-only, or a labeled empty state."""
    width, height = 720, 420
    picture = Image.new('RGB', (width, height), '#111e26')
    draw = ImageDraw.Draw(picture)
    draw.text((18, 15), message[:96], fill='#dbe8ec')
    draw.text((18, height - 24), 'SDK base URDF • read-only • body-frame grid, not floor contact',
              fill='#aebfc7')
    if angles is None:
        draw.text((70, height // 2), 'Robot geometry appears with fresh, verified joint telemetry',
                  fill='#cad6dc')
    else:
        try:
            import numpy as np
            mesh = _model_mesh()
            transforms = mesh.link_transforms(angles)
            az, el = math.radians(-55), math.radians(25)
            right = np.array([-math.sin(az), math.cos(az), 0.0])
            up = np.array([-math.sin(el)*math.cos(az), -math.sin(el)*math.sin(az), math.cos(el)])
            toward = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
            transformed = {name: vertices @ transforms[name][:3, :3].T + transforms[name][:3, 3]
                           for name, (vertices, _, _) in mesh.meshes.items()}
            points = np.concatenate(list(transformed.values()))
            raw = np.stack((points @ right, points @ up), axis=-1)
            minimum, maximum = raw.min(axis=0), raw.max(axis=0)
            span = np.maximum(maximum - minimum, .01)
            scale = min(width * .76 / span[0], height * .70 / span[1])
            center = np.array([width * .5, height * .53]) - (minimum + maximum) / 2 * np.array([scale, -scale])

            def project(values):
                xy = np.stack((values @ right, values @ up), axis=-1)
                return center + xy * np.array([scale, -scale])

            for coordinate in np.arange(-.8, .81, .2):
                for first, second in (((-.9, coordinate, -.53), (.9, coordinate, -.53)),
                                      ((coordinate, -.9, -.53), (coordinate, .9, -.53))):
                    a, b = project(np.array([first, second]))
                    draw.line((*a, *b), fill='#344650', width=1)
            triangles = []
            light = np.array([.4, -.5, .75])
            light /= np.linalg.norm(light)
            for name, (_, faces, color) in mesh.meshes.items():
                face_points = transformed[name][faces]
                normals = np.cross(face_points[:, 1]-face_points[:, 0],
                                   face_points[:, 2]-face_points[:, 0])
                visible = normals @ toward > 0
                face_points, normals = face_points[visible], normals[visible]
                if not len(face_points):
                    continue
                polygons = project(face_points.reshape(-1, 3)).reshape(-1, 3, 2)
                depths = face_points.mean(axis=1) @ toward
                shades = np.clip((normals @ light) / np.maximum(np.linalg.norm(normals, axis=1), 1e-9), 0, 1)
                for depth, polygon, shade in zip(depths, polygons, shades):
                    triangles.append((float(depth), polygon, color, float(shade)))
            triangles.sort(key=lambda item: item[0])
            for _, polygon, color, shade in triangles:
                rgb = tuple(int(int(color[i:i+2], 16) * (.52 + .48 * shade)) for i in (1, 3, 5))
                draw.polygon([tuple(point) for point in polygon], fill=rgb)
        except Exception as exc:
            draw.rectangle((0, 70, width, height - 50), fill='#111e26')
            draw.text((30, height // 2), f'SDK reference mesh unavailable: {exc}'[:100],
                      fill='#cad6dc')
    output = io.BytesIO()
    picture.save(output, format='PNG')
    return output.getvalue()


@lru_cache(maxsize=1)
def _model_mesh():
    from spot_model_view import SpotUrdfMesh
    return SpotUrdfMesh()


class Handler(BaseHTTPRequestHandler):
    server_version = 'SCOPE/0.1'

    @property
    def app(self):
        return self.server.app

    def log_message(self, format, *args):
        # Do not log URLs, robot addresses, or camera requests.
        pass

    def _send(self, status, content_type, data, extra=None):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy',
                         "default-src 'self'; img-src 'self' data: blob:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, payload):
        self._send(status, 'application/json; charset=utf-8', json.dumps(payload).encode())

    def _valid_host(self):
        host = self.headers.get('Host', '')
        return host in (f'127.0.0.1:{self.server.server_port}',
                        f'localhost:{self.server.server_port}')

    def do_GET(self):
        if not self._valid_host():
            return self._json(403, {'error': 'Invalid host'})
        path = urlsplit(self.path).path
        if path == '/':
            template = (WEB / 'index.html').read_text()
            body = template.replace('__SCOPE_TOKEN__', self.app.token).encode()
            return self._send(200, 'text/html; charset=utf-8', body)
        if path == '/favicon.ico':
            return self._send(204, 'image/x-icon', b'')
        if path in ('/style.css', '/app.js'):
            type_ = 'text/css' if path.endswith('.css') else 'text/javascript'
            return self._send(200, type_ + '; charset=utf-8', (WEB / path[1:]).read_bytes())
        if self.headers.get('X-SCOPE-Token') != self.app.token:
            return self._json(403, {'error': 'Invalid local session'})
        if path == '/api/state':
            return self._json(200, self.app.snapshot())
        if path.startswith('/api/frame/'):
            name = path.rsplit('/', 1)[-1]
            with self.app.lock:
                image = self.app.panorama if name == 'panorama' else self.app.frames.get(name)
            return self._send(200, 'image/jpeg', image) if image else self._json(404, {'error': 'Frame unavailable'})
        if path == '/api/model':
            with self.app.lock:
                angles, message = self.app.model_angles, self.app.model_status
            return self._send(200, 'image/png', model_png(angles, message))
        return self._json(404, {'error': 'Not found'})

    def do_POST(self):
        if not self._valid_host() or self.headers.get('Origin') != f'http://{self.headers.get("Host")}' or \
                self.headers.get('X-SCOPE-Token') != self.app.token:
            return self._json(403, {'error': 'Invalid local session'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 0 or length > MAX_BODY:
                raise ValueError('Request too large')
            data = json.loads(self.rfile.read(length) or b'{}')
            if not isinstance(data, dict):
                raise ValueError('Expected a JSON object')
            path = urlsplit(self.path).path
            if path == '/api/connect':
                self.app.connect(data.get('hostname', ''), data.get('username', ''),
                                 data.get('password', ''))
            elif path == '/api/mode':
                self.app.set_demo(data.get('demo'))
            elif path == '/api/disconnect':
                self.app.disconnect()
            elif path == '/api/drive':
                self.app.drive(data.get('controller'), data.get('sequence'),
                               data.get('keys'), data.get('released', False),
                               data.get('epoch'))
            elif path == '/api/presence':
                self.app.presence(data.get('controller'))
            elif path == '/api/action':
                self.app.action(data.get('name'), data)
            else:
                return self._json(404, {'error': 'Unknown command'})
            return self._json(200, {'ok': True})
        except (ValueError, TypeError) as exc:
            return self._json(400, {'error': str(exc)})
        except Exception:
            return self._json(500, {'error': 'Command failed; check connection status'})


def make_server(demo=False, port=0):
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.app = WebControl(demo=demo)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description='Run the SCOPE browser console on this laptop')
    parser.add_argument('--demo', action='store_true', help='Simulated images; no robot client or command')
    parser.add_argument('--port', type=int, default=0, help='Local port; 0 chooses an available port')
    parser.add_argument('--no-browser', action='store_true', help='Do not open the browser automatically')
    args = parser.parse_args()
    server = make_server(demo=args.demo, port=args.port)
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'SCOPE local console: {url}', flush=True)
    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.app.close()
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
