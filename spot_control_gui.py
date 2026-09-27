#!/usr/bin/env python3
"""Small, keyboard-driven Spot controller. Requires an independent class E-stop."""

import argparse
import io
import math
import threading
import time

import bosdyn.client
import bosdyn.client.util
import cv2
import numpy as np
from bosdyn.api import image_pb2, robot_state_pb2
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.geometry import EulerZXY
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QEvent, QObject, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QGridLayout, QHBoxLayout, QLabel,
                               QGraphicsScene, QGraphicsView, QMainWindow,
                               QPushButton, QSlider, QTabWidget, QVBoxLayout,
                               QWidget)
from spot_gesture import GestureGate, GestureVision, Observation


DEFAULT_SPEED = 0.20  # m/s, slower than the SDK WASD example's 0.5 m/s
MAX_SPEED = 0.35
COMMAND_DURATION = 0.35  # robot expires motion even if this program dies
COMMAND_PERIOD = 0.10
GESTURE_SPEED = 0.10
GESTURE_DURATION = 0.15
MAX_GESTURE_FRAME_AGE = 0.45
# Application limits, not certified physical limits. SDK tutorial shows +0.1 m;
# xbox_controller's +/-0.3 m is an example clamp, not a robot safety envelope.
HEIGHT_LIMIT_CM = 10
TILT_LIMIT_DEG = 5
PANORAMA_PERIOD = 2.0
FRONT_PAIR = ('frontleft_fisheye_image', 'frontright_fisheye_image')
CAMERAS = ('frontleft_fisheye_image', 'frontright_fisheye_image',
           'left_fisheye_image', 'right_fisheye_image', 'back_fisheye_image')
ROTATION = {'frontleft_fisheye_image': -78, 'frontright_fisheye_image': -102,
            'right_fisheye_image': 180}
KEYS = {Qt.Key_W: (1, 0), Qt.Key_Up: (1, 0),
        Qt.Key_S: (-1, 0), Qt.Key_Down: (-1, 0),
        Qt.Key_A: (0, 1), Qt.Key_Left: (0, 1),
        Qt.Key_D: (0, -1), Qt.Key_Right: (0, -1)}


def decode_visual_image(image):
    """Keep the source's grayscale or color format for live display."""
    if image.format == image_pb2.Image.FORMAT_JPEG:
        picture = Image.open(io.BytesIO(image.data))
        return picture.convert('L' if image.pixel_format ==
                               image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8 or
                               picture.mode == 'L' else 'RGB')
    if image.format == image_pb2.Image.FORMAT_RAW:
        modes = {image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8: 'L',
                 image_pb2.Image.PIXEL_FORMAT_RGB_U8: 'RGB',
                 image_pb2.Image.PIXEL_FORMAT_RGBA_U8: 'RGBA'}
        mode = modes.get(image.pixel_format)
        if mode:
            return Image.frombytes(mode, (image.cols, image.rows), image.data).convert(
                'RGB' if mode == 'RGBA' else mode)
    raise ValueError('Unsupported visual camera image format')


class UiSignals(QObject):
    status = Signal(str)
    connected = Signal(list, bool)
    powered = Signal()
    armed = Signal()
    failed = Signal(str)
    frame = Signal(object)
    panorama = Signal(object)
    panorama_status = Signal(str)
    posture_status = Signal(str)
    gesture_info = Signal(str)
    gesture_off = Signal(str)


class SpotSession:
    """Own all command RPCs on one thread; UI only changes the requested state."""

    def __init__(self, hostname, signals):
        self.hostname = hostname
        self.signals = signals
        self.lock = threading.Lock()
        self.wakeup = threading.Event()
        self.closing = threading.Event()
        self.keys = set()
        self.pending = None
        self.height = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.speed = DEFAULT_SPEED
        self.sources = [CAMERAS[0]]
        self.latest_frames = {}
        self.frame_times = {}
        self.auto_panorama = False
        self.panorama_requested = threading.Event()
        self.last_panorama_request = 0.0
        self.gesture_mode = False
        self.gesture_motion = 0
        self.gesture_valid_until = 0.0
        self.gesture_gate = GestureGate()
        self.gesture_depth_source = None
        self.client = None
        self.state_client = None
        self.powered = False
        self.last_motion_time = -float('inf')
        self.lease_keepalive = None
        self.robot = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.camera_thread = None
        self.gesture_thread = None
        self.panorama_thread = threading.Thread(target=self._panorama_loop, daemon=True)

    def start(self):
        self.panorama_thread.start()
        self.thread.start()

    def set_keys(self, keys):
        with self.lock:
            if self.gesture_mode:
                return
            self.keys = set(keys)
            if not keys:
                self.pending = 'stop'
        self.wakeup.set()

    def action(self, action):
        with self.lock:
            self.keys.clear()
            self.pending = action
        self.wakeup.set()

    def apply_posture(self, height_cm, roll_deg, pitch_deg):
        with self.lock:
            if self.keys or self.gesture_mode or self.closing.is_set() or self.pending == 'stop':
                self.signals.posture_status.emit(
                    'Apply blocked: movement, gesture mode, Stop, or exit is active')
                return False
            if not (0 <= height_cm <= HEIGHT_LIMIT_CM and
                    abs(roll_deg) <= TILT_LIMIT_DEG and abs(pitch_deg) <= TILT_LIMIT_DEG):
                self.signals.posture_status.emit('Apply blocked: requested offset is outside app limits')
                return False
            self.pending = ('posture', height_cm / 100, math.radians(roll_deg),
                            math.radians(pitch_deg))
        self.wakeup.set()
        return True

    def set_speed(self, speed):
        with self.lock:
            self.speed = max(0.05, min(MAX_SPEED, speed))

    def set_gesture_mode(self, enabled):
        with self.lock:
            if enabled and not self.gesture_depth_source:
                self.signals.gesture_off.emit('Aligned front depth source is unavailable')
                return
            self.gesture_mode = enabled
            self.gesture_motion = 0
            self.gesture_valid_until = 0.0
            self.gesture_gate.reset()
            self.keys.clear()
            self.pending = 'stop'
        self.wakeup.set()
        if enabled:
            self.signals.gesture_info.emit('ON — show neutral open palm first; waiting for clear person and hand')
        else:
            self.signals.gesture_info.emit('OFF — manual keyboard control')

    def set_auto_panorama(self, enabled):
        with self.lock:
            self.auto_panorama = enabled
        self.signals.panorama_status.emit(
            'Auto refresh on (about every 2 s); waiting for front cameras…' if enabled else
            'Auto refresh off. Front-camera approximation only.')
        if enabled:
            self.request_panorama()

    def request_panorama(self):
        with self.lock:
            self.last_panorama_request = time.monotonic()
        self.panorama_requested.set()

    def _panorama_loop(self):
        while not self.closing.is_set():
            if not self.panorama_requested.wait(0.2):
                continue
            self.panorama_requested.clear()
            with self.lock:
                gesture_active = self.gesture_mode
                images = [self.latest_frames.get(source) for source in FRONT_PAIR]
                times = [self.frame_times.get(source, 0) for source in FRONT_PAIR]
            if gesture_active:
                self.signals.panorama_status.emit('Paused while gesture mode uses the front camera')
                continue
            if any(image is None for image in images) or any(time.monotonic() - t > 1.0 for t in times):
                self.signals.panorama_status.emit('Waiting for fresh front-left and front-right frames')
                continue
            self.signals.panorama_status.emit('Stitching front cameras…')
            try:
                greyscale = all(image.mode == 'L' for image in images)
                frames = [cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
                          for image in images]
                status, panorama = cv2.Stitcher_create(cv2.Stitcher_PANORAMA).stitch(frames)
                if status != cv2.Stitcher_OK or panorama is None:
                    self.signals.panorama_status.emit(
                        'Could not align front frames; individual feeds continue')
                    continue
                picture = Image.fromarray(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
                if greyscale:
                    picture = picture.convert('L')
                self.signals.panorama.emit(picture)
                self.signals.panorama_status.emit('Front panorama updated (approximate, not calibrated)')
            except Exception as exc:
                self.signals.panorama_status.emit(f'Stitching unavailable: {exc}')

    def close(self):
        self.closing.set()
        self.panorama_requested.set()
        self.action('stop')
        # power_on can block for up to 20 s. Wait for the worker's final zero RPC.
        self.thread.join(timeout=25.0)

    def _command(self, command, end_time_secs=None):
        self.client.robot_command(command=command, end_time_secs=end_time_secs, timeout=0.7)

    def _zero(self):
        self._command(RobotCommandBuilder.synchro_velocity_command(
            0, 0, 0, body_height=self.height),
                      end_time_secs=time.time() + COMMAND_DURATION)

    def _run(self):
        try:
            self.signals.status.emit('Connecting and authenticating (watch Terminal for prompts)…')
            sdk = bosdyn.client.create_standard_sdk('SpotControlGUI')
            self.robot = sdk.create_robot(self.hostname)
            bosdyn.client.util.authenticate(self.robot)
            self.robot.time_sync.wait_for_sync(timeout_sec=10)
            image_client = self.robot.ensure_client(ImageClient.default_service_name)
            available = {source.name for source in image_client.list_image_sources(timeout=2)}
            cameras = [name for name in CAMERAS if name in available]
            if not cameras:
                raise RuntimeError('No built-in fisheye camera source was reported by Spot')
            with self.lock:
                self.sources = cameras
                if 'frontleft_fisheye_image' in available and \
                        'frontleft_depth_in_visual_frame' in available:
                    self.gesture_depth_source = 'frontleft_depth_in_visual_frame'
            self.client = self.robot.ensure_client(RobotCommandClient.default_service_name)
            self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
            lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
            self.lease_keepalive = LeaseKeepAlive(lease_client, must_acquire=True,
                                                   return_at_exit=True)
            self.powered = self.robot.is_powered_on()
            self.signals.connected.emit(cameras, self.powered)
            self.signals.status.emit('Connected. Use Power On and Stand if needed; hold a direction key to move.')
            self.camera_thread = threading.Thread(target=self._camera_loop,
                                                  args=(image_client,), daemon=True)
            self.camera_thread.start()
            self.gesture_thread = threading.Thread(target=self._gesture_loop,
                                                   args=(image_client,), daemon=True)
            self.gesture_thread.start()
            next_command = time.monotonic()
            last_gesture_direction = 0
            while not self.closing.is_set():
                with self.lock:
                    action, self.pending = self.pending, None
                    keys = set(self.keys)
                    height = self.height
                    speed = self.speed
                    gesture_mode = self.gesture_mode
                    direction = self.gesture_motion
                    valid_until = self.gesture_valid_until
                    if gesture_mode and direction and time.monotonic() >= valid_until:
                        self.gesture_motion = direction = 0
                        self.gesture_gate.stop_and_rearm()
                        self.signals.gesture_info.emit('Signal/depth stale — stopped; show neutral open palm')
                if action:
                    if action == 'stop':
                        if self.powered:
                            self._zero()
                        self.signals.status.emit('Stopped.')
                    elif action == 'power':
                        self.signals.status.emit('Powering on…')
                        self.robot.power_on(timeout_sec=20)
                        self.powered = True
                        self.signals.powered.emit()
                        self.signals.status.emit('Powered on. Press Stand before driving.')
                    elif action == 'stand':
                        self._zero()
                        self._command(RobotCommandBuilder.synchro_stand_command(
                            body_height=height,
                            footprint_R_body=EulerZXY(roll=self.roll, pitch=self.pitch)))
                        self.signals.armed.emit()
                        self.signals.status.emit('Stand command sent; requested posture may be limited by Spot')
                    elif isinstance(action, tuple) and action[0] == 'posture':
                        _, new_height, new_roll, new_pitch = action
                        if time.monotonic() - self.last_motion_time < 0.5:
                            self.signals.posture_status.emit(
                                'Apply blocked: wait until Spot has stopped')
                        elif not self._robot_is_stationary():
                            self.signals.posture_status.emit(
                                'Apply blocked: fresh robot state does not confirm stationary')
                        else:
                            with self.lock:
                                if (self.keys or self.gesture_mode or self.closing.is_set() or
                                        self.pending == 'stop'):
                                    self.signals.posture_status.emit(
                                        'Apply blocked: controls became active')
                                else:
                                    self._command(RobotCommandBuilder.synchro_stand_command(
                                        body_height=new_height,
                                        footprint_R_body=EulerZXY(
                                            roll=new_roll, pitch=new_pitch)))
                                    self.height, self.roll, self.pitch = (
                                        new_height, new_roll, new_pitch)
                                    self.signals.posture_status.emit(
                                        'Standing posture requested; actual pose may be limited by Spot')
                    next_command = time.monotonic() + COMMAND_PERIOD
                    if action == 'stop':
                        last_gesture_direction = 0
                elif gesture_mode and (time.monotonic() >= next_command or
                                       (last_gesture_direction and not direction)):
                    sent = False
                    # Serialize the final safety check with the UI's Stop/mode-off update.
                    with self.lock:
                        remaining = self.gesture_valid_until - time.monotonic()
                        if (self.gesture_mode and direction and
                                self.gesture_motion == direction and remaining > 0.02):
                            self._command(RobotCommandBuilder.synchro_velocity_command(
                                direction * GESTURE_SPEED, 0, 0, body_height=height),
                            end_time_secs=time.time() + min(GESTURE_DURATION, remaining))
                            self.last_motion_time = time.monotonic()
                            sent = True
                    if sent:
                        last_gesture_direction = direction
                    elif last_gesture_direction:
                        self._zero()
                        last_gesture_direction = 0
                    next_command = time.monotonic() + COMMAND_PERIOD
                elif keys and time.monotonic() >= next_command:
                    # Keep the final key check and RPC serialized with Stop/focus loss.
                    with self.lock:
                        live_keys = set(self.keys) if not self.gesture_mode and \
                            not self.closing.is_set() and self.pending != 'stop' else set()
                        if live_keys:
                            forward = int(Qt.Key_W in live_keys or Qt.Key_Up in live_keys) - int(
                                Qt.Key_S in live_keys or Qt.Key_Down in live_keys)
                            left = int(Qt.Key_A in live_keys or Qt.Key_Left in live_keys) - int(
                                Qt.Key_D in live_keys or Qt.Key_Right in live_keys)
                            # Normalize diagonals to the same overall speed cap.
                            scale = max(1, (forward * forward + left * left) ** 0.5)
                            command = RobotCommandBuilder.synchro_velocity_command(
                                self.speed * forward / scale, self.speed * left / scale, 0,
                                body_height=self.height)
                            self._command(command, end_time_secs=time.time() + COMMAND_DURATION)
                            self.last_motion_time = time.monotonic()
                    next_command = time.monotonic() + COMMAND_PERIOD
                self.wakeup.wait(timeout=max(0.02, min(0.05, next_command - time.monotonic())))
                self.wakeup.clear()
        except Exception as exc:
            with self.lock:
                self.keys.clear()
            self.signals.failed.emit(f'{type(exc).__name__}: {exc}')
        finally:
            self.closing.set()
            if self.client is not None and self.powered:
                try:
                    self._zero()
                except Exception:
                    pass  # The short command expiry remains the fallback.
            if self.lease_keepalive is not None:
                self.lease_keepalive.shutdown()

    def _robot_is_stationary(self):
        """Fail closed if odometry velocity or its timestamp is unavailable."""
        state = self.state_client.get_robot_state(timeout=0.7)
        if state.behavior_state.state != robot_state_pb2.BehaviorState.STATE_STANDING:
            return False
        kin = state.kinematic_state
        if not kin.HasField('velocity_of_body_in_odom') or \
                not kin.HasField('acquisition_timestamp'):
            return False
        now_robot = self._stamp_seconds(
            self.robot.time_sync.robot_timestamp_from_local_secs(time.time()))
        age = now_robot - self._stamp_seconds(kin.acquisition_timestamp)
        if not 0 <= age <= 0.6:
            return False
        velocity = kin.velocity_of_body_in_odom
        linear = velocity.linear
        angular = velocity.angular
        return (math.sqrt(linear.x**2 + linear.y**2 + linear.z**2) < 0.03 and
                math.sqrt(angular.x**2 + angular.y**2 + angular.z**2) < 0.05)

    @staticmethod
    def _stamp_seconds(stamp):
        return stamp.seconds + stamp.nanos / 1e9

    def _gesture_loop(self, image_client):
        vision = None
        while not self.closing.is_set():
            with self.lock:
                enabled = self.gesture_mode
            if not enabled:
                if vision is not None:
                    vision.close()
                    vision = None
                self.closing.wait(0.1)
                continue
            try:
                if vision is None:
                    vision = GestureVision()
                requests = [build_image_request('frontleft_fisheye_image', quality_percent=80),
                            build_image_request(self.gesture_depth_source)]
                responses = image_client.get_image(requests, timeout=0.8)
                by_name = {response.source.name: response for response in responses}
                visual = by_name['frontleft_fisheye_image']
                depth_response = by_name[self.gesture_depth_source]
                image = depth_response.shot.image
                if image.format != image_pb2.Image.FORMAT_RAW or \
                        image.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                    raise ValueError('Aligned depth is not raw uint16')
                depth = np.frombuffer(image.data, dtype='<u2').reshape(image.rows, image.cols)
                visual_image = decode_visual_image(visual.shot.image)
                rgb = np.asarray(visual_image.convert('RGB'))
                preview = visual_image.rotate(ROTATION['frontleft_fisheye_image'], expand=True)
                preview.thumbnail((720, 480))
                self.signals.frame.emit({'frontleft_fisheye_image': preview.copy()})
                if rgb.shape[:2] != depth.shape:
                    raise ValueError('Visual/depth image sizes do not match')
                visual_stamp = self._stamp_seconds(visual.shot.acquisition_time)
                depth_stamp = self._stamp_seconds(depth_response.shot.acquisition_time)
                robot_now = self._stamp_seconds(
                    self.robot.time_sync.robot_timestamp_from_local_secs(time.time()))
                age = robot_now - min(visual_stamp, depth_stamp)
                if abs(visual_stamp - depth_stamp) > 0.08 or not 0 <= age <= MAX_GESTURE_FRAME_AGE:
                    observation = Observation('unclear', reason='Image pair is stale or unsynchronized')
                else:
                    # Rotate both arrays identically so vision coordinates still match depth pixels.
                    rgb = np.ascontiguousarray(np.rot90(rgb, k=3))
                    depth = np.ascontiguousarray(np.rot90(depth, k=3))
                    observation = vision.inspect(rgb, depth, depth_response.source.depth_scale)
                robot_now = self._stamp_seconds(
                    self.robot.time_sync.robot_timestamp_from_local_secs(time.time()))
                age = robot_now - min(visual_stamp, depth_stamp)
                if not 0 <= age <= MAX_GESTURE_FRAME_AGE:
                    observation = Observation('unclear', reason='Processed image is stale')
                now = time.monotonic()
                with self.lock:
                    if not self.gesture_mode:
                        continue
                    direction, description = self.gesture_gate.observe(observation, now)
                    self.gesture_motion = direction
                    self.gesture_valid_until = now + max(0, MAX_GESTURE_FRAME_AGE - age)
                    if direction < 0:
                        self.gesture_valid_until = min(self.gesture_valid_until,
                                                       self.gesture_gate.backup_until)
                self.signals.gesture_info.emit(
                    f'ON — {description}' + (f' ({observation.reason})' if observation.reason else ''))
                self.wakeup.set()
            except Exception as exc:
                with self.lock:
                    self.gesture_mode = False
                    self.gesture_motion = 0
                    self.gesture_gate.stop_and_rearm()
                    self.pending = 'stop'
                self.signals.gesture_off.emit(f'Gesture mode disabled: {exc}')
                self.wakeup.set()
        if vision is not None:
            vision.close()

    def _camera_loop(self, image_client):
        while not self.closing.is_set():
            try:
                with self.lock:
                    active_gesture = self.gesture_mode
                    sources = list(self.sources)
                if active_gesture:
                    self.closing.wait(0.1)
                    continue
                requests = [build_image_request(source, quality_percent=55)
                            for source in sources]
                responses = image_client.get_image(requests, timeout=1.5)
                pictures = {}
                for response in responses:
                    source = response.source.name
                    picture = decode_visual_image(response.shot.image)
                    picture = picture.rotate(ROTATION.get(source, 0), expand=True)
                    picture.thumbnail((720, 480))
                    pictures[source] = picture.copy()
                with self.lock:
                    self.latest_frames.update(pictures)
                    now = time.monotonic()
                    self.frame_times.update({source: now for source in pictures})
                    auto_due = (self.auto_panorama and
                                now - self.last_panorama_request >= PANORAMA_PERIOD)
                    if auto_due:
                        self.last_panorama_request = now
                self.signals.frame.emit(pictures)
                if auto_due:
                    self.panorama_requested.set()
                self.closing.wait(0.15)
            except Exception as exc:
                self.signals.failed.emit(f'Camera/connection problem: {exc}')
                self.closing.set()
                self.wakeup.set()
                return


class PanoramaView(QGraphicsView):
    """Pan and zoom a stitched snapshot with the mouse wheel and drag."""

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(Qt.black)
        self.pixmap_item = None

    def set_picture(self, picture):
        pixmap = QPixmap.fromImage(ImageQt(picture))
        first_frame = self.pixmap_item is None
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(self.pixmap_item.boundingRect())
        if first_frame:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        self.scale(1.2 if event.angleDelta().y() > 0 else 1 / 1.2,
                   1.2 if event.angleDelta().y() > 0 else 1 / 1.2)


class BodyPosturePreview(QWidget):
    """Body-frame diagram only: no legs, feet, collision, or reachability model."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(460, 300)
        self.height_cm = self.roll_deg = self.pitch_deg = 0

    def set_request(self, height_cm, roll_deg, pitch_deg):
        self.height_cm, self.roll_deg, self.pitch_deg = height_cm, roll_deg, pitch_deg
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('#101820'))
        w, h = self.width(), self.height()
        center = QPointF(w * .5, h * .55)
        scale = min(w / 4.5, h / 3.0)

        def project(v):
            x, y, z = v
            return QPointF(center.x() + scale * (.78 * x - .58 * y),
                           center.y() + scale * (.30 * x + .32 * y - z))

        verts = [(x, y, z) for z in (-.22, .22)
                 for y in (-.48, .48) for x in (-.9, .9)]
        edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
                 if sum(verts[a][i] != verts[b][i] for i in range(3)) == 1]

        def draw_box(points, color, dashed=False):
            pen = QPen(QColor(color), 2)
            if dashed:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            for a, b in edges:
                painter.drawLine(project(points[a]), project(points[b]))

        draw_box(verts, '#58636b', True)
        q = EulerZXY(roll=math.radians(self.roll_deg),
                     pitch=math.radians(self.pitch_deg)).to_quaternion()

        def rotate(v):
            x, y, z = v
            return ((1 - 2 * (q.y*q.y + q.z*q.z))*x +
                    2 * (q.x*q.y - q.z*q.w)*y + 2 * (q.x*q.z + q.y*q.w)*z,
                    2 * (q.x*q.y + q.z*q.w)*x +
                    (1 - 2 * (q.x*q.x + q.z*q.z))*y + 2 * (q.y*q.z - q.x*q.w)*z,
                    2 * (q.x*q.z - q.y*q.w)*x + 2 * (q.y*q.z + q.x*q.w)*y +
                    (1 - 2 * (q.x*q.x + q.y*q.y))*z + self.height_cm / 10)

        draw_box([rotate(v) for v in verts], '#56d7d0')
        painter.setPen(QColor('#d9e4e8'))
        painter.drawText(12, 22, 'DESCRIPTIVE BODY-FRAME PREVIEW')
        painter.drawText(12, h - 34, 'Gray: nominal   Cyan: requested offset')
        painter.drawText(12, h - 15, 'Not to scale • no legs or feet • not a validated pose')
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self, hostname, offline=False):
        super().__init__()
        self.setWindowTitle('Spot control — OFFLINE PREVIEW' if offline else f'Spot control — {hostname}')
        self.resize(950, 760)
        self.offline = offline
        self.signals = UiSignals()
        self.session = SpotSession(hostname, self.signals)
        self.pressed = set()
        self.ready = False
        self.armed = False
        self.failed = False
        self.feed_labels = {}
        self.split_labels = {}

        root = QWidget()
        layout = QVBoxLayout(root)
        self.status = QLabel('OFFLINE PREVIEW — no robot connection or camera footage' if offline
                             else 'Starting connection…')
        layout.addWidget(self.status)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        posture_page = QWidget()
        posture_layout = QVBoxLayout(posture_page)
        self.posture_preview = BodyPosturePreview()
        posture_layout.addWidget(self.posture_preview, 1)
        posture_layout.addWidget(QLabel(
            'Requested body offset from nominal stand. Diagram only; actual pose may be limited by Spot.'))
        height_row = QHBoxLayout()
        height_row.addWidget(QLabel('Height offset:'))
        self.height = QSlider(Qt.Horizontal)
        self.height.setRange(0, HEIGHT_LIMIT_CM)
        self.height.setValue(0)
        self.height.setFocusPolicy(Qt.NoFocus)
        self.height.valueChanged.connect(self.posture_changed)
        height_row.addWidget(self.height, 1)
        self.height_label = QLabel('requested +0 cm')
        height_row.addWidget(self.height_label)
        posture_layout.addLayout(height_row)
        self.roll = QSlider(Qt.Horizontal)
        self.pitch = QSlider(Qt.Horizontal)
        self.roll_label = QLabel()
        self.pitch_label = QLabel()
        for title, slider, label in [('Roll', self.roll, self.roll_label),
                                     ('Pitch', self.pitch, self.pitch_label)]:
            row = QHBoxLayout()
            row.addWidget(QLabel(title + ' offset:'))
            slider.setRange(-TILT_LIMIT_DEG, TILT_LIMIT_DEG)
            slider.setValue(0)
            slider.setFocusPolicy(Qt.NoFocus)
            slider.valueChanged.connect(self.posture_changed)
            row.addWidget(slider, 1)
            row.addWidget(label)
            posture_layout.addLayout(row)
        self.apply_button = QPushButton('Apply requested standing posture')
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_requested_posture)
        posture_layout.addWidget(self.apply_button)
        self.posture_message = QLabel(
            'Preview only. Apply requires a stationary, powered, standing Spot.')
        self.posture_message.setWordWrap(True)
        posture_layout.addWidget(self.posture_message)
        self.tabs.addTab(posture_page, 'Posture preview')
        self.posture_changed()

        controls = QHBoxLayout()
        self.power = QPushButton('Power On')
        self.power.clicked.connect(lambda: self.session.action('power'))
        controls.addWidget(self.power)
        self.stand = QPushButton('Stand')
        self.stand.clicked.connect(lambda: self.session.action('stand'))
        controls.addWidget(self.stand)
        self.stop = QPushButton('STOP — zero velocity')
        self.stop.setStyleSheet('background: #c83131; color: white; font-weight: bold; padding: 12px;')
        self.stop.clicked.connect(self.stop_motion)
        controls.addWidget(self.stop, 1)
        layout.addLayout(controls)

        gesture_row = QHBoxLayout()
        self.gesture_toggle = QCheckBox('Enable supervised gesture mode')
        self.gesture_toggle.setEnabled(False)
        self.gesture_toggle.toggled.connect(self.toggle_gesture_mode)
        gesture_row.addWidget(self.gesture_toggle)
        self.gesture_indicator = QLabel('GESTURE MODE OFF')
        self.gesture_indicator.setStyleSheet('color: #b22; font-weight: bold;')
        gesture_row.addWidget(self.gesture_indicator, 1)
        layout.addLayout(gesture_row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel('Movement speed:'))
        self.speed = QSlider(Qt.Horizontal)
        self.speed.setRange(5, int(MAX_SPEED * 100))
        self.speed.setValue(int(DEFAULT_SPEED * 100))
        self.speed.setFocusPolicy(Qt.NoFocus)
        self.speed.valueChanged.connect(self.speed_changed)
        speed_row.addWidget(self.speed, 1)
        self.speed_label = QLabel(f'{DEFAULT_SPEED:.2f} m/s')
        speed_row.addWidget(self.speed_label)
        layout.addLayout(speed_row)
        layout.addWidget(QLabel('Hold WASD or arrow keys to drive. Release to stop. '
                                'Posture changes require Apply while stationary. Keep class E-stop open.'))
        self.setCentralWidget(root)

        self.power.setEnabled(False)
        self.stand.setEnabled(False)
        app = QApplication.instance()
        app.installEventFilter(self)
        self.signals.status.connect(self.status.setText)
        self.signals.connected.connect(self.on_connected)
        self.signals.powered.connect(self.on_powered)
        self.signals.armed.connect(self.on_armed)
        self.signals.failed.connect(self.on_failed)
        self.signals.frame.connect(self.show_frame)
        self.signals.panorama.connect(self.show_panorama)
        self.signals.panorama_status.connect(self.show_panorama_status)
        self.signals.posture_status.connect(self.posture_message.setText)
        self.signals.gesture_info.connect(self.show_gesture_info)
        self.signals.gesture_off.connect(self.gesture_error)
        if offline:
            self.gesture_indicator.setText('GESTURE MODE OFF — offline preview')
        else:
            self.session.start()

    def on_connected(self, cameras, powered):
        self.ready = True
        for source in cameras:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            image_label = self._new_image_label((640, 430))
            image_label.setText(f'Waiting for {self._camera_title(source)}…')
            page_layout.addWidget(image_label, 1)
            self.tabs.insertTab(len(self.feed_labels), page, self._camera_title(source))
            self.feed_labels[source] = image_label
        self.tabs.setCurrentIndex(0)

        split_page = QWidget()
        split_grid = QGridLayout(split_page)
        for index, source in enumerate(cameras):
            tile = QWidget()
            tile_layout = QVBoxLayout(tile)
            tile_layout.addWidget(QLabel(self._camera_title(source)))
            image_label = self._new_image_label((350, 225))
            image_label.setText('Waiting for camera…')
            tile_layout.addWidget(image_label, 1)
            split_grid.addWidget(tile, index // 2, index % 2)
            self.split_labels[source] = image_label
        self.tabs.addTab(split_page, 'Split screen')

        panorama_page = QWidget()
        panorama_layout = QVBoxLayout(panorama_page)
        self.panorama_image = PanoramaView()
        self.panorama_image.setMinimumSize(700, 400)
        panorama_layout.addWidget(self.panorama_image, 1)
        panorama_note = QLabel('Approximate front-left + front-right fisheye stitch. '
                               'No calibration or 360° coverage. Drag to pan; wheel to zoom.')
        panorama_note.setWordWrap(True)
        panorama_layout.addWidget(panorama_note)
        self.panorama_status = QLabel('Front panorama idle; individual feeds stay live.')
        panorama_layout.addWidget(self.panorama_status)
        self.auto_panorama = QCheckBox('Continuously update (about every 2 seconds)')
        self.auto_panorama.toggled.connect(self.session.set_auto_panorama)
        panorama_layout.addWidget(self.auto_panorama)
        self.stitch_button = QPushButton('Stitch current front frames once')
        self.stitch_button.clicked.connect(self.session.request_panorama)
        panorama_layout.addWidget(self.stitch_button)
        self.tabs.addTab(panorama_page, 'Panorama')
        if not all(source in cameras for source in FRONT_PAIR):
            self.auto_panorama.setEnabled(False)
            self.stitch_button.setEnabled(False)
            self.panorama_status.setText('Front camera pair unavailable; individual feeds continue')
        self.power.setEnabled(not powered)
        self.stand.setEnabled(powered)

    def on_powered(self):
        self.power.setEnabled(False)
        self.stand.setEnabled(True)

    def on_armed(self):
        self.armed = True
        self.power.setEnabled(False)
        self.apply_button.setEnabled(not self.pressed)
        self.gesture_toggle.setEnabled(bool(self.session.gesture_depth_source))

    def toggle_gesture_mode(self, enabled):
        if enabled and (not self.ready or not self.armed or self.failed):
            self.gesture_toggle.blockSignals(True)
            self.gesture_toggle.setChecked(False)
            self.gesture_toggle.blockSignals(False)
            return
        self.pressed.clear()
        self.session.set_gesture_mode(enabled)
        self.apply_button.setEnabled(self.ready and self.armed and not enabled)
        self.speed.setEnabled(not enabled)
        self.stand.setEnabled(self.armed and not enabled)
        self.gesture_indicator.setStyleSheet(
            'color: #b22; font-weight: bold;' if not enabled else
            'color: #a65b00; font-weight: bold;')

    def show_gesture_info(self, message):
        self.gesture_indicator.setText(message)

    def gesture_error(self, message):
        self.gesture_toggle.blockSignals(True)
        self.gesture_toggle.setChecked(False)
        self.gesture_toggle.blockSignals(False)
        self.gesture_indicator.setText(f'GESTURE MODE OFF — {message}')
        self.gesture_indicator.setStyleSheet('color: #b22; font-weight: bold;')
        self.apply_button.setEnabled(self.ready and self.armed and not self.failed)
        self.speed.setEnabled(True)
        self.stand.setEnabled(self.ready and self.armed and not self.failed)

    def on_failed(self, message):
        if self.failed:
            return
        self.failed = True
        self.stop_motion()
        self.ready = False
        self.armed = False
        self.power.setEnabled(False)
        self.stand.setEnabled(False)
        self.height.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.gesture_toggle.setEnabled(False)
        self.status.setText(f'Connection lost: {message}. Use the separate E-stop if needed.')

    def show_frame(self, picture):
        if self.failed:
            return
        for source, frame in picture.items():
            self._put_frame(self.feed_labels.get(source), frame)
            self._put_frame(self.split_labels.get(source), frame)

    def show_panorama(self, picture):
        self.panorama_image.set_picture(picture)

    def show_panorama_status(self, message):
        if hasattr(self, 'panorama_status'):
            self.panorama_status.setText(message)

    @staticmethod
    def _new_image_label(size):
        label = QLabel()
        label.setMinimumSize(*size)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet('background: #151a20; color: white;')
        return label

    @staticmethod
    def _camera_title(source):
        return source.replace('_fisheye_image', '').replace('_', ' ').title()

    @staticmethod
    def _put_frame(label, picture):
        if label is None:
            return
        pixmap = QPixmap.fromImage(ImageQt(picture))
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation))

    def posture_changed(self, value=None):
        self.height_label.setText(f'requested +{self.height.value()} cm')
        self.roll_label.setText(f'{self.roll.value():+d}°')
        self.pitch_label.setText(f'{self.pitch.value():+d}°')
        self.posture_preview.set_request(
            self.height.value(), self.roll.value(), self.pitch.value())

    def apply_requested_posture(self):
        if self.ready and self.armed and not self.failed and not self.pressed:
            if self.session.apply_posture(
                    self.height.value(), self.roll.value(), self.pitch.value()):
                self.posture_message.setText('Checking stationary state before sending…')

    def speed_changed(self, value):
        speed = value / 100
        self.speed_label.setText(f'{speed:.2f} m/s')
        self.session.set_speed(speed)

    def stop_motion(self):
        self.pressed.clear()
        if self.gesture_toggle.isChecked():
            self.gesture_toggle.setChecked(False)
        self.apply_button.setEnabled(self.ready and self.armed and not self.failed)
        if not self.offline:
            self.session.action('stop')

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.ApplicationDeactivate, QEvent.WindowDeactivate):
            self.stop_motion()
        elif event.type() in (QEvent.KeyPress, QEvent.KeyRelease) and event.key() in KEYS:
            if event.isAutoRepeat():
                return True
            if event.type() == QEvent.KeyPress and self.ready and self.armed and \
                    not self.failed and not self.gesture_toggle.isChecked():
                self.pressed.add(event.key())
                self.apply_button.setEnabled(False)
                self.session.set_keys(self.pressed)
            elif event.type() == QEvent.KeyRelease:
                self.pressed.discard(event.key())
                self.apply_button.setEnabled(self.ready and self.armed and not self.failed and
                                             not self.pressed and not self.gesture_toggle.isChecked())
                self.session.set_keys(self.pressed)
            return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        self.stop_motion()
        if not self.offline:
            self.session.close()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hostname', help='Spot hostname or IP; confirm with instructor')
    parser.add_argument('--offline-preview', action='store_true',
                        help='Show posture UI without connecting to a robot or showing camera footage')
    args = parser.parse_args()
    if not args.offline_preview and not args.hostname:
        parser.error('--hostname is required unless --offline-preview is set')
    app = QApplication([])
    window = MainWindow(args.hostname, offline=args.offline_preview)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
