#!/usr/bin/env python3
"""Small, keyboard-driven Spot controller. Requires an independent class E-stop."""

import argparse
import importlib.util
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
from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QGridLayout, QHBoxLayout,
                               QComboBox, QLabel, QDockWidget, QGraphicsScene, QGraphicsView, QMainWindow,
                               QMenu, QPushButton, QScrollArea, QSizePolicy, QSlider, QTabWidget,
                               QToolButton, QVBoxLayout, QWidget)
from spot_demo import make_demo_frames, make_demo_panorama
from spot_gesture import GestureGate, GestureVision, MODEL_DIR, Observation
from spot_model_view import SpotModelView, hardware_matches_base


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

GUI_STYLE = """
QMainWindow, QWidget#root { background: #f2f5f7; color: #172735; }
QFrame#header, QFrame#surface, QFrame#controlPanel, QFrame#statusBar {
    background: #ffffff; border: 1px solid #dce3e8; border-radius: 10px;
}
QLabel { color: #172735; }
QLabel#appTitle { color: #162b39; font-size: 20px; font-weight: 700; }
QLabel#sectionTitle { color: #162b39; font-size: 15px; font-weight: 700; }
QLabel#muted { color: #526574; font-size: 12px; }
QLabel#stateTitle { color: #162b39; font-weight: 700; }
QLabel#cameraStatus { color: #526574; font-size: 12px; }
QFrame#stateDot { border: none; border-radius: 5px; background: #9aa8b2; }
QPushButton { background: #ffffff; color: #172735; border: 1px solid #becbd4;
              border-radius: 6px; padding: 9px 12px; font-weight: 600; }
QPushButton:hover { background: #ecf4f5; border-color: #0b747b; }
QPushButton:focus { border: 2px solid #0b747b; }
QPushButton:disabled { color: #8998a2; background: #f2f4f5; border-color: #e2e7ea; }
QPushButton#stopButton { background: #ae2630; border-color: #91202a; color: white;
                         font-size: 15px; font-weight: 700; padding: 13px 18px; }
QPushButton#stopButton:hover { background: #951e28; }
QPushButton#stopButton:focus { border: 2px solid #172735; }
QPushButton#applyButton { background: #0b747b; border-color: #08626a; color: white; }
QPushButton#applyButton:hover { background: #08626a; }
QPushButton#applyButton:disabled { background: #dce7e8; border-color: #dce7e8;
                                  color: #71878a; }
QToolButton { background: #ffffff; color: #172735; border: 1px solid #becbd4;
              border-radius: 6px; padding: 10px 12px; font-weight: 600; }
QToolButton:hover { background: #ecf4f5; border-color: #0b747b; }
QComboBox { background: #ffffff; color: #172735; border: 1px solid #becbd4;
            border-radius: 6px; padding: 6px 8px; }
QComboBox QAbstractItemView { background: #ffffff; color: #172735;
                              selection-background-color: #dfe9eb; }
QTabWidget::pane { border: 1px solid #dce3e8; background: #ffffff; }
QTabBar::tab { background: #edf2f4; color: #405563; padding: 9px 13px;
               border: 1px solid #dce3e8; border-bottom: none; }
QTabBar::tab:selected { background: #ffffff; color: #0b6871; font-weight: 700; }
QTabBar::tab:hover:!selected { background: #dfe9eb; }
QScrollArea { border: none; background: transparent; }
QCheckBox { color: #172735; spacing: 8px; }
QSlider { min-height: 22px; }
QSlider::groove:horizontal { height: 6px; background: #d4e0e5; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #0b747b; border-radius: 3px; }
QSlider::handle:horizontal { width: 16px; margin: -6px 0; background: #ffffff;
                             border: 1px solid #0b747b; border-radius: 8px; }
QDockWidget { color: #162b39; font-weight: 700; }
QDockWidget::title { background: #e9eff2; padding: 7px 10px; }
"""


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
    model_state = Signal(object, str)
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
        self.model_verified = False
        self.model_reason = 'Robot model waiting for hardware configuration'
        self.powered = False
        self.armed = False
        self.last_motion_time = -float('inf')
        self.lease_keepalive = None
        self.robot = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.camera_thread = None
        self.gesture_thread = None
        self.state_thread = None
        self.panorama_thread = threading.Thread(target=self._panorama_loop, daemon=True)

    def start(self):
        self.panorama_thread.start()
        self.thread.start()

    def set_keys(self, keys, released=False):
        with self.lock:
            if self.gesture_mode or not self.armed or self.closing.is_set():
                return
            self.keys = set(keys)
            if released or not keys:
                self.pending = 'stop'
        self.wakeup.set()

    def action(self, action):
        with self.lock:
            self.keys.clear()
            self.pending = action
        self.wakeup.set()

    def apply_posture(self, height_cm, roll_deg, pitch_deg):
        with self.lock:
            if (not self.armed or not self.powered or self.keys or self.gesture_mode or
                    self.closing.is_set() or self.pending is not None):
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
            if enabled and (not self.armed or not self.powered or self.closing.is_set()):
                self.signals.gesture_off.emit('Stand and confirm readiness first')
                return
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
        cv2.setNumThreads(2)
        last_stitch_finished = 0.0
        while not self.closing.is_set():
            if not self.panorama_requested.wait(0.2):
                continue
            self.panorama_requested.clear()
            cooldown = PANORAMA_PERIOD - (time.monotonic() - last_stitch_finished)
            if cooldown > 0 and self.closing.wait(cooldown):
                break
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
                frames = []
                for image in images:
                    small = image.copy()
                    small.thumbnail((640, 420))
                    frames.append(cv2.cvtColor(np.asarray(small.convert('RGB')),
                                               cv2.COLOR_RGB2BGR))
                status, panorama = cv2.Stitcher_create(cv2.Stitcher_PANORAMA).stitch(frames)
                if status != cv2.Stitcher_OK or panorama is None:
                    self.signals.panorama_status.emit(
                        'Could not align front frames; individual feeds continue')
                    continue
                if panorama.shape[0] * panorama.shape[1] > 3_000_000:
                    self.signals.panorama_status.emit(
                        'Stitch exceeded preview size; individual feeds continue')
                    continue
                picture = Image.fromarray(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
                picture.thumbnail((1600, 700))
                if greyscale:
                    picture = picture.convert('L')
                self.signals.panorama.emit(picture)
                self.signals.panorama_status.emit('Front panorama updated (approximate, not calibrated)')
            except Exception as exc:
                self.signals.panorama_status.emit(f'Stitching unavailable: {exc}')
            finally:
                last_stitch_finished = time.monotonic()

    def close(self):
        self.closing.set()
        self.panorama_requested.set()
        self.action('stop')
        # Normal command RPCs time out in 0.7 s. Never freeze the UI during exit.
        self.thread.join(timeout=1.6)

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
            try:
                hardware = self.state_client.get_robot_hardware_configuration(timeout=1.5)
                self.model_verified = hardware_matches_base(hardware)
                if not self.model_verified:
                    self.model_reason = 'Robot skeleton differs from SDK base mesh; geometry hidden'
            except Exception as exc:
                self.model_reason = f'Robot skeleton not verified; geometry hidden: {exc}'
            if not self.model_verified:
                self.signals.model_state.emit(None, self.model_reason)
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
            self.state_thread = threading.Thread(target=self._state_loop, daemon=True)
            self.state_thread.start()
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
                        if not self.powered:
                            self.signals.status.emit('Stand blocked: Spot is not powered on')
                        else:
                            with self.lock:
                                self.armed = False
                            self._zero()
                            self._command(RobotCommandBuilder.synchro_stand_command(
                                body_height=height,
                                footprint_R_body=EulerZXY(roll=self.roll, pitch=self.pitch)))
                            self.signals.status.emit('Stand requested; waiting for fresh standing telemetry…')
                            if self._await_stand():
                                with self.lock:
                                    can_arm = not self.closing.is_set() and self.pending != 'stop'
                                    self.armed = can_arm
                                if can_arm:
                                    self.signals.armed.emit()
                                    self.signals.status.emit('Standing confirmed by fresh robot state')
                            else:
                                self.signals.status.emit(
                                    'Stand not confirmed; keyboard movement remains locked')
                    elif isinstance(action, tuple) and action[0] == 'posture':
                        _, new_height, new_roll, new_pitch = action
                        if not self.armed or not self.powered:
                            self.signals.posture_status.emit('Apply blocked: Stand is not confirmed')
                        elif time.monotonic() - self.last_motion_time < 0.5:
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
                        if (self.armed and self.powered and self.gesture_mode and direction and
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
                        live_keys = set(self.keys) if self.armed and self.powered and not self.gesture_mode and \
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
                self.armed = False
                self.gesture_mode = False
                self.gesture_motion = 0
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

    def _await_stand(self):
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not self.closing.is_set():
            with self.lock:
                if self.pending == 'stop':
                    return False
            try:
                if self._robot_is_stationary():
                    return True
            except Exception:
                pass
            if self.closing.wait(0.2):
                break
        return False

    def _state_loop(self):
        """Publish fresh measured joint angles for the read-only URDF view."""
        from spot_model_view import JOINT_NAMES
        while not self.closing.is_set():
            try:
                state = self.state_client.get_robot_state(timeout=0.7)
                kin = state.kinematic_state
                robot_now = self._stamp_seconds(
                    self.robot.time_sync.robot_timestamp_from_local_secs(time.time()))
                age = robot_now - self._stamp_seconds(kin.acquisition_timestamp)
                values = {joint.name: joint.position.value for joint in kin.joint_states
                          if joint.HasField('position') and joint.name in JOINT_NAMES}
                if 0 <= age <= 0.6 and len(values) == len(JOINT_NAMES):
                    self.signals.model_state.emit(
                        values if self.model_verified else None,
                        'Measured joint state • recent robot telemetry' if self.model_verified
                        else self.model_reason)
                else:
                    self.signals.model_state.emit(None, 'Robot model waiting for fresh joint telemetry')
                    with self.lock:
                        if self.armed:
                            self.keys.clear()
                            self.gesture_motion = 0
                            self.armed = False
                            self.pending = 'stop'
                            self.closing.set()
                            self.signals.failed.emit('Robot state became stale; movement disabled')
                            self.wakeup.set()
                            return
            except Exception as exc:
                if self.closing.is_set():
                    return
                self.signals.model_state.emit(None, f'Robot model telemetry unavailable: {exc}')
                with self.lock:
                    self.keys.clear()
                    self.gesture_motion = 0
                    self.pending = 'stop'
                    self.armed = False
                self.signals.failed.emit(f'Robot state connection problem: {exc}')
                self.closing.set()
                self.wakeup.set()
                return
            self.closing.wait(0.5)

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
                # Gesture worker supplies front-left visual frames; keep other feeds live.
                if active_gesture:
                    sources = [source for source in sources
                               if source != 'frontleft_fisheye_image']
                if not sources:
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
                if self.closing.is_set():
                    return
                self.signals.failed.emit(f'Camera/connection problem: {exc}')
                self.closing.set()
                self.wakeup.set()
                return


class CameraImageLabel(QLabel):
    """Keep the full source frame visible when a camera tile is resized."""

    def __init__(self, minimum_size):
        super().__init__()
        self.setMinimumSize(*minimum_size)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet('background: #15212a; color: #d9e4e8; border-radius: 5px;')
        self._source_pixmap = None

    def set_picture(self, picture):
        self._source_pixmap = (picture if isinstance(picture, QPixmap) else
                               QPixmap.fromImage(ImageQt(picture)))
        self._refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._source_pixmap is not None and self.width() > 0 and self.height() > 0:
            self.setPixmap(self._source_pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                                                      Qt.SmoothTransformation))


class PanoramaView(QGraphicsView):
    """Pan and zoom a stitched snapshot with the mouse wheel and drag."""

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(Qt.black)
        self.pixmap_item = None
        message = self.scene.addText('Waiting for front camera frames and alignment…')
        message.setDefaultTextColor(QColor('#d9e4e8'))

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


class MainWindow(QMainWindow):
    def __init__(self, hostname, offline=False, demo=False):
        super().__init__()
        self.setWindowTitle('Spot Control Console — DEMO' if demo else
                            'Spot Control Console — OFFLINE PREVIEW' if offline else
                            f'Spot Control Console — {hostname}')
        self.resize(1240, 820)
        self.setMinimumSize(1040, 620)
        self.setStyleSheet(GUI_STYLE)
        self.offline = offline or demo
        self.demo = demo
        self.signals = UiSignals()
        self.session = SpotSession(hostname, self.signals)
        self.pressed = set()
        self.ready = False
        self.armed = False
        self.failed = False
        self.feed_labels = {}
        self.split_labels = {}
        self.feed_status = {}
        self.split_status = {}

        root = QWidget()
        root.setObjectName('root')
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QFrame()
        header.setObjectName('header')
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(18, 12, 18, 12)
        heading = QVBoxLayout()
        app_title = QLabel('Spot Control Console')
        app_title.setObjectName('appTitle')
        heading.addWidget(app_title)
        sub = QLabel('Built-in cameras, keyboard drive, and standing posture')
        sub.setObjectName('muted')
        heading.addWidget(sub)
        header_row.addLayout(heading, 1)
        self.layout_button = QToolButton()
        self.layout_button.setText('Panels and layout')
        self.layout_button.setPopupMode(QToolButton.InstantPopup)
        self.layout_menu = QMenu(self.layout_button)
        self.layout_button.setMenu(self.layout_menu)
        header_row.addWidget(self.layout_button)
        self.stop = QPushButton('STOP MOVEMENT')
        self.stop.setObjectName('stopButton')
        self.stop.setMinimumWidth(205)
        self.stop.setToolTip('Immediately request zero velocity and clear held keys and gesture mode. '
                             'The separate class E-stop remains required.')
        self.stop.clicked.connect(self.stop_motion)
        self.layout_menu.aboutToShow.connect(self.stop_motion)
        header_row.addWidget(self.stop)
        layout.addWidget(header)

        status_bar = QFrame()
        status_bar.setObjectName('statusBar')
        status_row = QHBoxLayout(status_bar)
        status_row.setContentsMargins(16, 9, 16, 9)
        self.state_dot = QFrame()
        self.state_dot.setObjectName('stateDot')
        self.state_dot.setFixedSize(10, 10)
        status_row.addWidget(self.state_dot)
        status_text = QVBoxLayout()
        status_text.setSpacing(2)
        self.state_title = QLabel()
        self.state_title.setObjectName('stateTitle')
        status_text.addWidget(self.state_title)
        self.status = QLabel()
        self.status.setObjectName('muted')
        self.status.setWordWrap(True)
        status_text.addWidget(self.status)
        status_row.addLayout(status_text, 1)
        estop = QLabel('Keep the class E-stop open in a separate Terminal')
        estop.setObjectName('muted')
        estop.setWordWrap(True)
        estop.setMaximumWidth(245)
        status_row.addWidget(estop)
        layout.addWidget(status_bar)

        self.dock_host = QMainWindow()
        self.dock_host.setDockNestingEnabled(True)
        self.dock_host.setDockOptions(QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks)
        layout.addWidget(self.dock_host, 1)
        camera_surface = QFrame()
        camera_surface.setObjectName('surface')
        camera_layout = QVBoxLayout(camera_surface)
        camera_layout.setContentsMargins(16, 14, 16, 16)
        camera_layout.setSpacing(8)
        camera_title = QLabel('Built-in cameras')
        camera_title.setObjectName('sectionTitle')
        camera_layout.addWidget(camera_title)
        camera_intro = QLabel('Choose a camera, all feeds, or an approximate front stitch.')
        camera_intro.setObjectName('muted')
        camera_intro.setWordWrap(True)
        camera_layout.addWidget(camera_intro)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel('View'))
        self.source_select = QComboBox()
        self.source_select.setFocusPolicy(Qt.NoFocus)
        self.source_select.setToolTip('Choose a built-in feed, split screen, or front panorama.')
        source_row.addWidget(self.source_select, 1)
        camera_layout.addLayout(source_row)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().hide()
        self.source_select.currentIndexChanged.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self.source_select.setCurrentIndex)
        camera_layout.addWidget(self.tabs, 1)
        model_surface = QFrame()
        model_surface.setObjectName('surface')
        model_layout = QVBoxLayout(model_surface)
        model_layout.setContentsMargins(16, 12, 16, 12)
        model_layout.setSpacing(6)
        model_title = QLabel('Robot geometry')
        model_title.setObjectName('sectionTitle')
        model_layout.addWidget(model_title)
        model_note = QLabel('Read-only SDK model. Drag to orbit or pan; scroll to zoom. '
                            'Legs update only from fresh telemetry with a matching skeleton.')
        model_note.setObjectName('muted')
        model_note.setWordWrap(True)
        model_layout.addWidget(model_note)
        self.model_view = SpotModelView()
        model_layout.addWidget(self.model_view, 1)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        control_panel = QFrame()
        control_panel.setObjectName('controlPanel')
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(18, 16, 18, 16)
        control_layout.setSpacing(10)
        controls_scroll.setWidget(control_panel)
        self.camera_dock = self._make_dock('Camera', 'cameraDock', camera_surface)
        self.model_dock = self._make_dock('Robot Model', 'modelDock', model_surface)
        self.controls_dock = self._make_dock('Controls', 'controlsDock', controls_scroll)
        self.controls_dock.setMinimumWidth(320)
        self._build_layout_menu()
        self._place_docks('Balanced')
        self.settings = QSettings(QSettings.IniFormat, QSettings.UserScope,
                                  'SCOPE', 'Spot Control Console')
        saved = self.settings.value('dashboard/state')
        self._pending_layout = None if saved and self.dock_host.restoreState(saved, 1) else 'Balanced'

        ready_title = QLabel('Readiness')
        ready_title.setObjectName('sectionTitle')
        control_layout.addWidget(ready_title)
        ready_note = QLabel('Connect, power on if needed, then Stand. Drive unlocks only after fresh standing telemetry.')
        ready_note.setObjectName('muted')
        ready_note.setWordWrap(True)
        control_layout.addWidget(ready_note)
        self.power = QPushButton('Power On')
        self.power.setToolTip('Power Spot motors on after connecting. This does not make Spot stand or drive.')
        self.power.clicked.connect(lambda: self.session.action('power'))
        control_layout.addWidget(self.power)
        self.stand = QPushButton('Stand')
        self.stand.setToolTip('Request a high-level stand command. Driving unlocks only after fresh robot state confirms standing.')
        self.stand.clicked.connect(lambda: self.session.action('stand'))
        control_layout.addWidget(self.stand)
        control_layout.addSpacing(8)

        drive_title = QLabel('Keyboard movement')
        drive_title.setObjectName('sectionTitle')
        control_layout.addWidget(drive_title)
        drive_note = QLabel('Hold W / ↑ forward, S / ↓ back, A / ← left, D / → right. Release to stop.')
        drive_note.setObjectName('muted')
        drive_note.setWordWrap(True)
        drive_note.setToolTip('Direction keys work after Stand is confirmed. Releasing a key sends zero velocity; '
                               'focus loss and app exit also request zero velocity.')
        control_layout.addWidget(drive_note)
        speed_heading = QHBoxLayout()
        speed_heading.addWidget(QLabel('Speed limit'))
        self.speed_label = QLabel(f'{DEFAULT_SPEED:.2f} m/s')
        speed_heading.addWidget(self.speed_label, 1, Qt.AlignRight)
        control_layout.addLayout(speed_heading)
        self.speed = QSlider(Qt.Horizontal)
        self.speed.setRange(5, int(MAX_SPEED * 100))
        self.speed.setValue(int(DEFAULT_SPEED * 100))
        self.speed.setFocusPolicy(Qt.NoFocus)
        self.speed.setToolTip('Requested keyboard driving speed in meters per second, '
                              'from 0.05 to 0.35 m/s. Diagonal movement is normalized.')
        self.speed.valueChanged.connect(self.speed_changed)
        control_layout.addWidget(self.speed)
        speed_note = QLabel('0.05–0.35 m/s • commands expire after 0.35 s')
        speed_note.setObjectName('muted')
        speed_note.setWordWrap(True)
        control_layout.addWidget(speed_note)
        control_layout.addSpacing(8)

        posture_layout = control_layout
        posture_heading = QLabel('Standing body posture')
        posture_heading.setObjectName('sectionTitle')
        posture_layout.addWidget(posture_heading)
        posture_intro = QLabel('Set a standing body request, then Apply while Spot is stationary. These are offsets, not measured pose.')
        posture_intro.setObjectName('muted')
        posture_intro.setWordWrap(True)
        posture_layout.addWidget(posture_intro)
        height_row = QHBoxLayout()
        posture_layout.addWidget(QLabel('Height offset from nominal stand'))
        self.height = QSlider(Qt.Horizontal)
        self.height.setRange(0, HEIGHT_LIMIT_CM)
        self.height.setValue(0)
        self.height.setFocusPolicy(Qt.NoFocus)
        self.height.setToolTip('Requested height relative to nominal standing height, in 1 cm steps. '
                               'Spot may limit the actual movement. No verified safe lowering range is available.')
        self.height.valueChanged.connect(self.posture_changed)
        height_row.addWidget(self.height, 1)
        self.height_label = QLabel('requested +0 cm')
        height_row.addWidget(self.height_label)
        posture_layout.addLayout(height_row)
        height_note = QLabel('0 to +10 cm is an app request limit. A 1 cm step does not guarantee 1 cm of movement.')
        height_note.setObjectName('muted')
        height_note.setWordWrap(True)
        posture_layout.addWidget(height_note)
        self.roll = QSlider(Qt.Horizontal)
        self.pitch = QSlider(Qt.Horizontal)
        self.roll_label = QLabel()
        self.pitch_label = QLabel()
        for title, slider, label in [('Roll', self.roll, self.roll_label),
                                     ('Pitch', self.pitch, self.pitch_label)]:
            row = QHBoxLayout()
            row.addWidget(QLabel(title + ' offset'))
            slider.setRange(-TILT_LIMIT_DEG, TILT_LIMIT_DEG)
            slider.setValue(0)
            slider.setFocusPolicy(Qt.NoFocus)
            slider.setToolTip(f'Requested standing body {title.lower()} angle in degrees, '
                              'limited to ±5° by this app. It is not a validated pose.')
            slider.valueChanged.connect(self.posture_changed)
            row.addWidget(slider, 1)
            row.addWidget(label)
            posture_layout.addLayout(row)
        self.apply_button = QPushButton('Apply requested standing posture')
        self.apply_button.setObjectName('applyButton')
        self.apply_button.setEnabled(False)
        self.apply_button.setToolTip('Send one high-level stand request only after Spot is powered, '
                                     'standing, stationary, and manual or gesture movement is inactive.')
        self.apply_button.clicked.connect(self.apply_requested_posture)
        posture_layout.addWidget(self.apply_button)
        self.posture_message = QLabel(
            'Sliders edit the requested posture. Click Apply to send the high-level stand command to a stationary Spot.')
        self.posture_message.setWordWrap(True)
        posture_layout.addWidget(self.posture_message)
        self.posture_changed()


        gesture_title = QLabel('Supervised gesture mode')
        gesture_title.setObjectName('sectionTitle')
        control_layout.addWidget(gesture_title)
        self.gesture_toggle = QCheckBox('Enable supervised gesture mode')
        self.gesture_toggle.setEnabled(False)
        self.gesture_toggle.setToolTip('Opt in after Stand. Uses the built-in front-left camera and aligned depth; '
                                       'keyboard driving is disabled while active.')
        self.gesture_toggle.toggled.connect(self.toggle_gesture_mode)
        control_layout.addWidget(self.gesture_toggle)
        self.gesture_indicator = QLabel('GESTURE MODE OFF')
        self.gesture_indicator.setWordWrap(True)
        self.gesture_indicator.setStyleSheet('color: #a65b00; font-weight: 600;')
        control_layout.addWidget(self.gesture_indicator)
        control_layout.addStretch(1)
        self.setCentralWidget(root)

        self.power.setEnabled(False)
        self.stand.setEnabled(False)
        self._set_connection_state('demo' if demo else 'offline' if offline else 'connecting')
        app = QApplication.instance()
        app.installEventFilter(self)
        self.signals.status.connect(self.status.setText)
        self.signals.model_state.connect(self.show_model_state)
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
        if demo:
            self._start_demo()
        elif offline:
            self.gesture_indicator.setText('GESTURE MODE OFF — offline preview')
        else:
            self.session.start()

    def _make_dock(self, title, name, content):
        dock = QDockWidget(title, self.dock_host)
        dock.setObjectName(name)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable |
                         QDockWidget.DockWidgetClosable)
        dock.setWidget(content)
        return dock

    def _build_layout_menu(self):
        for dock in (self.camera_dock, self.model_dock, self.controls_dock):
            self.layout_menu.addAction(dock.toggleViewAction())
        self.layout_menu.addSeparator()
        for name in ('Balanced', 'Camera Focus', 'Model Focus'):
            self.layout_menu.addAction(name, lambda checked=False, preset=name:
                                       self._place_docks(preset))
        self.layout_menu.addSeparator()
        self.layout_menu.addAction('Restore Default Layout', self.restore_default_layout)

    def _place_docks(self, preset):
        docks = (self.camera_dock, self.model_dock, self.controls_dock)
        for dock in docks:
            self.dock_host.removeDockWidget(dock)
            dock.setFloating(False)
            self.dock_host.addDockWidget(Qt.LeftDockWidgetArea, dock)
            dock.show()
        self.dock_host.splitDockWidget(self.camera_dock, self.model_dock, Qt.Horizontal)
        if preset == 'Balanced':
            self.dock_host.splitDockWidget(self.model_dock, self.controls_dock, Qt.Horizontal)
        elif preset == 'Camera Focus':
            self.dock_host.splitDockWidget(self.model_dock, self.controls_dock, Qt.Vertical)
        else:
            self.dock_host.splitDockWidget(self.camera_dock, self.controls_dock, Qt.Vertical)
        self._pending_layout = preset
        if self.isVisible():
            QTimer.singleShot(0, self._size_docks)

    def _size_docks(self):
        preset = self._pending_layout
        if preset:
            width = self.dock_host.width()
            height = self.dock_host.height()
            if preset == 'Balanced':
                self.dock_host.resizeDocks(
                    [self.camera_dock, self.model_dock, self.controls_dock],
                    [int(width * .47), int(width * .30), int(width * .23)], Qt.Horizontal)
            elif preset == 'Camera Focus':
                self.dock_host.resizeDocks(
                    [self.camera_dock, self.model_dock],
                    [int(width * .60), int(width * .40)], Qt.Horizontal)
                self.dock_host.resizeDocks(
                    [self.model_dock, self.controls_dock],
                    [int(height * .62), int(height * .38)], Qt.Vertical)
            else:
                self.dock_host.resizeDocks(
                    [self.camera_dock, self.model_dock],
                    [int(width * .42), int(width * .58)], Qt.Horizontal)
                self.dock_host.resizeDocks(
                    [self.camera_dock, self.controls_dock],
                    [int(height * .60), int(height * .40)], Qt.Vertical)
            self._pending_layout = None

    def restore_default_layout(self):
        self._place_docks('Balanced')
        self.settings.remove('dashboard/state')

    def showEvent(self, event):
        super().showEvent(event)
        if self._pending_layout:
            QTimer.singleShot(0, self._size_docks)

    def _set_connection_state(self, state):
        states = {
            'connecting': ('Connecting to Spot', '#b7791f',
                           'Authenticating and discovering built-in cameras. Watch Terminal for prompts.'),
            'connected': ('Connected • not standing', '#b7791f',
                          'Use Power On if needed, then Stand to unlock keyboard movement.'),
            'powered': ('Powered • stand required', '#b7791f',
                        'Press Stand before using the keyboard or applying posture.'),
            'ready': ('Standing confirmed • keyboard unlocked', '#08756e',
                      'Hold a direction key to move; release it to request zero velocity.'),
            'failed': ('Disconnected • movement disabled', '#b4232a',
                       'Connection lost. Use the separate class E-stop if needed.'),
            'offline': ('Offline posture preview', '#566b7a',
                        'No robot connection, commands, or camera imagery.'),
            'demo': ('Offline demo • simulated camera scenes', '#566b7a',
                     'No robot connection or commands. Camera imagery is original grayscale test artwork.'),
        }
        title, color, detail = states[state]
        self.state_title.setText(title)
        self.state_dot.setStyleSheet(f'background: {color}; border: none; border-radius: 5px;')
        self.status.setText(detail)

    def _start_demo(self):
        """Exercise the UI with static synthetic frames; never start the robot worker."""
        self.on_connected(list(CAMERAS), False)
        self.ready = False
        self.power.setEnabled(False)
        self.stand.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.gesture_toggle.setEnabled(False)
        self.gesture_indicator.setText('OFF • demo has no gesture recognition')
        self._demo_frames = make_demo_frames()
        self._demo_panorama = make_demo_panorama()
        self.model_view.show_demo_reference()
        QTimer.singleShot(0, lambda: self.show_frame(self._demo_frames))
        self._demo_timer = QTimer(self)
        self._demo_timer.setInterval(int(PANORAMA_PERIOD * 1000))
        self._demo_timer.timeout.connect(self._demo_tick)
        self._demo_timer.start()
        QTimer.singleShot(0, self._demo_tick)
        self._set_connection_state('demo')

    def _demo_tick(self):
        if self.auto_panorama.isChecked():
            self.show_panorama(self._demo_panorama)
            self.show_panorama_status('Static front composite refreshed for UI review; '
                                      'live stitching is not exercised in demo')

    def _demo_auto_changed(self, enabled):
        if enabled and hasattr(self, '_demo_panorama'):
            self._demo_tick()
        elif not enabled:
            self.show_panorama_status('Demo auto refresh off; front composite remains visible')

    def _demo_stitch_once(self):
        self.show_panorama(self._demo_panorama)
        self.show_panorama_status('Static simulated front composite shown; '
                                  'live stitching is not exercised in demo')

    def on_connected(self, cameras, powered):
        if self.failed:
            return
        self.ready = True
        if not self.demo:
            self._set_connection_state('powered' if powered else 'connected')
        for source in cameras:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(12, 12, 12, 12)
            page_layout.setSpacing(8)
            status = QLabel('Loading first image from Spot…' if not self.demo else
                            'Simulated grayscale scene • no Spot camera connection')
            status.setObjectName('cameraStatus')
            page_layout.addWidget(status)
            self.feed_status[source] = status
            image_label = self._new_image_label((280, 110))
            image_label.setText('Waiting for first camera image…' if not self.demo else
                                'Loading simulated scene…')
            image_label.setToolTip('Built-in fisheye camera. Grayscale remains grayscale when supplied by Spot.'
                                   if not self.demo else 'Original simulated grayscale test scene; not Spot footage.')
            page_layout.addWidget(image_label, 1)
            self.tabs.insertTab(len(self.feed_labels), page, self._camera_title(source))
            self.tabs.setTabToolTip(len(self.feed_labels),
                                    f'Show the {self._camera_title(source).lower()} built-in fisheye feed.')
            self.feed_labels[source] = image_label
        self.tabs.setCurrentIndex(0)

        split_page = QWidget()
        split_page.setMinimumHeight(480)
        split_grid = QGridLayout(split_page)
        split_grid.setContentsMargins(12, 12, 12, 12)
        split_grid.setSpacing(10)
        for index, source in enumerate(cameras):
            tile = QWidget()
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(0, 0, 0, 0)
            tile_layout.setSpacing(4)
            tile_title = QLabel(self._camera_title(source))
            tile_title.setObjectName('sectionTitle')
            tile_layout.addWidget(tile_title)
            status = QLabel('Loading…' if not self.demo else 'Simulated scene')
            status.setObjectName('cameraStatus')
            tile_layout.addWidget(status)
            self.split_status[source] = status
            image_label = self._new_image_label((240, 130))
            image_label.setText('Waiting for image…' if not self.demo else 'Loading demo…')
            image_label.setToolTip('Built-in fisheye view in the split screen.' if not self.demo else
                                   'Original simulated grayscale scene; not Spot footage.')
            tile_layout.addWidget(image_label, 1)
            split_grid.addWidget(tile, index // 2, index % 2)
            self.split_labels[source] = image_label
        split_scroll = QScrollArea()
        split_scroll.setWidgetResizable(True)
        split_scroll.setWidget(split_page)
        self.tabs.insertTab(len(cameras), split_scroll, 'Split screen')
        self.tabs.setTabToolTip(len(cameras), 'View every available built-in fisheye feed at once.')

        panorama_page = QWidget()
        panorama_page.setMinimumHeight(470)
        panorama_layout = QVBoxLayout(panorama_page)
        panorama_layout.setContentsMargins(12, 12, 12, 12)
        panorama_layout.setSpacing(8)
        pano_title = QLabel('Approximate front panorama')
        pano_title.setObjectName('sectionTitle')
        panorama_layout.addWidget(pano_title)
        self.panorama_image = PanoramaView()
        self.panorama_image.setMinimumSize(280, 180)
        self.panorama_image.setToolTip('Drag to pan, scroll to zoom. The front-left and front-right '
                                       'images are aligned by features, without calibrated 360° coverage.')
        panorama_layout.addWidget(self.panorama_image, 1)
        panorama_note = QLabel('Approximate front-left + front-right fisheye stitch. '
                               'No calibration or 360° coverage. Drag to pan; wheel to zoom.')
        panorama_note.setObjectName('muted')
        panorama_note.setWordWrap(True)
        panorama_layout.addWidget(panorama_note)
        self.panorama_status = QLabel('Front panorama idle; individual feeds stay live.')
        self.panorama_status.setObjectName('cameraStatus')
        self.panorama_status.setWordWrap(True)
        panorama_layout.addWidget(self.panorama_status)
        self.auto_panorama = QCheckBox('Continuously update (about every 2 seconds)')
        self.auto_panorama.setToolTip('Request a front-pair stitch at most about once every 2 seconds '
                                      'on a background thread. Individual feeds continue if alignment fails.')
        self.auto_panorama.toggled.connect(self._demo_auto_changed if self.demo else
                                            self.session.set_auto_panorama)
        panorama_layout.addWidget(self.auto_panorama)
        self.stitch_button = QPushButton('Stitch current front frames once')
        self.stitch_button.setToolTip('Try aligning fresh front-left and front-right frames once. '
                                      'The result is approximate and may fail in low-detail scenes.')
        self.stitch_button.clicked.connect(self._demo_stitch_once if self.demo else
                                           self.session.request_panorama)
        panorama_layout.addWidget(self.stitch_button)
        panorama_scroll = QScrollArea()
        panorama_scroll.setWidgetResizable(True)
        panorama_scroll.setWidget(panorama_page)
        self.tabs.insertTab(len(cameras) + 1, panorama_scroll, 'Panorama')
        self.tabs.setTabToolTip(len(cameras) + 1,
                                'Approximate front-left and front-right stitch; no 360° coverage.')
        self.source_select.addItems([self.tabs.tabText(index)
                                     for index in range(self.tabs.count())])
        self.tabs.tabBar().hide()
        self.source_select.setCurrentIndex(0)
        if not all(source in cameras for source in FRONT_PAIR):
            self.auto_panorama.setEnabled(False)
            self.stitch_button.setEnabled(False)
            self.panorama_status.setText('Front camera pair unavailable; individual feeds continue')
        else:
            self.auto_panorama.setChecked(True)
        self.power.setEnabled(not powered and not self.demo)
        self.stand.setEnabled(powered and not self.demo)

    def on_powered(self):
        if self.failed:
            return
        self.power.setEnabled(False)
        self.stand.setEnabled(True)
        self._set_connection_state('powered')

    def on_armed(self):
        if self.failed:
            return
        self.armed = True
        self.power.setEnabled(False)
        self.apply_button.setEnabled(not self.pressed)
        models_ready = (importlib.util.find_spec('mediapipe') is not None and
                        (MODEL_DIR / 'gesture_recognizer.task').is_file() and
                        (MODEL_DIR / 'pose_landmarker_lite.task').is_file())
        self.gesture_toggle.setEnabled(bool(self.session.gesture_depth_source) and models_ready)
        if not self.session.gesture_depth_source:
            self.gesture_indicator.setText('GESTURE MODE OFF • aligned front depth unavailable')
        elif not models_ready:
            self.gesture_indicator.setText('GESTURE MODE OFF • model files or MediaPipe missing; see README')
        self._set_connection_state('ready')

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
        self.gesture_indicator.setStyleSheet('color: #a65b00; font-weight: 600;')
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
        self.roll.setEnabled(False)
        self.pitch.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.gesture_toggle.setEnabled(False)
        self._set_connection_state('failed')
        self.status.setText(f'{message}. Use the separate class E-stop if needed.')
        self.model_view.set_measured_state(None, 'Disconnected • model telemetry unavailable')
        for label in (*self.feed_status.values(), *self.split_status.values()):
            label.setText('Feed stopped • last image may remain visible')
        if hasattr(self, 'panorama_status'):
            self.panorama_status.setText('Disconnected • last stitch may remain visible')

    def show_frame(self, picture):
        if self.failed:
            return
        for source, frame in picture.items():
            pixmap = QPixmap.fromImage(ImageQt(frame))
            self._put_frame(self.feed_labels.get(source), pixmap)
            self._put_frame(self.split_labels.get(source), pixmap)
            message = ('Simulated grayscale scene • not Spot footage' if self.demo else
                       'Receiving built-in Spot camera images')
            if source in self.feed_status:
                self.feed_status[source].setText(message)
            if source in self.split_status:
                self.split_status[source].setText('Simulated scene' if self.demo else 'Receiving images')

    def show_model_state(self, angles, message):
        if not self.failed:
            self.model_view.set_measured_state(angles, message)

    def show_panorama(self, picture):
        self.panorama_image.set_picture(picture)

    def show_panorama_status(self, message):
        if hasattr(self, 'panorama_status'):
            self.panorama_status.setText(('SIMULATED • ' if self.demo else '') + message)

    @staticmethod
    def _new_image_label(size):
        return CameraImageLabel(size)

    @staticmethod
    def _camera_title(source):
        names = {'frontleft_fisheye_image': 'Front left',
                 'frontright_fisheye_image': 'Front right',
                 'left_fisheye_image': 'Left',
                 'right_fisheye_image': 'Right',
                 'back_fisheye_image': 'Back'}
        return names.get(source, source.replace('_', ' ').title())

    @staticmethod
    def _put_frame(label, picture):
        if label is None:
            return
        label.set_picture(picture)

    def posture_changed(self, value=None):
        self.height_label.setText(f'requested +{self.height.value()} cm')
        self.roll_label.setText(f'{self.roll.value():+d}°')
        self.pitch_label.setText(f'{self.pitch.value():+d}°')

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
        elif event.type() == QEvent.MouseButtonPress and obj is self.source_select:
            self.stop_motion()
        elif event.type() in (QEvent.KeyPress, QEvent.KeyRelease) and event.key() in KEYS:
            if QApplication.activePopupWidget() is not None or self.source_select.view().isVisible():
                self.stop_motion()
                return False
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
                self.session.set_keys(self.pressed, released=True)
            return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        self.stop_motion()
        self.settings.setValue('dashboard/state', self.dock_host.saveState(1))
        if self.demo:
            self._demo_timer.stop()
        elif not self.offline:
            self.session.close()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hostname', help='Spot hostname or IP; confirm with instructor')
    parser.add_argument('--offline-preview', action='store_true',
                        help='Show posture controls without connecting to a robot or showing camera footage')
    parser.add_argument('--demo', action='store_true',
                        help='Show simulated grayscale camera scenes without any robot connection or commands')
    args = parser.parse_args()
    if not args.offline_preview and not args.demo and not args.hostname:
        parser.error('--hostname is required unless --offline-preview or --demo is set')
    app = QApplication([])
    window = MainWindow(args.hostname, offline=args.offline_preview, demo=args.demo)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
