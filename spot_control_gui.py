#!/usr/bin/env python3
"""Basic Spot keyboard controller with a built-in camera view."""

import argparse
import io
import threading
import time

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
                               QPushButton, QSlider, QVBoxLayout, QWidget)


CAMERA = 'frontleft_fisheye_image'
COMMAND_DURATION = 0.35
COMMAND_PERIOD = 0.10
DEFAULT_SPEED = 0.20
KEYS = {Qt.Key_W: (1, 0), Qt.Key_Up: (1, 0),
        Qt.Key_S: (-1, 0), Qt.Key_Down: (-1, 0),
        Qt.Key_A: (0, 1), Qt.Key_Left: (0, 1),
        Qt.Key_D: (0, -1), Qt.Key_Right: (0, -1)}


class UiSignals(QObject):
    status = Signal(str)
    connected = Signal(bool)
    powered = Signal()
    armed = Signal()
    failed = Signal(str)
    frame = Signal(object)


class SpotSession:
    """Keep robot requests off the UI thread."""

    def __init__(self, hostname, signals):
        self.hostname = hostname
        self.signals = signals
        self.lock = threading.Lock()
        self.wakeup = threading.Event()
        self.closing = threading.Event()
        self.keys = set()
        self.pending = None
        self.speed = DEFAULT_SPEED
        self.powered = False
        self.client = None
        self.lease_keepalive = None
        self.robot = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.camera_thread = None

    def start(self):
        self.thread.start()

    def set_keys(self, keys):
        with self.lock:
            self.keys = set(keys)
            if not keys:
                self.pending = 'stop'
        self.wakeup.set()

    def action(self, action):
        with self.lock:
            self.keys.clear()
            self.pending = action
        self.wakeup.set()

    def set_speed(self, speed):
        with self.lock:
            self.speed = max(0.05, min(0.35, speed))

    def _command(self, command, end_time_secs=None):
        self.client.robot_command(command=command, end_time_secs=end_time_secs, timeout=0.7)

    def _zero(self):
        self._command(RobotCommandBuilder.synchro_velocity_command(0, 0, 0),
                      end_time_secs=time.time() + COMMAND_DURATION)

    def _camera_loop(self, image_client):
        while not self.closing.is_set():
            try:
                response = image_client.get_image(
                    [build_image_request(CAMERA, quality_percent=55)], timeout=1.5)[0]
                image = response.shot.image
                if image.format == image_pb2.Image.FORMAT_JPEG:
                    picture = Image.open(io.BytesIO(image.data)).convert('L')
                elif image.format == image_pb2.Image.FORMAT_RAW and (
                        image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8):
                    picture = Image.frombytes('L', (image.cols, image.rows), image.data)
                else:
                    raise ValueError('Unsupported visual camera image')
                picture.thumbnail((720, 480))
                self.signals.frame.emit(picture.copy())
                self.closing.wait(0.15)
            except Exception as exc:
                self.signals.failed.emit(f'Camera problem: {exc}')
                self.closing.set()
                self.wakeup.set()
                return

    def _run(self):
        try:
            self.signals.status.emit('Connecting; enter credentials in Terminal when prompted')
            sdk = bosdyn.client.create_standard_sdk('SpotControlGUI')
            self.robot = sdk.create_robot(self.hostname)
            bosdyn.client.util.authenticate(self.robot)
            self.robot.time_sync.wait_for_sync(timeout_sec=10)
            image_client = self.robot.ensure_client(ImageClient.default_service_name)
            available = {source.name for source in image_client.list_image_sources(timeout=2)}
            if CAMERA not in available:
                raise RuntimeError('Front-left built-in camera is unavailable')
            self.client = self.robot.ensure_client(RobotCommandClient.default_service_name)
            lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
            self.lease_keepalive = LeaseKeepAlive(lease_client, must_acquire=True,
                                                   return_at_exit=True)
            self.powered = self.robot.is_powered_on()
            self.signals.connected.emit(self.powered)
            self.camera_thread = threading.Thread(target=self._camera_loop,
                                                  args=(image_client,), daemon=True)
            self.camera_thread.start()
            next_command = time.monotonic()
            while not self.closing.is_set():
                with self.lock:
                    action, self.pending = self.pending, None
                    keys = set(self.keys)
                if action == 'stop':
                    if self.powered:
                        self._zero()
                    self.signals.status.emit('Stopped')
                elif action == 'power':
                    self.robot.power_on(timeout_sec=20)
                    self.powered = True
                    self.signals.powered.emit()
                elif action == 'stand':
                    self._zero()
                    self._command(RobotCommandBuilder.synchro_stand_command())
                    self.signals.armed.emit()
                elif keys and time.monotonic() >= next_command:
                    with self.lock:
                        live_keys = (set(self.keys) if not self.closing.is_set() and
                                     self.pending != 'stop' else set())
                        if live_keys:
                            forward = int(Qt.Key_W in live_keys or Qt.Key_Up in live_keys) - int(
                                Qt.Key_S in live_keys or Qt.Key_Down in live_keys)
                            left = int(Qt.Key_A in live_keys or Qt.Key_Left in live_keys) - int(
                                Qt.Key_D in live_keys or Qt.Key_Right in live_keys)
                            scale = max(1, (forward * forward + left * left) ** 0.5)
                            self._command(RobotCommandBuilder.synchro_velocity_command(
                                self.speed * forward / scale, self.speed * left / scale, 0),
                                end_time_secs=time.time() + COMMAND_DURATION)
                    next_command = time.monotonic() + COMMAND_PERIOD
                self.wakeup.wait(0.05)
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
                    pass
            if self.lease_keepalive is not None:
                self.lease_keepalive.shutdown()

    def close(self):
        self.closing.set()
        self.action('stop')
        if self.thread.is_alive():
            self.thread.join(timeout=25)


class MainWindow(QMainWindow):
    def __init__(self, hostname, offline=False):
        super().__init__()
        self.setWindowTitle('Spot Control Console')
        self.signals = UiSignals()
        self.session = SpotSession(hostname, self.signals)
        self.held = set()
        self.armed = False
        root = QWidget()
        layout = QVBoxLayout(root)
        self.status = QLabel('Offline preview' if offline else 'Connecting')
        layout.addWidget(self.status)
        self.camera = QLabel('Waiting for front-left camera')
        self.camera.setMinimumSize(640, 360)
        self.camera.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera, 1)
        controls = QHBoxLayout()
        self.power = QPushButton('Power On')
        self.stand = QPushButton('Stand')
        self.stop = QPushButton('STOP MOVEMENT')
        self.stand.setEnabled(False)
        for button in (self.power, self.stand, self.stop):
            controls.addWidget(button)
        layout.addLayout(controls)
        layout.addWidget(QLabel('Speed limit: 0.05–0.35 m/s'))
        self.speed = QSlider(Qt.Horizontal)
        self.speed.setRange(5, 35)
        self.speed.setValue(20)
        layout.addWidget(self.speed)
        layout.addWidget(QLabel('Hold W/A/S/D or arrow keys to move'))
        self.setCentralWidget(root)
        self.power.clicked.connect(lambda: self.session.action('power'))
        self.stand.clicked.connect(lambda: self.session.action('stand'))
        self.stop.clicked.connect(self.stop_motion)
        self.speed.valueChanged.connect(lambda value: self.session.set_speed(value / 100))
        self.signals.status.connect(self.status.setText)
        self.signals.failed.connect(self.status.setText)
        self.signals.connected.connect(self.on_connected)
        self.signals.powered.connect(lambda: self.stand.setEnabled(True))
        self.signals.armed.connect(self.on_armed)
        self.signals.frame.connect(self.show_frame)
        QApplication.instance().installEventFilter(self)
        if offline:
            self.power.setEnabled(False)
            self.stop.setEnabled(False)
        else:
            self.session.start()

    def on_connected(self, powered):
        self.status.setText('Connected; keep the class E-stop running separately')
        self.stand.setEnabled(powered)

    def on_armed(self):
        self.armed = True
        self.status.setText('Stand requested; hold a direction key to move')

    def show_frame(self, picture):
        pixmap = QPixmap.fromImage(ImageQt(picture))
        self.camera.setPixmap(pixmap.scaled(self.camera.size(), Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation))

    def stop_motion(self):
        self.held.clear()
        self.session.action('stop')

    def eventFilter(self, obj, event):
        if event.type() == QEvent.WindowDeactivate:
            self.stop_motion()
        if self.armed and event.type() in (QEvent.KeyPress, QEvent.KeyRelease):
            if event.key() in KEYS:
                if event.type() == QEvent.KeyPress:
                    self.held.add(event.key())
                else:
                    self.held.discard(event.key())
                self.session.set_keys(self.held)
                return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        self.stop_motion()
        self.session.close()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hostname', help='Spot hostname or IP; confirm with instructor')
    parser.add_argument('--offline-preview', action='store_true')
    args = parser.parse_args()
    if not args.hostname and not args.offline_preview:
        parser.error('--hostname is required unless --offline-preview is set')
    app = QApplication([])
    window = MainWindow(args.hostname, offline=args.offline_preview)
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
