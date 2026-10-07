"""Offline checks for display and posture guards; never creates a robot client."""

import io
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from bosdyn.api import image_pb2, robot_state_pb2
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from spot_control_gui import (HEIGHT_LIMIT_CM, MainWindow, SpotSession, UiSignals,
                              decode_visual_image)


class GuiOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings_dir = tempfile.TemporaryDirectory()
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, cls.settings_dir.name)
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        cls.settings_dir.cleanup()

    def setUp(self):
        QSettings(QSettings.IniFormat, QSettings.UserScope,
                  'SCOPE', 'Spot Control Console').clear()

    def test_camera_decoder_retains_grayscale(self):
        raw = image_pb2.Image(format=image_pb2.Image.FORMAT_RAW,
                              pixel_format=image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8,
                              rows=2, cols=2, data=bytes([0, 50, 100, 255]))
        self.assertEqual(decode_visual_image(raw).mode, 'L')
        self.assertEqual(decode_visual_image(raw).getpixel((1, 1)), 255)

        encoded = io.BytesIO()
        Image.new('L', (2, 2), 90).save(encoded, format='JPEG')
        jpeg = image_pb2.Image(format=image_pb2.Image.FORMAT_JPEG,
                               pixel_format=image_pb2.Image.PIXEL_FORMAT_UNKNOWN,
                               data=encoded.getvalue())
        self.assertEqual(decode_visual_image(jpeg).mode, 'L')

    def test_offline_preview_has_no_connection_or_apply(self):
        window = MainWindow(None, offline=True)
        try:
            self.assertFalse(window.session.thread.is_alive())
            self.assertFalse(window.apply_button.isEnabled())
            self.assertFalse(window.power.isEnabled())
            self.assertFalse(window.stand.isEnabled())
            self.assertEqual((window.height.minimum(), window.height.maximum()),
                             (0, HEIGHT_LIMIT_CM))
            window.height.setValue(5)
            self.assertIn('requested +5 cm', window.height_label.text())
        finally:
            window.close()

    def test_full_demo_never_starts_robot_worker(self):
        with patch('spot_control_gui.bosdyn.client.create_standard_sdk',
                   side_effect=AssertionError('robot SDK creation in demo')):
            window = MainWindow(None, demo=True)
            try:
                window.show()
                self.app.processEvents()
                self.assertEqual(window.tabs.count(), 7)
                self.assertEqual(window.source_select.count(), 7)
                self.assertTrue(window.auto_panorama.isChecked())
                self.assertIsNone(window.session.client)
                self.assertFalse(window.session.thread.is_alive())
                self.assertFalse(window.session.panorama_thread.is_alive())
                self.assertFalse(window.power.isEnabled())
                self.assertFalse(window.stand.isEnabled())
                self.assertFalse(window.apply_button.isEnabled())
                self.assertFalse(window.gesture_toggle.isEnabled())
                self.assertIsNone(window.session.pending)
                self.assertIsNotNone(window.model_view.angles)
                self.assertIn('not measured', window.model_view.message)
                self.assertTrue(all(frame.mode == 'L' for frame in window._demo_frames.values()))
                self.assertIn('simulated', window.state_title.text().lower())
                window.tabs.setCurrentIndex(5)
                self.app.processEvents()
                for label in window.split_labels.values():
                    self.assertLessEqual(label.pixmap().width(), label.width())
                    self.assertLessEqual(label.pixmap().height(), label.height())
            finally:
                window.close()

    def test_docks_presets_and_saved_visibility(self):
        window = MainWindow(None, demo=True)
        try:
            window.resize(1100, 740)
            window.show()
            self.app.processEvents()
            self.app.processEvents()
            self.assertTrue(all(dock.isVisible() for dock in
                                (window.camera_dock, window.model_dock, window.controls_dock)))
            self.assertLess(window.camera_dock.geometry().right(),
                            window.model_dock.geometry().left())
            self.assertLess(window.model_dock.geometry().right(),
                            window.controls_dock.geometry().left())
            window.source_select.setCurrentIndex(5)
            self.assertEqual(window.tabs.currentIndex(), 5)
            self.assertTrue(window.model_dock.isVisible())
            window._place_docks('Model Focus')
            for _ in range(5):
                self.app.processEvents()
            self.assertGreater(window.model_dock.width(), window.camera_dock.width())
            window.restore_default_layout()
            for _ in range(5):
                self.app.processEvents()
            self.assertGreater(window.camera_dock.width(), window.model_dock.width())
            window.model_dock.hide()
        finally:
            window.close()
        reopened = MainWindow(None, demo=True)
        try:
            reopened.show()
            self.app.processEvents()
            self.assertFalse(reopened.model_dock.isVisible())
            reopened.model_dock.toggleViewAction().trigger()
            self.assertTrue(reopened.model_dock.isVisible())
            reopened.restore_default_layout()
            self.app.processEvents()
            self.assertTrue(reopened.model_dock.isVisible())
        finally:
            reopened.close()

    def test_posture_request_rejected_while_moving_or_stopping(self):
        session = SpotSession(None, UiSignals())
        self.assertFalse(session.apply_posture(5, 0, 0))
        session.powered = session.armed = True
        self.assertFalse(session.apply_posture(-1, 0, 0))
        self.assertFalse(session.apply_posture(HEIGHT_LIMIT_CM + 1, 0, 0))
        session.set_keys({ord('W')})
        self.assertFalse(session.apply_posture(5, 0, 0))
        session.action('stop')
        self.assertFalse(session.apply_posture(5, 0, 0))
        session.pending = None
        self.assertTrue(session.apply_posture(5, 2, -2))
        self.assertEqual(session.pending[0], 'posture')

    def test_key_release_queues_zero_and_unconfirmed_stand_cannot_drive(self):
        session = SpotSession(None, UiSignals())
        session.set_keys({ord('W')})
        self.assertEqual(session.keys, set())
        session.powered = session.armed = True
        session.set_keys({ord('W')})
        self.assertEqual(session.keys, {ord('W')})
        session.set_keys({ord('A')}, released=True)
        self.assertEqual(session.keys, {ord('A')})
        self.assertEqual(session.pending, 'stop')

    def test_stationary_requires_fresh_standing_odometry(self):
        session = SpotSession(None, UiSignals())
        state = robot_state_pb2.RobotState()
        state.behavior_state.state = robot_state_pb2.BehaviorState.STATE_STANDING
        state.kinematic_state.acquisition_timestamp.seconds = 100
        state.kinematic_state.velocity_of_body_in_odom.linear.x = 0.0
        session.state_client = Mock(get_robot_state=Mock(return_value=state))
        session.robot = SimpleNamespace(time_sync=SimpleNamespace(
            robot_timestamp_from_local_secs=lambda _: SimpleNamespace(seconds=100,
                                                                       nanos=200_000_000)))
        self.assertTrue(session._robot_is_stationary())
        state.kinematic_state.velocity_of_body_in_odom.linear.x = 0.1
        self.assertFalse(session._robot_is_stationary())
        state.kinematic_state.velocity_of_body_in_odom.linear.x = 0.0
        state.kinematic_state.acquisition_timestamp.seconds = 98
        self.assertFalse(session._robot_is_stationary())

    def test_panorama_alignment_failure_does_not_fail_session(self):
        signals = UiSignals()
        statuses = []
        failures = []
        signals.panorama_status.connect(statuses.append)
        signals.failed.connect(failures.append)
        session = SpotSession(None, signals)
        with session.lock:
            for source in ('frontleft_fisheye_image', 'frontright_fisheye_image'):
                session.latest_frames[source] = Image.new('L', (64, 64), 80)
                session.frame_times[source] = time.monotonic()
        with patch('spot_control_gui.cv2.Stitcher_create',
                   return_value=Mock(stitch=Mock(return_value=(1, None)))):
            session.panorama_thread.start()
            session.request_panorama()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not any('Could not align' in s for s in statuses):
                self.app.processEvents()
                time.sleep(0.01)
            session.closing.set()
            session.panorama_requested.set()
            session.panorama_thread.join(timeout=1)
        self.assertTrue(any('Could not align' in s for s in statuses))
        self.assertEqual(failures, [])

    def test_camera_failure_disarms_before_reporting_failure(self):
        signals = UiSignals()
        session = SpotSession(None, signals)
        session.powered = session.armed = True
        session.keys = {ord('W')}
        reported = []
        signals.failed.connect(lambda message: reported.append(
            (message, session.armed, set(session.keys), session.closing.is_set())))
        session._camera_loop(Mock(get_image=Mock(side_effect=RuntimeError('camera lost'))))
        self.assertEqual(len(reported), 1)
        self.assertFalse(reported[0][1])
        self.assertEqual(reported[0][2], set())
        self.assertTrue(reported[0][3])


if __name__ == '__main__':
    unittest.main()
