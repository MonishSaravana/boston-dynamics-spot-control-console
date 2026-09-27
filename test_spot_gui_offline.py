"""Offline checks for display and posture guards; never creates a robot client."""

import io
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from bosdyn.api import image_pb2, robot_state_pb2
from PIL import Image
from PySide6.QtWidgets import QApplication

from spot_control_gui import (HEIGHT_LIMIT_CM, MainWindow, SpotSession, UiSignals,
                              decode_visual_image)


class GuiOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

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

    def test_posture_request_rejected_while_moving_or_stopping(self):
        session = SpotSession(None, UiSignals())
        self.assertFalse(session.apply_posture(-1, 0, 0))
        self.assertFalse(session.apply_posture(HEIGHT_LIMIT_CM + 1, 0, 0))
        session.set_keys({ord('W')})
        self.assertFalse(session.apply_posture(5, 0, 0))
        session.action('stop')
        self.assertFalse(session.apply_posture(5, 0, 0))
        session.pending = None
        self.assertTrue(session.apply_posture(5, 2, -2))
        self.assertEqual(session.pending[0], 'posture')

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


if __name__ == '__main__':
    unittest.main()
