"""Automatic alignment, E-stop gating, saved login and non-blocking shutdown, offline."""
from dataclasses import replace
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scope import robot_login
from scope.alignment import AlignmentCheck, edge_alignment, passes, registered_depth_for
from scope.console_backend import DemoBackend, SpotBackend
from scope.workspace import Workspace

VISUAL, DEPTH = "frontleft_fisheye_image", "frontleft_depth_in_visual_frame"


def frames(shift=0, count=6):
    base = DemoBackend().frame
    depth = np.roll(base.depth_m, shift, axis=1)
    return [replace(base, depth_m=depth, frame_id=f"f{i}") for i in range(count)]


class AlignmentTests(unittest.TestCase):
    def test_registered_pair_names(self):
        self.assertEqual(registered_depth_for(VISUAL), DEPTH)
        self.assertIsNone(registered_depth_for("hand_color_image"))

    def test_aligned_depth_passes_and_shifted_depth_fails(self):
        aligned = frames()[0]
        self.assertTrue(passes(*edge_alignment(aligned.rgb, aligned.depth_m)))
        shifted = frames(shift=8)[0]
        self.assertFalse(passes(*edge_alignment(shifted.rgb, shifted.depth_m)))

    def test_check_needs_several_distinct_frames(self):
        check = AlignmentCheck()
        aligned = frames()
        self.assertEqual(check.observe(aligned[0]), "checking")
        self.assertEqual(check.observe(aligned[0]), "checking")   # same frame is not counted twice
        check.observe(aligned[1])
        self.assertEqual(check.observe(aligned[2]), "passed")

    def test_check_gives_up_on_misaligned_depth_or_no_frames(self):
        check = AlignmentCheck()
        for frame in frames(shift=8, count=25):
            check.observe(frame)
        self.assertEqual(check.state, "failed")
        now = [0.]
        idle = AlignmentCheck(clock=lambda: now[0])
        now[0] = 21.
        self.assertEqual(idle.give_up_if_slow(), "failed")


class FakeSource:
    """Shape of SpotReadOnlySource that SpotBackend.tick uses."""

    def __init__(self, shift=0):
        self.sources = {VISUAL: None, DEPTH: None}
        self.policy = {name: SimpleNamespace(process=True, display=True, acquire=True)
                       for name in self.sources}
        self.alignment_verified, self.alignment_method = False, None
        self.frames = iter(frames(shift, count=500))
        rgb = frames()[0].rgb
        self.latest = {VISUAL: SimpleNamespace(pixels=rgb, source=VISUAL, intrinsics=None,
                                               acquisition_robot_s=1., receive_host_s=time.monotonic()),
                       DEPTH: SimpleNamespace(acquisition_robot_s=1.)}
        self.robot = None

    def configure(self, *args, **kwargs):
        pass

    def poll(self):
        return self.latest

    def robot_pose(self):
        raise ValueError("no robot in tests")

    def rgbd(self, visual, depth, assume_aligned=False):
        if not (self.alignment_verified or assume_aligned):
            raise ValueError("Depth alignment requires physical verification")
        return next(self.frames)

    def mark_processed(self, *args):
        pass

    def status(self):
        return {}


class LiveAlignmentTests(unittest.TestCase):
    def test_registered_pair_is_confirmed_and_starts_the_map(self):
        backend = SpotBackend("", VISUAL, DEPTH, source=FakeSource())
        try:
            deadline = time.monotonic()+10
            while time.monotonic() < deadline and backend.pipeline is None:
                backend.tick()
                time.sleep(.05)
            self.assertEqual(backend.source.alignment_method, "automatic")
            self.assertIsNotNone(backend.pipeline)
            self.assertTrue(backend.pipeline.runtime.state("mapping").config.enabled)
        finally:
            backend.close()

    def test_unregistered_pair_is_never_confirmed_automatically(self):
        backend = SpotBackend("", VISUAL, "right_depth", source=SimpleNamespace(
            sources={VISUAL: None, "right_depth": None}, configure=lambda *a, **k: None,
            alignment_verified=False, alignment_method=None))
        self.assertIsNone(backend.alignment)
        backend.close()


class CloseTests(unittest.TestCase):
    def test_close_stops_go_immediately_and_finishes_in_background(self):
        backend = SpotBackend.__new__(SpotBackend)
        backend.mode = "SPOT_SUPERVISED"
        backend.executor = Mock()
        backend.pool = Mock()
        backend.lease_keepalive = None
        backend.pipeline = SimpleNamespace(close=lambda: time.sleep(2))
        start = time.monotonic()
        backend.close()
        self.assertLess(time.monotonic()-start, .5)
        backend.executor.stop.assert_called_once()


class LoginTests(unittest.TestCase):
    def status(self, name, silence_s):
        duration = SimpleNamespace(seconds=int(silence_s), nanos=int((silence_s % 1)*1e9))
        return SimpleNamespace(endpoints=[SimpleNamespace(
            endpoint=SimpleNamespace(name=name), time_since_valid_response=duration)])

    def test_estop_must_be_ours_and_checking_in(self):
        self.assertTrue(robot_login.estop_active(self.status("Estop NoGUI", .3)))
        self.assertFalse(robot_login.estop_active(self.status("Estop NoGUI", 9.)))
        self.assertFalse(robot_login.estop_active(self.status("tablet", .1)))

    def test_blank_password_uses_keychain_or_explains(self):
        workspace = SimpleNamespace(control=SimpleNamespace(demo=False, session=None), backend=None)
        with patch.object(robot_login, "saved_password", return_value=None):
            with self.assertRaisesRegex(ValueError, "Start Spot.command"):
                Workspace._connect(workspace, {"hostname": "", "username": "", "password": ""}, 0)

    def test_auto_connect_waits_for_estop_then_takes_robot_control(self):
        from concurrent.futures import Future
        done = Future()
        done.set_result(None)
        workspace = SimpleNamespace(lock=threading.RLock(), message="", submit=Mock(return_value=done))
        app = SimpleNamespace(workspace=workspace, closed=threading.Event())
        with patch.object(robot_login, "saved_password", return_value="pw"), \
                patch.object(robot_login, "wait_for_estop", return_value=True):
            self.assertTrue(robot_login.auto_connect(app, "192.168.80.3", "user"))
        name, data = workspace.submit.call_args[0]
        self.assertEqual((name, data["mode"], data["command_authority"]), ("connect", "robot_control", True))
        with patch.object(robot_login, "saved_password", return_value="pw"), \
                patch.object(robot_login, "wait_for_estop", return_value=False):
            workspace.submit.reset_mock()
            self.assertFalse(robot_login.auto_connect(app, "192.168.80.3", "user"))
            workspace.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
