import time
import unittest
from dataclasses import replace

import numpy as np

from scope.live_interaction import (
    DryRunExecutor, InteractionError, InteractionSession, RobotPose,
    SelectionSource, TargetCandidate, VirtualExecutor,
    candidates_from_pointing, candidates_from_query, direct_candidate)
from scope.mapping import MapConfig, VoxelMap
from scope.spot_live_source import SpotImage, SpotReadOnlySource, pair_to_frame
from scope.spot_navigation import SpotTrajectoryExecutor


def free_map():
    mapping = VoxelMap(MapConfig((-4., -4., 0.), (80, 80, 20), .1))
    mapping.frees[:] = 1
    mapping.revision = 1
    mapping.world_frame = "odom"
    return mapping


def entity():
    return {"entity_id": "chair_1", "label": "chair",
            "bounds_low_m": [-.3, -.3, 0.], "bounds_high_m": [.3, .3, 1.],
            "confidence": .8, "last_seen_s": 10.}


def candidate(now, **kw):
    value = direct_candidate(entity(), now, "odom")
    return replace(value, **kw)


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.executor = VirtualExecutor()
        self.session = InteractionSession(self.executor, clock=lambda: self.now)
        self.mapping = free_map()
        self.robot = RobotPose(-2., 0., 0., self.now, "odom")

    def test_query_confirm_preview_go_are_separate(self):
        query = {"state": "FOUND", "source_timestamp_s": 7., "user_phrase": "chair",
                 "candidates": [{"entity_id": "chair_1", "score": .8,
                                 "detector_match": "chair", "box_xyxy": (1, 2, 3, 4)}]}
        offered = candidates_from_query(query, [entity()], self.now, "odom")
        self.assertEqual(offered[0].source, SelectionSource.QUERY)
        self.session.offer(offered)
        self.assertFalse(self.executor.commands)
        self.session.confirm()
        self.assertFalse(self.executor.commands)
        proposal = self.session.propose(self.robot, self.mapping)
        self.assertNotAlmostEqual(proposal.x_m, 0.)
        self.assertFalse(self.executor.commands)
        self.assertEqual(self.session.go(self.robot, self.mapping), "SIMULATED_ARRIVAL")
        self.assertEqual(len(self.executor.commands), 1)
        with self.assertRaises(InteractionError):
            self.session.go(self.robot, self.mapping)

    def test_pointing_abstains_or_uses_same_flow(self):
        ranking = {"state": "DEGRADED", "scores": {"chair_1": .72, "unknown": .28}}
        offered, unresolved = candidates_from_pointing(
            ranking, [entity()], self.now, "odom")
        self.assertFalse(unresolved)
        self.assertEqual(offered[0].source, SelectionSource.POINTING)
        self.session.offer(offered)
        self.session.confirm()
        self.session.propose(self.robot, self.mapping)
        self.assertFalse(self.executor.commands)
        abstain, unresolved = candidates_from_pointing(
            {"state": "ABSTAIN", "scores": {}}, [entity()], self.now, "odom")
        self.assertTrue(unresolved)
        self.assertFalse(abstain)

    def test_ambiguous_reject_and_direct(self):
        second = replace(candidate(self.now), entity_id="chair_2")
        self.session.offer([candidate(self.now), second], ambiguous=True)
        self.assertIsNone(self.session.selected)
        with self.assertRaises(InteractionError):
            self.session.confirm()
        self.session.select("chair_2")
        self.session.reject()
        self.assertEqual(self.session.selected.entity_id, "chair_1")
        self.assertEqual(self.session.selected.source, SelectionSource.DIRECT)
        self.assertFalse(self.executor.commands)

    def test_stale_lost_moved_map_and_state_fail_closed(self):
        self.session.offer([candidate(self.now, track_state="TRACK_LOST")])
        with self.assertRaisesRegex(InteractionError, "STALE or lost"):
            self.session.confirm()
        self.session.offer([candidate(self.now)])
        self.session.confirm()
        self.session.propose(self.robot, self.mapping)
        self.now += 1.6
        with self.assertRaisesRegex(InteractionError, "STALE"):
            self.session.go(self.robot, self.mapping)
        self.assertFalse(self.executor.commands)
        self.now = 10.
        self.session.offer([candidate(self.now)])
        self.session.confirm()
        self.session.propose(self.robot, self.mapping)
        self.mapping.revision += 1
        self.mapping.frees[:] = 0
        with self.assertRaisesRegex(InteractionError, "DESTINATION BLOCKED"):
            self.session.go(self.robot, self.mapping)
        self.assertFalse(self.executor.commands)
        self.mapping = free_map()
        self.session.offer([candidate(self.now)])
        self.session.confirm()
        with self.assertRaisesRegex(InteractionError, "ROBOT STATE STALE"):
            self.session.propose(replace(self.robot, observed_host_s=8.), self.mapping)
        self.assertFalse(self.executor.commands)

    def test_refresh_keeps_stable_approval_and_invalidates_movement(self):
        original = candidate(self.now)
        self.session.offer([original])
        self.session.confirm()
        self.session.propose(self.robot, self.mapping)
        self.now += 1.
        self.robot = replace(self.robot, observed_host_s=self.now)
        self.assertTrue(self.session.refresh_observation(candidate(self.now)))
        self.assertIsNotNone(self.session.proposal)
        self.assertFalse(self.session.refresh_observation(
            candidate(self.now, bounds_low_m=(.2, .2, 0.))))
        self.assertIsNone(self.session.proposal)
        self.assertIsNone(self.session.confirmed)
        self.assertFalse(self.executor.commands)

    def test_depth_map_unknown_blocked_and_stop(self):
        no_depth = candidate(self.now, bounds_low_m=None, bounds_high_m=None)
        self.session.offer([no_depth])
        self.session.confirm()
        with self.assertRaisesRegex(InteractionError, "DEPTH UNAVAILABLE"):
            self.session.propose(self.robot, self.mapping)
        self.session.offer([candidate(self.now)])
        self.session.confirm()
        with self.assertRaisesRegex(InteractionError, "MAP UNAVAILABLE"):
            self.session.propose(self.robot, None)
        wrong_frame = free_map()
        wrong_frame.world_frame = "vision"
        with self.assertRaisesRegex(InteractionError, "MAP FRAME"):
            self.session.propose(self.robot, wrong_frame)
        blocked = free_map()
        blocked.frees[:] = 0
        with self.assertRaisesRegex(InteractionError, "DESTINATION BLOCKED"):
            self.session.propose(self.robot, blocked)
        self.session.stop()
        self.assertFalse(self.executor.commands)

    def test_dry_run_has_no_robot_commands(self):
        dry = DryRunExecutor()
        session = InteractionSession(dry, clock=lambda: self.now)
        session.offer([candidate(self.now)])
        session.confirm()
        session.propose(self.robot, self.mapping)
        self.assertEqual(session.go(self.robot, self.mapping), "WOULD_EXECUTE_NO_MOTION")
        self.assertEqual(len(dry.proposals), 1)
        self.assertFalse(self.executor.commands)


class SpotNormalizationTests(unittest.TestCase):
    def test_verified_fisheye_pair_rectifies_to_pinhole_contract(self):
        from scope.data import Intrinsics
        K = Intrinsics(64, 48, 50., 50., 31.5, 23.5)
        visual = SpotImage("visual", np.full((48, 64, 1), 120, np.uint8),
            100., 10., 123., "camera", np.eye(4), "odom", K, None, 12., 2.,
            "kannala_brandt", (0., 0., 0., 0.))
        depth = SpotImage("depth", np.full((48, 64, 1), 1000, np.uint16),
            100., 10., 123., "camera", np.eye(4), "odom", K, 1000., 12., 2.,
            "kannala_brandt", (0., 0., 0., 0.))
        frame = pair_to_frame(visual, depth, alignment_verified=True)
        frame.validate()
        self.assertAlmostEqual(float(frame.depth_m[24, 32]), 1.)
        self.assertEqual(frame.source, "SPOT_LIVE")

    def test_paired_sources_are_requested_together(self):
        from bosdyn.api import image_pb2
        class Client:
            def __init__(self): self.calls = []
            def list_image_sources(self, **kwargs):
                return [image_pb2.ImageSource(name=name, cols=4, rows=3)
                        for name in ("visual", "depth")]
            def get_image(self, requests, **kwargs):
                self.calls.append([r.image_source_name for r in requests])
                result = []
                for _ in requests:
                    response = image_pb2.ImageResponse(status=image_pb2.ImageResponse.STATUS_OK)
                    response.shot.acquisition_time.seconds = 100
                    response.shot.image.CopyFrom(image_pb2.Image(rows=3, cols=4,
                        format=image_pb2.Image.FORMAT_RAW,
                        pixel_format=image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8,
                        data=bytes(12)))
                    result.append(response)
                return result
        client = Client()
        source = SpotReadOnlySource(client)
        source.poll()
        self.assertEqual(client.calls, [["visual", "depth"]])
        self.assertEqual(source.health, {"visual": "OK", "depth": "OK"})

    def test_one_camera_failure_does_not_stop_others(self):
        from bosdyn.api import image_pb2
        class Client:
            def list_image_sources(self, **kwargs):
                return [image_pb2.ImageSource(name=name, cols=4, rows=3)
                        for name in ("good", "broken")]
            def get_image(self, requests, **kwargs):
                if requests[0].image_source_name == "broken":
                    raise RuntimeError("network camera error")
                response = image_pb2.ImageResponse(status=image_pb2.ImageResponse.STATUS_OK)
                response.shot.acquisition_time.seconds = 100
                response.shot.image.CopyFrom(image_pb2.Image(rows=3, cols=4,
                    format=image_pb2.Image.FORMAT_RAW,
                    pixel_format=image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8,
                    data=bytes(12)))
                return [response]
        source = SpotReadOnlySource(Client())
        source.configure("good", display=False, max_hz=2.)
        source.poll()
        self.assertEqual(source.health["good"], "OK")
        self.assertEqual(source.health["broken"], "FAILED")
        self.assertFalse(source.status()["good"]["displayed"])
        self.assertEqual(source.timings.snapshot()["sensor_rpc/good"]["count"], 1)

    def test_grayscale_registered_depth_requires_explicit_verification(self):
        from scope.data import Intrinsics
        K = Intrinsics(4, 3, 3., 3., 1.5, 1.)
        visual = SpotImage("camera", np.zeros((3, 4, 1), np.uint8), 100., 10.,
                           123., "camera", np.eye(4), "odom", K, None, 12., 2.)
        depth = SpotImage("depth", np.full((3, 4, 1), 1000, np.uint16), 100.01, 10.,
                          123., "camera", np.eye(4), "odom", K, 1000., 12., 2.)
        with self.assertRaisesRegex(ValueError, "verification"):
            pair_to_frame(visual, depth)
        frame = pair_to_frame(visual, depth, alignment_verified=True)
        self.assertEqual(frame.rgb.shape, (3, 4, 3))
        self.assertAlmostEqual(float(frame.depth_m[0, 0]), 1.)
        with self.assertRaisesRegex(ValueError, "timing"):
            pair_to_frame(visual, replace(depth, acquisition_robot_s=101.),
                          alignment_verified=True)

    def test_no_transform_or_scale_cannot_produce_geometry(self):
        from scope.data import Intrinsics
        K = Intrinsics(4, 3, 3., 3., 1.5, 1.)
        visual = SpotImage("camera", np.zeros((3, 4, 3), np.uint8), 100., 10.,
                           123., "camera", None, "odom", K, None, 12., 2.)
        depth = SpotImage("depth", np.full((3, 4, 1), 1000, np.uint16), 100., 10.,
                          123., "camera", np.eye(4), "odom", K, None, 12., 2.)
        with self.assertRaisesRegex(ValueError, "transform"):
            pair_to_frame(visual, depth, alignment_verified=True)
        with self.assertRaisesRegex(ValueError, "scale"):
            pair_to_frame(replace(visual, T_root_sensor=np.eye(4)), depth,
                          alignment_verified=True)


class NavigationTests(unittest.TestCase):
    def test_stop_before_dispatch_sends_no_trajectory(self):
        class Client:
            def __init__(self): self.sent = []
            def robot_command(self, **kw):
                self.sent.append(kw["command"])
                return len(self.sent)
        client = Client()
        executor = SpotTrajectoryExecutor(client,
            lambda: RobotPose(-2., 0., 0., 10., "odom"), lambda _: True,
            enabled=True, clock=lambda: 10.)
        session = InteractionSession(VirtualExecutor(), clock=lambda: 10.)
        session.offer([candidate(10.)]); session.confirm()
        destination = session.propose(RobotPose(-2., 0., 0., 10., "odom"), free_map())
        executor.stop()
        with self.assertRaisesRegex(InteractionError, "Stop requested"):
            executor.execute(destination)
        self.assertTrue(client.sent)
        self.assertTrue(all(cmd.synchronized_command.mobility_command.HasField("se2_velocity_request")
                            for cmd in client.sent))

    def test_sdk_adapter_is_disabled_until_commissioned(self):
        class Client:
            calls = 0
            def robot_command(self, **kwargs):
                self.calls += 1
        client = Client()
        executor = SpotTrajectoryExecutor(client, lambda: None, lambda _: True)
        with self.assertRaisesRegex(InteractionError, "disabled"):
            executor.execute(object())
        self.assertEqual(client.calls, 0)

    def test_sdk_trajectory_has_short_expiry_speed_limit_and_zero(self):
        from types import SimpleNamespace as Obj
        from bosdyn.api import basic_command_pb2
        proposal_session = InteractionSession(VirtualExecutor(), clock=lambda: 10.)
        proposal_session.offer([candidate(10.)])
        proposal_session.confirm()
        destination = proposal_session.propose(RobotPose(-2., 0., 0., 10., "odom"), free_map())
        class Client:
            def __init__(self): self.sent = []
            def robot_command(self, **kw):
                self.sent.append(kw)
                return len(self.sent)
            def robot_command_feedback(self, *args, **kw):
                traj = Obj(status=1, STATUS_AT_GOAL=1,
                           body_movement_status=2, BODY_STATUS_SETTLED=2)
                mobility = Obj(status=basic_command_pb2.RobotCommandFeedbackStatus.STATUS_PROCESSING,
                               se2_trajectory_feedback=traj)
                return Obj(feedback=Obj(synchronized_feedback=Obj(mobility_command_feedback=mobility)))
        client = Client()
        clock = lambda: 10.
        executor = SpotTrajectoryExecutor(client,
            lambda: RobotPose(destination.x_m, destination.y_m,
                              destination.yaw_rad, 10., "odom"), lambda _: True,
            enabled=True, clock=clock)
        self.assertEqual(executor.execute(destination), "ARRIVED_MEASURED_FEEDBACK")
        self.assertEqual(len(client.sent), 2)
        command = client.sent[0]["command"]
        request = command.synchronized_command.mobility_command.se2_trajectory_request
        self.assertEqual(request.se2_frame_name, "odom")
        self.assertAlmostEqual(request.trajectory.points[0].pose.position.x, destination.x_m)
        self.assertLess(client.sent[0]["end_time_secs"]-time.time(), 1.)
        self.assertTrue(client.sent[1]["command"].synchronized_command.mobility_command.HasField("se2_velocity_request"))


class ConsoleIntegrationTests(unittest.TestCase):
    def test_simulated_buttons_share_confirm_then_go(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets
        from scope.m5_console import DemoBackend, InteractionWindow
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        backend = DemoBackend()
        window = InteractionWindow(backend)
        try:
            window.query.setText("chair")
            window.find()
            self.assertEqual(len(window.session.candidates), 2)
            self.assertFalse(backend.executor.commands)
            window.entity_clicked(window.entity_list.item(0))
            self.assertEqual(window.session.selected.source, SelectionSource.QUERY)
            window.confirm_clicked()
            self.assertIsNotNone(window.session.proposal)
            self.assertFalse(backend.executor.commands)
            window.go_clicked()
            self.assertEqual(len(backend.executor.commands), 1)
            window.point_clicked()
            self.assertEqual(window.session.selected.source, SelectionSource.POINTING)
            self.assertEqual(len(backend.executor.commands), 1)
            window.entity_clicked(window.entity_list.item(2))
            self.assertEqual(window.session.selected.source, SelectionSource.DIRECT)
            self.assertEqual(len(backend.executor.commands), 1)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
