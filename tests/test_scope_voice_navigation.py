import json
import math
import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace as Obj
from unittest import mock
from urllib.request import Request, urlopen

import numpy as np

from scope.commands import Action, Relation, parse_command
from scope.live_interaction import (
    LEG_MAX_M, InteractionError, InteractionSession, RobotPose, VirtualExecutor,
    _clear_disk, direct_candidate, propose_destination)
from scope.mapping import MapConfig, VoxelMap
from scope.path_planning import densify
from scope.spot_navigation import SpotTrajectoryExecutor
from scope.voice import Transcriber


def free_map():
    mapping = VoxelMap(MapConfig((-4., -4., 0.), (80, 80, 20), .1))
    mapping.frees[:] = 1
    mapping.revision = 1
    mapping.world_frame = "odom"
    return mapping


def couch(now, low=(-.9, -.4, 0.), high=(.9, .4, .9)):
    entity = {"entity_id": "couch_1", "label": "couch", "bounds_low_m": list(low),
              "bounds_high_m": list(high), "confidence": .8, "last_seen_s": None}
    return direct_candidate(entity, now, "odom")


def mark(mapping, low, high, occupied=True):
    cfg = mapping.config
    lo = np.floor((np.asarray(low) - cfg.origin) / cfg.voxel_m).astype(int)
    hi = np.ceil((np.asarray(high) - cfg.origin) / cfg.voxel_m).astype(int)
    if occupied:
        mapping.hits[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
    else:
        mapping.frees[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 0


def length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


class CommandTests(unittest.TestCase):
    def test_relations_and_targets(self):
        cases = {
            "go behind the couch": ("couch", Relation.BEHIND),
            "Hey Spot, go behind the couch.": ("couch", Relation.BEHIND),
            "walk to the left of the table": ("table", Relation.LEFT),
            "go to the right hand side of the bin": ("bin", Relation.RIGHT),
            "go next to the red chair": ("red chair", Relation.NEAR),
            "go to the kitchen door": ("kitchen door", Relation.FRONT),
            "find the backpack": ("backpack", Relation.FRONT),
            "chair": ("chair", Relation.FRONT),
        }
        for text, (target, relation) in cases.items():
            parsed = parse_command(text)
            self.assertEqual((parsed.action, parsed.target, parsed.relation),
                             (Action.FIND, target, relation), text)

    def test_pointing_and_control_words(self):
        parsed = parse_command("go behind that")
        self.assertTrue(parsed.pointing)
        self.assertEqual(parsed.relation, Relation.BEHIND)
        self.assertEqual(parse_command("Stop!").action, Action.STOP)
        self.assertEqual(parse_command("stop moving now").action, Action.STOP)
        self.assertEqual(parse_command("Go.").action, Action.GO)
        self.assertEqual(parse_command("yes").action, Action.CONFIRM)
        self.assertEqual(parse_command("not that one").action, Action.REJECT)
        with self.assertRaises(ValueError):
            parse_command("  ")


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.robot = RobotPose(-2.5, 0., 0., self.now, "odom")

    def propose(self, mapping, relation="front", robot=None, reference=None, target=None):
        session = InteractionSession(VirtualExecutor(), clock=lambda: self.now)
        session.offer([target or couch(self.now)])
        return propose_destination(session.confirm(), robot or self.robot, mapping, self.now,
                                   relation=relation, reference_xy=reference)

    def assert_route_clear(self, mapping, p):
        self.assertTrue(all(_clear_disk(mapping, x, y, .42 + p.clearance_m)
                            for x, y in p.route_xy_m))
        self.assertTrue(all(math.dist(a, b) <= .1001 for a, b in
                            zip(p.route_xy_m, p.route_xy_m[1:])))

    def test_relations_choose_the_requested_side(self):
        mapping = free_map()
        mark(mapping, (-.9, -.4, 0.), (.9, .4, .9))
        front = self.propose(mapping, "front")
        self.assertLess(front.x_m, -.9)
        self.assertEqual(front.purpose, "ARRIVE")
        behind = self.propose(mapping, "behind", robot=replace(self.robot, x_m=-1.5, y_m=-1.3))
        # From the south-west, behind is to the north-east of the couch.
        self.assertGreater(behind.goal_xy_m[0] + behind.goal_xy_m[1], 1.)
        left = self.propose(mapping, "left")
        right = self.propose(mapping, "right")
        # Facing +x from x=-2.5, left is +y and right is -y.
        self.assertGreater(left.goal_xy_m[1], .4)
        self.assertLess(right.goal_xy_m[1], -.4)
        for p in (front, behind, left, right):
            self.assert_route_clear(mapping, p)
            self.assertLessEqual(length(p.route_xy_m), LEG_MAX_M + 1e-6)

    def test_route_goes_around_an_obstacle(self):
        mapping = free_map()
        mark(mapping, (-.9, -.4, 0.), (.9, .4, .9))
        p = self.propose(mapping, "behind")
        self.assertGreater(len(p.waypoints), 1)
        self.assert_route_clear(mapping, p)
        self.assertAlmostEqual(p.waypoints[-1][2], p.yaw_rad)

    def test_long_route_is_split_into_legs(self):
        mapping = free_map()
        target = couch(self.now, (2.5, -.3, 0.), (3.0, .3, .9))
        robot = replace(self.robot, x_m=-3.0)
        p = self.propose(mapping, "front", robot=robot, target=target)
        self.assertEqual(p.purpose, "LEG")
        self.assertAlmostEqual(length(p.route_xy_m), LEG_MAX_M, places=3)
        self.assertGreater(p.goal_xy_m[0], p.x_m)

    def test_unmapped_side_first_walks_to_a_viewpoint(self):
        mapping = free_map()
        mark(mapping, (-.9, -.4, 0.), (.9, .4, .9))
        mark(mapping, (-1.5, .4, 0.), (1.5, 3., 2.), occupied=False)
        p = self.propose(mapping, "behind", robot=replace(self.robot, x_m=0., y_m=-1.8))
        self.assertEqual(p.purpose, "LOOK")
        self.assertIn("not mapped", p.note)
        self.assert_route_clear(mapping, p)
        # The viewpoint faces the unmapped place, not the couch.
        heading = math.atan2(p.goal_xy_m[1] - p.y_m, p.goal_xy_m[0] - p.x_m)
        self.assertAlmostEqual(p.yaw_rad, heading)

    def test_unknown_robot_footprint_and_blocked_room_fail_closed(self):
        blocked = free_map()
        blocked.frees[:] = 0
        with self.assertRaisesRegex(InteractionError, "DESTINATION BLOCKED"):
            self.propose(blocked, "behind")

    def test_reference_keeps_behind_fixed_across_legs(self):
        mapping = free_map()
        mark(mapping, (-.9, -.4, 0.), (.9, .4, .9))
        session = InteractionSession(VirtualExecutor(), clock=lambda: self.now)
        session.set_relation(Relation.BEHIND)
        session.offer([couch(self.now)])
        session.confirm()
        first = session.propose(self.robot, mapping)
        moved = replace(self.robot, x_m=0., y_m=-2.)
        second = session.propose(moved, mapping)
        self.assertEqual(first.reference_xy_m, second.reference_xy_m)
        self.assertGreater(second.goal_xy_m[0], .9)

    def test_go_refuses_a_route_that_differs_from_the_preview(self):
        mapping = free_map()
        mark(mapping, (-.9, -.4, 0.), (.9, .4, .9))
        executor = VirtualExecutor()
        session = InteractionSession(executor, clock=lambda: self.now)
        session.set_relation(Relation.BEHIND)
        session.offer([couch(self.now)])
        session.confirm()
        shown = session.propose(self.robot, mapping)
        # Same goal, reached round the other side of the couch.
        other = replace(shown, route_xy_m=tuple((x, -y) for x, y in shown.route_xy_m),
                        waypoints=tuple((x, -y, yaw) for x, y, yaw in shown.waypoints))
        with mock.patch("scope.live_interaction.propose_destination", return_value=other):
            with self.assertRaisesRegex(InteractionError, "ROUTE CHANGED"):
                session.go(self.robot, mapping)
        self.assertIsNone(session.proposal)
        self.assertIsNone(executor.pose)


class WaypointExecutorTests(unittest.TestCase):
    def test_bent_route_is_commanded_corner_by_corner(self):
        from bosdyn.api import basic_command_pb2
        mapping = free_map()
        mark(mapping, (-.9, -.4, 0.), (.9, .4, .9))
        now = [10.]
        robot = RobotPose(-2.5, 0., 0., 10., "odom")
        session = InteractionSession(VirtualExecutor(), clock=lambda: now[0])
        session.set_relation(Relation.BEHIND)
        session.offer([couch(10.)])
        session.confirm()
        destination = session.propose(robot, mapping)
        self.assertGreater(len(destination.waypoints), 1)
        position = [robot.x_m, robot.y_m, 0.]

        class Client:
            def __init__(self):
                self.goals = []
            def robot_command(self, **kw):
                request = kw["command"].synchronized_command.mobility_command
                if request.HasField("se2_trajectory_request"):
                    point = request.se2_trajectory_request.trajectory.points[0].pose
                    self.goals.append((point.position.x, point.position.y))
                    # The fake robot reaches each commanded point before the next read.
                    position[:2] = point.position.x, point.position.y
                    position[2] = point.angle
                return len(self.goals)
            def robot_command_feedback(self, *args, **kw):
                trajectory = Obj(status=0, STATUS_AT_GOAL=1, body_movement_status=0,
                                 BODY_STATUS_SETTLED=2)
                mobility = Obj(status=basic_command_pb2.RobotCommandFeedbackStatus.STATUS_PROCESSING,
                               se2_trajectory_feedback=trajectory)
                return Obj(feedback=Obj(synchronized_feedback=Obj(mobility_command_feedback=mobility)))

        def wait(seconds):
            now[0] += seconds
            return False

        client = Client()
        executor = SpotTrajectoryExecutor(
            client, lambda: RobotPose(position[0], position[1], position[2], now[0], "odom"),
            lambda _: True, enabled=True, clock=lambda: now[0], wait=wait)
        self.assertEqual(executor.execute(destination), "ARRIVED_MEASURED_POSE")
        corners = list(dict.fromkeys(client.goals))
        self.assertEqual(len(corners), len(destination.waypoints))
        for (x, y), corner in zip(corners, destination.waypoints):
            self.assertAlmostEqual(x, corner[0], places=5)
            self.assertAlmostEqual(y, corner[1], places=5)


class CornerShortcutTests(unittest.TestCase):
    def test_next_corner_waits_until_the_shortcut_is_clear(self):
        from bosdyn.api import basic_command_pb2
        mapping = free_map()
        mark(mapping, (-.9, -.4, 0.), (.9, .4, .9))
        now = [10.]
        robot = RobotPose(-2.5, 0., 0., 10., "odom")
        session = InteractionSession(VirtualExecutor(), clock=lambda: now[0])
        session.set_relation(Relation.BEHIND)
        session.offer([couch(10.)])
        session.confirm()
        destination = session.propose(robot, mapping)
        self.assertGreater(len(destination.waypoints), 1)
        planned = np.asarray(densify(destination.route_xy_m, .02))

        def on_route(route):
            points = np.asarray(densify(route.route_xy_m, .02))
            gaps = np.linalg.norm(points[:, None] - planned[None], axis=2).min(axis=1)
            return bool(gaps.max() <= .06)

        position = [robot.x_m, robot.y_m, 0.]
        segments = []

        class Client:
            def robot_command(self, **kw):
                request = kw["command"].synchronized_command.mobility_command
                if request.HasField("se2_trajectory_request"):
                    point = request.se2_trajectory_request.trajectory.points[0].pose
                    goal = (point.position.x, point.position.y)
                    start = tuple(position[:2])
                    segments.append((start, goal))
                    # Each command first gets Spot 0.25 m short of the corner, then onto it.
                    gap = math.dist(start, goal)
                    t = 1. if gap <= .26 else (gap - .25) / gap
                    position[0] = start[0] + t * (goal[0] - start[0])
                    position[1] = start[1] + t * (goal[1] - start[1])
                    position[2] = point.angle
                return len(segments)
            def robot_command_feedback(self, *args, **kw):
                trajectory = Obj(status=0, STATUS_AT_GOAL=1, body_movement_status=0,
                                 BODY_STATUS_SETTLED=2)
                mobility = Obj(status=basic_command_pb2.RobotCommandFeedbackStatus.STATUS_PROCESSING,
                               se2_trajectory_feedback=trajectory)
                return Obj(feedback=Obj(synchronized_feedback=Obj(mobility_command_feedback=mobility)))

        def wait(seconds):
            now[0] += seconds
            return False

        executor = SpotTrajectoryExecutor(
            Client(), lambda: RobotPose(position[0], position[1], position[2], now[0], "odom"),
            on_route, enabled=True, clock=lambda: now[0], wait=wait)
        self.assertEqual(executor.execute(destination), "ARRIVED_MEASURED_POSE")
        for start, goal in segments:
            self.assertTrue(on_route(Obj(route_xy_m=densify((start, goal)))), (start, goal))


class DemoCommandTests(unittest.TestCase):
    def setUp(self):
        from scope_web import WebControl
        self.app = WebControl(demo=True)
        self.workspace = self.app.workspace
        self.workspace.initialization.result(5)

    def tearDown(self):
        self.app.close()

    def action(self, name, **data):
        data.setdefault('revision', self.workspace.revision)
        return self.workspace.submit(name, data).result(10)

    def test_go_behind_the_couch_walks_leg_by_leg(self):
        self.action('query', phrase='go behind the couch')
        state = self.workspace.snapshot()
        self.assertEqual(state['command']['relation'], 'behind')
        self.assertEqual(state['selected']['entity_id'], 'couch')
        self.action('confirm')
        purposes = []
        for _ in range(8):
            deadline = time.monotonic() + 5
            while self.workspace.snapshot()['destination'] is None and time.monotonic() < deadline:
                time.sleep(.05)
            destination = self.workspace.snapshot()['destination']
            self.assertIsNotNone(destination, self.workspace.snapshot()['message'])
            self.assertEqual(destination['relation'], 'behind')
            purposes.append(destination['purpose'])
            self.action('go', controller='tab')
            if destination['purpose'] == 'ARRIVE':
                break
        self.assertEqual(purposes[-1], 'ARRIVE')
        self.assertGreater(len(purposes), 1)
        robot = self.workspace.backend.robot
        self.assertGreater(robot.y_m, 1.45)  # past the back of the couch

    def test_picking_a_target_directly_drops_the_earlier_side(self):
        self.action('query', phrase='go behind the couch')
        self.assertEqual(self.workspace.snapshot()['relation'], 'behind')
        self.action('select', entity_id='table')
        state = self.workspace.snapshot()
        self.assertEqual(state['selected']['entity_id'], 'table')
        self.assertEqual(state['relation'], 'front')
        self.assertIsNone(state['command'])

    def test_control_words_are_not_searches(self):
        with self.assertRaisesRegex(ValueError, 'go command'):
            self.action('query', phrase='go')


class VoiceEndpointTests(unittest.TestCase):
    def test_audio_is_transcribed_and_parsed(self):
        from scope_web import make_server

        class Model:
            def transcribe(self, audio, **kw):
                self.samples = audio.size
                return iter([Obj(text=" Go behind the couch.", no_speech_prob=.01)]), None

        server = make_server(demo=True)
        server.app.voice = Transcriber(factory=Model)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f'http://127.0.0.1:{server.server_port}'
        headers = {'Origin': base, 'X-SCOPE-Token': server.app.token,
                   'Content-Type': 'application/octet-stream'}
        tone = (np.sin(np.arange(16000) / 5) * 8000).astype('<i2').tobytes()
        try:
            with urlopen(Request(base + '/api/voice', data=tone, headers=headers)) as response:
                result = json.load(response)
            self.assertEqual(result['text'], 'Go behind the couch.')
            self.assertEqual(result['command'], {'action': 'find', 'target': 'couch',
                                                 'relation': 'behind', 'pointing': False})
            from urllib.error import HTTPError
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base + '/api/voice', data=bytes(32000), headers=headers))
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base + '/api/voice', data=tone,
                                headers={**headers, 'X-SCOPE-Token': 'wrong'}))
            self.assertEqual(error.exception.code, 403)
            error.exception.close()
        finally:
            server.shutdown(); server.app.close(); server.server_close()


if __name__ == '__main__':
    unittest.main()
