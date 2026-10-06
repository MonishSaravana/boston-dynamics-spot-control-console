"""Offline integration checks for the unified browser; never connect to Spot."""
from dataclasses import replace
from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scope.live_interaction import DryRunExecutor, RobotPose
from scope.workspace import GuardedDispatch, RunLibrary
from scope_web import WebControl, make_server


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.app = WebControl(demo=True)
        self.workspace = self.app.workspace
        self.workspace.initialization.result(5)
        self.workspace.submit('query', {'phrase': 'chair'}).result(5)

    def tearDown(self):
        self.app.close()

    def action(self, name, **data):
        data.setdefault('revision', self.workspace.revision)
        return self.workspace.submit(name, data).result(10)

    def target(self):
        self.action('select', entity_id='chair_a')
        self.action('confirm')

    def test_typed_confirm_preview_virtual_go_is_single_use(self):
        state = self.workspace.snapshot()
        self.assertEqual({c['entity_id'] for c in state['candidates']}, {'chair_a', 'chair_b'})
        self.assertIsNone(state['selected'])
        self.target()
        self.assertTrue(self.workspace.snapshot()['can_go'])
        self.action('go', controller='tab')
        self.assertEqual(self.workspace.snapshot()['message'], 'SIMULATED_ARRIVAL')
        self.assertFalse(self.workspace.snapshot()['can_go'])
        with self.assertRaisesRegex(ValueError, 'No confirmed destination'):
            self.action('go', controller='tab')
        self.assertIsNone(self.app.session)

    def test_pointing_direct_selection_reject_share_the_candidate_path(self):
        self.action('point')
        self.assertEqual(self.workspace.snapshot()['selected']['source'], 'POINTING')
        self.action('select', entity_id='backpack')
        self.assertEqual(self.workspace.snapshot()['selected']['source'], 'DIRECT')
        self.action('reject')
        self.assertIsNone(self.workspace.snapshot()['selected'])

    def test_observe_and_demo_cannot_enable_robot_control(self):
        self.action('mode', mode='observe')
        self.target()
        with self.assertRaisesRegex(ValueError, 'no movement authorization'):
            self.action('go', controller='tab')
        with self.assertRaisesRegex(ValueError, 'command authority'):
            self.action('mode', mode='robot_control')
        self.assertIsNone(self.app.session)

    def test_dry_run_uses_no_physical_executor(self):
        self.target()
        dry = DryRunExecutor()
        self.workspace.backend.executor = dry
        self.workspace.interaction.executor = dry
        self.action('go', controller='tab')
        self.assertEqual(len(dry.proposals), 1)
        self.assertIsNone(self.app.session)
        self.assertEqual(self.workspace.snapshot()['message'], 'WOULD_EXECUTE_NO_MOTION')

    def test_stop_cancels_go_after_slow_route_check_without_waiting(self):
        self.target()
        entered, release = threading.Event(), threading.Event()
        original = self.workspace.backend.ready_for_go
        def slow(candidate):
            entered.set()
            release.wait(2)
            return original(candidate)
        self.workspace.backend.ready_for_go = slow
        executor = self.workspace.backend.executor
        future = self.workspace.submit('go', {'revision': self.workspace.revision, 'controller': 'tab'})
        self.assertTrue(entered.wait(2))
        start = time.monotonic()
        self.app.stop()
        self.assertLess(time.monotonic() - start, .2)
        release.set()
        with self.assertRaisesRegex(ValueError, 'cancelled'):
            future.result(5)
        self.assertFalse(executor.commands)

    def test_target_revision_rejects_delayed_approval_from_another_tab(self):
        old = self.workspace.revision
        self.action('select', entity_id='chair_a')
        with self.assertRaisesRegex(ValueError, 'selection changed'):
            self.action('confirm', revision=old)
        self.assertIsNone(self.workspace.interaction.confirmed)

    def test_known_target_behind_robot_proposes_facing_heading(self):
        b = self.workspace.backend
        self.action('select', entity_id='chair_a')
        # Fixture is behind the robot's heading. Camera visibility is not a
        # prerequisite of direct world-frame selection or destination planning.
        b.robot = replace(b.robot, yaw_rad=-2.2)
        self.action('confirm')
        p = self.workspace.snapshot()['destination']
        self.assertGreater(abs(p['yaw_rad'] - b.robot.yaw_rad), 2)
        self.assertEqual(p['world_frame'], 'simulated-world')

    def test_navigation_presence_timeout_stops_and_clears_owner(self):
        fake = Mock()
        self.workspace.backend.executor = fake
        self.workspace.navigation_controller = 'tab'
        self.workspace.last_presence = time.monotonic() - 1
        self.workspace.watchdog()
        self.assertIsNone(self.workspace.navigation_controller)
        fake.stop.assert_called()

    def test_disconnect_does_not_hold_command_lock_while_worker_changes_mode(self):
        entered, release = threading.Event(), threading.Event()
        original = self.app.stop
        def paused_stop():
            if threading.current_thread() is self.workspace.thread:
                entered.set()
                release.wait(3)
            original()
        with patch.object(self.app, 'stop', paused_stop):
            mode = self.workspace.submit('mode', {'mode': 'observe'})
            self.assertTrue(entered.wait(2))
            disconnect = threading.Thread(target=self.app.disconnect)
            disconnect.start()
            deadline = time.monotonic() + 2
            while not self.app.disconnecting and time.monotonic() < deadline:
                time.sleep(.01)
            release.set()
            disconnect.join(3)
            self.assertFalse(disconnect.is_alive())
            mode.result(3)
        self.assertIsNone(self.workspace.backend)
        self.app.set_demo(True)
        self.assertFalse(self.app.disconnecting)

    def test_stop_with_full_queue_still_cancels_pending_go(self):
        self.target()
        entered, release = threading.Event(), threading.Event()
        original = self.workspace.backend.ready_for_go
        def slow(candidate):
            entered.set()
            release.wait(3)
            return original(candidate)
        self.workspace.backend.ready_for_go = slow
        go = self.workspace.submit('go', {'revision': self.workspace.revision, 'controller': 'tab'})
        self.assertTrue(entered.wait(2))
        for _ in range(32):
            self.workspace.tasks.put_nowait(('query', {'phrase': 'chair'}, self.workspace.epoch, Future()))
        self.app.stop()
        release.set()
        with self.assertRaisesRegex(ValueError, 'cancelled'):
            go.result(3)

    def test_navigation_stop_failure_still_requests_manual_zero(self):
        manual = Mock()
        self.app.session = manual
        with patch.object(self.workspace.backend.executor, 'stop', side_effect=RuntimeError('offline')):
            self.app.stop()
        manual.action.assert_called_with('stop')

    def test_go_refuses_a_manual_request_already_in_progress(self):
        self.workspace.mode = 'robot_control'
        self.app.session = Mock(lock=threading.RLock(), armed=True, powered=True,
                                pending=None, active_action='stand', keys=set(), gesture_mode=False)
        self.app.connected = self.app.armed = True
        executor = Mock()
        guard = GuardedDispatch(self.workspace, executor, self.workspace.epoch, 'tab')
        with self.assertRaisesRegex(ValueError, 'manual request'):
            guard.execute(Mock())
        executor.execute.assert_not_called()


class RunLibraryTests(unittest.TestCase):
    def test_roots_reject_traversal_and_symlink_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / 'runs'; root.mkdir()
            outside = base / 'private.json'; outside.write_text('{}')
            (root / 'escape.json').symlink_to(outside)
            (root / 'metrics.json').write_text('{"source":"SIMULATED"}')
            library = RunLibrary([root])
            self.assertEqual(len(library.scan()), 1)
            self.assertEqual(library.inspect('0/metrics.json')['source'], 'SIMULATED')
            for key in ('0/../private.json', '0/escape.json', '9/metrics.json'):
                with self.assertRaises(ValueError):
                    library.inspect(key)


class WorkspaceHttpTests(unittest.TestCase):
    def test_routes_and_stop_reject_a_delayed_workspace_go(self):
        server = make_server(demo=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f'http://127.0.0.1:{server.server_port}'
        headers = {'Origin': base, 'X-SCOPE-Token': server.app.token, 'Content-Type': 'application/json'}
        def post(name, **data):
            return urlopen(Request(base + '/api/workspace/action',
                data=json.dumps({'name': name, **data}).encode(), headers=headers))
        try:
            with urlopen(base) as response:
                self.assertIn(b'page-maps', response.read())
            w = server.app.workspace
            w.initialization.result(5)
            with post('query', epoch=w.epoch, phrase='chair'):
                pass
            epoch = w.epoch
            server.app.stop()
            with self.assertRaises(HTTPError) as error:
                post('go', epoch=epoch, revision=w.revision, controller='tab')
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
            with urlopen(Request(base + '/api/runs', headers=headers)) as response:
                self.assertIn('artifacts', json.load(response))
        finally:
            server.shutdown(); server.app.close(); server.server_close()


if __name__ == '__main__':
    unittest.main()
