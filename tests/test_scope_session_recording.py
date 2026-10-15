"""Live session recording, checked offline with the synthetic room only."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scope.console_backend import DemoBackend
from scope.session_recording import SessionRecorder, copy_map
from scope.workspace import RunLibrary, Workspace

ROOT = Path(__file__).resolve().parents[1]


def control():
    return SimpleNamespace(demo=True, disconnecting=False, failed=False, session=None,
                           cameras=[], stop=lambda: None)


def demo_sample(backend, host_s, with_map=False):
    frame = backend.tick()
    sample = {"host_s": host_s, "jpeg": None, "frame": frame, "geometry": True,
              "robot": {"x_m": backend.robot.x_m, "y_m": backend.robot.y_m,
                        "yaw_rad": backend.robot.yaw_rad},
              "entities": backend.entities}
    if with_map:
        positions, colors, _ = backend.mapping.cloud()
        sample["map_cloud"] = (positions, colors)
        sample["topdown"] = backend.mapping.topdown()
    return sample


def verify_rrd(path):
    executable = Path(sys.executable).with_name("rerun")
    if not executable.exists():
        executable = shutil.which("rerun")
    if not executable:
        return None
    return subprocess.run([str(executable), "rrd", "verify", str(path)],
                          capture_output=True, text=True, timeout=60)


class SessionRecorderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.backend = DemoBackend()

    def recorder(self, **kwargs):
        recorder = SessionRecorder(self.root, info={"backend_mode": "SIMULATED",
                                                    "world_frame": "simulated-world"}, **kwargs)
        self.addCleanup(recorder.join, 10)
        return recorder

    def test_session_folder_holds_recording_map_entities_and_manifest(self):
        recorder = self.recorder()
        for i in range(3):
            self.assertTrue(recorder.offer(demo_sample(self.backend, recorder.started_host_s + i,
                                                       with_map=i == 0)))
        recorder.finish(copy_map(self.backend.mapping), self.backend.entities, "operator")
        self.assertTrue(recorder.join(20))
        status = recorder.status()
        self.assertEqual(status["state"], "saved", status["error"])
        self.assertEqual(status["samples"], 3)
        folder = recorder.folder
        self.assertTrue(folder.name.startswith("live-"))
        self.assertEqual((folder / ".gitignore").read_text(), "*\n")
        manifest = json.loads((folder / "manifest.json").read_text())
        self.assertEqual(manifest["stopped_by"], "operator")
        self.assertEqual(manifest["map_revision"], self.backend.mapping.revision)
        self.assertNotIn("hostname", manifest)
        entities = json.loads((folder / "entities.json").read_text())["entities"]
        self.assertEqual({e["entity_id"] for e in entities},
                         {e["entity_id"] for e in self.backend.entities})
        from scope.storage import load_map
        saved = load_map(folder)
        np.testing.assert_array_equal(saved.hits, self.backend.mapping.hits)
        result = verify_rrd(folder / "session.rrd")
        if result is not None:
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_offer_never_waits_for_a_slow_disk(self):
        release = threading.Event()
        recorder = self.recorder()
        with patch.object(SessionRecorder, "_log", lambda *a: release.wait(10)):
            sample = demo_sample(self.backend, recorder.started_host_s)
            start = time.perf_counter()
            results = [recorder.offer(dict(sample, host_s=sample["host_s"] + i)) for i in range(40)]
            elapsed = time.perf_counter() - start
            release.set()
            recorder.finish(None, [], "operator")
            self.assertTrue(recorder.join(20))
        self.assertLess(elapsed, .5)
        self.assertIn(False, results)
        self.assertGreater(recorder.status()["dropped"], 0)
        self.assertEqual(recorder.status()["state"], "saved")

    def test_size_limit_stops_accepting_samples(self):
        recorder = self.recorder(max_bytes=1)
        recorder.offer(demo_sample(self.backend, recorder.started_host_s))
        deadline = time.monotonic() + 10
        while not recorder.limit_reached and time.monotonic() < deadline:
            time.sleep(.05)
        self.assertTrue(recorder.limit_reached)
        self.assertFalse(recorder.due(time.monotonic() + 100))
        self.assertFalse(recorder.offer(demo_sample(self.backend, recorder.started_host_s + 1)))
        self.assertIn("limit", recorder.status()["warning"])
        recorder.finish(None, [], "size limit")
        self.assertTrue(recorder.join(20))

    def test_sampling_is_rate_limited(self):
        recorder = self.recorder(sample_hz=2)
        now = recorder.started_host_s
        self.assertTrue(recorder.due(now))
        recorder.offer(demo_sample(self.backend, now))
        self.assertFalse(recorder.due(now + .2))
        self.assertTrue(recorder.due(now + .5))
        recorder.finish(None, [], "operator")


class WorkspaceRecordingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.workspace = Workspace(control(), demo=True, runs_dirs=[self.root])
        self.addCleanup(self.workspace.close)
        self.workspace.initialization.result(5)

    def record(self, enabled):
        return self.workspace.submit('record', {'enabled': enabled}).result(10)

    def wait_for_samples(self, count=2):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = self.workspace.snapshot()['recording']
            if status['samples'] >= count:
                return status
            time.sleep(.1)
        self.fail('Recorder received no samples')

    def test_record_toggle_saves_a_run_listed_in_history(self):
        self.assertIsNone(self.workspace.snapshot()['recording'])
        self.record(True)
        with self.assertRaisesRegex(ValueError, 'Already recording'):
            self.record(True)
        self.wait_for_samples()
        self.record(False)
        self.assertTrue(self.workspace.recorder.join(20))
        status = self.workspace.snapshot()['recording']
        self.assertEqual(status['state'], 'saved', status['error'])
        self.assertEqual(status['reason'], 'operator')
        listed = RunLibrary([self.root]).scan()
        names = {item['name'] for item in listed if item['run'] == status['folder']}
        self.assertTrue({'session.rrd', 'map.npz', 'entities.json', 'manifest.json'} <= names)
        rrd = next(item for item in listed if item['name'] == 'session.rrd')
        self.assertEqual(rrd['kind'], 'Viewer recording')
        with patch('scope.viewer.subprocess.Popen') as popen:
            RunLibrary([self.root]).open_viewer(rrd['id'])
        self.assertTrue(popen.call_args[0][0][-1].endswith('session.rrd'))

    def test_stop_does_not_wait_for_or_end_the_recording(self):
        self.record(True)
        self.wait_for_samples(1)
        with patch.object(SessionRecorder, 'finish') as finish:
            start = time.perf_counter()
            self.workspace.stop()
            self.assertLess(time.perf_counter() - start, .1)
        finish.assert_not_called()
        self.assertEqual(self.workspace.snapshot()['recording']['state'], 'recording')

    def test_leaving_the_source_saves_the_recording(self):
        self.record(True)
        self.wait_for_samples(1)
        recorder = self.workspace.recorder
        self.workspace.submit('disconnect', {}).result(10)
        self.assertTrue(recorder.join(20))
        self.assertEqual(recorder.status()['reason'], 'disconnected')
        self.assertTrue((recorder.folder / 'map.npz').exists())
        with self.assertRaisesRegex(ValueError, 'offline demo before recording'):
            self.record(True)

    def test_finish_without_recording_and_invalid_values_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'No recording in progress'):
            self.record(False)
        with self.assertRaisesRegex(ValueError, 'true or false'):
            self.workspace.submit('record', {'enabled': 'yes'}).result(10)

    def test_size_limit_finishes_and_saves_the_map(self):
        self.record(True)
        recorder = self.workspace.recorder
        recorder.max_bytes = 1  # The next size check trips the limit.
        self.assertTrue(recorder.join(20))
        self.assertEqual(recorder.status()['reason'], 'size limit')
        self.assertTrue((recorder.folder / 'map.npz').exists())


class PrivacyTests(unittest.TestCase):
    def test_default_run_folder_is_ignored_by_git(self):
        if shutil.which('git') is None or not (ROOT / '.git').exists():
            self.skipTest('Not a Git checkout')
        for path in ('runs/live-20261015-120000/session.rrd',
                     'runs/live-20261015-120000/map.npz',
                     'runs/live-20261015-120000/manifest.json'):
            result = subprocess.run(['git', 'check-ignore', '-q', path], cwd=ROOT)
            self.assertEqual(result.returncode, 0, f'{path} is not ignored')


if __name__ == '__main__':
    unittest.main()
