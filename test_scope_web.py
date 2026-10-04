"""Offline checks for the local browser console and its update path."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scope_web import DRIVE_TIMEOUT, WebControl, make_server


class FakeSession:
    def __init__(self):
        self.calls = []

    def set_keys(self, keys, released=False):
        self.calls.append(('keys', set(keys), released))

    def action(self, name):
        self.calls.append(('action', name))

    def set_gesture_mode(self, enabled):
        self.calls.append(('gesture', enabled))

    def close(self):
        self.calls.append(('close',))


class BrowserConsoleTests(unittest.TestCase):
    def test_demo_serves_simulated_frames_without_robot_client(self):
        server = make_server(demo=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            request = Request(base + '/api/state', headers={'X-SCOPE-Token': server.app.token})
            with urlopen(request) as response:
                state = json.load(response)
            self.assertTrue(state['demo'])
            self.assertFalse(state['connected'])
            self.assertFalse(state['armed'])
            self.assertEqual(len(state['cameras']), 5)
            request = Request(base + '/api/frame/frontleft_fisheye_image',
                              headers={'X-SCOPE-Token': server.app.token})
            with urlopen(request) as response:
                self.assertEqual(response.headers['Content-Type'], 'image/jpeg')
                self.assertTrue(response.read().startswith(b'\xff\xd8'))
            with self.assertRaises(ValueError):
                server.app.connect('192.168.80.3', 'user', 'password')
        finally:
            server.shutdown()
            server.app.close()
            server.server_close()
            thread.join(timeout=2)

    def test_local_command_requires_origin_and_token(self):
        server = make_server(demo=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            for origin, token, expected in (
                ('https://other.example', server.app.token, 403),
                (base, 'wrong', 403),
                (base, server.app.token, 200),
            ):
                request = Request(base + '/api/action', data=b'{"name":"stop"}',
                                  headers={'Origin': origin, 'X-SCOPE-Token': token,
                                           'Content-Type': 'application/json'})
                if expected == 200:
                    with urlopen(request) as response:
                        self.assertEqual(response.status, expected)
                else:
                    with self.assertRaises(HTTPError) as error:
                        urlopen(request)
                    self.assertEqual(error.exception.code, expected)
                    error.exception.close()
        finally:
            server.shutdown()
            server.app.close()
            server.server_close()
            thread.join(timeout=2)

    def test_drive_heartbeat_timeout_clears_held_keys(self):
        app = WebControl()
        fake = FakeSession()
        app.session = fake
        app.connected = app.powered = app.armed = True
        try:
            app.drive('tab-1', 1, ['w'], False, app.control_epoch)
            deadline = time.monotonic() + DRIVE_TIMEOUT + 1
            while time.monotonic() < deadline and app.held:
                time.sleep(0.02)
            self.assertFalse(app.held)
            self.assertIn(('action', 'stop'), fake.calls)
            self.assertTrue(any(call[0] == 'keys' and not call[1] and call[2]
                                for call in fake.calls))
        finally:
            app.close()

    def test_drive_rejects_unconfirmed_or_gesture_control(self):
        app = WebControl()
        fake = FakeSession()
        app.session = fake
        app.connected = True
        try:
            with self.assertRaises(ValueError):
                app.drive('tab', 1, ['w'], False, app.control_epoch)
            app.armed = True
            app.gesture_active = True
            with self.assertRaises(ValueError):
                app.drive('tab', 2, ['w'], False, app.control_epoch)
            self.assertFalse(app.held)
        finally:
            app.close()

    def test_stop_rejects_delayed_drive_request(self):
        app = WebControl()
        fake = FakeSession()
        app.session = fake
        app.connected = app.powered = app.armed = True
        try:
            old_epoch = app.control_epoch
            app.drive('tab-1', 1, ['w'], False, old_epoch)
            app.stop()
            with self.assertRaises(ValueError):
                app.drive('tab-1', 2, ['w'], False, old_epoch)
            with self.assertRaises(ValueError):
                app.action('stand', {'epoch': old_epoch})
            self.assertFalse(app.held)
            self.assertEqual(fake.calls[-1], ('action', 'stop'))
        finally:
            app.close()

    def test_gesture_presence_timeout_disables_mode(self):
        app = WebControl()
        fake = FakeSession()
        app.session = fake
        app.connected = app.powered = app.armed = True
        app.gesture_available = True
        try:
            app.action('gesture', {'enabled': True, 'controller': 'tab-1',
                                   'epoch': app.control_epoch})
            self.assertTrue(app.gesture_active)
            with self.assertRaises(ValueError):
                app.presence('other-tab')
            deadline = time.monotonic() + DRIVE_TIMEOUT + 1
            while time.monotonic() < deadline and app.gesture_active:
                time.sleep(0.02)
            self.assertFalse(app.gesture_active)
            self.assertIn(('gesture', False), fake.calls)
            self.assertIn(('action', 'stop'), fake.calls)
        finally:
            app.close()


class InstallerTests(unittest.TestCase):
    @staticmethod
    def launcher_module():
        script = Path(__file__).resolve().parent / 'scripts' / 'launch_scope.py'
        spec = importlib.util.spec_from_file_location('scope_launcher', script)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        return launcher

    def test_rerun_updates_one_checkout(self):
        launcher = self.launcher_module()
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            origin, seed, installed = base / 'origin.git', base / 'seed', base / 'installed'
            subprocess.run(['git', 'init', '--bare', str(origin)], check=True, capture_output=True)
            subprocess.run(['git', 'init', '-b', 'main', str(seed)], check=True, capture_output=True)
            subprocess.run(['git', 'config', 'user.name', 'SCOPE Test'], cwd=seed, check=True)
            subprocess.run(['git', 'config', 'user.email', 'scope-test@example.invalid'], cwd=seed, check=True)
            (seed / 'version.txt').write_text('first')
            subprocess.run(['git', 'add', 'version.txt'], cwd=seed, check=True)
            subprocess.run(['git', 'commit', '-m', 'First'], cwd=seed, check=True, capture_output=True)
            subprocess.run(['git', 'remote', 'add', 'origin', f'file://{origin}'], cwd=seed, check=True)
            subprocess.run(['git', 'push', '-u', 'origin', 'main'], cwd=seed, check=True, capture_output=True)
            with patch.object(launcher, 'INSTALL_DIR', base), \
                 patch.object(launcher, 'REPOSITORY', f'file://{origin}'), \
                 patch.object(launcher, 'CHECKOUT', installed):
                launcher.update_checkout()
                self.assertEqual((installed / 'version.txt').read_text(), 'first')
                launcher.update_checkout()
                self.assertEqual(len(list(base.glob('installed*'))), 1)
                (seed / 'version.txt').write_text('second')
                subprocess.run(['git', 'commit', '-am', 'Second'], cwd=seed,
                               check=True, capture_output=True)
                subprocess.run(['git', 'push'], cwd=seed, check=True, capture_output=True)
                launcher.update_checkout()
                self.assertEqual((installed / 'version.txt').read_text(), 'second')

    def test_dependencies_install_once_until_requirements_change(self):
        launcher = self.launcher_module()
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            checkout, venv = base / 'source', base / 'venv'
            checkout.mkdir()
            (venv / 'bin').mkdir(parents=True)
            (venv / 'bin' / 'python').write_text('')
            requirements = checkout / 'requirements-console.txt'
            requirements.write_text('Pillow==12.3.0\n')
            fake_run = Mock()
            with patch.object(launcher, 'INSTALL_DIR', base), \
                 patch.object(launcher, 'CHECKOUT', checkout), \
                 patch.object(launcher, 'VENV', venv), \
                 patch.object(launcher, 'run', fake_run):
                launcher.ensure_environment()
                launcher.ensure_environment()
                self.assertEqual(fake_run.call_count, 1)
                requirements.write_text('Pillow==12.3.1\n')
                launcher.ensure_environment()
                self.assertEqual(fake_run.call_count, 2)


if __name__ == '__main__':
    unittest.main()
