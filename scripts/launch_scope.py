#!/usr/bin/env python3
"""Install/update one managed SCOPE checkout, then open the local browser console."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import webbrowser


REPOSITORY = 'https://github.com/MonishSaravana/boston-dynamics-spot-control-console.git'
INSTALL_DIR = Path.home() / 'Library' / 'Application Support' / 'SCOPE'
CHECKOUT = INSTALL_DIR / 'source'
VENV = INSTALL_DIR / 'venv'
SHORTCUT = Path.home() / 'Applications' / 'Open SCOPE.command'
PORT = 8765


def run(*args, cwd=None, timeout=None):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    return subprocess.run(args, cwd=cwd, env=env, check=True, timeout=timeout)


def output(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()


def update_checkout():
    if not CHECKOUT.exists():
        print('Downloading SCOPE into one managed location…', flush=True)
        with tempfile.TemporaryDirectory(prefix='.download-', dir=INSTALL_DIR) as temp:
            staging = Path(temp) / 'source'
            run('git', 'clone', '--depth', '1', '--branch', 'main', REPOSITORY,
                str(staging), timeout=120)
            staging.replace(CHECKOUT)
        return
    if not (CHECKOUT / '.git').is_dir():
        raise RuntimeError(f'{CHECKOUT} exists but is not a Git checkout; nothing was overwritten')
    remote = output('git', 'remote', 'get-url', 'origin', cwd=CHECKOUT)
    if remote != REPOSITORY:
        raise RuntimeError('The installed checkout has a different Git remote; nothing was changed')
    if output('git', 'status', '--porcelain', '--untracked-files=no', cwd=CHECKOUT):
        print('Installed source has local edits; keeping that version and skipping the update.', flush=True)
        return
    try:
        print('Checking for SCOPE updates…', flush=True)
        run('git', 'fetch', '--quiet', 'origin', 'main', cwd=CHECKOUT, timeout=20)
        current = output('git', 'rev-parse', 'HEAD', cwd=CHECKOUT)
        latest = output('git', 'rev-parse', 'FETCH_HEAD', cwd=CHECKOUT)
        if current != latest:
            run('git', 'merge', '--ff-only', 'FETCH_HEAD', cwd=CHECKOUT, timeout=20)
            print('SCOPE source updated.', flush=True)
        else:
            print('SCOPE source is current.', flush=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print('Update check failed; using the installed source. No live connection has started.', flush=True)


def ensure_environment():
    requirements = CHECKOUT / 'requirements-console.txt'
    if not requirements.is_file():
        raise RuntimeError('The installed SCOPE source is missing requirements-console.txt')
    fingerprint = hashlib.sha256(requirements.read_bytes() +
                                 sys.version.encode()).hexdigest()
    marker = INSTALL_DIR / 'requirements.sha256'
    python = VENV / 'bin' / 'python'
    if not python.is_file():
        print('Creating the SCOPE Python environment…', flush=True)
        run(sys.executable, '-m', 'venv', str(VENV))
    if not marker.is_file() or marker.read_text().strip() != fingerprint:
        print('Installing the browser console dependencies…', flush=True)
        run(str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
            '-r', str(requirements))
        marker.write_text(fingerprint + '\n')
    return python


def ensure_shortcut():
    SHORTCUT.parent.mkdir(parents=True, exist_ok=True)
    target = CHECKOUT / 'Open SCOPE.command'
    content = '#!/bin/zsh\n# SCOPE managed shortcut\nexec ' + shlex.quote(str(target)) + ' "$@"\n'
    if not SHORTCUT.exists() or SHORTCUT.read_text() != content:
        if SHORTCUT.exists() and not SHORTCUT.read_text().startswith('#!/bin/zsh\n# SCOPE managed shortcut\n'):
            print(f'Existing {SHORTCUT} was left alone; run {target} directly.', flush=True)
            return
        SHORTCUT.write_text(content)
        SHORTCUT.chmod(0o755)
    print(f'Future launches: {SHORTCUT}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='Open the offline demo; no robot client')
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        raise SystemExit('This launcher is currently for macOS; the Python web console can run separately.')
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    lock = (INSTALL_DIR / 'launcher.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('SCOPE is already open; showing the local page.', flush=True)
        webbrowser.open(f'http://127.0.0.1:{PORT}/')
        return 0
    try:
        update_checkout()
        python = ensure_environment()
        ensure_shortcut()
        command = [str(python), str(CHECKOUT / 'scope_web.py'), '--port', str(PORT)]
        if args.demo:
            command.append('--demo')
        print('Opening SCOPE on this laptop. Keep this launcher open while using the browser.', flush=True)
        return subprocess.call(command)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        print(f'SCOPE could not start: {exc}', file=sys.stderr)
        raise SystemExit(1)
