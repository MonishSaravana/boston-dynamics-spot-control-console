"""Class robot address, username, and the password saved in the macOS Keychain.

The password is never stored in the repository or sent to the browser. The
server reads it from the Keychain when the connect form leaves it blank. Save it
once with `Start Spot.command`, or:

    security add-generic-password -s "SCOPE Spot" -a user -w
"""
import subprocess
import sys

DEFAULT_HOST = "192.168.80.3"
DEFAULT_USER = "user"
KEYCHAIN_SERVICE = "SCOPE Spot"
ESTOP_ENDPOINT_NAME = "Estop NoGUI"   # Name used by the SDK's estop_nogui.py example.


def saved_password(username):
    if sys.platform != "darwin" or not username:
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", username, "-w"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (result.stdout.rstrip("\n") or None) if result.returncode == 0 else None


def estop_active(status, name=ESTOP_ENDPOINT_NAME, max_silence_s=2.):
    """True when the named E-stop endpoint is registered and checking in."""
    for item in status.endpoints:
        silence = item.time_since_valid_response
        if (item.endpoint.name == name and
                silence.seconds + silence.nanos/1e9 <= max_silence_s):
            return True
    return False


def wait_for_estop(hostname, username, password, closed, report, interval_s=2.):
    """Block until the class E-stop window is active, or `closed` is set."""
    import bosdyn.client
    from bosdyn.client.estop import EstopClient
    client = None
    while not closed.is_set():
        try:
            if client is None:
                robot = bosdyn.client.create_standard_sdk("scope-estop-check").create_robot(hostname)
                robot.authenticate(username, password)
                client = robot.ensure_client(EstopClient.default_service_name)
            if estop_active(client.get_status(timeout=2)):
                return True
            report("Waiting for the E-stop window to start…")
        except Exception as exc:
            client = None
            report(f"Waiting for Spot at {hostname}: {type(exc).__name__}")
        closed.wait(interval_s)
    return False


def auto_connect(app, hostname, username):
    """Connect in Robot control once the separate class E-stop is active."""
    workspace = app.workspace

    def report(message):
        with workspace.lock:
            workspace.message = message
    password = saved_password(username)
    if not password:
        report("No saved Spot password; use Connection, or run Start Spot.command once")
        return False
    if not wait_for_estop(hostname, username, password, app.closed, report):
        return False
    report("E-stop active; connecting in Robot control…")
    try:
        workspace.submit("connect", {"hostname": hostname, "username": username,
                                     "password": password, "mode": "robot_control",
                                     "command_authority": True}).result(timeout=60)
    except Exception as exc:
        report(f"Automatic connection failed: {exc}. Use Connection to retry.")
        return False
    return True
