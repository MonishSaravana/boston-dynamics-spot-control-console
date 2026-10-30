#!/bin/zsh
# Start the class E-stop and the SCOPE console for a robot session.
# This window is the E-stop: space = STOP, r = release, q = quit.
# Quitting the E-stop, or the E-stop failing, also closes SCOPE.
set -u
setopt no_bg_nice   # keep SCOPE at normal priority in the background

cd -- "$(dirname -- "$0")"
host="${SPOT_HOST:-192.168.80.3}"
user_name="${SPOT_USER:-user}"
service="SCOPE Spot"
logs="$HOME/Library/Logs/SCOPE"
python=".venv/bin/python"
estop="spot-sdk/python/examples/estop/estop_nogui.py"

pause() { read -r '?Press Return to close…'; }
fail() { echo; echo "$@"; pause; exit 1; }

# 1. This laptop: environment, SDK example, and no second copy already running.
if [[ ! -x "$python" ]]; then
    fail "No Python environment yet. In this folder run:" \
         "  python3 -m venv .venv && .venv/bin/python -m pip install -r requirements-console.txt"
fi
if [[ ! -f "$estop" ]]; then
    fail "The Spot SDK checkout is missing. Put it at ./spot-sdk so $estop exists (see README)."
fi
if ! "$python" -c 'import bosdyn.client, cv2, PIL' 2>/dev/null; then
    fail "The Python environment is missing console packages. Run:" \
         "  .venv/bin/python -m pip install -r requirements-console.txt"
fi
if pgrep -f 'scope_web.py --auto-connect' >/dev/null 2>&1; then
    fail "SCOPE is already running from another window. Quit that E-stop window (q) first."
fi

# 2. The robot: reachable on this network before anything is started.
echo "Looking for Spot at $host…"
if ! nc -z -G 3 "$host" 443 >/dev/null 2>&1; then
    fail "Spot at $host is not reachable." \
         "  Join Spot's Wi-Fi (the network name is printed on the robot), then run this again." \
         "  If the robot has another address, run: SPOT_HOST=<address> zsh \"Start Spot.command\""
fi

# 3. The password: saved once in the macOS Keychain, never in a file.
if ! security find-generic-password -s "$service" -a "$user_name" >/dev/null 2>&1; then
    echo "Save the Spot password for '$user_name' once. It goes in your macOS Keychain, not in any file."
    security add-generic-password -s "$service" -a "$user_name" -w || { pause; exit 1; }
fi
password="$(security find-generic-password -s "$service" -a "$user_name" -w)" || { pause; exit 1; }

# 4. SCOPE in the background. It gets the password through its own environment
#    so it does not depend on Keychain access from a background process.
mkdir -p "$logs"
echo "Starting SCOPE (log: $logs/scope.log)…"
SPOT_PASSWORD="$password" "$python" scope_web.py --auto-connect "$host" --user "$user_name" \
    >"$logs/scope.log" 2>&1 &
app=$!
trap 'kill $app 2>/dev/null; wait $app 2>/dev/null' EXIT INT TERM
sleep 2
if ! kill -0 "$app" 2>/dev/null; then
    echo "SCOPE stopped right after starting. Last log lines:"
    tail -n 15 "$logs/scope.log"
    fail "Fix the problem above, then run this again."
fi

echo
echo "E-stop for Spot at $host. Keep this window open."
echo "  space = STOP    r = release    q = quit (also closes SCOPE)"
echo "SCOPE connects in Robot control once this E-stop is active. Then press Power On and Stand."
echo
BOSDYN_CLIENT_USERNAME="$user_name" BOSDYN_CLIENT_PASSWORD="$password" "$python" "$estop" "$host"
status=$?
unset password

if (( status != 0 )); then
    echo
    echo "The E-stop stopped (exit $status), so SCOPE was closed too."
    echo "  MotorsOnError: Spot's motors are on from the tablet. Power them off there, then run this again."
    echo "  InvalidLoginError: wrong saved password. Fix it with:"
    echo "    security delete-generic-password -s \"$service\" -a \"$user_name\""
    echo "  Robot unreachable: join Spot's Wi-Fi and check the address ($host)."
    pause
fi
