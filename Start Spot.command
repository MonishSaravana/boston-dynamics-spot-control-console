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

if [[ ! -x "$python" || ! -f "$estop" ]]; then
    echo "Set up first: create .venv and put the Spot SDK in ./spot-sdk (see README)."
    pause; exit 1
fi

if ! security find-generic-password -s "$service" -a "$user_name" >/dev/null 2>&1; then
    echo "Save the Spot password for '$user_name' once. It goes in your macOS Keychain, not in any file."
    security add-generic-password -s "$service" -a "$user_name" -w || { pause; exit 1; }
fi
password="$(security find-generic-password -s "$service" -a "$user_name" -w)" || { pause; exit 1; }

mkdir -p "$logs"
echo "Starting SCOPE (log: $logs/scope.log)…"
"$python" scope_web.py --auto-connect "$host" --user "$user_name" >"$logs/scope.log" 2>&1 &
app=$!
trap 'kill $app 2>/dev/null; wait $app 2>/dev/null' EXIT INT TERM

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
