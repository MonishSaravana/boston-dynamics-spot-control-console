#!/bin/zsh
set -eu

script_path="$0"
while [ -L "$script_path" ]; do
    script_dir="$(cd -- "$(dirname -- "$script_path")" && pwd)"
    script_path="$(readlink "$script_path")"
    [[ "$script_path" = /* ]] || script_path="$script_dir/$script_path"
done
script_dir="$(cd -- "$(dirname -- "$script_path")" && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; raise SystemExit(not (3, 11) <= sys.version_info[:2] <= (3, 14))' 2>/dev/null; then
        exec "$candidate" "$script_dir/scripts/launch_scope.py" "$@"
    fi
done

echo "SCOPE needs Python 3.11–3.14. Install Python, then open this file again."
read -r '?Press Return to close…'
exit 1
