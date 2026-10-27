#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/../.." && pwd)
if [[ -n "${PYTHON:-}" ]]; then
    tui_python=${PYTHON}
elif [[ -x "${repo_dir}/.venv/bin/python3" ]]; then
    tui_python="${repo_dir}/.venv/bin/python3"
else
    tui_python=python3
fi
exec "${tui_python}" "${script_dir}/ninfer_tui.py" "$@"
