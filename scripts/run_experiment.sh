#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_setup=0
task_preview=0
task_args=()
for task_arg in "$@"; do
  case "$task_arg" in
    --setup) task_setup=1 ;;
    --dry-run|--help|-h) task_preview=1; task_args+=("$task_arg") ;;
    *) task_args+=("$task_arg") ;;
  esac
done
task_bootstrap_python="${MODALITYGAP_BOOTSTRAP_PYTHON:-python3}"
task_python="$task_root/.venv/bin/python"
if [[ "$task_setup" == 1 && "$task_preview" == 0 ]]; then
  command -v flock >/dev/null || { echo 'flock is required (Linux util-linux).' >&2; exit 1; }
  exec 9>"$task_root/.environment-setup.lock"
  flock -n 9 || { echo 'Another environment installation is running.' >&2; exit 1; }
  "$task_bootstrap_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "The dependency locks require Python 3.12; set MODALITYGAP_BOOTSTRAP_PYTHON=python3.12"'
  if [[ ! -x "$task_python" ]]; then
    "$task_bootstrap_python" -m venv "$task_root/.venv"
  fi
  "$task_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing .venv is not Python 3.12"'
  "$task_python" -m pip install -r "$task_root/requirements-cu128.lock"
  "$task_python" -m pip install --no-deps -e "$task_root"
  flock -u 9
  exec 9>&-
fi
if [[ ! -x "$task_python" ]]; then
  if [[ "$task_preview" == 1 ]]; then
    task_python="$task_bootstrap_python"
  else
    echo 'Missing .venv. Run this command with --setup first.' >&2
    exit 1
  fi
fi
export PYTHONPATH="$task_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$task_python" "$task_root/scripts/run_experiment.py" "${task_args[@]}"
