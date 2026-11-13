#!/usr/bin/env bash
set -euo pipefail
TASK=/mnt/d/motif-phase4-work/route-a-repair-20261112
ENV=/mnt/d/ultra-phase1-env/env
mkdir -p "$TASK/logs" "$TASK/cache/tmp"
exec 9>"$TASK/logs/workflow.lock"
if ! flock -n 9; then
  echo "This repair workflow is already running. No second run was started."
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$TASK/cache/tmp"
export ROUTE_A_OUTPUT="$TASK" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$TASK/code:/mnt/d/motif-phase4-work/query-community-v1-20261108/experiments/query_community_v1/cache/dependencies"
cd "$TASK"
exec "$ENV/bin/python" -B -u workflow.py "$@"
