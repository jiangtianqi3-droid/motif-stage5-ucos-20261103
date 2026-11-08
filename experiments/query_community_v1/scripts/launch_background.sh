#!/usr/bin/env bash
set -euo pipefail
QC_ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$QC_ROOT/logs"
QC_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
QC_LOG="$QC_ROOT/logs/full-$QC_STAMP.log"
setsid nohup bash "$QC_ROOT/run.sh" scripts/run_community_experiments.py "$@" >"$QC_LOG" 2>&1 < /dev/null &
QC_PID=$!
printf '%s\n' "$QC_PID" > "$QC_ROOT/logs/full.pid"
printf '%s\n' "$QC_LOG" > "$QC_ROOT/logs/full.logpath"
printf 'PID=%s\nLOG=%s\n' "$QC_PID" "$QC_LOG"
# One launch verification, no subsequent polling or waits.
kill -0 "$QC_PID"
