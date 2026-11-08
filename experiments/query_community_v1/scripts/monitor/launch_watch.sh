#!/usr/bin/env bash
set -euo pipefail
QC_ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
QC_LOG="$QC_ROOT/logs/watch-$(date -u +%Y%m%dT%H%M%SZ).log"
setsid nohup bash "$QC_ROOT/run.sh" scripts/monitor/watch_experiment.py "$@" >"$QC_LOG" 2>&1 < /dev/null &
QC_WATCH_PID=$!
printf '%s\n' "$QC_WATCH_PID" > "$QC_ROOT/logs/watch.pid"
printf '%s\n' "$QC_LOG" > "$QC_ROOT/logs/watch.logpath"
printf 'WATCH_PID=%s\nWATCH_LOG=%s\n' "$QC_WATCH_PID" "$QC_LOG"
kill -0 "$QC_WATCH_PID"
