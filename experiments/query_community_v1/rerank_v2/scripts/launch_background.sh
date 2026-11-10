#!/usr/bin/env bash
set -euo pipefail
ROUTE_ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
ROUTE_OUTPUT="${ROUTE_A_OUTPUT:-/mnt/d/motif-phase4-work/route-a-rerank-20261109}"
mkdir -p "$ROUTE_OUTPUT/logs"
ROUTE_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROUTE_LOG="$ROUTE_OUTPUT/logs/full-$ROUTE_STAMP.log"
setsid nohup bash "$ROUTE_ROOT/run.sh" scripts/run_route_a_rerank.py "$@" >"$ROUTE_LOG" 2>&1 < /dev/null &
ROUTE_PID=$!
printf '%s\n' "$ROUTE_PID" > "$ROUTE_OUTPUT/logs/full.pid"
printf '%s\n' "$ROUTE_LOG" > "$ROUTE_OUTPUT/logs/full.logpath"
printf 'PID=%s\nLOG=%s\n' "$ROUTE_PID" "$ROUTE_LOG"
kill -0 "$ROUTE_PID"
