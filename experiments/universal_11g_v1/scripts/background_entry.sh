#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$ROOT/logs"
exec bash "$ROOT/scripts/run.sh" "$ROOT/scripts/run_universal_11g.py" "$@" >> "$ROOT/logs/full.log" 2>&1
