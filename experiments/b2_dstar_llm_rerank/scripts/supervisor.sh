#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$ROOT/logs"
cd "$ROOT"
# An exclusive lock in the Python entrypoint prevents duplicate runs. Do not
# automatically retry invariant failures or ambiguous paid API requests.
bash scripts/run.sh scripts/run_experiment.py >> logs/full.log 2>&1
