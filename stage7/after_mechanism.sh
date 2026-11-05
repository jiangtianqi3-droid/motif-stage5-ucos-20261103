#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$HERE/logs"
exec > >(tee -a "$HERE/logs/workflow.log") 2>&1
bash "$HERE/run.sh" -m stage7.run source-validation
bash "$HERE/run.sh" -m stage7.run gpu-checks
bash "$HERE/run.sh" -m stage7.run tie-checks
bash "$HERE/run.sh" -m stage7.mechanism_summary
bash "$HERE/run.sh" -m stage7.preflight
bash "$HERE/run.sh" -m stage7.validate_preflight
bash "$HERE/run.sh" -m stage7.run protocol
bash "$HERE/run.sh" -m stage7.run adapt
bash "$HERE/run.sh" -m stage7.run freeze
cd "$(dirname "$HERE")"
FETCH_PYTHON=${DRTA_FETCH_PYTHON:-/mnt/e/Python/Python39/python.exe}
if [[ -f "$FETCH_PYTHON" ]]; then
    PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 "$FETCH_PYTHON" -B -m stage7.formal_fetch
else
    bash "$HERE/run.sh" -m stage7.formal_fetch
fi
bash "$HERE/run.sh" -m stage7.path_compat formal
bash "$HERE/run.sh" -m stage7.alpha0_audit
if [[ -f "$FETCH_PYTHON" ]]; then
    PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 "$FETCH_PYTHON" -B -m stage7.run analyze
    PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 "$FETCH_PYTHON" -B -m stage7.plot
else
    bash "$HERE/run.sh" -m stage7.run analyze
fi
bash "$HERE/run.sh" -m stage7.path_compat verify
