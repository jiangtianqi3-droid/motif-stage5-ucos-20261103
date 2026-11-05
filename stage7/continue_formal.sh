#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec > >(tee -a "$HERE/logs/workflow.log") 2>&1
cd "$(dirname "$HERE")"
bash "$HERE/run.sh" -m stage7.path_compat formal
bash "$HERE/run.sh" -m stage7.alpha0_audit
ANALYSIS_PYTHON=${DRTA_FETCH_PYTHON:-/mnt/e/Python/Python39/python.exe}
PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 "$ANALYSIS_PYTHON" -B -m stage7.run analyze
PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 "$ANALYSIS_PYTHON" -B -m stage7.plot
bash "$HERE/run.sh" -m stage7.path_compat verify
