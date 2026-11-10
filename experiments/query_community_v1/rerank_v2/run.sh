#!/usr/bin/env bash
set -euo pipefail
ROUTE_ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
ROUTE_ENV=/mnt/d/ultra-phase1-env/env
ROUTE_OLD=/mnt/d/motif-phase4-work/query-community-v1-20261108/experiments/query_community_v1
export ROUTE_A_OUTPUT="${ROUTE_A_OUTPUT:-/mnt/d/motif-phase4-work/route-a-rerank-20261109}"
export CUDA_HOME="$ROUTE_ENV" TORCH_EXTENSIONS_DIR="$ROUTE_A_OUTPUT/cache/torch_extensions"
export CPATH=/mnt/d/ultra-phase1-env/cuda-headers/include TMPDIR="$ROUTE_A_OUTPUT/cache/tmp"
export PATH="$ROUTE_ENV/bin:$PATH" MAX_JOBS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_CUDA_ARCH_LIST=8.6 PYTHONDONTWRITEBYTECODE=1
export CC=/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/gcc-11 CXX=/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/g++-11
export TRITON_CACHE_DIR="$ROUTE_A_OUTPUT/cache/triton" MOTIF_EVAL_CACHE=0
export MPLCONFIGDIR="$ROUTE_A_OUTPUT/cache/matplotlib" XDG_CACHE_HOME="$ROUTE_A_OUTPUT/cache/xdg"
export PYTHONPATH="$ROUTE_OLD/cache/dependencies:$ROUTE_ROOT"
mkdir -p "$TMPDIR"
cd "$ROUTE_ROOT"
exec "$ROUTE_ENV/bin/python" -u "$@"
