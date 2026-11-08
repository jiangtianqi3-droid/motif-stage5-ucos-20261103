#!/usr/bin/env bash
set -euo pipefail
QC_ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
QC_ENV=/mnt/d/ultra-phase1-env/env
export CUDA_HOME="$QC_ENV" TORCH_EXTENSIONS_DIR="$QC_ROOT/cache/torch_extensions"
export CPATH=/mnt/d/ultra-phase1-env/cuda-headers/include TMPDIR="$QC_ROOT/cache/tmp"
export PATH="$QC_ENV/bin:$PATH" MAX_JOBS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_CUDA_ARCH_LIST=8.6 PYTHONDONTWRITEBYTECODE=1
export CC=/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/gcc-11 CXX=/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/g++-11
export TRITON_CACHE_DIR="$QC_ROOT/cache/triton" MOTIF_EVAL_CACHE=0
export MPLCONFIGDIR="$QC_ROOT/cache/matplotlib" XDG_CACHE_HOME="$QC_ROOT/cache/xdg"
export PYTHONPATH="$QC_ROOT/cache/dependencies:$QC_ROOT"
mkdir -p "$TMPDIR"
cd "$QC_ROOT"
exec "$QC_ENV/bin/python" -u "$@"
