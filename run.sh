#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/d/motif-phase4-work/topology-ucos-stage5-20261103
ENV=/mnt/d/ultra-phase1-env/env
export CUDA_HOME="$ENV" TORCH_EXTENSIONS_DIR="$ROOT/torch_extensions"
export CPATH=/mnt/d/ultra-phase1-env/cuda-headers/include TMPDIR=/mnt/d/ultra-phase1-env/tmp
export PYTHONPATH="$ROOT/runtime:$ROOT" PATH="$ENV/bin:$PATH" MAX_JOBS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_CUDA_ARCH_LIST=8.6 PYTHONDONTWRITEBYTECODE=1
export CC=/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/gcc-11 CXX=/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/g++-11
export TRITON_CACHE_DIR="$ROOT/triton_cache" MOTIF_EVAL_CACHE=0
cd "$ROOT"
exec "$ENV/bin/python" -u "$@"
