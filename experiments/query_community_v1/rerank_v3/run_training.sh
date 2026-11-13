#!/usr/bin/env bash
set -euo pipefail
TASK=/mnt/d/motif-phase4-work/route-a-repair-20261112
ENV=/mnt/d/ultra-phase1-env/env
OLD=/mnt/d/motif-phase4-work/topology-ucos-stage5-20261103
export CUDA_HOME="$ENV" TORCH_EXTENSIONS_DIR="$TASK/cache/torch_extensions"
export CPATH=/mnt/d/ultra-phase1-env/cuda-headers/include TMPDIR="$TASK/cache/tmp"
export PYTHONPATH="$TASK/training:$OLD/runtime" PATH="$ENV/bin:$PATH" MAX_JOBS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_CUDA_ARCH_LIST=8.6 PYTHONDONTWRITEBYTECODE=1
export CC=/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/gcc-11 CXX=/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/g++-11
export TRITON_CACHE_DIR="$TASK/cache/triton" MOTIF_EVAL_CACHE=0
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
cd "$TASK/training"
exec "$ENV/bin/python" -u run_repair.py
