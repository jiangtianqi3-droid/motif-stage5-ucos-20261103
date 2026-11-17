#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
env=/mnt/d/ultra-phase1-env/env
stage=/mnt/d/motif-phase4-work/motif-stage7-asymmetric-20261106
poc="$root/../motif_pcst_api_poc_v1"
mkdir -p "$root/results/runtime/tmp"
export PYTHONPATH="$root:$root/../motif_pcst_structural_p1:$poc:$poc/dependencies:$stage/runtime:$stage"
export PYTHONDONTWRITEBYTECODE=1 MOTIF_EVAL_CACHE=0
export CUDA_HOME="$env" TORCH_EXTENSIONS_DIR="$root/results/runtime/torch"
export TRITON_CACHE_DIR="$root/results/runtime/triton" TMPDIR="$root/results/runtime/tmp"
export CPATH=/mnt/d/ultra-phase1-env/cuda-headers/include PATH="$env/bin:$PATH"
export MAX_JOBS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_CUDA_ARCH_LIST=8.6
export CC=/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/gcc-11 CXX=/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/g++-11
cd "$root"
exec "$env/bin/python" -u "$@"
