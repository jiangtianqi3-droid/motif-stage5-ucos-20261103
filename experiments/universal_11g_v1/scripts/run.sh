#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
ENV=/mnt/d/ultra-phase1-env/env
OLD=/mnt/d/motif-phase4-work/motif-stage7-asymmetric-20261106
PREVIOUS="$ROOT/../conceptnet_zeroshot_v1"
mkdir -p "$ROOT/cache/tmp" "$ROOT/cache/torch_extensions" "$ROOT/cache/triton"
export CUDA_HOME="$ENV" TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions"
export CPATH=/mnt/d/ultra-phase1-env/cuda-headers/include TMPDIR="$ROOT/cache/tmp"
export PYTHONPATH="$ROOT/src:$PREVIOUS/cache/deps:$OLD/runtime:$OLD:$PREVIOUS/src"
export PATH="$ENV/bin:$PATH" MAX_JOBS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_CUDA_ARCH_LIST=8.6 PYTHONDONTWRITEBYTECODE=1
export CC=/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/gcc-11 CXX=/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/g++-11
export TRITON_CACHE_DIR="$ROOT/cache/triton" MOTIF_EVAL_CACHE=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TZ=Asia/Shanghai
export HF_HOME="$ROOT/cache/huggingface" XDG_CACHE_HOME="$ROOT/cache/xdg" MPLCONFIGDIR="$ROOT/cache/matplotlib"
cd "$ROOT"
exec "$ENV/bin/python" -u "$@"
