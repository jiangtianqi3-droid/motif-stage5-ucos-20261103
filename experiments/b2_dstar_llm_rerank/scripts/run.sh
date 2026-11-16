#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
ENV=/mnt/d/ultra-phase1-env/env
S7=/mnt/d/motif-phase4-work/motif-stage7-asymmetric-20261106
mkdir -p "$ROOT/cache/tmp" "$ROOT/cache/torch_extensions" "$ROOT/cache/triton" "$ROOT/results" "$ROOT/logs"
export CUDA_HOME="$ENV" CPATH=/mnt/d/ultra-phase1-env/cuda-headers/include
export PATH="$ENV/bin:$PATH" TMPDIR="$ROOT/cache/tmp"
export PYTHONPATH="$ROOT/src:$ROOT/cache/deps:$ROOT/../universal_11g_v1/src:$S7/runtime:$S7:$ROOT/../conceptnet_zeroshot_v1/src"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions" TRITON_CACHE_DIR="$ROOT/cache/triton"
export TIKTOKEN_CACHE_DIR="$ROOT/cache/tokenizer" HF_HOME="$ROOT/cache/huggingface" XDG_CACHE_HOME="$ROOT/cache/xdg"
export CC=/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/gcc-11 CXX=/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/g++-11
export MAX_JOBS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_CUDA_ARCH_LIST=8.6 PYTHONDONTWRITEBYTECODE=1
export MOTIF_EVAL_CACHE=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TZ=Asia/Shanghai
cd "$ROOT"
exec "$ENV/bin/python" -u "$@"
