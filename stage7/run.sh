#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE=$(dirname "$HERE")
ENV=${DRTA_ENV:-/mnt/d/ultra-phase1-env/env}
export CUDA_HOME="$ENV" TORCH_EXTENSIONS_DIR="$HERE/torch_extensions"
export CPATH="${DRTA_CUDA_HEADERS:-/mnt/d/ultra-phase1-env/cuda-headers/include}" TMPDIR="${DRTA_TMPDIR:-/mnt/d/ultra-phase1-env/tmp}"
export PYTHONPATH="$BASE/runtime:$BASE${PYTHONPATH:+:$PYTHONPATH}" PATH="$ENV/bin:$PATH"
export MAX_JOBS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_CUDA_ARCH_LIST=8.6 PYTHONDONTWRITEBYTECODE=1
export CC="${DRTA_CC:-/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/gcc-11}" CXX="${DRTA_CXX:-/mnt/d/ultra-phase1-env/ubuntu-gcc/root/usr/bin/g++-11}"
export TRITON_CACHE_DIR="$HERE/triton_cache" MOTIF_EVAL_CACHE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BASE"
exec "$ENV/bin/python" -u "$@"
