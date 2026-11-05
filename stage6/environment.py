import os
import platform
import sys
import time
import numpy
import scipy
import torch
import torch_geometric
import triton
from stage6.util import ROOT, dump


def run():
    dump(ROOT / 'results/environment.json', dict(
        recorded_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), python=sys.version,
        platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
        pyg=torch_geometric.__version__, numpy=numpy.__version__, scipy=scipy.__version__, triton=triton.__version__,
        gpu=torch.cuda.get_device_name(), gpu_total_memory=torch.cuda.get_device_properties(0).total_memory,
        precision='float32; no autocast', matmul_precision=torch.get_float32_matmul_precision(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        threads=torch.get_num_threads(), cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        evaluation_cache=os.environ.get('MOTIF_EVAL_CACHE'),
        ranking_ties='pessimistic: candidate score >= true score; filtered positives excluded',
        numeric_limit='native HyperRelConvSumAggrSegmented forward contains atomic floating reductions'))


if __name__ == '__main__':
    run()
