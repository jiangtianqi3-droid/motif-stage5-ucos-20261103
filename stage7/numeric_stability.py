"""Fixed per-query GEMM geometry; no changed weights or ranking convention."""
import types
import torch
import torch.nn.functional as F


def configure(tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')


def query_linear(self, x):
    # Entity projection and score MLP use [queries, nodes, features].
    # Each GEMM has the same rows for an individual query at every outer batch.
    if x.ndim >= 3:
        return torch.stack([F.linear(v.contiguous(), self.weight, self.bias) for v in x])
    return F.linear(x, self.weight, self.bias)


def install(model):
    configure(False)
    for layer in model.entity_model.modules():
        if isinstance(layer, torch.nn.Linear):
            layer.forward = types.MethodType(query_linear, layer)
    return model
