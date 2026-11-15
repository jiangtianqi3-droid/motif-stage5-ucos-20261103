"""Separate adaptive-backbone ablation. Original Stage-7 anchors are untouched."""
import types
import torch
from torch.nn import functional as F
from weighted import aggregate

def forward(self, hyper, query):
    q = query + 1
    state = torch.zeros(len(q), int(hyper.num_nodes) + 1, self.dims[0], device=q.device)
    state.scatter_(1, q[:, None, None].expand(len(q), 1, self.dims[0]), 1.)
    for layer in self.layers:
        old = state
        gated = hasattr(layer, 'topology_gate')
        a = layer.topology_gate(state[torch.arange(len(q), device=q.device), q], hyper.topology_z, hyper.type_bounds) if gated else state.new_ones(len(q), hyper.plan['E'])
        scale = state.new_tensor([1, 1, 1, .15, .15, .15, .15])
        out = aggregate(state, layer.rel_embedding.weight * scale[:, None], layer.pos_embedding.weight,
                        a, hyper.plan, getattr(self, 'incident_normalization', 'type'), not gated)
        output = layer.dropout(layer.linear(torch.cat([out, state], -1)))
        if layer.norm_type == 'layer_norm':
            output = layer.norm(output)
        state = F.relu(output)
        if self.short_cut:
            state = state + old
    return self.mlp(state)[:, 1:]

def install_adaptive(model, data, family):
    from structural_calibrator import install_plus
    model.relation_model.forward = types.MethodType(forward, model.relation_model)
    install_plus(model, data)
