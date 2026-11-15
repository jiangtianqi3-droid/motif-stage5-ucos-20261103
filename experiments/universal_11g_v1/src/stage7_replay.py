"""Replay the original ConceptNet query batches with bounded GPU memory."""
import types

import numpy as np
import torch
from motif import tasks

from backbone import Backbone
from evaluator import resources


def last_hidden_bellmanford(self, data, h_index, r_index, separate_grad=False):
    # Stage-7's model scores only the last hidden layer. Keep the same operators,
    # batch dimensions and accumulation order while releasing earlier activations.
    assert not self.concat_hidden and not separate_grad and not torch.is_grad_enabled()
    query = self.query[torch.arange(len(r_index), device=r_index.device), r_index]
    boundary = torch.zeros(len(r_index), data.num_nodes, self.dims[0], device=h_index.device)
    boundary.scatter_add_(1, h_index[:, None, None].expand(-1, 1, query.shape[-1]), query[:, None, :])
    size = (data.num_nodes, data.num_nodes)
    weight = torch.ones(data.num_edges, device=h_index.device)
    state = boundary
    for layer in self.layers:
        hidden = layer(state, query, boundary, data.edge_index, data.edge_type, size, weight)
        if self.short_cut and hidden.shape == state.shape:
            hidden = hidden + state
        state = hidden
    return {'node_feature': torch.cat([state, query[:, None, :].expand_as(state)], dim=-1),
            'edge_weights': [weight] * len(self.layers)}


class Stage7Replay(Backbone):
    def __init__(self, z, family, seed):
        torch.cuda.empty_cache()
        total = torch.cuda.get_device_properties(0).total_memory
        budget = (4 * 1024 - 64) * 1024**2
        torch.cuda.set_per_process_memory_fraction(min(1., budget / total), device=0)
        super().__init__(z, family, seed)
        entity = self.model.entity_model
        entity._s7_original_bellmanford = types.MethodType(last_hidden_bellmanford, entity)

    @torch.no_grad()
    def scores(self, facts):
        assert 0 < len(facts) <= 16
        triples = torch.as_tensor(facts[:, [0, 2, 1]], device='cuda')
        scores = []
        for batch in tasks.all_negative(self.data, triples):
            score = self.model(self.data, batch)
            assert torch.isfinite(score).all()
            scores.append(score.cpu().numpy())
            resources()
        return np.stack(scores, axis=1).reshape(-1, self.z['n'])
