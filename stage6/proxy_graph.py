"""Rebuild topology on the actual residual graph; cache identity is graph-specific."""
import numpy as np
import torch
from torch_geometric.data import Data
from topology import extract, graph_key
from plans import build_plan, to_device


def build_graph(graph, queries, truths, cache, device='cuda'):
    sig = extract(graph.edge_index, graph.edge_type, graph.n, graph.R, cache=cache)
    assert str(sig['key']) == graph_key(graph.edge_index, graph.edge_type, graph.n, graph.R)
    he, ht = sig['edge_index'], sig['edge_type']
    assert np.all(ht[1:] >= ht[:-1])
    plan = build_plan(he, ht, graph.R)
    h = Data(edge_index=torch.as_tensor(he.astype(np.int64), device=device),
             edge_type=torch.as_tensor(ht.astype(np.int64), device=device), num_nodes=graph.R, num_relations=7)
    h.plan = to_device(plan, device)
    h.topology_z = torch.as_tensor(sig['z'].astype(np.float32), device=device)
    h.type_bounds = np.r_[0, np.cumsum(np.bincount(ht, minlength=7))].tolist()
    raw = sig['raw'][:, 3]
    cv = float(raw.std() / max(abs(raw.mean()), 1e-6)) if len(raw) else 0.
    h.alpha_adaptive = .15 + .10 * cv / (1 + cv)
    h.topology_cv = cv
    q = torch.as_tensor(np.asarray(queries).copy(), dtype=torch.long, device=device)
    d = Data(edge_index=torch.tensor(graph.edge_index.copy(), device=device),
             edge_type=torch.tensor(graph.edge_type.copy(), device=device), num_nodes=graph.n, num_relations=graph.R,
             target_edge_index=q[:, [0, 2]].T, target_edge_type=q[:, 1], relation_hypergraph=h)
    f = torch.as_tensor(np.asarray(truths).copy(), dtype=torch.long, device=device)
    truth = Data(edge_index=f[:, [0, 2]].T, edge_type=f[:, 1], num_nodes=graph.n)
    return d, truth, dict(topology_key=str(sig['key']), hyperedges=len(ht), topology_seconds=float(sig['seconds']),
                          cache_hit=bool(sig['cache_hit']), residual_sha256=graph.key)
