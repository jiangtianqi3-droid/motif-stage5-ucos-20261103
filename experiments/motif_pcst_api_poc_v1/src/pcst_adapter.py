"""PCST transformation for directed, parallel knowledge-graph facts.

Adapted from G-Retriever retrieval_via_pcst, MIT, Copyright (c) 2024
Xiaoxin He. See THIRD_PARTY_PROVENANCE.md for source and license.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Fact:
    id: str
    src: str
    relation: str
    dst: str


def transform(nodes, facts, node_prizes, edge_prizes, edge_costs):
    """Build nonnegative undirected solver input while preserving fact identity."""
    if len(set(nodes)) != len(nodes) or len({f.id for f in facts}) != len(facts):
        raise ValueError('node and fact IDs must be unique')
    if not (len(node_prizes) == len(nodes) and len(edge_prizes) == len(edge_costs) == len(facts)):
        raise ValueError('inconsistent lengths')
    for x in list(node_prizes) + list(edge_prizes) + list(edge_costs):
        if not math.isfinite(x) or x < 0:
            raise ValueError('prizes and costs must be finite and nonnegative')
    index = {v: i for i, v in enumerate(nodes)}
    prizes, edges, costs, edge_map, virtual_map = list(node_prizes), [], [], {}, {}
    for i, f in enumerate(facts):
        if f.src not in index or f.dst not in index:
            raise ValueError('fact endpoint is absent')
        u, v = index[f.src], index[f.dst]
        if u == v:
            raise ValueError('self-loop unsupported; handle explicitly before PCST')
        if edge_prizes[i] <= edge_costs[i]:
            edge_map[len(edges)] = i
            edges.append((u, v)); costs.append(edge_costs[i] - edge_prizes[i])
        else:
            z = len(prizes)
            virtual_map[z] = i
            prizes.append(edge_prizes[i] - edge_costs[i])
            edges.extend([(u, z), (z, v)]); costs.extend([0., 0.])
    return prizes, edges, costs, edge_map, virtual_map


def restore(nodes, facts, vertices, selected_edges, edge_map, virtual_map):
    selected = {edge_map[i] for i in selected_edges if i in edge_map}
    selected.update(virtual_map[v] for v in vertices if v in virtual_map)
    result = [facts[i] for i in sorted(selected)]
    selected_nodes = {nodes[v] for v in vertices if 0 <= v < len(nodes)}
    for fact in result:
        selected_nodes.update((fact.src, fact.dst))
    return selected_nodes, result


def solve(nodes, facts, node_prizes, edge_prizes, edge_costs, root=None):
    import numpy as np
    from pcst_fast import pcst_fast
    prizes, edges, costs, emap, vmap = transform(nodes, facts, node_prizes, edge_prizes, edge_costs)
    if not edges:
        return ({root} if root is not None else set()), []
    root_index = -1 if root is None else nodes.index(root)
    vertices, chosen = pcst_fast(np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                                 np.asarray(prizes, dtype=np.float64),
                                 np.asarray(costs, dtype=np.float64), root_index, 1, 'gw', 0)
    return restore(nodes, facts, vertices, chosen, emap, vmap)
