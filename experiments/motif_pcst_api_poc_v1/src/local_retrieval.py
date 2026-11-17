"""Shared directed coarse graph and deterministic budgeted evidence selection."""
from collections import defaultdict
from math import log1p
from random import Random
from time import perf_counter

import numpy as np

from src.evidence_serializer import MAX_INPUT, estimate, messages
from src.graph_loader import fact
from src.pcst_adapter import solve
from query_community_v1.src.local_ppr import local_ppr

MAX_NODES, MAX_EDGES, MAX_FACTS = 300, 800, 60
ARMS = ('A2', 'A3', 'A4', 'A5', 'A4-no-motif', 'A4-no-coverage', 'A4-no-hub')


def directed_paths(graph, head, goals, edge_value, permitted=None):
    """Best support among shortest paths of one to three hops, retaining edge identity."""
    layers = {head: (0., ())}
    seen = {head}
    answer = {}
    goals = set(goals) - {head}
    if not goals:
        return answer
    for _ in range(3):
        next_layer = {}
        for u, (value, path) in layers.items():
            for i in graph.edges_at(u):
                i = int(i)
                if permitted is not None and i not in permitted:
                    continue
                v = int(graph.edge_index[1, i])
                if v in seen:
                    continue
                item = (value - edge_value(i), path + (i,))
                if v not in next_layer or item < next_layer[v]:
                    next_layer[v] = item
        for v in goals & next_layer.keys():
            answer[v] = next_layer[v][1]
        seen.update(next_layer)
        layers = next_layer
        if not layers or goals <= answer.keys():
            break
    return answer


def coarse(graph, weights, head, candidates):
    forward = local_ppr(weights.transition, head)
    reverse = defaultdict(float)
    for candidate in candidates:
        for v, mass in local_ppr(weights.reverse_transition, candidate).items():
            reverse[v] = max(mass, reverse[v])
    u, v = graph.edge_index
    focus = {head, *candidates}
    # The row-normalized transition uses uniform observed facts. Hub cost is applied in PCST.
    affinity = {x: forward.get(x, 0.) + 0.5 * reverse[x] for x in set(forward) | set(reverse)}
    paths = directed_paths(graph, head, candidates, lambda i: weights.edge_probability[i])
    mandatory = {i for path in paths.values() for i in path}
    for i in mandatory:
        focus.update((int(u[i]), int(v[i])))
    nodes = focus | set(sorted(affinity.keys() - focus, key=lambda x: (-affinity[x], x))[:MAX_NODES-len(focus)])
    edges = np.flatnonzero(np.isin(u, list(nodes)) & np.isin(v, list(nodes)))
    def priority(i):
        return (-max(forward.get(int(u[i]), 0.), forward.get(int(v[i]), 0.))
                - .5 * max(reverse[int(u[i])], reverse[int(v[i])]), int(i))
    local = sorted(map(int, edges), key=lambda i: (i not in mandatory, priority(i)))[:MAX_EDGES]
    return nodes, local, forward, reverse, paths


def budget(graph, head, rel, mapping, groups):
    ordered = {}
    for group in groups:
        additions = {i: fact(graph, i) for i in group if i not in ordered}
        if len(ordered) + len(additions) > MAX_FACTS:
            continue
        trial = {**ordered, **additions}
        if estimate(messages(head, rel, mapping, list(trial.values()))) <= MAX_INPUT:
            ordered = trial
    return list(ordered)


def pcst(graph, nodes, local, head, candidates, node_prize, bonus, paths, balance, pack):
    reserved = []
    covered = {head}
    if balance:
        # Public candidate permutation fixes tie/order priority. Include a path atomically.
        reserved = pack([paths[c] for c in candidates if c in paths])
        for i in reserved:
            covered.update(map(int, graph.edge_index[:, i]))
    remap = lambda x: head if x in covered else x
    cost = 0.36
    reserve_set = set(reserved)
    adapted = []
    for i in local:
        original = fact(graph, i)
        u, v = remap(original.src), remap(original.dst)
        if i not in reserve_set and u != v:
            adapted.append(type(original)(i, u, original.relation, v))
    remaining = sorted({remap(v) for v in nodes})
    rewards = [node_prize[v] for v in remaining]
    selected_nodes, chosen = solve(remaining, adapted, rewards, [bonus[f.id] for f in adapted],
                                   [cost] * len(adapted), root=head if balance else None)
    selected = {f.id for f in chosen}
    # Append connected extensions in order of attachment so budget trimming does not break them.
    active = covered if balance else ({head} if head in selected_nodes else set())
    groups = [reserved] if reserved else []
    while selected:
        touching = [i for i in selected if set(graph.edge_index[:, i]) & active]
        if not touching and active and balance:
            break
        i = min(touching or selected, key=lambda i: (-bonus[i], i))
        # Pack the growing prefix to keep a connected result across truncation.
        if len(pack([*groups, (i,)])) == sum(map(len, groups)) + 1:
            groups.append((i,))
            active.update(map(int, graph.edge_index[:, i]))
        elif balance:
            break
        selected.remove(i)
    if balance:
        # Disconnected candidates contribute at most one real local incident fact each.
        for c in candidates:
            if c not in paths and c != head:
                incident = [i for i in local if c in graph.edge_index[:, i]]
                if incident:
                    groups.append((min(incident, key=lambda i: (-bonus[i], i)),))
    return pack(groups)


def retrieve(graph, weighter, head, rel, mapping, random_seed):
    start = perf_counter()
    candidates = list(mapping.values())
    weights = weighter.compute()
    nodes, local, fwd, back, reachable = coarse(graph, weights, head, candidates)
    coarse_ms = 1000 * (perf_counter() - start)
    rho = weighter.relation_importance(rel)
    path_support = defaultdict(int)
    for path in reachable.values():
        for i in path:
            path_support[i] += 1
    candidate_set = set(candidates)
    maximum = max(1, int(graph.out_degree.max()))
    hub_cut = float(np.quantile(graph.out_degree, .9))
    def base(v):
        return 0.12 + min(1., 3 * fwd.get(v, 0.) + back[v]) + .45 * (v in candidate_set) + 1. * (v == head)
    def node_prize(v, hub):
        penalty = .24 * log1p(graph.out_degree[v]) / log1p(maximum) if hub and v != head and v not in candidate_set else 0.
        return max(0., base(v) + .36 * np.sqrt(fwd.get(v, 0.) * back[v]) - penalty)
    def pack(groups):
        return budget(graph, head, rel, mapping, groups)
    output = {}
    for arm in ARMS:
        began = perf_counter()
        if arm == 'A2':
            chosen = pack([(i,) for i in local])
        elif arm == 'A5':
            order = local.copy()
            Random(random_seed).shuffle(order)
            size = len(output['A4']['ids'])
            chosen = pack([(i,) for i in order])[:size]
        else:
            ordinary = arm == 'A3'
            topology = arm not in ('A3', 'A4-no-motif')
            balance = arm not in ('A3', 'A4-no-coverage')
            bonus = {i: 0. if ordinary else
                     .44 * float(rho[graph.edge_type[i]]) * topology + .26 * min(1., path_support[i] / 3.)
                     for i in local}
            prizes = {v: base(v) if ordinary else node_prize(v, arm != 'A4-no-hub') for v in nodes}
            supported = directed_paths(graph, head, candidates, lambda i: bonus[i], set(local))
            chosen = pcst(graph, nodes, local, head, candidates, prizes, bonus, supported, balance, pack)
        selection = set(chosen)
        if len(selection) != len(chosen) or not selection.issubset(local):
            raise ValueError('invalid evidence fact IDs')
        found = directed_paths(graph, head, candidates, lambda _: 0., selection)
        picked_nodes = set(map(int, graph.edge_index[:, chosen].ravel()))
        canonical_facts = {(u, r, v) if r < graph.R // 2 else (v, graph.inverse(r), u)
                           for i in chosen for f in [fact(graph, i)] for u, r, v in [(f.src, f.relation, f.dst)]}
        intermediates = picked_nodes - candidate_set - {head}
        serialized = [fact(graph, i) for i in chosen]
        output[arm] = dict(ids=chosen, paths=found, reachable=len(reachable), covered=len(found),
                           local_nodes=len(nodes), local_edges=len(local), nodes=len(picked_nodes),
                           edges=len(chosen), motif_support=float(np.mean(rho[graph.edge_type[chosen]])) if chosen else 0.,
                           hub_ratio=sum(graph.out_degree[v] > hub_cut for v in intermediates) / max(1, len(intermediates)),
                           duplication=1-len(canonical_facts)/max(1,len(chosen)) if chosen else 0.,
                           no_evidence=not bool(found), path_lengths=[len(q) for q in found.values()],
                           input_bound=estimate(messages(head, rel, mapping, serialized)),
                           coarse_ms=coarse_ms, milliseconds=1000*(perf_counter()-began)+coarse_ms)
    return output
