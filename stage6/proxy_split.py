"""Sequential constrained deletion, stratified before sampling, never relaxed."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from stage6.observed_loader import from_forward
from stage6.util import identity


def make_split(graph, seed, rate=.15, cap=256, min_remaining=2):
    f = graph.forward
    rng = np.random.default_rng(seed)
    deg = np.bincount(f[:, [0, 2]].ravel(), minlength=graph.n)
    freq = np.bincount(f[:, 1], minlength=graph.R // 2)
    # Two bins on relation frequency, two on min endpoint degree, round robin.
    rf = (freq[f[:, 1]] > np.median(freq[freq > 0])).astype(int)
    dd = np.minimum(deg[f[:, 0]], deg[f[:, 2]])
    db = (dd > np.median(dd)).astype(int)
    buckets = [rng.permutation(np.flatnonzero(rf * 2 + db == k)).tolist() for k in range(4)]
    order = []
    for j in range(max(map(len, buckets), default=0)):
        order.extend(bucket[j] for bucket in buckets if j < len(bucket))
    wanted = min(cap, int(np.floor(rate * len(f))))
    # Preserve a deterministic spanning forest of the undirected simple graph.
    # This guarantees component preservation under joint removal (parallel facts safe).
    parent = np.arange(graph.n)
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    protected = set()
    for i in rng.permutation(len(f)):
        a, b = root(f[i, 0]), root(f[i, 2])
        if a != b:
            parent[a] = b
            protected.add(int(i))
    chosen = []
    for i in order:
        if len(chosen) == wanted:
            break
        h, r, t = f[i]
        delta_h, delta_t = (2, 0) if h == t else (1, 1)
        if i in protected or freq[r] <= 1 or deg[h] - delta_h < min_remaining or deg[t] - delta_t < min_remaining:
            continue
        chosen.append(i)
        deg[h] -= delta_h
        deg[t] -= delta_t
        freq[r] -= 1
    # Split once. Both sets are removed before either is scored.
    chosen = rng.permutation(np.asarray(chosen, dtype=np.int64))
    cut = int(np.floor(.6 * len(chosen)))
    keep = np.ones(len(f), dtype=bool)
    keep[chosen] = False
    hidden = f[chosen]
    rem = from_forward(f[keep], graph.n, graph.R,
                       {'parent': graph.key, 'split_seed': seed, 'rate': rate})
    assert not set(map(tuple, rem.forward.tolist())) & set(map(tuple, hidden.tolist()))
    assert np.all(np.bincount(rem.edge_index[1], minlength=graph.n)[np.bincount(graph.edge_index[1], minlength=graph.n) > 0] > 0)
    def components(g):
        a = coo_matrix((np.ones(len(g.edge_type)), g.edge_index), shape=(g.n, g.n)).tocsr()
        return connected_components(a, directed=False)[0]
    assert components(graph) == components(rem)
    meta = dict(seed=seed, rate=rate, requested=wanted, actual=len(hidden), selection=cut,
                confirmation=len(hidden) - cut, hidden_sha256=identity(hidden),
                selection_sha256=identity(hidden[:cut]), confirmation_sha256=identity(hidden[cut:]),
                graph_sha256=graph.key, residual_sha256=rem.key,
                direct_leaks=0, inverse_leaks=0, component_preserved=True,
                relation_coverage_preserved=True, unique_observed_facts=len(f))
    return rem, hidden, cut, meta
