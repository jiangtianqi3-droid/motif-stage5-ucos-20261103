import sys

import numpy as np

from src.io_utils import ROOT, read, sha, snapshot, RESULTS
from src.recovery import load
from src.pcst_adapter import Fact

sys.path.insert(0, str(ROOT.parent))
from query_community_v1.src.graph_adapter import ObservedGraph
from query_community_v1.src.edge_weighter import CommunityEdgeWeighter


def open_graph(key, model):
    z = load(snapshot(key, 'observed.npz'))
    graph = ObservedGraph(z['edge_index'], z['edge_type'], int(z['n']), int(z['R']))
    graph.assert_inverse_pairs()
    sig = load(snapshot(key, 'canonical_topology.npz'))
    entry = next(x for x in read(snapshot(key, 'observed.npz').parents[1] / 'manifest.json')['datasets']
                 if x['dataset'] == key)
    if graph.key != entry['graph_key'] or str(sig['key']) != entry['canonical_key']:
        raise ValueError('observation and motif graph identifiers differ')
    records = read(RESULTS / 'recovery/recovery.json')['records']
    rec = next(r for r in records if r['dataset'] == key and r['model'] == model)
    from pathlib import Path
    cache = Path(rec['path'])
    if sha(cache) != rec['cache_sha256']:
        raise ValueError('candidate cache checksum mismatch')
    ranks = np.load(cache, mmap_mode='r', allow_pickle=False)
    queries = np.asarray(read(RESULTS / 'recovery' / key / 'queries.json'), dtype=np.int64)
    if ranks.shape != (*queries.shape[:2], graph.n):
        raise ValueError('query and score shapes disagree')
    return graph, CommunityEdgeWeighter(graph, sig), ranks, queries


def fact(graph, i):
    u, v = graph.edge_index[:, i]
    return Fact(int(i), int(u), int(graph.edge_type[i]), int(v))
