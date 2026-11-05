"""Nested physical masks, with no query-budget cap on removed facts."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from stage6.observed_loader import from_forward
from stage7.util import identity


def remaining_graph(graph, hidden_indices, facts=None):
    """Delete every masked copy, retain multiplicity of all unmasked facts."""
    raw=np.c_[graph.edge_index[0,graph.edge_type<graph.R//2],
              graph.edge_type[graph.edge_type<graph.R//2],graph.edge_index[1,graph.edge_type<graph.R//2]]
    unique,counts=np.unique(raw,axis=0,return_counts=True)
    if facts is not None:assert np.array_equal(unique,facts)
    keep=np.ones(len(unique),dtype=bool);keep[np.asarray(hidden_indices,dtype=np.int64)]=False
    forward=np.repeat(unique[keep],counts[keep],axis=0)
    return from_forward(forward,graph.n,graph.R,{'parent':graph.key})


def nested_masks(graph, seed, rates=(.05,.10,.15), min_remaining=2):
    f = graph.forward
    rng = np.random.default_rng(seed)
    degree = np.bincount(f[:, [0,2]].ravel(), minlength=graph.n)
    frequency = np.bincount(f[:,1], minlength=graph.R//2)
    parent = np.arange(graph.n)
    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    protected = set()
    for i in rng.permutation(len(f)):
        h, _, t = f[i]
        a,b = root(h),root(t)
        if a != b:
            parent[a] = b
            protected.add(int(i))
    desired = [int(np.floor(p*len(f))) for p in rates]
    maximum = max(desired)
    chosen = []
    excluded = dict(forest=0, relation=0, endpoint=0)
    for i in rng.permutation(len(f)):
        if len(chosen) >= maximum:
            break
        h,r,t = f[i]
        dh,dt = (2,0) if h==t else (1,1)
        if i in protected:
            excluded['forest'] += 1
        elif frequency[r] <= 1:
            excluded['relation'] += 1
        elif degree[h]-dh < min_remaining or degree[t]-dt < min_remaining:
            excluded['endpoint'] += 1
        else:
            chosen.append(int(i))
            degree[h] -= dh
            degree[t] -= dt
            frequency[r] -= 1
    masks, metadata = [], []
    previous = 0
    for p,want in zip(rates,desired):
        hidden = np.asarray(chosen[:want],dtype=np.int64)
        keep = np.ones(len(f),dtype=bool)
        keep[hidden] = False
        rem = remaining_graph(graph,hidden,f)
        hidden_set = set(map(tuple,f[hidden]))
        assert not hidden_set & set(map(tuple,rem.forward))
        aug = set(zip(rem.edge_index[0],rem.edge_type,rem.edge_index[1]))
        assert not any((t,r+graph.R//2,h) in aug for h,r,t in f[hidden])
        residual = np.bincount(rem.edge_index[0],minlength=graph.n)
        original = np.bincount(graph.edge_index[0],minlength=graph.n)
        assert np.all(residual[original>0]>0)
        def components(g):
            return connected_components(coo_matrix((np.ones(len(g.edge_type)),g.edge_index),shape=(g.n,g.n)).tocsr(),directed=False)[0]
        assert components(rem)==components(graph)
        masks.append(hidden)
        metadata.append(dict(rate=p,desired=want,hidden=len(hidden),actual_rate=len(hidden)/len(f),
            added=len(hidden)-previous,hidden_sha256=identity(f[hidden]),residual_sha256=rem.key,
            excluded=excluded,forest_facts=len(protected),direct_inverse_leaks=0,components_preserved=True))
        metadata[-1]['remaining_duplicate_multiplicity_preserved']=True
        previous=len(hidden)
    return masks,metadata
