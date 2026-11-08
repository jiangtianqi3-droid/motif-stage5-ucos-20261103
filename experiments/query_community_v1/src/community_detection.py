"""Sparse weighted Louvain fallback; original directed multiedges never modified."""
import time
import networkx as nx
import numpy as np
import scipy.sparse as sp

def detect_communities(graph,weights=None,seed=1024):
    began=time.perf_counter()
    u,v=graph.edge_index
    w=np.ones(len(u)) if weights is None else np.asarray(weights,dtype=float)
    assert w.shape==u.shape and np.isfinite(w).all() and (w>=0).all()
    a=sp.coo_matrix((w,(u,v)),shape=(graph.n,graph.n)).tocsr()
    # Sum the two directed masses, consolidating all relation labels for clustering only.
    projection=(a+a.T).tocsr()
    net=nx.from_scipy_sparse_array(projection,create_using=nx.Graph)
    if net.size(weight='weight'):
        groups=nx.community.louvain_communities(net,weight='weight',seed=seed)
        groups=sorted(groups,key=lambda x:min(x))
        modularity=nx.community.modularity(net,groups,weight='weight')
    else:
        groups=[{i} for i in range(graph.n)];modularity=0.
    labels=np.empty(graph.n,dtype=np.int64)
    for c,members in enumerate(groups): labels[list(members)]=c
    members=[sorted(c) for c in groups]
    intra=labels[u]==labels[v]
    hubs=np.argsort(-(graph.in_degree+graph.out_degree),kind='stable')[:max(1,int(np.ceil(.05*graph.n)))]
    return dict(labels=labels,members=members,statistics=dict(algorithm='networkx_louvain_sparse',seed=seed,
        modularity=float(modularity),sizes=[len(c) for c in groups],communities=len(groups),
        intra_edge_fraction=float(intra.mean()) if len(intra) else 0.,
        inter_edge_fraction=float((~intra).mean()) if len(intra) else 0.,
        hub_community_counts=np.bincount(labels[hubs],minlength=len(groups)).tolist() if graph.n else [],
        seconds=time.perf_counter()-began))
