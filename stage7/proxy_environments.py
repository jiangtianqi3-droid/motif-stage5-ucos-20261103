"""Four query distributions on one shared hidden-fact pool (matched graph)."""
import numpy as np


def bins(values):
    values = np.asarray(values)
    bounds = np.quantile(values,[1/3,2/3]) if len(values) else [0,0]
    return np.searchsorted(bounds,values,side='left'), np.asarray(bounds)


def features(graph, hyperedge):
    f = graph.forward
    deg = np.bincount(f[:,[0,2]].ravel(),minlength=graph.n)
    eb,ebounds = bins(deg[deg>0])
    bounds = np.quantile(deg[deg>0],[1/3,2/3])
    db = np.searchsorted(bounds,deg,side='left')
    support = np.bincount(hyperedge[hyperedge>=0],minlength=graph.R)
    paired_support = support[:graph.R//2] + support[graph.R//2:]
    mb,mbounds = bins(paired_support)
    return {'A':np.zeros(len(f),dtype=int), 'B':f[:,1],
            'C':db[f[:,0]]*3+db[f[:,2]], 'D':mb[f[:,1]]}, dict(
                degree_bounds=bounds.tolist(), motif_bounds=mbounds.tolist(),
                motif_support=paired_support.tolist(), entity_degree=deg.tolist())


def sample(pool, labels, n, rng, natural=False):
    pool = np.asarray(pool,dtype=np.int64)
    if natural:
        return rng.permutation(pool)[:n]
    buckets = [rng.permutation(pool[labels[pool]==v]).tolist() for v in np.unique(labels[pool])]
    # Shuffle strata once; equal mass until a stratum is exhausted, no replacement.
    buckets = [buckets[i] for i in rng.permutation(len(buckets))]
    out = []
    for at in range(max(map(len,buckets),default=0)):
        for b in buckets:
            if at<len(b):
                out.append(b[at])
                if len(out)==n:
                    return np.asarray(out,dtype=np.int64)
    return np.asarray(out,dtype=np.int64)


def environments(graph, base_pool, hyperedge, seed):
    f = graph.forward
    labels,description = features(graph,hyperedge)
    # Budget controls evaluated queries ONLY; all physical masks reach their
    # requested rates when graph constraints permit. It never caps deletion.
    budget = min(192,max(64,int(np.ceil(2*np.sqrt(len(f))))))
    pool = np.random.default_rng(seed+701).permutation(base_pool)
    cut = int(np.floor(.6*len(pool)))
    select_pool,confirm_pool = pool[:cut],pool[cut:]
    ns,nc = min(int(np.floor(.6*budget)),len(select_pool)),min(budget-int(np.floor(.6*budget)),len(confirm_pool))
    envs = {}
    stratum_counts = {}
    for i,name in enumerate('ABCD'):
        rng = np.random.default_rng(seed+1001+37*i)
        s = sample(select_pool,labels[name],ns,rng,name=='A')
        c = sample(confirm_pool,labels[name],nc,rng,name=='A')
        assert not set(s)&set(c)
        envs[name] = {'selection':s,'confirmation':c}
        stratum_counts[name]=dict(available={str(int(k)):int((labels[name][base_pool]==k).sum()) for k in np.unique(labels[name][base_pool])},
            selection={str(int(k)):int((labels[name][s]==k).sum()) for k in np.unique(labels[name][s])},
            confirmation={str(int(k)):int((labels[name][c]==k).sum()) for k in np.unique(labels[name][c])})
    assert not set(np.concatenate([v['selection'] for v in envs.values()])) & set(np.concatenate([v['confirmation'] for v in envs.values()]))
    return envs,dict(query_budget=budget,selection_per_environment=ns,confirmation_per_environment=nc,
        shared_global_selection_confirmation_split=True,common_query_pool_all_densities=True,
        matched_physical_graph_all_environments=True,stratum_counts=stratum_counts,**description)
