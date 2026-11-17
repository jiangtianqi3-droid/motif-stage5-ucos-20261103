"""Path-only feature extraction. Selected fact IDs are the mandatory input."""
from collections import defaultdict
from math import sqrt

import numpy as np
from scipy.special import logsumexp

from p1.paths import reliability, path_choices
from src.graph_loader import fact


def all_paths(graph, head, candidates, selected):
    adj = defaultdict(list)
    candidates = set(candidates)-{head}
    for i in sorted(set(selected)):
        if not 0 <= i < len(graph.edge_type):
            raise ValueError('invalid observation ID')
        adj[int(graph.edge_index[0,i])].append(i)
    out=defaultdict(list)
    def visit(u,seen,path):
        if len(path)==3:
            return
        for i in adj[u]:
            v=int(graph.edge_index[1,i])
            if v in seen:
                continue
            p=path+(i,)
            if v in candidates:
                out[v].append(p)
            visit(v,seen|{v},p)
    visit(head,{head},())
    return out


def support(graph,r,path,table):
    hubs=[int(graph.in_degree[graph.edge_index[1,i]]) for i in path[:-1]]
    return (reliability(table,graph,r,path)-.5)/(sqrt(len(path))*(1+np.log1p(max(hubs,default=0))/8))


def profile(graph,r,paths,prior,table):
    best={}
    for p in paths:
        pattern=tuple(map(int,graph.edge_type[list(p)]))
        s=float(support(graph,r,p,table))
        value=(s,float(np.mean(prior[list(pattern)])),p)
        if pattern not in best or (s,-len(p),p)>(best[pattern][0],-len(best[pattern][2]),best[pattern][2]):
            best[pattern]=value
    return sorted(best.values(),key=lambda x:(-x[0],len(x[2]),x[2]))


def aggregate(values, method):
    if not len(values):
        return 0.
    x=np.sort(values)[::-1]
    if method=='S0':
        return float(x[0])
    if method.startswith('S1-'):
        return float(x[:int(method.split('-')[1])].mean())
    tau=float(method.split('-')[1])
    return float(tau*(logsumexp(x/tau)-np.log(len(x))))


def candidate_scores(profiles,agg,motif_weight,relative=False):
    scores={}
    valid={}
    for c,paths in profiles.items():
        valid[c]=bool(paths)
        scores[c]=aggregate([s+motif_weight*(m-.5) for s,m,_ in paths],agg)
    active=[scores[c] for c in scores if valid[c]]
    avg=float(np.mean(active)) if active else 0.
    other={c:max((scores[i] for i in scores if i!=c and valid[i]),default=scores[c]) for c in scores}
    return {c:dict(score=scores[c]-(avg if relative and valid[c] else 0.),valid=valid[c],
                   relative_mean=scores[c]-avg if valid[c] else 0.,
                   relative_max=scores[c]-other[c] if valid[c] else 0.) for c in scores}


def path_facts(graph, ids):
    return [[f.id,f.src,f.relation,f.dst] for f in (fact(graph,i) for i in sorted(set(ids)))]
