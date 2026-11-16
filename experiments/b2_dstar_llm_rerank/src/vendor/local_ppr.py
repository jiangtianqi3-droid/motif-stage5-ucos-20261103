"""Truncated sparse PPR. alpha is restart probability, not continuation probability."""
import numpy as np
import scipy.sparse as sp

def local_ppr(transition,seed,alpha=.2,hops=3):
    assert .15<=alpha<=.3 and hops in (2,3)
    n=transition.shape[0]
    assert 0<=seed<n
    p=sp.csr_matrix(([1.],([0],[int(seed)])),shape=(1,n))
    restart=p.copy()
    rows=np.asarray(transition.sum(1)).ravel()
    for _ in range(hops):
        # Dangling probability returns to the public seed, preserving probability mass.
        dangling=float(p[:,rows==0].sum())
        p=(1-alpha)*(p@transition)+(alpha+(1-alpha)*dangling)*restart
        p.eliminate_zeros()
    assert np.isfinite(p.data).all() and (p.data>=0).all() and np.isclose(p.sum(),1.)
    return {int(v):float(s) for v,s in zip(p.indices,p.data)}

def intersections(forward,reverse):
    return sorted(((v,float(np.sqrt(forward[v]*reverse[v]))) for v in forward.keys() & reverse.keys()),
                  key=lambda x:(-x[1],x[0]))

def local_relation_distribution(graph,weights,ppr,head,signature=None):
    # Directed 2-hop locality is independent of any candidate or answer.
    nodes={int(head)};front={int(head)}
    for _ in range(2):
        nxt=set()
        for u in front: nxt.update(map(int,graph.edge_index[1,graph.edges_at(u)]))
        front=nxt-nodes;nodes.update(nxt)
    # Restrict to the strongest 50 nodes under the same head-only PPR ranking.
    selected={head}|set(sorted(nodes-{head},key=lambda v:(-ppr.get(v,0.),v))[:49])
    u,v=graph.edge_index
    active=np.isin(u,list(selected)) & np.isin(v,list(selected))
    ids=np.flatnonzero(active)
    mass=np.array([ppr.get(int(u[e]),0.) for e in ids])*weights.weights[ids]
    pi=np.bincount(graph.edge_type[ids],weights=mass,minlength=graph.R).astype(float)
    if pi.sum(): pi/=pi.sum()
    presence=[]
    if signature is not None:
        local=set(map(int,graph.edge_type[ids]))
        for k in range(7):
            he=signature['edge_index'][:,signature['edge_type']==k]
            valid=he>=0
            present=np.isin(he,list(local)) | ~valid
            presence.append(float(present.all(0).mean()) if he.shape[1] else 0.)
    return pi,presence,selected,ids
