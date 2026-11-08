"""Deterministic hyperedge -> relation -> entity-edge mapping."""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from scipy.special import expit

@dataclass
class EdgeWeights:
    weights: np.ndarray
    rho: np.ndarray
    degree_factor: np.ndarray
    transition: sp.csr_matrix
    edge_probability: np.ndarray
    reverse_transition: sp.csr_matrix
    row_sums: np.ndarray

class CommunityEdgeWeighter:
    def __init__(self,graph,signature=None,eta_out=.5,eta_rel=.5,eta_in=0.,eta=.5,support='log_count'):
        self.g=graph;self.sig=signature
        self.eta_out,self.eta_rel,self.eta_in,self.eta=eta_out,eta_rel,eta_in,eta
        if signature is not None:
            he=signature['edge_index'];E=he.shape[1]
            slots,ids=np.nonzero(he>=0)
            self.incidence=sp.coo_matrix((np.ones(len(ids)),(he[slots,ids],ids)),shape=(graph.R,E)).tocsr()
            # Each motif contributes once to each distinct relation, even repeated slots.
            self.incidence.data[:]=1
            self.support=signature['raw'][:,0] if support=='log_count' else np.ones(E)
            self.support=np.asarray(self.support,dtype=np.float64)

    def relation_importance(self,q,submode='topo_signature',gates=None):
        sig=self.sig
        if sig is None: raise ValueError('topology_aware requires audited signatures')
        if submode=='topo_signature':
            # m~q := direct relation-hypergraph cooccurrence, q is an active slot of m.
            active=np.asarray(self.incidence.getrow(int(q)).toarray()).ravel()>0
            a=self.support*active
            val=expit(sig['z'][:,3]+sig['z'][:,2])
        elif submode=='gated_model':
            if gates is None: raise ValueError('gated_model requires actual frozen model gates')
            val=np.asarray(gates,dtype=float);a=self.support
            assert val.shape==a.shape and np.isfinite(val).all() and (val>=0).all()
        else: raise ValueError(submode)
        den=np.asarray(self.incidence@a).ravel()
        rho=np.asarray(self.incidence@(a*val)).ravel()/(den+1e-12)
        # Exactly the requested epsilon-denominator formula: no support gives rho=0.
        assert np.isfinite(rho).all() and (rho>=0).all()
        return rho

    @staticmethod
    def bounded_map(rho):
        # Min/max over the entire public relation vocabulary. Constant input is neutral.
        span=float(np.ptp(rho))
        return np.ones_like(rho) if span<1e-12 else .1+1.9*(rho-rho.min())/span

    def compute(self,mode='uniform',q=0,submode='topo_signature',gates=None):
        g=self.g;u,v=g.edge_index;r=g.edge_type
        rho=np.ones(g.R)
        degree=np.ones(len(r))
        if mode=='uniform': w=degree.copy()
        elif mode=='degree_aware':
            degree=(1.+g.out_degree[u])**(-self.eta_out)*(1.+g.edge_rel_out)**(-self.eta_rel)*(1.+g.in_degree[v])**(-self.eta_in)
            w=degree/degree.mean() if len(degree) else degree
        elif mode=='topology_aware':
            rho=self.relation_importance(q,submode,gates)
            degree=(1.+g.out_degree[u])**(-self.eta)
            w=self.bounded_map(rho)[r]*degree
        else: raise ValueError(mode)
        assert np.isfinite(w).all() and (w>=0).all()
        rows=np.bincount(u,weights=w,minlength=g.n)
        prob=np.divide(w,rows[u],out=np.zeros_like(w),where=rows[u]>0)
        transition=sp.coo_matrix((prob,(u,v)),shape=(g.n,g.n)).tocsr()
        rs=np.asarray(transition.sum(1)).ravel()
        assert np.allclose(rs[rows>0],1) and np.all(rs[rows==0]==0)
        reverse_rows=np.bincount(v,weights=w,minlength=g.n)
        reverse_prob=np.divide(w,reverse_rows[v],out=np.zeros_like(w),where=reverse_rows[v]>0)
        reverse=sp.coo_matrix((reverse_prob,(v,u)),shape=(g.n,g.n)).tocsr()
        return EdgeWeights(w,rho,degree,transition,prob,reverse,rows)
