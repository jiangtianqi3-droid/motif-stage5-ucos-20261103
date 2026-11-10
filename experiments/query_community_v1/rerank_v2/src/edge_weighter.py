"""Unified F operator; historical C/D/E3a remain byte-for-byte unchanged."""
import numpy as np
import scipy.sparse as sp
from .util import historical_package
historical_package()
from qc_v1.edge_weighter import CommunityEdgeWeighter, EdgeWeights

class UnifiedEdgeWeighter(CommunityEdgeWeighter):
    def compute(self, mode='ppr_topology_unified', q=0, submode='topo_signature', gates=None):
        if mode != 'ppr_topology_unified': return super().compute(mode, q, submode, gates)
        g = self.g; u, v = g.edge_index; r = g.edge_type
        rho = self.relation_importance(q)
        degree = (1. + g.out_degree[u]) ** (-.25) * (1. + g.edge_rel_out) ** (-.5) * (1. + g.out_degree[v]) ** (-.25)
        raw = self.bounded_map(rho)[r] * degree
        w = raw / raw.mean() if len(raw) else raw
        rows = np.bincount(u, weights=w, minlength=g.n)
        probability = np.divide(w, rows[u], out=np.zeros_like(w), where=rows[u] > 0)
        forward = sp.coo_matrix((probability, (u, v)), shape=(g.n, g.n)).tocsr()
        back_rows = np.bincount(v, weights=w, minlength=g.n)
        back_p = np.divide(w, back_rows[v], out=np.zeros_like(w), where=back_rows[v] > 0)
        reverse = sp.coo_matrix((back_p, (v, u)), shape=(g.n, g.n)).tocsr()
        for mat, mass in [(forward, rows), (reverse, back_rows)]:
            sums = np.asarray(mat.sum(1)).ravel()
            assert np.allclose(sums[mass > 0], 1.) and np.all(sums[mass == 0] == 0)
        assert np.isfinite(w).all() and (w > 0).all()
        return EdgeWeights(w, rho, degree, forward, probability, reverse, rows)

def weights_for(factory, method, q):
    return factory.compute(dict(R1='uniform', R2='degree_aware', R3='topology_aware', R4='ppr_topology_unified')[method], q)
