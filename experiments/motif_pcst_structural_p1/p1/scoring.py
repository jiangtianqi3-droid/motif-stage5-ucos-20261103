"""Answer-free, deterministic evidence scores and fixed-slot rank fusion."""
from math import sqrt
import numpy as np

from p1.paths import reliability
from src.ranking_fusion import writeback


def evidence(graph, query_rel, candidates, selected, table):
    result = {}
    for c in candidates:
        paths = selected['paths'].get(c, [])
        values = []
        for path in paths:
            degree = max([int(graph.in_degree[graph.edge_index[1,i]]) for i in path[:-1]],default=0)
            support = reliability(table,graph,query_rel,path)
            weight = (support-.5)/(sqrt(len(path))*(1+np.log1p(degree)/8))
            values.append(float(weight))
        # One best distinct path limits the advantage from high counts. Missing paths
        # receive a neutral additive contribution, with explicit missingness exported.
        result[c] = dict(score=max(values) if values else 0.,valid=bool(values),paths=len(paths))
    return result


def permutations(raw,candidates,values,lam,margin):
    arr = np.asarray(candidates,dtype=np.int64)
    # Convert the reciprocal rank surrogate into its original, pessimistic rank tiers.
    by_rank = np.rint(1/raw[arr]-60).astype(np.int64)
    base = (by_rank.max()-by_rank)/max(1,by_rank.max()-by_rank.min())
    scores = np.asarray([values[c]['score'] for c in candidates])
    prior = [c for c in candidates]
    enforce = lambda v: [candidates[i] for i in np.argsort(-v,kind='stable')]
    # Under raw-rank ties base sorting preserves the stable candidate order.
    fused = enforce(base+lam*scores)
    gate = len(base)>1 and base[0]-base[1]<=margin and any(v['valid'] for v in values.values())
    return dict(evidence_only=enforce(scores),fusion=fused,gated=fused if gate else prior)


def filtered_after(raw,candidates,order,target,excluded,rank):
    if order==candidates:
        return rank(raw,target,excluded)
    full = np.lexsort((np.arange(len(raw)),-raw)).tolist()
    written = writeback(full,candidates,order)
    if written[:len(candidates)]!=order or written[len(candidates):]!=full[len(candidates):]:
        raise ValueError('invalid complete ordering')
    updated = raw.copy()
    updated[order] = raw[candidates]
    return rank(updated,target,excluded)


def raw_order(row):
    if not np.all(np.isfinite(row)) or np.any(row<=0):
        raise ValueError('invalid reciprocal rank surrogate')
    return np.lexsort((np.arange(len(row)),-row)).tolist()
