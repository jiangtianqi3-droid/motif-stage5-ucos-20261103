"""Canonical facts and deterministic local matched controls."""
from collections import defaultdict, Counter, deque

import numpy as np

from p1.assets import verify
from p1.retrieval import true_fact
from src.io_utils import identity


class FactIndex:
    def __init__(self, graph):
        self.graph = graph
        self.records = defaultdict(list)
        self.id_to_fact = []
        for i in range(len(graph.edge_type)):
            f = true_fact(graph,i)
            self.records[f].append(i)
            self.id_to_fact.append(f)
        self.facts = tuple(sorted(self.records))
        self.fact_id = {f:i for i,f in enumerate(self.facts)}
        self.directed_to_id = np.array([self.fact_id[f] for f in self.id_to_fact],np.int64)
        self.by_id = {i:tuple(sorted(self.records[f])) for i,f in enumerate(self.facts)}
        for f,ids in self.records.items():
            u,r,v=f
            expected={(u,r,v),(v,graph.inverse(r),u)}
            observed={(int(graph.edge_index[0,i]),int(graph.edge_type[i]),int(graph.edge_index[1,i])) for i in ids}
            verify(observed==expected,'canonical fact missing its inverse')

    def canonical(self, edges):
        return tuple(sorted({int(self.directed_to_id[int(i)]) for i in edges}))

    def closure(self, facts):
        return np.array(sorted(i for f in facts for i in self.by_id[int(f)]),dtype=np.int64)

    def subgraph_ids(self, selected):
        return self.closure(self.canonical(selected))

    def complement(self, facts):
        return np.flatnonzero(~np.isin(self.directed_to_id,np.array(tuple(facts),dtype=np.int64)))

    def assert_paired(self, ids):
        ids=set(map(int,ids))
        verify(all(set(self.by_id[f]).issubset(ids) for f in self.canonical(ids)),
               'fact deletion or subgraph omitted inverse/duplicate record')


def strata(index, local_facts, head):
    """Directed min edge distance in inverse-closed Gq, preserving 1/2/3/>3 buckets."""
    graph=index.graph
    ids=index.closure(local_facts)
    adjacency=defaultdict(set)
    for i in ids:
        u,v=map(int,graph.edge_index[:,i])
        adjacency[u].add(v)
    distance={head:0}
    queue=deque([head])
    while queue:
        u=queue.popleft()
        if distance[u]>=3:
            continue
        for v in adjacency[u]:
            if v not in distance:
                distance[v]=distance[u]+1
                queue.append(v)
    relation_counts=Counter(index.facts[f][1] for f in range(len(index.facts)))
    logfreq=np.log1p(list(relation_counts.values()))
    cuts=np.quantile(logfreq,[1/3,2/3])
    rows={}
    for f in local_facts:
        u,r,v=index.facts[f]
        hop=min(4,1+min(distance.get(u,3),distance.get(v,3)))
        freq=int(np.searchsorted(cuts,np.log1p(relation_counts[r]),side='right'))
        rows[f]=(hop,freq,r)
    return rows,dict(relation_counts),cuts.tolist()


def matched_random(selected, local_facts, strata_by_fact, seed):
    """First exclude S, then relax strata, only last permit local overlap."""
    selected=set(selected)
    all_local=set(local_facts)
    verify(selected<=all_local,'retrieval fact outside local graph')
    rng=np.random.default_rng(seed)
    needed=sorted(selected)
    # Seeded order avoids giving low fact IDs priority at sparse strata.
    needed=[needed[i] for i in rng.permutation(len(needed))]
    unused=all_local-selected
    taken=set()
    detail=[]
    for reference in needed:
        hop,freq,relation=strata_by_fact[reference]
        levels=[
            ('exact_relation_hop',lambda f:strata_by_fact[f][0]==hop and strata_by_fact[f][2]==relation),
            ('hop_frequency',lambda f:strata_by_fact[f][:2]==(hop,freq)),
            ('hop_only',lambda f:strata_by_fact[f][0]==hop),
            ('frequency_only',lambda f:strata_by_fact[f][1]==freq),
            ('local_only',lambda f:True)]
        choice=None
        for name,condition in levels:
            candidates=sorted(f for f in unused-taken if condition(f))
            if candidates:
                choice=int(rng.choice(candidates))
                break
        if choice is None:
            candidates=sorted(all_local-taken)
            verify(bool(candidates),'local graph cannot match original unique fact budget')
            choice=int(rng.choice(candidates))
            name='local_overlap_last_resort'
        taken.add(choice)
        detail.append(dict(reference_fact_id=reference,random_fact_id=choice,relaxation=name,
                           reference_hop=hop,random_hop=strata_by_fact[choice][0],
                           reference_frequency=freq,random_frequency=strata_by_fact[choice][1],
                           reference_relation=relation,random_relation=strata_by_fact[choice][2],
                           overlap_selected=choice in selected))
    verify(len(taken)==len(selected) and taken<=all_local,'matched random invalid budget')
    return tuple(sorted(taken)),detail
