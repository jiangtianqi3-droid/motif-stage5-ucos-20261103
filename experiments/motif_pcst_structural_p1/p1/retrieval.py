"""Nested R0-R5 on one shared local graph, with a fixed real-fact budget."""
from collections import Counter, defaultdict
from time import perf_counter

import numpy as np

from p1.paths import path_choices, reliability
from src.graph_loader import fact
from src.local_retrieval import coarse
from src.pcst_adapter import Fact, solve as poc_solve
from query_community_v1.src.edge_weighter import CommunityEdgeWeighter

ARMS = ('R0','R1','R2','R3','R4','R5')
ABLATIONS = ('R5-no-reliability','R5-no-discrimination','R5-no-dedup')
COST = .34


def true_fact(g,i):
    u,v = map(int,g.edge_index[:,i])
    r = int(g.edge_type[i])
    return (u,r,v) if r<g.R//2 else (v,g.inverse(r),u)


def capped(groups, budget, canonical=None):
    picked = {}
    seen = set()
    for group in groups:
        new = []
        keys = []
        for i in dict.fromkeys(group):
            if i in picked:
                continue
            key = canonical[i] if canonical is not None else i
            if key in seen or key in keys:
                continue
            new.append(i)
            keys.append(key)
        if len(picked)+len(new)<=budget:
            picked.update(dict.fromkeys(new))
            seen.update(keys)
    return list(picked)


def verify_connected(g, selected, head):
    reached = {head}
    remaining = set(selected)
    while remaining:
        touching = {i for i in remaining if reached.intersection(g.edge_index[:,i])}
        if not touching:
            raise ValueError('PCST evidence disconnected from initialized head')
        reached.update(int(v) for i in touching for v in g.edge_index[:,i])
        remaining -= touching


def adapt_solver(nodes, facts, prizes, bonus, costs, root):
    # A virtual node can be returned as a leaf. Reconstituting its original edge
    # adds the missing endpoint at zero converted cost. Recheck root connectivity
    # on ORIGINAL IDs after contraction; virtual IDs never reach the evidence.
    return poc_solve(nodes,facts,prizes,bonus,costs,root=root)


def choose_pcst(g,nodes,ids,head,order,prizes,bonuses,costs,paths,budget,balance,dedup):
    reserve_groups = []
    if balance:
        # Candidate rank is a fixed tie-break; path admission is atomic.
        for c in order:
            if paths.get(c):
                reserve_groups.append(paths[c][0])
    # Retain all directed traversals of protected paths, including a needed inverse.
    backbone = capped(reserve_groups,budget)
    canonical = {i:true_fact(g,i) for i in ids}
    keep = set(backbone)
    contracted = {head}
    for i in keep:
        contracted.update(map(int,g.edge_index[:,i]))
    def vertex(v):
        return head if v in contracted else v
    parents = {}
    for i in ids:
        f = fact(g,i)
        s,t = vertex(f.src),vertex(f.dst)
        if s!=t and i not in keep:
            parents[i] = Fact(i,s,f.relation,t)
    verts = sorted({vertex(v) for v in nodes})
    facts = list(parents.values())
    _, selected = adapt_solver(verts,facts,[prizes[v] for v in verts],
                               [bonuses[f.id] for f in facts],[costs[f.id] for f in facts],head)
    remaining = {f.id for f in selected}
    edges = backbone[:]
    used = {canonical[i] for i in edges}
    while remaining and len(edges)<budget:
        touching = [i for i in remaining if contracted.intersection(g.edge_index[:,i])]
        if not touching:
            raise ValueError('restored solver result is disconnected')
        i = min(touching,key=lambda k:(costs[k]-bonuses[k],k))
        remaining.remove(i)
        # Dropping an inverse or identical fact is only safe when the existing fact
        # already joins these exact endpoints in the optimization skeleton.
        if dedup and canonical[i] in used:
            continue
        edges.append(i)
        used.add(canonical[i])
        contracted.update(map(int,g.edge_index[:,i]))
    verify_connected(g,edges,head)
    return edges,len(backbone)


class Retriever:
    def __init__(self,graph,gate_vals,table,config):
        self.g = graph
        self.gates = gate_vals
        self.table = table
        self.cfg = config
        self.weights = CommunityEdgeWeighter(graph).compute()
        self.hub_cut = float(np.quantile(graph.in_degree,.9))
        self.hub = np.log1p(graph.in_degree)/max(1.,float(np.log1p(graph.in_degree).max()))

    def query(self,head,rel,candidates,budget,arms=ARMS,matched=False):
        g = self.g
        began = perf_counter()
        nodes,ids,fwd,back,reachable = coarse(g,self.weights,head,candidates)
        # Allocate one slot per directed observation, even if raw records repeat it.
        by_triple = {}
        for i in ids:
            f = fact(g,i)
            by_triple.setdefault((f.src,f.relation,f.dst),i)
        ids = list(by_triple.values())
        if len(nodes)>self.cfg['max_local_nodes'] or len(ids)>self.cfg['max_local_edges']:
            raise ValueError('local search budget exceeded')
        focus = {head,*candidates}
        gate = self.gates[rel]
        motif = gate['layers'].mean(0)
        base = {v:.10+min(1.,2.5*fwd.get(v,0.)+.6*back[v])+.45*(v in focus) for v in nodes}
        def ppr_edge(i):
            u,v = g.edge_index[:,i]
            return fwd.get(u,0.)*self.weights.edge_probability[i]+.35*max(back[u],back[v])
        values = {i:ppr_edge(i) for i in ids}
        oracle = path_choices(g,head,candidates,ids,values.__getitem__)
        roster = defaultdict(set)
        trust = defaultdict(float)
        for c,paths in oracle.items():
            for path in paths:
                weight = reliability(self.table,g,rel,path)
                for i in path:
                    roster[i].add(c)
                    trust[i] = max(trust[i],weight)
        motifs = {i:float(motif[g.edge_type[i]]) for i in ids}
        common_ms = (perf_counter()-began)*1000
        selected = {}
        for arm in arms:
            start = perf_counter()
            k = 5 if arm in ABLATIONS else int(arm[1])
            costs = {i:COST+(.26*self.hub[g.edge_index[1,i]] if k>=4 and g.edge_index[1,i] not in focus else 0.)
                     for i in ids}
            discr = {i:1/len(roster[i]) if roster[i] else 0. for i in ids}
            bonuses = {i:(.52*motifs[i] if k>=3 else 0.)
                       + ((.22*trust[i] if arm!='R5-no-reliability' else 0.)
                          + (.20*discr[i] if arm!='R5-no-discrimination' else 0.) if k>=5 else 0.)
                       for i in ids}
            if k==0:
                chosen = sorted(ids,key=lambda i:(-values[i],i))[:budget]
                protected = 0
            else:
                # R2 fixes the same ordered candidate backbones in R3/R4/R5 so that the
                # nested contrast measures extension rewards, with equal path obligations.
                chosen,protected = choose_pcst(g,nodes,ids,head,candidates,base,bonuses,costs,oracle,
                                                budget,k>=2,k>=5 and arm!='R5-no-dedup')
            if len(chosen)>budget or len(set(chosen))!=len(chosen) or not set(chosen).issubset(ids):
                raise ValueError('invalid evidence IDs')
            selected[arm] = dict(ids=chosen,protected=protected,costs=[costs[i] for i in chosen],
                                 bonuses=[bonuses[i] for i in chosen],
                                 ms=common_ms+(perf_counter()-start)*1000)
        comp = min(len(x['ids']) for x in selected.values()) if matched else None
        output = {}
        for arm,val in selected.items():
            output[arm] = self.describe(head,rel,candidates,reachable,motif,values,budget,val,base)
            if matched:
                ids = val['ids'][:comp]
                matched_val = dict(val,ids=ids,protected=min(comp,val['protected']),
                                   costs=val['costs'][:comp],bonuses=val['bonuses'][:comp])
                output[arm+'-equal'] = self.describe(head,rel,candidates,reachable,motif,values,comp,matched_val,base)
        return output

    def describe(self,head,rel,candidates,reachable,motif,priority,budget,val,prizes):
        g = self.g
        chosen = val['ids']
        paths = path_choices(g,head,candidates,chosen,priority.__getitem__)
        nodes = {head}
        nodes.update(map(int,g.edge_index[:,chosen].ravel()))
        hubs = nodes-{head,*candidates}
        supports = [reliability(self.table,g,rel,p) for pp in paths.values() for p in pp]
        metrics = dict(selected_nodes=len(nodes),selected_observed_edges=len(chosen),
                       path_count=len(supports),
                       path_length_distribution=dict(Counter(len(p) for pp in paths.values() for p in pp)),
                       covered=len(paths),reachable=len(reachable),
                       candidate_coverage_all=len(paths)/len(candidates),
                       candidate_coverage_reachable=len(paths)/len(reachable) if reachable else None,
                       motif_support=float(np.mean(motif[g.edge_type[chosen]])) if chosen else 0.,
                       path_reliability=float(np.mean(supports)) if supports else None,
                       hub_ratio=sum(g.in_degree[v]>self.hub_cut for v in hubs)/max(1,len(hubs)),
                       evidence_redundancy=1-len({true_fact(g,i) for i in chosen})/len(chosen) if chosen else 0.,
                       budget_utilization=len(chosen)/budget if budget else 0.,
                       fact_budget=budget, protected_facts=val['protected'],retrieval_latency_ms=val['ms'])
        return dict(**val,paths=paths,nodes=sorted(nodes),node_prizes={v:prizes[v] for v in nodes},
                    motif=[float(x) for x in motif[g.edge_type[chosen]]], metrics=metrics)
