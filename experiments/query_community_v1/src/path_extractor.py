"""Exact directed <=3-hop joins, then query-wide node/edge budgeting."""
import heapq
import time
import numpy as np
from .local_ppr import local_ppr,intersections,local_relation_distribution

def top_paths(graph,weights,h,t,max_hops=3,limit=10,allowed_nodes=None):
    """No beam pruning: a high-value bridge is never dropped for high fanout.

    Enumerate all directed simple paths up to the requested length using a
    two-step forward / one-step reverse join. Memory is O(edges + top-M).
    """
    if h==t: return []
    edge=graph.edge_index
    incoming={}
    for e in graph.edges_at(t,reverse=True):
        b=int(edge[0,e])
        incoming.setdefault(b,[]).append(int(e))
    first_ids=graph.edges_at(h)
    left_cost=int(graph.out_degree[edge[1,first_ids]].sum())
    last_ids=graph.edges_at(t,reverse=True)
    right_cost=int(graph.in_degree[edge[0,last_ids]].sum())
    reverse_join=right_cost<left_cost
    heap=[]
    def keep(ids):
        nodes=[h]+[int(edge[1,e]) for e in ids]
        if len(set(nodes))!=len(nodes): return
        if allowed_nodes is not None and any(v not in allowed_nodes for v in nodes): return
        strength=float(np.prod(weights.weights[list(ids)]))
        key=(strength,tuple(-e for e in ids))
        item=(key,tuple(ids))
        if len(heap)<limit: heapq.heappush(heap,item)
        elif item>heap[0]: heapq.heapreplace(heap,item)
    outgoing={}
    for e1 in first_ids:
        e1=int(e1);a=int(edge[1,e1])
        outgoing.setdefault(a,[]).append(e1)
        if allowed_nodes is not None and a not in allowed_nodes: continue
        if a==t: keep((e1,));continue
        if a==h: continue
        if max_hops>=2:
            for last in incoming.get(a,[]): keep((e1,last))
        if max_hops>=3 and not reverse_join:
            for e2 in graph.edges_at(a):
                e2=int(e2);b=int(edge[1,e2])
                if b in (h,a,t): continue
                if allowed_nodes is not None and b not in allowed_nodes: continue
                for last in incoming.get(b,[]): keep((e1,e2,last))
    if max_hops>=3 and reverse_join:
        # Algebraically identical reverse two-step / forward one-step join;
        # choose the cheaper direction without dropping a single possible path.
        for b,lasts in incoming.items():
            if b in (h,t): continue
            if allowed_nodes is not None and b not in allowed_nodes: continue
            for e2 in graph.edges_at(b,reverse=True):
                e2=int(e2);a=int(edge[0,e2])
                if a in (h,b,t): continue
                for first in outgoing.get(a,[]):
                    for last in lasts: keep((first,e2,last))
    return [ids for _,ids in sorted(heap,reverse=True)]

def describe_path(graph,weights,ids,labels):
    triples=[];edges=[];relations=[];bridge=False;bridge_nodes=set()
    for e in ids:
        u,v=map(int,graph.edge_index[:,e]);r=int(graph.edge_type[e]);cross=bool(labels[u]!=labels[v])
        triples.append([u,r,v]);relations.append(r);bridge|=cross
        if cross: bridge_nodes.update([u,v])
        edges.append(dict(edge_id=int(e),head=u,relation=r,tail=v,rho=float(weights.rho[r]),
            degree_weight=float(weights.degree_factor[e]),weight=float(weights.weights[e]),
            transition_probability=float(weights.edge_probability[e]),is_bridge=cross))
    return dict(triples=triples,edge_ids=list(map(int,ids)),relation_types=relations,edges=edges,
        path_weight=float(np.prod(weights.weights[list(ids)])),
        transition_path_probability=float(np.prod(weights.edge_probability[list(ids)])),contains_bridge=bridge,
        bridge_nodes=sorted(bridge_nodes))

def retrieve(graph,weights,labels,h,q,candidates,method,signature=None,max_nodes=50,max_edges=100,
             max_paths=10,hops=3,alpha=.2,path_cache=None,tail_cache=None):
    began=time.perf_counter()
    assert max_nodes>=1 and max_edges>=0 and max_paths>=1
    head=local_ppr(weights.transition,h,alpha,hops)
    pi,presence,local_nodes,local_ids=local_relation_distribution(graph,weights,head,h,signature)
    allowed=None
    if method=='naive_khop':
        # Fixed deterministic breadth-first ID ordering, no candidate seeding.
        nodes={h};front={h}
        for _ in range(2):
            nxt=set()
            for u in sorted(front): nxt.update(map(int,graph.edge_index[1,graph.edges_at(u)]))
            front=nxt-nodes;nodes.update(nxt)
        allowed={h}|set(sorted(nodes-{h})[:max_nodes-1])
    elif method=='global_community':
        nodes=np.flatnonzero(labels==labels[h]).tolist()
        allowed={h}|set(sorted((v for v in nodes if v!=h),key=lambda v:(-head.get(v,0.),v))[:max_nodes-1])
    evidence=[];path_queue=[]
    for i,candidate in enumerate(candidates):
        t=candidate['entity']
        if tail_cache is not None and t in tail_cache:
            tail=tail_cache[t]
        else:
            tail=local_ppr(weights.reverse_transition,t,alpha,hops)
            if tail_cache is not None: tail_cache[t]=tail
        meeting=intersections(head,tail)
        if path_cache is not None and t in path_cache:
            paths=path_cache[t][:max_paths]
        else:
            paths=top_paths(graph,weights,h,t,hops,10 if path_cache is not None else max_paths,allowed)
            if path_cache is not None: path_cache[t]=paths
        record=dict(candidate,community_head=int(labels[h]),community_tail=int(labels[t]),
            is_cross_community=bool(labels[h]!=labels[t]),
            ppr_affinity_score=float(sum(s for _,s in meeting)),meeting_nodes=[dict(entity=v,score=s) for v,s in meeting[:10]],
            evidence_paths=[],max_path_weight=0.,status='no_evidence')
        evidence.append(record)
        for ids in paths:
            desc=describe_path(graph,weights,ids,labels)
            path_queue.append((desc['path_weight'],i,tuple(ids),desc))
    selected_nodes={h};selected_edges=set()
    # Query-wide budgets, shared fairly through a first-path round and remaining paths.
    first=[];rest=[];seen=set()
    for row in sorted(path_queue,key=lambda x:(-x[0],x[1],x[2])):
        if row[1] not in seen: first.append(row);seen.add(row[1])
        else: rest.append(row)
    for _,i,ids,desc in first+rest:
        nodes={int(graph.edge_index[s,e]) for e in ids for s in (0,1)}
        if len(selected_nodes|nodes)>max_nodes or len(selected_edges|set(ids))>max_edges: continue
        selected_nodes.update(nodes);selected_edges.update(ids)
        evidence[i]['evidence_paths'].append(desc)
    # Preserve directed multiedges among the selected head context after all evidence.
    context_nodes=allowed if allowed is not None else local_nodes
    context_ids=[int(e) for e in local_ids if int(graph.edge_index[0,e]) in context_nodes and int(graph.edge_index[1,e]) in context_nodes]
    if method=='naive_khop':
        context_ids=[int(e) for u in sorted(allowed) for e in graph.edges_at(u) if int(graph.edge_index[1,e]) in allowed]
    for e in sorted(context_ids,key=lambda e:(-weights.weights[e]*head.get(int(graph.edge_index[0,e]),0.),e)):
        nodes=set(map(int,graph.edge_index[:,e]))
        if len(selected_nodes|nodes)<=max_nodes and len(selected_edges|{e})<=max_edges:
            selected_nodes.update(nodes);selected_edges.add(e)
    for record in evidence:
        paths=record['evidence_paths']
        if paths:
            record['status']='ok';record['max_path_weight']=max(p['path_weight'] for p in paths)
        # This check also prevents any exported path outside the query subgraph budget.
        assert all(set(p['edge_ids'])<=selected_edges for p in paths)
    assert len(selected_nodes)<=max_nodes and len(selected_edges)<=max_edges
    return dict(schema_version='query-community-1.0',graph_key=graph.key,method=method,
        query_info=dict(head=int(h),relation=int(q),pi_h=pi.tolist(),anchor_presence_7=presence,
                        anchor_presence_definition='fraction of relation motifs whose distinct relation types all occur in head-only local 2hop edges'),
        candidate_evidence=evidence,subgraph=dict(nodes=sorted(selected_nodes),edge_ids=sorted(selected_edges),
            directed_edges=[dict(edge_id=e,head=int(graph.edge_index[0,e]),relation=int(graph.edge_type[e]),tail=int(graph.edge_index[1,e])) for e in sorted(selected_edges)]),
        budget=dict(max_nodes=max_nodes,max_edges=max_edges,max_paths_per_candidate=max_paths,hops=hops),
        seconds=time.perf_counter()-began)
