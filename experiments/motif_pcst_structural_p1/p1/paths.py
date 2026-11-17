"""Directed path instances and source-only reliability estimates."""
from collections import Counter, defaultdict
from math import log2

import numpy as np

from src.graph_loader import ObservedGraph
from src.local_retrieval import directed_paths


def signature(g, q, path, degree=True):
    # Within-graph equality or inversion only. No relation numbers cross graphs.
    inv = g.inverse(q)
    roles = tuple('q' if r==q else 'i' if r==inv else 'f' if r<g.R//2 else 'b'
                  for r in g.edge_type[list(path)])
    # Use a portable degree bucket for the highest intermediate branching factor.
    hubs = [int(g.out_degree[g.edge_index[1,i]]) for i in path[:-1]]
    hub = min(5,int(log2(max(hubs,default=1))))
    return ''.join(roles)+('/'+str(hub) if degree else '')


def masked_graph(graph, facts):
    removed = set(facts) | {(t,graph.inverse(r),h) for h,r,t in facts}
    edges = graph.edge_index
    keep = np.array([(int(u),int(r),int(v)) not in removed
                     for u,r,v in zip(edges[0],graph.edge_type,edges[1])])
    output = ObservedGraph(edges[:,keep],graph.edge_type[keep],graph.n,graph.R)
    output.assert_inverse_pairs()
    output.assert_hidden_absent(facts)
    return output


def path_choices(g, head, candidates, allowed, priority, limit=4):
    """Up to four distinct relation-type paths per target and at most three hops."""
    adjacency = defaultdict(list)
    for i in sorted(allowed):
        adjacency[int(g.edge_index[0,i])].append(i)
    for u in adjacency:
        adjacency[u].sort(key=lambda i: (-priority(i),i))
    targets = set(candidates)-{head}
    results = defaultdict(dict)
    frontier = [(head,0.,(),frozenset({head}))]
    for hop in range(3):
        next_front = defaultdict(list)
        for u,score,path,visited in frontier:
            for i in adjacency[u]:
                v = int(g.edge_index[1,i])
                if v in visited:
                    continue
                p = path+(i,)
                weight = score+priority(i)
                if v in targets:
                    key = tuple(map(int,g.edge_type[list(p)]))
                    old = results[v].get(key)
                    metric = (len(p),-weight,p)
                    if old is None or metric<old:
                        results[v][key] = metric
                if hop<2:
                    next_front[v].append((v,weight,p,visited|{v}))
        frontier = [x for records in next_front.values()
                    for x in sorted(records,key=lambda y:(-y[1],y[2]))[:4]]
    result = {c: [x[2] for x in sorted(records.values())[:limit]] for c,records in results.items()}
    # The shortest-path oracle guarantees that a beam cutoff cannot erase reachability.
    exact = directed_paths(g,head,candidates,priority,set(allowed))
    for c,p in exact.items():
        if c not in result:
            result[c]=[p]
    return result


def counts_from_rows(rows):
    wins, total = Counter(),Counter()
    for row in rows:
        if not row['has_path']:
            continue
        for key in {row['signature'],row['coarse']}:
            wins[key] += int(row['observed_positive'])
            total[key] += 1
    return {key:[wins[key],n] for key,n in total.items()}


def reliability(table,g,q,path):
    fine, coarse = signature(g,q,path),signature(g,q,path,False)
    yes,n = table.get(coarse,(0,0))
    backoff = (yes+1)/(n+2)
    yes,n = table.get(fine,(0,0))
    return (yes+4*backoff)/(n+4)


def source_probes(graph, seed, nfit, ntest):
    rng = np.random.default_rng(seed)
    e,r = graph.edge_index,graph.edge_type
    direct = sorted({(int(u),int(q),int(v)) for u,q,v in zip(e[0],r,e[1]) if q<graph.R//2 and u!=v})
    chosen = [direct[i] for i in rng.choice(len(direct),nfit+ntest,replace=False)]
    g = masked_graph(graph,chosen)
    known = defaultdict(set)
    for u,q,v in zip(e[0],r,e[1]):
        known[u,q].add(v)
    rows = []
    for j,(u,q,v) in enumerate(chosen):
        for h,rel,t in ((u,q,v),(v,g.inverse(q),u)):
            shell = {h}
            for _ in range(3):
                shell |= {int(g.edge_index[1,i]) for a in shell for i in g.edges_at(a)}
            negative_pool = sorted(shell-known[h,rel]-{h,t})
            controls = [negative_pool[i] for i in rng.choice(len(negative_pool),min(3,len(negative_pool)),replace=False)]
            paths = directed_paths(g,h,[t,*controls],lambda _:0.)
            for c in [t,*controls]:
                p = paths.get(c)
                rows.append(dict(fold='fit' if j<nfit else 'probe', sample=j, head=h, relation=rel, c=c,
                                 has_path=bool(p), observed_positive=(c==t), signature=signature(g,rel,p) if p else '',
                                 coarse=signature(g,rel,p,False) if p else '',
                                 inverse_pair_masked=True, path_ids=list(p or ())))
    return g,chosen,rows
