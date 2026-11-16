"""Observation-only directed retrieval; damping applies exclusively to senders."""
from collections import Counter, defaultdict
from types import SimpleNamespace
import math
import numpy as np
import scipy.sparse as sp
from vendor.community_detection import detect_communities
from vendor.local_ppr import local_ppr

class EvidenceGraph:
    def __init__(self, z, config):
        self.n, self.R = int(z['n']), int(z['R'])
        self.half = self.R // 2
        self.config = config
        self.edge = np.asarray(z['edge_index']); self.rel = np.asarray(z['edge_type'])
        self.out = [[] for _ in range(self.n)]
        self.inc = [[] for _ in range(self.n)]
        self.profiles = [Counter() for _ in range(self.n)]
        self.tails = defaultdict(set)
        for e, (u, v) in enumerate(self.edge.T):
            u, v, r = int(u), int(v), int(self.rel[e])
            self.out[u].append(e); self.inc[v].append(e)
            self.profiles[u][r] += 1; self.tails[(u, r)].add(v)
        self.degree = np.array([len(x) for x in self.out])
        self.hub_threshold=max(2,float(np.quantile(self.degree,.95)))
        self.base_weights = np.array([(1+self.degree[u])**-config['eta_out'] *
            (1+self.profiles[u][int(r)])**-config['eta_rel'] for u, r in zip(self.edge[0], self.rel)])
        # Relation relevance is derived only from observed target endpoints.
        self.range_profile = [Counter() for _ in range(self.R)]
        reltails = defaultdict(set)
        for (u, r), ts in self.tails.items(): reltails[r].update(ts)
        for r, nodes in reltails.items():
            for v in nodes: self.range_profile[r].update(self.profiles[v])
        self.he = np.asarray(z['hyper_edge']); self.ht = np.asarray(z['hyper_type']); self.raw = np.asarray(z['raw'])
        self.motif_ids = [np.flatnonzero((self.he == r).any(0)) for r in range(self.R)]
        self._community = None
        self._transitions = {}

    def communities(self):
        if self._community is None:
            g = SimpleNamespace(edge_index=self.edge, n=self.n, out_degree=self.degree,
                in_degree=np.array([len(x) for x in self.inc]))
            self._community = detect_communities(g, self.base_weights)['labels']
        return self._community

    def weights(self, r):
        profile = self.range_profile[r]
        total = max(1, sum(profile.values()))
        relevance = np.array([.25 + .75 * profile.get(i, 0) / total for i in range(self.R)])
        relevance[r] = 1.
        return self.base_weights * relevance[self.rel]

    def transition(self, r):
        if r not in self._transitions:
            w = self.weights(r)
            a = sp.coo_matrix((w, self.edge), shape=(self.n, self.n)).tocsr()
            sums = np.asarray(a.sum(1)).ravel()
            self._transitions[r] = sp.diags(1/np.maximum(sums, 1e-30)) @ a
            if len(self._transitions) > 4: self._transitions.pop(next(iter(self._transitions)))
        return self._transitions[r]

    def signature(self, node, k=5):
        return sorted(self.profiles[node].items(), key=lambda x: (-x[1], x[0]))[:k]

    def retrieve(self, h, r, candidates, community=False):
        weights = self.weights(r)
        targets = set(candidates)
        found = {t: [] for t in candidates}
        selected = None; pprs = {}
        labels = self.communities() if community else None
        if community:
            transition = self.transition(r)
            hp = local_ppr(transition, h)
            selected = {h} | targets
            for t in candidates:
                tp = local_ppr(transition, t)
                combined = {u: hp.get(u, 0) + tp.get(u, 0) for u in hp.keys() | tp.keys()}
                nodes = sorted(combined, key=lambda u: (-combined[u], u))[:self.config['local_nodes']]
                selected.update(nodes); pprs[t] = combined
            # Preserve boundary bridge witnesses even when outside PPR top nodes.
            for t in candidates:
                for e in self.inc[t]: selected.add(int(self.edge[0, e]))
            for e in self.out[h]: selected.add(int(self.edge[1, e]))
        frontier = [(1., (h,), ())]
        beam = self.config.get('path_beam', 512)
        expanded = 0; pruned = False
        for depth in range(1, 4):
            next_front = []
            for weight, nodes, edges in frontier:
                u = nodes[-1]
                for e in self.out[u]:
                    v = int(self.edge[1, e])
                    if v in nodes: continue
                    if selected is not None and v not in selected: continue
                    new_weight = weight * float(weights[e]); new_edges = edges + (e,)
                    expanded += 1
                    if v in targets:
                        bridge = bool(labels is not None and any(labels[self.edge[0, x]] != labels[self.edge[1, x]] for x in new_edges))
                        hub=any(self.degree[x]>=self.hub_threshold for x in nodes[1:])
                        ppr_support=sum(pprs.get(v,{}).get(x,0.) for x in nodes[1:])
                        found[v].append(dict(edges=new_edges, weight=new_weight, nodes=nodes+(v,),
                            retrieval_score=new_weight*(1+ppr_support),
                            bridge=bridge, level='direct_composition' if depth <= 2 and not hub else 'indirect_long_or_hub_path'))
                    if depth < 3: next_front.append((new_weight, nodes+(v,), new_edges))
            next_front.sort(key=lambda x: (-x[0], x[2]))
            pruned |= len(next_front) > beam
            frontier = next_front[:beam]
        result = {}
        for t in candidates:
            paths = sorted(found[t], key=lambda p: (-p['retrieval_score'], p['edges']))
            unique = []; signatures = set()
            for p in paths:
                signature = tuple(int(self.rel[e]) for e in p['edges'])
                if signature in signatures: continue
                signatures.add(signature); unique.append(p)
                if len(unique) == 3: break
            # Keep a bridge witness if diversity selection otherwise discarded it.
            bridge_paths = [p for p in paths if p['bridge']]
            if bridge_paths and not any(p['bridge'] for p in unique):
                unique = unique[:2] + [bridge_paths[0]]
            actual = set(self.profiles[t]); expected = set(self.range_profile[r])
            result[t] = dict(paths=unique, no_evidence=not unique,
                signature=self.signature(t, 3), domain_range_warning=bool(expected and actual and not actual.intersection(expected)),
                type_unknown=not bool(actual), cross_community=bool(labels is not None and labels[h] != labels[t]),
                motifs=self.motifs(h, r, t, unique), search_pruned=pruned, expanded_edges=expanded)
        return result

    def motifs(self, h, r, t, paths):
        # Instantiate actual local directed motif witnesses (not mere label presence).
        nodes = {h, t}
        for p in paths: nodes.update(p['nodes'])
        local_edges = set()
        for u in nodes: local_edges.update(self.out[u]); local_edges.update(self.inc[u])
        out = defaultdict(set); inc = defaultdict(set)
        for e in local_edges:
            u, v = map(int, self.edge[:, e]); rr = int(self.rel[e])
            out[u].add(rr); inc[v].add(rr)
        matches = []
        local_relations=set(map(int,self.rel[list(local_edges)]))
        for i in self.motif_ids[r]:
            labels = list(map(int, self.he[:, i])); kind = int(self.ht[i])
            if not all(x<0 or x in local_relations for x in labels):continue
            a, b, c = labels
            if kind < 3:
                left, right = [(out, out), (inc, inc), (out, inc)][kind]
                active = any(a in left[u] and b in right[u] for u in nodes)
            else:
                left, right = [(inc, out), (inc, inc), (out, inc), (out, out)][kind-3]
                active = any(int(self.rel[e]) == b and a in left[int(self.edge[0,e])] and c in right[int(self.edge[1,e])] for e in local_edges)
            if active:
                matches.append(dict(kind=['HH','TT','HT','TFH','TFT','HFT','HFH'][kind],
                    relations=[x for x in labels if x >= 0], support=round(float(np.expm1(self.raw[i,0])), 3),
                    exclusivity=round(float(self.raw[i,3]), 6), source='observed_E3a_topology'))
        return sorted(matches, key=lambda x: (-x['exclusivity'], -x['support'], x['relations']))[:3]
