"""Bounded CPU-only workers; isolated relation jobs and compressed evidence."""
from collections import OrderedDict
import copy
import time
import numpy as np
from .util import historical_package
historical_package()
from .edge_weighter import UnifiedEdgeWeighter,weights_for
from qc_v1.path_extractor import retrieve

def initialize(graph,signature,labels,key,work):
    global G,SIG,LABELS,KEY,WORK,FACTORY
    G,SIG,LABELS,KEY,WORK=graph,signature,labels,key,work
    FACTORY=UnifiedEdgeWeighter(graph,signature)

def relation_job(payload):
    from types import SimpleNamespace
    from .pipeline import ev_path,save_evidence,legacy_record
    from .util import read,peak_memory,RSS_LIMIT
    q,rows=payload;completed=0;reused=0;seconds=0.;audits=[]
    dataset=SimpleNamespace(key=KEY,g=G)
    for method in ['R1','R2','R3','R4']:
        weights=weights_for(FACTORY,method,q);cache=OrderedDict()
        if method=='R4':
            old=FACTORY.compute('topology_aware',q);delta=weights.transition-old.transition
            u=G.edge_index[0];without=weights.weights/(1+G.out_degree[u])**(-.25)
            mass=np.bincount(u,weights=without,minlength=G.n)
            error=float(np.max(np.abs(without/mass[u]-weights.edge_probability),initial=0))
            if error>=1e-12:raise AssertionError('Row-constant cancellation invariant')
            audits.append(dict(relation=int(q),edge_weight_mean=float(weights.weights.mean()),max_forward_difference_vs_E3a=float(np.max(np.abs(delta.data),initial=0)),sender_cancellation_error=error,row_stochastic_assertions_passed=True))
            del old,delta,without,mass
        for split,i,d in rows:
            path=ev_path(WORK,method,split,i,d)
            if not path.exists():
                began=time.perf_counter();c=read(WORK/'candidates'/split/f'{i}_{d}.json')
                identity=(c['query_info']['head'],tuple(v['entity'] for v in c['candidates']))
                record=cache.get(identity)
                if record is not None:
                    record=copy.deepcopy(record);cache.move_to_end(identity);reused+=1
                else:
                    record=legacy_record(dataset,WORK,method,split,i,d,c)
                    if record is not None:reused+=1
                    else:record=retrieve(G,weights,LABELS,c['query_info']['head'],q,c['candidates'],method,SIG,max_paths=10)
                    cache[identity]=copy.deepcopy(record)
                    if len(cache)>32:cache.popitem(last=False)
                for ev,candidate in zip(record['candidate_evidence'],c['candidates']):ev.update(candidate)
                record['query_info'].update(c['query_info'])
                provenance=record.get('provenance',{})
                record['provenance']=dict(provenance,ground_truth_used=False,candidate_filter='observation+valid only at test',model=c['model'])
                save_evidence(path,record);seconds+=time.perf_counter()-began
            completed+=1
        del weights,cache
    peak=peak_memory()
    if peak>RSS_LIMIT:raise MemoryError(f'Retrieval worker peak {peak} exceeds {RSS_LIMIT}')
    return dict(records=completed,reused=reused,seconds=seconds,audits=audits,peak_rss_bytes=peak)
