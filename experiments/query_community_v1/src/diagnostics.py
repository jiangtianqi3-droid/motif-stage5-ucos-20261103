"""Offline evaluator: the only module allowed to inspect evaluation answers."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from .graph_adapter import HISTORY,runtime_path,dump

def load_offline_truth(key,graph):
    old=HISTORY/'topology-semantic-stage4-20261102/semantic'
    provenance=json.loads((old/'target_mapping_provenance.json').read_text(encoding='utf-8-sig'))[key]
    vocab=json.loads((old/'relation_mappings.json').read_text(encoding='utf-8-sig'))[key]
    rels={v['original_name']:v['id'] for v in vocab[:graph.R//2]}
    raw=runtime_path(provenance['raw']);entities={}
    def parse(name):
        triples=[]
        for line in (raw/name).read_text(encoding='utf-8').splitlines():
            h,r,t=line.split()
            for token in (h,t):
                if token not in entities: entities[token]=len(entities)
            triples.append([entities[h],rels[r],entities[t]])
        return np.asarray(triples,dtype=np.int64).reshape(-1,3)
    facts=parse(provenance['fact_file'])
    assert np.array_equal(facts[:,[0,2]].T,graph.edge_index[:,:len(facts)])
    assert np.array_equal(facts[:,1],graph.edge_type[:len(facts)])
    names=['valid_ind.txt','test_ind.txt'] if provenance['kind']=='grail' else ['inf_valid.txt','inf_test.txt']
    valid,test=[parse(name) for name in names]
    graph.assert_hidden_absent(np.r_[valid,test])
    # Check exact historical target order (Grail merges inductive valid/test).
    with np.load(HISTORY/'motif-stage6-entity-degree-p0p1-v2/data'/f'{key}.npz') as z:
        queries=z['queries'].copy()
    assert np.array_equal(queries,np.r_[valid,test] if provenance['kind']=='grail' else test)
    return queries,dict(valid_hidden=len(valid),test_hidden=len(test),direct_overlap=0,inverse_overlap=0)

def evaluate_record(record,answer,labels,hubs):
    candidates=record['candidate_evidence']
    truth=next((c for c in candidates if c['entity']==int(answer)),None)
    hit=bool(truth and truth['status']=='ok')
    cross=bool(labels[record['query_info']['head']]!=labels[answer])
    # Offline ground-truth exclusion happens here, after evidence has been fixed.
    negatives=[c['max_path_weight'] for c in candidates if c['base_rank']<=10 and c['entity']!=int(answer)]
    expansion=bool(truth and 11<=truth['base_rank']<=30)
    outranks=bool(expansion and truth['status']=='ok' and negatives and truth['max_path_weight']>max(negatives))
    top=[c for c in candidates if c['base_rank']<=10]
    values=np.array([c['max_path_weight'] for c in top])
    truth_at=[i for i,c in enumerate(top) if c['entity']==int(answer)]
    structure_rank=None
    if truth_at:
        i=truth_at[0]
        structure_rank=float(1+np.sum(values>values[i])+.5*(np.sum(values==values[i])-1))
    corr=None
    if len(values)>1 and np.ptp(values)>0:
        corr=float(np.corrcoef(rankdata(-values),np.arange(len(values)))[0,1])
    nodes=record['subgraph']['nodes']
    return dict(query_id=record['query_info']['query_id'],fact_index=record['query_info']['fact_index'],
        direction=record['query_info']['direction'],nodes=len(nodes),edges=len(record['subgraph']['edge_ids']),
        hub_fraction=float(sum(v in hubs for v in nodes)/len(nodes)) if nodes else 0.,
        hit=int(hit),cross=int(cross),cross_hit=int(cross and hit),candidate_hit=int(truth is not None),
        expansion_eligible=int(expansion),expansion_success=int(outranks),
        evidence_rr=1/structure_rank if structure_rank else 0.,top10_answer_present=int(bool(truth_at)),
        score_spearman=corr,seconds=record['seconds'])

def aggregate(rows):
    n=len(rows)
    def avg(k): return float(np.mean([r[k] for r in rows])) if n else None
    def ratio(a,b):
        den=sum(r[b] for r in rows)
        return sum(r[a] for r in rows)/den if den else None
    corr=[r['score_spearman'] for r in rows if r['score_spearman'] is not None]
    return dict(queries=n,mean_nodes=avg('nodes'),mean_edges=avg('edges'),hub_fraction=avg('hub_fraction'),
        path_hit_rate=avg('hit'),candidate_recall=avg('candidate_hit'),
        cross_queries=sum(r['cross'] for r in rows),cross_answer_recall=ratio('cross_hit','cross'),
        expansion_eligible=sum(r['expansion_eligible'] for r in rows),expansion_success=sum(r['expansion_success'] for r in rows),
        expansion_potential=ratio('expansion_success','expansion_eligible'),
        top10_evidence_mrr_conditional=ratio('evidence_rr','top10_answer_present'),
        base_structure_spearman=float(np.mean(corr)) if corr else None,
        mean_seconds=avg('seconds'))

def paired_interval(rows_a,rows_b,key,seed=1024,replicates=2000):
    a={r['query_id']:r for r in rows_a};b={r['query_id']:r for r in rows_b}
    assert a.keys()==b.keys()
    # Cluster both query directions by original fact, to avoid false independence.
    facts={}
    for q in sorted(a):
        facts.setdefault(a[q]['fact_index'],[]).append(b[q][key]-a[q][key])
    delta=np.array([np.mean(v) for v in facts.values()])
    rng=np.random.default_rng(seed)
    draws=rng.choice(delta,size=(replicates,len(delta)),replace=True).mean(1)
    return dict(delta=float(delta.mean()),lo=float(np.quantile(draws,.025)),hi=float(np.quantile(draws,.975)),
                clusters=len(delta),replicates=replicates,seed=seed)
