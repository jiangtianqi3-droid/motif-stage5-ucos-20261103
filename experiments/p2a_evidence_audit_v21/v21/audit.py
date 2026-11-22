"""Legacy paths/utilities, evidence hashes, canonical fact differences and fixed aggregators."""
from collections import defaultdict
from statistics import mean

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

from v21.graph import FactIndex
from v21.common import verify, identity, dump, write_csv
from p1.assets import Evaluator
from p1.scoring import evidence, filtered_after
from p2.features import all_paths, support
from p2.audit import local, inspect_selected
from query_community_v1.src.edge_weighter import CommunityEdgeWeighter

AGGREGATORS=('Max','Sum','LogSumExp','NormalizedLogSumExp')


def aggregate(util, name, tau):
    if not len(util):
        return 0.
    x=np.asarray(util,dtype=float)
    verify(np.isfinite(x).all(),'nonfinite original path utility')
    if name=='Max': return float(x.max())
    if name=='Sum': return float(x.sum())
    lse=float(tau*logsumexp(x/tau))
    return lse if name=='LogSumExp' else lse-float(tau*np.log(len(x)))


def order(candidates, scores):
    return [candidates[i] for i in np.argsort(-np.array([scores[c] for c in candidates]),kind='stable')]


def correlation(x,y):
    if len(set(x))<2 or len(set(y))<2:
        return None
    return float(spearmanr(x,y).statistic)


def category(edge_change,path_change,feature_change,score_change,order_change):
    if not edge_change: return 'T0'
    if not path_change: return 'T1'
    if not feature_change: return 'T2'
    if not score_change and not order_change: return 'T3'
    return 'T4'


def path_identity(index, g, path):
    return tuple((int(index.directed_to_id[i]),int(g.edge_type[i])) for i in path)


def collect(view, selection, index, cfg, directory, table):
    g=view.graph
    weights=CommunityEdgeWeighter(g).compute()
    offline=Evaluator(view.asset)
    trace=[]
    aggregation=[]
    signatures=[]
    selections={}
    fixed=[]
    per_candidate=[]
    tau=cfg['audit']['aggregation_temperature']
    for number,(i,d) in enumerate(selection):
        job,raw=view.job(i,d)
        qid=job['query_id']
        h,r,candidates=job['head'],job['relation'],job['candidates']
        target,excluded,before=offline.query(i,d,h,r,raw)
        fixed.append(dict(ordinal=number,query_id=qid,fact_index=i,direction=d,query_head=h,
                          query_relation=r,candidate_ids=candidates,top30=job['top30']))
        nodes,local_ids,fwd,back,priority=local(g,weights,h,candidates)
        verify(all(set(arm['ids'])<=set(local_ids) for arm in job['arms'].values()
                   if isinstance(arm,dict) and 'ids' in arm),'cached arm escaped local Gq')
        selections[qid]=dict(local_ids=local_ids,job=job)
        content={}
        prior=view.gate(r)
        for arm in cfg['retrieval_groups']:
            ids=job['arms'][arm]['ids']
            facts=index.canonical(ids)
            paths,legacy,full,features,profiles=inspect_selected(g,job,ids,priority,prior,table)
            verify({str(k):[list(path) for path in pp] for k,pp in paths.items()}==job['arms'][arm]['paths'],
                   f'legacy path mismatch {qid}/{arm}')
            verify(all(abs(legacy[c]['score']-job['arms'][arm]['values'][str(c)]['score'])<1e-12
                       for c in candidates),'legacy utility mismatch')
            utilities={c:[support(g,r,path,table) for path in paths.get(c,[])] for c in candidates}
            path_hash=identity({c:[path_identity(index,g,p) for p in paths.get(c,[])] for c in candidates})
            payload=dict(query_id=qid,method=arm,selected_ids=ids,canonical_fact_ids=facts,
                         nodes=sorted({int(v) for e in ids for v in g.edge_index[:,e]}),
                         relations=sorted(set(map(int,g.edge_type[ids]))),
                         paths={c:[list(p) for p in paths.get(c,[])] for c in candidates},utilities=utilities)
            # Replay from the same input twice; also prove different methods have independently owned collections.
            hash0=identity(payload)
            hash1=identity(dict(payload))
            again=inspect_selected(g,job,list(ids),priority,prior,table)
            verify(identity(again[1])==identity(legacy) and hash0==hash1,'repeated scoring input changed')
            signatures.append(dict(query_id=qid,method=arm,input_hash=hash0,path_hash=path_hash,repeat_equal=True,
                 observed_records=len(ids),unique_facts=len(facts),inverse_records=int(sum(g.edge_type[e]>=g.R//2 for e in ids)),
                 full_message_records=len(index.closure(facts)),nodes=payload['nodes'],relations=payload['relations'],
                 path_set=payload['paths'],scoring_input=payload))
            content[arm]=dict(ids=ids,facts=facts,paths=paths,full=full,features=features,utilities=utilities,
                              legacy=legacy,path_hash=path_hash,payload=payload)
            if arm not in ('R2','R3'):
                continue
            agg_scores={}
            for name in AGGREGATORS:
                values={c:aggregate(utilities[c],name,tau) for c in candidates}
                sorted_c=order(candidates,values)
                rank=filtered_after(raw,candidates,sorted_c,target,excluded,offline.rank)
                top=next((c for c in sorted_c if c not in excluded or c==target),sorted_c[0])
                aggregation.append(dict(query_id=qid,fact_index=i,direction=d,method=arm,aggregator=name,
                     path_input_hash=identity([paths,utilities]),candidate_order=sorted_c,rank=rank,rr=1/rank,
                     hits1=rank<=1,hits10=rank<=10,top1=sorted_c[0],baseline_rank=before,
                     top1_changed=sorted_c[0]!=candidates[0],correct_in_top12=target in candidates,
                     path_count=sum(map(len,utilities.values())),candidate_scores=values))
                for c in candidates:
                    per_candidate.append(dict(query_id=qid,method=arm,aggregator=name,candidate_id=c,
                       paths=len(utilities[c]),score=values[c],outdegree=int(g.out_degree[c]),indegree=int(g.in_degree[c])))
                agg_scores[name]=sorted_c
            content[arm]['agg_order']=agg_scores
        for a,b in (('R2','R3'),('R3','R4')):
            x,y=content[a],content[b]
            added=set(y['facts'])-set(x['facts'])
            removed=set(x['facts'])-set(y['facts'])
            delta=set(x['facts'])^set(y['facts'])
            full_old={path_identity(index,g,p) for pp in x['full'].values() for p in pp}
            full_new={path_identity(index,g,p) for pp in y['full'].values() for p in pp}
            path_added=full_new-full_old
            path_used={int(index.directed_to_id[e]) for pp in y['full'].values() for p in pp for e in p}
            participating=added & path_used
            old_templates={tuple(map(int,g.edge_type[list(p)])) for pp in x['full'].values() for p in pp}
            new_templates={tuple(map(int,g.edge_type[list(p)])) for pp in y['full'].values() for p in pp}
            local_paths=all_paths(g,h,candidates,local_ids)
            local_used={int(index.directed_to_id[e]) for pp in local_paths.values() for p in pp for e in p}
            feature_delta=max(float(np.linalg.norm(y['features'][c]-x['features'][c])) for c in candidates)
            score_delta=max(abs(y['legacy'][c]['score']-x['legacy'][c]['score']) for c in candidates)
            orders=[order(candidates,{c:v['score'] for c,v in u['legacy'].items()}) for u in (x,y)]
            ranks=[filtered_after(raw,candidates,o,target,excluded,offline.rank) for o in orders]
            tag=category(bool(delta),full_old!=full_new,feature_delta>1e-12,score_delta>1e-12,orders[0]!=orders[1])
            participants=sum(any(added & {int(index.directed_to_id[e]) for p in y['full'].get(c,[]) for e in p})
                             for c in candidates)
            trace.append(dict(query_id=qid,fact_index=i,direction=d,method_a=a,method_b=b,
                  category=tag,unique_facts_before=len(x['facts']),unique_facts_after=len(y['facts']),
                  directed_records_before=len(x['ids']),directed_records_after=len(y['ids']),
                  added_facts=len(added),removed_facts=len(removed),added_fact_ids=sorted(added),
                  candidate_path_added_facts=len(participating),peripheral_added_facts=len(added)-len(participating),
                  path_participation=len(participating)/len(added) if added else None,
                  potential_local_path_participation=len(added&local_used)/len(added) if added else None,
                  participating_candidates=participants,new_paths=len(path_added),new_relation_templates=len(new_templates-old_templates),
                  new_entity_motif_instances=None,feature_delta=feature_delta,score_delta=score_delta,
                  input_hash_before=identity(x['payload']),input_hash_after=identity(y['payload']),
                  top1_changed=orders[0][0]!=orders[1][0],target_rank_before=ranks[0],target_rank_after=ranks[1],
                  mrr_changed=ranks[0]!=ranks[1],target_path_added=bool(added & {int(index.directed_to_id[e])
                       for p in y['full'].get(target,[]) for e in p}),order_changed=orders[0]!=orders[1]))
    write_csv(directory/'propagation/fixed_queries.csv',fixed)
    write_csv(directory/'propagation/t0_t4_audit.csv',trace)
    write_csv(directory/'propagation/input_consistency.csv',signatures)
    write_csv(directory/'aggregation/aggregation_per_query.csv',aggregation)
    write_csv(directory/'aggregation/candidate_values.csv',per_candidate)
    dump(directory/'propagation/scoring_inputs.json',signatures)
    summary={}
    for a,b in (('R2','R3'),('R3','R4')):
        rows=[r for r in trace if r['method_a']==a]
        summary[f'{a}_{b}']=dict(queries=len(rows),types={f'T{k}':sum(r['category']==f'T{k}' for r in rows) for k in range(5)},
             type_rates={f'T{k}':mean(r['category']==f'T{k}' for r in rows) for k in range(5)},
             mean_added_facts=mean(r['added_facts'] for r in rows),
             no_added_facts=sum(r['added_facts']==0 for r in rows),
             mean_path_participation=mean(r['path_participation'] for r in rows if r['path_participation'] is not None)
               if any(r['path_participation'] is not None for r in rows) else None,
             added_on_paths=sum(r['candidate_path_added_facts'] for r in rows),
             added_total=sum(r['added_facts'] for r in rows),
             mean_feature_delta=mean(r['feature_delta'] for r in rows),mean_score_delta=mean(r['score_delta'] for r in rows))
    dump(directory/'propagation/summary.json',summary)
    metrics=[]
    for arm in ('R2','R3'):
        for name in AGGREGATORS:
            rows=[r for r in aggregation if r['method']==arm and r['aggregator']==name]
            other={r['query_id']:r for r in aggregation if r['method']=='R2' and r['aggregator']==name}
            candidates_data=[r for r in per_candidate if r['method']==arm and r['aggregator']==name]
            correlations={}
            for field in ('paths','outdegree','indegree'):
                correlations[field]=correlation([r[field] for r in candidates_data],[r['score'] for r in candidates_data])
            metrics.append(dict(method=arm,aggregator=name,queries=len(rows),MRR=mean(r['rr'] for r in rows),
                 Hits1=mean(r['hits1'] for r in rows),Hits10=mean(r['hits10'] for r in rows),
                 mean_rank=mean(r['rank'] for r in rows),top1_change_rate=mean(r['top1_changed'] for r in rows),
                 r2_r3_order_agreement=mean(r['candidate_order']==other[r['query_id']]['candidate_order'] for r in rows),
                 **{f'spearman_{k}':v for k,v in correlations.items()}))
    write_csv(directory/'aggregation/aggregation_metrics.csv',metrics)
    return selections,trace,metrics
