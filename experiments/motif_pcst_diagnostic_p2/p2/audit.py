"""Replay selected-graph paths, compare adjacent P1 arms and probe causal inputs."""
from collections import Counter, defaultdict
from statistics import mean
from time import perf_counter
import argparse

import numpy as np

from p2.assets import RESULTS, GRAPHS, PAIR_GRAPHS, Target, config, preflight, verify
from p2.features import all_paths, path_facts, profile, support
from p1.assets import Evaluator
from p1.paths import path_choices
from p1.retrieval import choose_pcst, true_fact, COST
from p1.scoring import evidence, permutations, filtered_after
from src.local_retrieval import coarse
from src.graph_loader import ObservedGraph
from src.io_utils import dump, read, write_csv, identity
from query_community_v1.src.edge_weighter import CommunityEdgeWeighter

PAIRS=(('R2','R3'),('R3','R4'),('R4','R5'))
EPS=1e-12


def selected_paths(graph, head, candidates, ids, priority):
    paths=path_choices(graph,head,candidates,ids,priority)
    verify(all(set(path).issubset(ids) for pp in paths.values() for path in pp),'P1 path escaped selected graph')
    return paths


def local(graph, weights, head, candidates):
    nodes,ids,fwd,back,_=coarse(graph,weights,head,candidates)
    ids=list({}.fromkeys(ids))
    seen={}
    for i in ids:
        u,v=map(int,graph.edge_index[:,i])
        seen.setdefault((u,int(graph.edge_type[i]),v),i)
    ids=list(seen.values())
    def priority(i):
        u,v=graph.edge_index[:,i]
        return fwd.get(u,0.)*weights.edge_probability[i]+.35*max(back[u],back[v])
    return nodes,ids,fwd,back,priority


def inspect_selected(graph,job,ids,priority,prior,table,unclipped=True):
    h,r,c=job['head'],job['relation'],job['candidates']
    paths=selected_paths(graph,h,c,ids,priority)
    legacy=evidence(graph,r,c,{'paths':paths},table)
    paths_all=all_paths(graph,h,c,ids) if unclipped else paths
    features={}
    profiles={}
    active=[v['score'] for v in legacy.values() if v['valid']]
    avg=mean(active) if active else 0.
    for x in c:
        pp=paths.get(x,[])
        scores=sorted((support(graph,r,p,table) for p in pp),reverse=True)
        top4=np.pad(scores[:4],(0,max(0,4-len(scores))))
        prof=profile(graph,r,paths_all.get(x,[]),prior,table)
        profiles[x]=prof
        hubs=[int(graph.in_degree[graph.edge_index[1,i]]) for p in pp for i in p[:-1]]
        motif=mean([p[1] for p in prof]) if prof else 0.
        features[x]=np.r_[len(pp),len(prof),len(paths_all.get(x,[])),top4,motif,
                           legacy[x]['score']-avg if pp else 0.,mean(hubs) if hubs else 0.,not pp]
    return paths,legacy,paths_all,features,profiles


def category(edges,feature,score,order):
    if not edges: return 'T0'
    if order: return 'T4'
    if not feature: return 'T1'
    if not score: return 'T2'
    return 'T3'


def edge_diff(a,b):
    a,b=set(a),set(b)
    return sorted(b-a),sorted(a-b),len(a&b)/max(1,len(a|b)) if a or b else 1.


def changes(a,b,c):
    return [abs(b[x]['score']-a[x]['score']) for x in c]


def replay(view, job, raw, table, weights, lambda_p1):
    g=view.graph
    h,r,c=job['head'],job['relation'],job['candidates']
    nodes,local_ids,fwd,back,p=local(g,weights,h,c)
    prior=view.gate(r)
    out={}
    for arm in ('R2','R3','R4','R5'):
        original=job['arms'][arm]
        verify(len(set(original['ids']))==len(original['ids']) and set(original['ids']).issubset(local_ids),
               'P1 evidence escaped local graph')
        paths, legacy, allp, feats, prof=inspect_selected(g,job,original['ids'],p,prior,table)
        verify({str(k):[list(x) for x in pp] for k,pp in paths.items()}==original['paths'],
               f"selected-path replay failed: {job['query_id']}/{arm}")
        verify(all(abs(v['score']-original['values'][str(k)]['score'])<EPS and
                   v['valid']==original['values'][str(k)]['valid'] for k,v in legacy.items()), 'legacy evidence score differs')
        orders=permutations(raw,c,legacy,lambda_p1,.23)
        verify(all(orders[k]==original['orders'][k] for k in orders),'legacy permutation replay mismatch')
        out[arm]=dict(paths=paths,legacy=legacy,all=allp,features=feats,profiles=prof,orders=orders,ids=original['ids'])
    return out, (nodes,local_ids,fwd,back,p)


def probe(view,job,raw,base,context,weights,table,seed):
    g=view.graph
    h,r,c=job['head'],job['relation'],job['candidates']
    ids=base['ids']
    if not ids:
        return []
    nodes,local_ids,fwd,back,p=context
    prior=view.gate(r)
    rng=np.random.default_rng(seed)
    motif=lambda i:float(prior[g.edge_type[i]])
    highest=max(ids,key=lambda i:(motif(i),-i))
    all_used={i for pp in base['paths'].values() for path in pp for i in path}
    key_paths=[p for pp in base['paths'].values() for p in pp]
    head_key=max(key_paths,key=lambda path:(support(g,r,path,table),path),default=())
    critical=next(iter(head_key),None)
    non_key=sorted(set(ids)-all_used)
    replacements=sorted(set(local_ids)-set(ids),key=lambda i:(motif(i),p(i),i))
    scenarios=[('delete_max_motif',[i for i in ids if i!=highest],highest,None)]
    if critical is not None:
        scenarios.append(('delete_best_path_edge',[i for i in ids if i!=critical],critical,None))
    if non_key:
        n=int(rng.choice(non_key))
        scenarios.append(('delete_random_nonpath',[i for i in ids if i!=n],n,None))
        if replacements:
            scenarios.append(('replace_low_relevance',[i for i in ids if i!=n]+[replacements[0]],n,replacements[0]))
    costs={i:COST for i in local_ids}
    focus={h,*c}
    prizes={v:.10+min(1.,2.5*fwd.get(v,0.)+.6*back[v])+.45*(v in focus) for v in nodes}
    oracle=path_choices(g,h,c,local_ids,p)
    # Reset PCST with the unperturbed edge rewards first to verify the intervention baseline.
    bonuses={i:.52*motif(i) for i in local_ids}
    replay_ids,_=choose_pcst(g,nodes,local_ids,h,c,prizes,bonuses,costs,oracle,32,True,False)
    verify(replay_ids==ids,'R3 solver replay differs')
    shuffle=dict(zip(local_ids,map(float,rng.permutation(list(bonuses.values())))))
    shuffled,_=choose_pcst(g,nodes,local_ids,h,c,prizes,shuffle,costs,oracle,32,True,False)
    scenarios.append(('shuffle_rewards_reselect',shuffled,None,None))
    result=[]
    def record(scope,name,graph,chosen,priority,delete,add,original_ids=None):
        _,rev,_,_,_=inspect_selected(graph,job,chosen,priority,prior,table,False)
        delta=changes(base['legacy'],rev,c)
        order=permutations(raw,c,rev,0.,.23)['evidence_only']
        result.append(dict(dataset=view.key,query_id=job['query_id'],scope=scope,intervention=name,
                           deleted_edge=delete,added_edge=add,original_edge_ids=ids,
                           selected_edge_ids=chosen if original_ids is None else original_ids,
                           candidate_scores_changed=sum(x>EPS for x in delta),
                           max_score_delta=max(delta),ranking_changed=order!=base['orders']['evidence_only']))
    for name,chosen,deletion,addition in scenarios:
        record('selected_graph_counterfactual',name,g,chosen,p,deletion,addition)
    # Remove the original fact and its canonical inverse/duplicates in the observation graph.
    # Rebuild degree and PPR; retain the frozen global B2 prior to isolate the graph change.
    truth=true_fact(g,highest)
    u,q,v=truth
    excluded={(u,q,v),(v,g.inverse(q),u)}
    edge=g.edge_index
    keep=[i for i,(a,b,r0) in enumerate(zip(edge[0],edge[1],g.edge_type)) if (a,r0,b) not in excluded]
    gmask=ObservedGraph(edge[:,keep],g.edge_type[keep],g.n,g.R)
    gmask.assert_inverse_pairs()
    neww=CommunityEdgeWeighter(gmask).compute()
    mn,mid,mf,mb,mp=local(gmask,neww,h,c)
    mo=path_choices(gmask,h,c,mid,mp)
    mm={i:.52*float(prior[gmask.edge_type[i]]) for i in mid}
    mc={i:COST for i in mid}
    mpr={v:.10+min(1.,2.5*mf.get(v,0.)+.6*mb[v])+.45*(v in focus) for v in mn}
    new_ids,_=choose_pcst(gmask,mn,mid,h,c,mpr,mm,mc,mo,32,True,False)
    record('original_graph_counterfactual','delete_fact_and_inverse_reselect',gmask,new_ids,mp,highest,None,
           [keep[i] for i in new_ids])
    return result


def run(folder,root,cfg,mode,status=lambda *args,**kwargs:None):
    summary_path=folder/'propagation_summary.json'
    if summary_path.is_file() and read(summary_path)['mode']==mode:
        return read(summary_path)
    out,probes,summary=[],[],[]
    t0=perf_counter()
    table_obj=read(root/'source_calibration.json')
    table=table_obj['table']
    cases={}
    outside_rank_changes=0
    graphs=PAIR_GRAPHS if mode=='smoke' else GRAPHS
    for key in graphs:
        view=Target(key,root,folder)
        g=view.graph
        w=CommunityEdgeWeighter(g).compute()
        indices=view.indices[:cfg['audit_queries_per_primary']//2] if mode=='smoke' else view.indices
        target_rows=[]
        # Replay paths and scores before accessing the offline target evaluator.
        for i in indices:
            for d in (0,1):
                job,raw=view.job(i,d)
                arms,context=replay(view,job,raw,table,w,table_obj['lambda_selected'])
                target_rows.append((i,d,job,arms))
                if key in PAIR_GRAPHS and len(target_rows)<=cfg['audit_queries_per_primary']:
                    # Sparse original-graph probes, paired with selected-graph interventions.
                    if len(target_rows)%7==1:
                        probes.extend(probe(view,job,raw,arms['R3'],context,w,table,cfg['seed']+i*2+d))
                if len(target_rows)%100==0:
                    status(f'audit/{key}',len(target_rows),2*len(indices))
        evaluator=Evaluator(view.asset)
        measures={p:[] for p in PAIRS}
        for i,d,job,arms in target_rows:
            raw=view.raw[i][d]
            c=job['candidates']
            target,excluded,_=evaluator.query(i,d,job['head'],job['relation'],raw)
            for a,b in PAIRS:
                aa,bb=arms[a],arms[b]
                added,removed,jaccard=edge_diff(aa['ids'],bb['ids'])
                n=len(added)+len(removed)
                ds=changes(aa['legacy'],bb['legacy'],c)
                fa=[float(np.linalg.norm(bb['features'][x]-aa['features'][x])) for x in c]
                order=aa['orders']['evidence_only']!=bb['orders']['evidence_only']
                fused=aa['orders']['fusion']!=bb['orders']['fusion']
                tag=category(n,max(fa)>EPS,max(ds)>EPS,order)
                ra,rb=[filtered_after(raw,c,u['orders']['evidence_only'],target,excluded,evaluator.rank) for u in (aa,bb)]
                rfa,rfb=[filtered_after(raw,c,u['orders']['fusion'],target,excluded,evaluator.rank) for u in (aa,bb)]
                outside_rank_changes+=int(rb!=ra and target not in c)
                touched=False
                pre_path,post_path=False,False
                for x,score_change,feature_delta in zip(c,ds,fa):
                    pa,pb=aa['paths'].get(x,[]),bb['paths'].get(x,[])
                    ua,ub=aa['all'].get(x,[]),bb['all'].get(x,[])
                    path_added={j for p in ub for j in p}&set(added)
                    path_used={j for p in pb for j in p}&set(added)
                    pre_path|=bool(path_added)
                    post_path|=bool(path_used)
                    changed_path_edges=len({j for p in pa for j in p}^{j for p in pb for j in p})
                    touched|=bool(changed_path_edges)
                    sa,sb=aa['legacy'][x]['score'],bb['legacy'][x]['score']
                    edge_records=dict(edges_added=added,edges_removed=removed,edge_jaccard=jaccard)
                    row=dict(dataset=key,query_id=job['query_id'],candidate_id=x,method_a=a,method_b=b,
                             selected_edges_changed=n,candidate_path_edges_changed=changed_path_edges,
                             feature_delta=feature_delta,score_delta=score_change,normalized_score_delta=score_change,
                             score_before=sa,score_after=sb,trace_category=tag,**edge_records,
                             rank_delta=aa['orders']['evidence_only'].index(x)-bb['orders']['evidence_only'].index(x),
                             raw_candidate_rank=c.index(x)+1,evidence_rank_before=aa['orders']['evidence_only'].index(x)+1,
                             evidence_rank_after=bb['orders']['evidence_only'].index(x)+1,
                             fused_rank_before=aa['orders']['fusion'].index(x)+1,fused_rank_after=bb['orders']['fusion'].index(x)+1,
                             full_entity_rank_before=ra if x==target else None,full_entity_rank_after=rb if x==target else None,
                             full_fused_target_rank_before=rfa,full_fused_target_rank_after=rfb,
                             target_in_candidates=target in c,features_before=aa['features'][x].tolist(),
                             features_after=bb['features'][x].tolist(),paths_before=pa,paths_after=pb,
                             new_path_edges_all=sorted(path_added),new_path_edges_retained=sorted(path_used),
                             p1_score_is_tied=sum(abs(v['score']-sb)<EPS for v in bb['legacy'].values())>1)
                    out.append(row)
                    for case_name,condition in [(tag, n>0),('no_path',not pb),
                                 ('reachable_low_support',bool(pb) and sb<-.10),('positive',x==target and rb<ra),
                                 ('negative',x==target and rb>ra)]:
                        case_key=f'{key}/{a}-{b}/{case_name}'
                        if condition and case_key not in cases:
                            cases[case_key]=dict(row,facts=path_facts(g,set(aa['ids'])|set(bb['ids'])),
                                                 motif_by_edge={j:float(view.gate(job['relation'])[g.edge_type[j]])
                                                                for j in set(aa['ids'])|set(bb['ids'])})
                measures[a,b].append(dict(edges=n,feat=max(fa),score=max(ds),rank=order,fused=fused,t=tag,
                      positive=rb<ra,negative=rb>ra,touched=touched,new_any_path=pre_path,
                      new_retained_path=post_path,target_outside=target not in c,
                      target_rank_before=ra,target_rank_after=rb))
        for (a,b),rows in measures.items():
            counts=Counter(x['t'] for x in rows)
            avg=lambda attr:mean(x[attr] for x in rows)
            summary.append(dict(dataset=key,method_a=a,method_b=b,queries=len(rows),
              types={f'T{k}':counts[f'T{k}'] for k in range(5)},type_rates={f'T{k}':counts[f'T{k}']/len(rows) for k in range(5)},
              edge_change_rate=mean(x['edges']>0 for x in rows),feature_change_rate=mean(x['feat']>EPS for x in rows),
              score_change_rate=mean(x['score']>EPS for x in rows),ranking_change_rate=avg('rank'),fused_change_rate=avg('fused'),
              mean_edge_symmetric_difference=avg('edges'),mean_feature_delta=avg('feat'),mean_score_delta=avg('score'),
              new_any_path_rate=avg('new_any_path'),new_retained_path_rate=avg('new_retained_path'),
              changed_candidate_path_rate=avg('touched'),target_outside_rate=avg('target_outside'),
              selected_target_same_rank_rate=mean(x['target_rank_before']==x['target_rank_after'] for x in rows),
              positive_rank_changes=sum(x['positive'] for x in rows),negative_rank_changes=sum(x['negative'] for x in rows),
              order_changed_rank_unchanged=sum(x['rank'] and not(x['positive'] or x['negative']) for x in rows)))
    write_csv(folder/'propagation_trace.csv',out)
    write_csv(folder/'counterfactual_probe.csv',probes)
    dump(folder/'cases.json',cases)
    value=dict(mode=mode,rows=summary,probes=len(probes),seconds=perf_counter()-t0,
               exact_legacy_replay=True,outside_rank_changes=outside_rank_changes)
    dump(summary_path,value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['smoke','full'],default='smoke')
    mode=parser.parse_args().mode
    cfg=config()
    fingerprint,root,_=preflight(cfg)
    print(run(RESULTS/fingerprint/mode,root,cfg,mode))
