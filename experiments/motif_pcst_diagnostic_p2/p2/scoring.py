"""Frozen source calibration and fixed scorer ablations on the P1 query rows."""
from collections import defaultdict
from statistics import mean
from time import perf_counter

import numpy as np
from scipy.stats import rankdata

from p2.assets import GRAPHS, PAIR_GRAPHS, Target, verify
from p2.features import all_paths, profile, candidate_scores
from p1.assets import Assets, Evaluator, prepare
from p1.config import load_config as p1_config
from p1.retrieval import Retriever
from p1.scoring import raw_order, filtered_after, evidence, permutations
from p1.run import boot_delta
from src.io_utils import dump, read, write_csv

MOTIFS={'M0':('R3',False),'M1':('R3',True),'M2':('R2',True),'M3':('R2',False)}
MODES=('F0','F1','F2','F3','F4','F5')


def pair_profiles(graph, h, r, candidates, arms, gate, table):
    data={}
    for arm in ('R2','R3'):
        paths=all_paths(graph,h,candidates,arms[arm]['ids'])
        data[arm]={c:profile(graph,r,paths.get(c,[]),gate,table) for c in candidates}
    return data


def proxy(raw, cs):
    r=np.rint(1/raw[cs]-60).astype(np.int64)
    return (r.max()-r)/max(1,r.max()-r.min())


def fuse(raw, cs, scores, mode, lam, scale, legacy_order=None):
    p=proxy(raw,cs)
    ev=np.array([scores[c]['score'] for c in cs])
    valid=np.array([scores[c]['valid'] for c in cs])
    adj=np.clip(ev/max(scale,1e-6),-3.,3.)
    active=False
    if mode=='F0':
        return cs,False,False,False
    if mode=='F1':
        return legacy_order, bool(lam and ev.any()),legacy_order!=cs,False
    if mode=='F5':
        if not valid.any():
            return cs,False,False,False
        # Evidence values define the order. The frozen order is the explicit
        # fallback for a tied evidence tier, including no evidence.
        order=[cs[j] for j in np.argsort(-ev,kind='stable')]
        return order,True,bool(np.ptp(ev)>1e-12),False
    if lam==0 or not valid.any():
        return cs,False,False,False
    if mode=='F4':
        viable=np.sort(adj[valid])
        active=bool(len(viable)>=3 and viable[-1]>0 and viable[-1]-viable[-2]>=.15
                    and p[0]-p[1]<=.23)
        if not active:
            return cs,False,False,False
    if mode=='F3':
        base_rank=rankdata(-p,method='average')
        evidence_rank=rankdata(-ev,method='average')
        delta=lam/(5+evidence_rank)
        result=1/(5+base_rank)+delta
        adjusted=bool(np.ptp(delta)>1e-12)
    else:
        delta=lam*adj
        result=p+delta
        adjusted=bool(np.any(np.abs(delta)>1e-12))
    order=[cs[j] for j in np.argsort(-result,kind='stable')]
    return order,adjusted,bool(np.ptp(delta)>1e-12),active


def for_variant(profiles,motif,agg,relative,cfg):
    arm,use=MOTIFS[motif]
    return candidate_scores(profiles[arm],agg,cfg['motif_weight'] if use else 0.,relative)


def cache_rank(raw, cs, target, excluded, fn):
    cache={}
    before=fn(raw,target,excluded)
    def calculate(order):
        key=tuple(order)
        if key not in cache:
            val=filtered_after(raw,cs,order,target,excluded,fn)
            verify(target in cs or val==before,'outside K rank changed')
            cache[key]=val
        return cache[key]
    return calculate


def calibrate(folder,root,path,cfg):
    destination=folder/'source_lock.json'
    if destination.is_file():
        return read(destination)
    started=perf_counter()
    source=Assets('wn18rr_train',folder)
    old=read(root/'source_calibration.json')
    reference=old['validation_indices']
    used=set(reference)
    rng=np.random.default_rng(cfg['seed']+163)
    remainder=[i for i in rng.permutation(len(source.queries)).tolist() if i not in used]
    indices=reference+remainder[:cfg['source_validation_facts']-len(reference)]
    score,gates=prepare(source,indices)
    retriever=Retriever(source.graph,gates,path['table'],p1_config()['retrieval'])
    potential=[]
    for i in indices:
        for d,(h,r) in enumerate(source.queries[i]):
            raw=score[i][d]
            cs=raw_order(raw)[:12]
            arms=retriever.query(int(h),int(r),cs,32,('R2','R3'))
            prof=pair_profiles(source.graph,int(h),int(r),cs,arms,gates[int(r)]['layers'].mean(0),path['table'])
            potential.append((i,d,int(h),int(r),raw,cs,prof))
    # Labels are only opened after all candidate profiles are available.
    evaluator=Evaluator(source)
    prepared=[]
    for i,d,h,r,raw,cs,prof in potential:
        target,excluded,before=evaluator.query(i,d,h,r,raw)
        fn=cache_rank(raw,cs,target,excluded,evaluator.rank)
        prepared.append((i,raw,cs,prof,fn,before))
    aggs=('S0','S1-2','S1-4',*[f'S2-{t}' for t in cfg['tau_grid']])
    lambdas=cfg['lambda_grid']
    refs=set(reference)
    trials=[]
    def trial(m,a,relative,fusion,lambdas):
        scores_list=[for_variant(prof,m,a,relative,cfg) for _,_,_,prof,_,_ in prepared]
        magnitudes=[abs(row['score']) for scores in scores_list for row in scores.values() if row['valid']]
        scale=max(.025,float(np.quantile(magnitudes,.75))) if magnitudes else .025
        results=[]
        for lam in lambdas:
            stats=[]
            for (i,raw,cs,_,fn,before),ev in zip(prepared,scores_list):
                order,on,shift,active=fuse(raw,cs,ev,fusion,lam,scale)
                rank=fn(order)
                stats.append((i,rank,before,on,shift,order!=cs,active))
            record=dict(motif=m,agg=a,relative=relative,fusion=fusion,lam=lam,scale=scale,
                        queries=len(stats),mrr=mean(1/r for _,r,*_ in stats),
                        mrr_original96=mean(1/r for i,r,*_ in stats if i in refs),
                        mrr_additional=mean(1/r for i,r,*_ in stats if i not in refs),
                        nonzero=sum(s[3] for s in stats),score_change=sum(s[4] for s in stats),
                        rank_change=sum(s[5] for s in stats),gate_active=sum(s[6] for s in stats),
                        positive=sum(s[1]<s[2] for s in stats),negative=sum(s[1]>s[2] for s in stats))
            trials.append(record)
            results.append(record)
        return results
    # Controlled steps reuse one arm/setting. No target scores enter these choices.
    first=[]
    for a in aggs:
        first+=trial('M3',a,False,'F2',lambdas)
    pick=lambda seq:max(seq,key=lambda r:(round(r['mrr'],12),-r['lam']))
    agg=pick(first)['agg']
    variants=[]
    for m in MOTIFS:
        for rel in (False,True) if m=='M3' else (False,):
            variants+=trial(m,agg,rel,'F2',lambdas)
    decision=pick(variants)
    # If all F2 are equivalent at lambda=0, retain the motif-free M3 reference.
    if decision['lam']==0:
        decision=pick([r for r in variants if r['motif']=='M3' and not r['relative']])
    controls={'F2':decision}
    for f in ('F3','F4','F5'):
        controls[f]=pick(trial(decision['motif'],agg,decision['relative'],f,lambdas if f!='F5' else (0.,)))
    preferred=pick([controls[f] for f in ('F2','F3','F4')])
    from src.io_utils import identity
    result=dict(primary=preferred,grid=controls,path_aggregation=agg,original_p1_lambda=old['lambda_selected'],
                original_p1_grid=old['source_grid'],source='wn18rr_train',old_queries=len(reference)*2,
                queries=len(prepared),selection_only='source validation, choose lowest lambda on MRR ties',
                indices=indices,source_path_table_sha=identity(path['table']),seconds=perf_counter()-started)
    write_csv(folder/'source_fusion_grid.csv',trials)
    dump(destination,result)
    return result


def configs_for(lock, k, baseline_only=False):
    primary=lock['primary']
    if k!=12:
        return [('P2_candidate_sensitivity',primary)]
    result=[('P2_primary',primary)]
    result.extend(('P2_fusion_'+f,dict(primary,**lock['grid'].get(f,{})) | {'fusion': f})
                  for f in MODES if f!='F1')
    for a in ('S0','S1-2','S1-4','S2-0.08','S2-0.25'):
        result.append((f'agg_{a}',dict(primary,agg=a,fusion='F5',lam=0.)))
    for m in MOTIFS:
        result.append((f'motif_{m}',dict(primary,motif=m,fusion='F5',lam=0.,relative=False)))
        result.append((f'motif_{m}_continuous',dict(primary,motif=m,**{z:lock['grid']['F2'][z] for z in ('fusion','lam','scale')},relative=False)))
    for rel in (True,False):
        for f in ('F2','F5'):
            result.append((f'relative_{rel}_{f}',dict(primary,relative=rel,fusion=f,
                           lam=lock['grid']['F2']['lam'] if f=='F2' else 0.,scale=lock['grid']['F2']['scale'])))
    return result


def transfer_graph(view,folder,lock,path,cfg,k=12):
    root=folder/'transfer'/view.key/f'k{k}'
    done=root/'summary.json'
    if done.is_file():
        return read(done)
    began=perf_counter()
    idx=view.indices if k==12 else read(view.dir.parent.parent/'sensitivity'/view.key/'samples.json')['indices']
    stages=[]
    for i in idx:
        for d in (0,1):
            job,raw=view.job(i,d,k)
            cs=job['candidates']
            profile_map=pair_profiles(view.graph,job['head'],job['relation'],cs,job['arms'],view.gate(job['relation']),path['table'])
            stages.append((i,d,job,raw,profile_map))
    evaluator=Evaluator(view.asset)
    rows=[]
    memory=[]
    for i,d,job,raw,prof in stages:
        cs=job['candidates']
        top30=job['top30']
        target,exclude,before=evaluator.query(i,d,job['head'],job['relation'],raw)
        calculate=cache_rank(raw,cs,target,exclude,evaluator.rank)
        b0=int(evaluator.b0[i,d])
        m5=evaluator.m5[:,i,d]
        shared=dict(dataset=view.key,k=k,query_id=job['query_id'],fact_index=i,direction=d,
                    original_rank=before,b0_rank=b0,m5_mean_rr=float((1/m5).mean()),
                    outside_candidates=target not in cs,recall12=target in top30[:12],recall30=target in top30)
        def save(scorer,conf,rank,on=False,shift=False,changed=False,gate=False,empty=None,gr=None):
            rows.append(dict(shared,scorer=scorer,motif_mode=conf.get('motif',''),path_aggregation=conf.get('agg',''),
                fusion_mode=conf.get('fusion',''),lambda_selected=conf.get('lam',0.),rank=rank,
                rr=1/rank if gr is None else float((1/gr).mean()),
                hits1=float(rank<=1) if gr is None else float((gr<=1).mean()),
                hits3=float(rank<=3) if gr is None else float((gr<=3).mean()),
                hits10=float(rank<=10) if gr is None else float((gr<=10).mean()),
                positive=rank<before if gr is None else float((gr<before).mean()),
                negative=rank>before if gr is None else float((gr>before).mean()),
                adjustment=on,score_change=shift,rank_change=changed,gate=gate,no_evidence_fraction=empty))
        if k==12:
            save('B0',{},b0)
            save('frozen_B2+Dstar',{},before)
            save('M5_three_seed',{},float(m5.mean()),gr=m5)
            for arm in ('R1','R2','R5'):
                original=job['arms'][arm]
                cs_missing=mean(not v['valid'] for v in original['values'].values())
                for mode,name in [('F1','fusion'),('F5','evidence_only')]:
                    order=original['orders'][name]
                    save(f'P1_{arm}_{mode}',dict(fusion=mode,lam=lock['original_p1_lambda']),
                         calculate(order),False,False,order!=cs,False,cs_missing)
        cache={}
        for name,setting in configs_for(lock,k):
            key=(setting['motif'],setting['agg'],setting['relative'])
            if key not in cache:
                cache[key]=for_variant(prof,*key,cfg)
            values=cache[key]
            lam=setting['lam']
            order,on,shift,gate=fuse(raw,cs,values,setting['fusion'],lam,setting['scale'])
            save(name,setting,calculate(order),on,shift,order!=cs,gate,mean(not v['valid'] for v in values.values()))
            if name=='P2_primary':
                arm,_=MOTIFS[setting['motif']]
                memory.append(dict(query_id=job['query_id'],head=job['head'],relation=job['relation'],candidates=cs,
                    selected_edge_ids=job['arms'][arm]['ids'],paths={c:[p for _,_,p in q] for c,q in prof[arm].items()},
                    scores=values,observation=view.provenance[1],checkpoint=view.provenance[0],
                    entity_embedding_ref='P1 frozen query-conditioned BellmanFord node_feature recompute'))
    groups=defaultdict(list)
    for r in rows:
        groups[r['scorer']].append(r)
    metrics=[]
    for name,m in groups.items():
        title=m[0]
        ds=[(r,r['rr']-1/r['original_rank']) for r in m]
        gaps_b0=[(r,r['rr']-1/r['b0_rank']) for r in m]
        gaps_m5=[(r,r['rr']-r['m5_mean_rr']) for r in m]
        valid=[r['no_evidence_fraction'] for r in m if r['no_evidence_fraction'] is not None]
        metrics.append(dict(dataset=view.key,k=k,scorer=name,motif_mode=title['motif_mode'],
            path_aggregation=title['path_aggregation'],fusion_mode=title['fusion_mode'],lambda_selected=title['lambda_selected'],
            MRR=mean(r['rr'] for r in m),Hits1=mean(r['hits1'] for r in m),Hits3=mean(r['hits3'] for r in m),
            Hits10=mean(r['hits10'] for r in m),recall12=mean(r['recall12'] for r in m),recall30=mean(r['recall30'] for r in m),
            delta_b2=mean(d for _,d in ds),delta_b0=mean(d for _,d in gaps_b0),delta_m5=mean(d for _,d in gaps_m5),
            ci_b2=boot_delta(ds,cfg['seed'],cfg['bootstrap']),ci_b0=boot_delta(gaps_b0,cfg['seed'],cfg['bootstrap']),
            ci_m5=boot_delta(gaps_m5,cfg['seed'],cfg['bootstrap']),positive_flips=sum(r['positive'] for r in m),
            negative_flips=sum(r['negative'] for r in m),queries=len(m),queries_with_nonzero_adjustment=sum(r['adjustment'] for r in m),
            queries_with_score_change=sum(r['score_change'] for r in m),queries_with_rank_change=sum(r['rank_change'] for r in m),
            gate_activation_rate=mean(r['gate'] for r in m),no_evidence_fraction=mean(valid) if valid else None))
    write_csv(root/'per_query_rank_changes.csv',rows)
    if memory:
        dump(root/'scorer_memory.json',memory)
    result=dict(dataset=view.key,k=k,metrics=metrics,queries=len(stages),seconds=perf_counter()-began,
         nodes=view.graph.n,edges=len(view.graph.edge_type),relations=view.graph.R,
         degree_p90=float(np.quantile(view.graph.out_degree,.9)))
    dump(done,result)
    return result
