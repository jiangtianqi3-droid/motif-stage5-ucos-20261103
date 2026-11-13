"""Streaming evidence, validation-only selection, calibrated LLM and offline ranks."""
import concurrent.futures
import csv
import dataclasses
import time
import gzip
import os
import numpy as np
from .util import *
from .edge_weighter import UnifiedEdgeWeighter, weights_for
from .structural_reranker import Config, grid, rerank, trigger
from .offline import Dataset, metrics, bootstrap
from .inference import infer
from .verbalizer import Verbalizer
from .llm_reranker import Client, confidence
from qc_v1.path_extractor import retrieve
from .selection import validation_partition, query_key, sample_query_groups, conservative_choice
from .statistics import comparison

def selected_jobs(indices,smoke=False):
    jobs=[(int(i),d) for i in indices for d in (0,1)]
    return jobs[:5] if smoke else jobs

def ev_path(work,method,split,i,d): return work/'evidence'/method/split/f'{i}_{d}.json.gz'

def evidence(work,method,split,i,d):
    path=ev_path(work,method,split,i,d)
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)

def save_evidence(path,record):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=1) as f:json.dump(record,f,ensure_ascii=False,allow_nan=False)
    tmp.replace(path)

def legacy_record(dataset,work,method,split,i,d,candidate):
    # Retrieval is a deterministic function of graph, relation and ordered entity IDs.
    # Reuse only exact ordered pools; refresh frozen logits for the new checkpoint.
    old=HISTORY/'route-a-rerank-20261109/results/runs/f20ab88ff89d82d8506c'/dataset.key/('B0' if work.name=='B0' else 'T-4D')
    cp=old/'candidates'/split/f'{i}_{d}.json';ep=old/'evidence'/method/split/f'{i}_{d}.json'
    if not cp.exists() or not ep.exists():return None
    previous=read(cp)
    if previous['query_info']!=candidate['query_info']:return None
    if [c['entity'] for c in previous['candidates']]!=[c['entity'] for c in candidate['candidates']]:return None
    record=read(ep)
    if record['graph_key']!=dataset.g.key:raise RuntimeError('Reused evidence graph mismatch')
    for ev,c in zip(record['candidate_evidence'],candidate['candidates']):ev.update(c)
    record['provenance']['reused_retrieval_only']=str(ep)
    record['provenance']['reused_evidence_sha256']=sha(ep)
    record['provenance']['model']=candidate['model']
    return record

def obtain_evidence(dataset,g,sig,work,selections,run,smoke):
    return parallel_evidence(dataset,g,sig,work,selections,run,smoke)

def parallel_evidence(dataset,g,sig,work,selections,run,smoke):
    import multiprocessing
    from .retrieval_jobs import initialize,relation_job
    community=read(OLD/'results/runs/d48ae432a2533cdaaec0'/dataset.key/'community.json')
    labels=np.asarray(community['labels'],dtype=np.int64)
    if labels.shape!=(g.n,):raise ValueError('Community labels do not match graph')
    jobs={}
    for split,indices in selections.items():
        for i,d in selected_jobs(indices,smoke and split=='test'):
            q=int(dataset.public[split][i,d,1]);jobs.setdefault(q,[]).append((split,i,d))
    total=4*sum(map(len,jobs.values()));complete=0;reused=0;seconds=0.;audits=[];peak=0
    workers=min(4,len(jobs))
    if not workers:raise ValueError('No evidence queries')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize,initargs=(g,sig,labels,dataset.key,work)) as pool:
        pending={pool.submit(relation_job,payload) for payload in sorted(jobs.items())}
        while pending:
            done,pending=concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                result=future.result();complete+=result['records'];reused+=result['reused'];seconds+=result['seconds']
                audits.extend(result['audits']);peak=max(peak,result['peak_rss_bytes'])
            status(run,state='running',stage='structural_evidence',dataset=dataset.key,family=work.name,done=complete,total=total,
                   pending_relations=len(pending),workers=workers,reused=reused)
    dump(work/'retrieval_completion.json',dict(records=total,reused=reused,worker_seconds=seconds,workers=workers,peak_worker_rss_bytes=peak,K=30,M=10))
    dump(work/'operator_audit.json',audits)

def score_file(work,split,i,d): return np.load(work/'scores'/split/f'{i}_{d}.npy',allow_pickle=False)

def select_structure(dataset,work,selections,run,smoke):
    frozen=work/'selected_structure.json'
    if frozen.exists(): return read(frozen)
    candidates=list(grid()); selected={}
    partition=read(work/'validation_partition.json')
    jobs=selected_jobs(partition['structure'],smoke)
    for method in METHODS:
        values=[];groups=[];base_values=[]
        for i,d in jobs:
            record=evidence(work,method,'valid',i,d); ids=np.array([c['entity'] for c in record['candidate_evidence']],dtype=int)
            scores=score_file(work,'valid',i,d)
            base_values.append(1/dataset.rank('valid',i,d,scores,ids,validation_role='structure'));row=[]
            for j,cfg in enumerate(candidates):
                result=rerank(record,cfg)
                row.append(1/dataset.rank('valid',i,d,scores,ids,result['order'],validation_role='structure'))
            values.append(row);groups.append(query_key(dataset.public['valid'],i,d))
            del record,scores
        choice=conservative_choice(np.asarray(values).reshape(-1,len(candidates)),groups)
        best=choice['chosen'];means=choice['means'] or [0.]*len(candidates)
        selected[method]=dict(config=candidates[best].dict(),validation_MRR=means[best],base_MRR=means[0],selection_audit=choice,
                              validation_queries=len(jobs),grid_size=len(candidates),selection_split='valid',test_used=False,
                              grid=[dict(config=c.dict(),MRR=float(s)) for c,s in zip(candidates,means)])
        status(run,state='running',stage='validation_selection',dataset=dataset.key,family=work.name,method=method,selected=selected[method]['config'])
    dump(frozen,selected)
    return selected

def request_llm(dataset,work,indices,cfg,verbalizer,client,split,run,smoke=False,limit=None):
    jobs=selected_jobs(indices,smoke); pending=[]; eligible=0; sent=0; cache_hits=0
    active_rows=[]
    for i,d in jobs:
        record=evidence(work,'R4',split,i,d);result=rerank(record,cfg)
        if trigger(record,result,cfg):
            h,q=map(int,dataset.public[split][i,d]);active_rows.append((i,d,h,q))
    sampled=sample_query_groups(active_rows,limit)
    selected_aliases=[alias for key,aliases in sampled for alias in aliases]
    if limit is not None and len(sampled)<32:
        return dict(queries=len(jobs),eligible_query_groups=len(sampled),eligible=len(sampled),processed=0,cache_hits=0,
                    selected_aliases=[],selection='insufficient independent eligible queries; no API calls',network_requests=0)
    pool_limit=4
    def one(aliases):
        i,d=aliases[0]
        record=evidence(work,'R4',split,i,d); result=rerank(record,cfg); pool=trigger(record,result,cfg)
        if not pool: return None
        response=client.call(verbalizer.prompt(record,result,pool),[record['candidate_evidence'][j]['entity'] for j in pool])
        return aliases,response
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4,int(os.environ.get('LLM_MAX_CONCURRENCY','4')))) as executor:
        def collect():
            nonlocal sent,cache_hits
            for future in pending:
                value=future.result()
                if value is not None:
                    aliases,response=value
                    for i,d in aliases:dump(work/'llm'/split/f'{i}_{d}.json',response)
                    cache_hits+=int(response['cache_hit']);sent+=1
            pending.clear()
        for key,aliases in sampled:
            eligible+=1
            pending.append(executor.submit(one,aliases))
            if len(pending)>=pool_limit:
                collect()
                status(run,state='running',stage='llm_evidence',dataset=dataset.key,family=work.name,split=split,
                       eligible=eligible,processed=sent,network_requests=client.network_requests)
        collect()
    return dict(queries=len(jobs),eligible_query_groups=len({(h,q) for i,d,h,q in active_rows}),eligible=eligible,processed=sent,cache_hits=cache_hits,
                selected_aliases=selected_aliases,selection='seeded uniform unique public queries',network_requests=client.network_requests)

def calibrate_llm(dataset,work,selections,selection,run,smoke):
    frozen=work/'selected_llm.json'
    if frozen.exists(): return read(frozen)
    cfg=Config(**selection['R4']['config']); client=Client()
    verbalizer=Verbalizer(dataset.key,dataset.names,dataset.relations,dataset.g.R,work/'verbalizer_audit.json')
    partition=read(work/'validation_partition.json')
    if partition['unique_queries']['calibration']<32:
        result=dict(strength=0.,reason='fewer_than_32_independent_calibration_queries',selection_split='calibration',test_used=False,stats={})
        dump(frozen,result);return result
    stats=request_llm(dataset,work,partition['calibration'],cfg,verbalizer,client,'valid',run,smoke,limit=64)
    strengths=[0.,.05]; values=[];groups=[];fingerprints=set()
    for i,d in stats['selected_aliases']:
        record=evidence(work,'R4','valid',i,d); ids=np.array([c['entity'] for c in record['candidate_evidence']],dtype=int)
        scores=score_file(work,'valid',i,d); path=work/'llm/valid'/f'{i}_{d}.json'
        response=read(path);conf=confidence(response);fingerprints.add(response['fingerprint'])
        values.append([1/dataset.rank('valid',i,d,scores,ids,rerank(record,cfg,conf,lam)['order'],validation_role='calibration') for lam in strengths])
        groups.append(query_key(dataset.public['valid'],i,d))
        del record,scores
    choice=conservative_choice(np.asarray(values).reshape(-1,len(strengths)),groups,min_groups=32)
    chosen=choice['chosen'] if len(fingerprints)>=32 else 0
    result=dict(strength=strengths[chosen],grid=[dict(strength=lam,MRR=float(s)) for lam,s in zip(strengths,choice['means'])],selection_audit=choice,
                reason=choice['reason'] if len(fingerprints)>=32 else 'fewer_than_32_unique_calibration_prompts',
                unique_prompts=len(fingerprints),validation_queries=len(values),calibration_calls_max=64,selection_split='calibration',test_used=False,stats=stats,model=client.model)
    dump(frozen,result)
    return result

def evaluate(dataset,work,selections,selection,llm,run,smoke):
    jobs=selected_jobs(selections['test'],smoke)
    methods=METHODS+(['R5'] if llm is not None else [])
    ranks={name:[] for name in ['M0','ReverseThenM0',*methods]}; expansions={m:dict(eligible=0,promoted=0,demoted=0) for m in methods}
    reverse_flags=[];seen_flags=[];partition=read(work/'validation_partition.json')
    development_keys={query_key(dataset.public['valid'],i,d) for role in ['structure','calibration'] for i in partition[role] for d in (0,1)}
    structural_diag={m:dict(top10_answers=0,top10_rr_sum=0.,rank11_30_answers=0,strict_expansion_potential=0) for m in METHODS}
    llm_diag=dict(queries=len(jobs),eligible=0,calls=0,abstain=0,parse_failures=0,positive_flips=0,negative_flips=0,total_tokens=0,total_latency_seconds=0.,cache_hits=0)
    filtered_diff=dict(queries=len(jobs),candidate_pools_differ=0,filtered_top30_missing_entities=0)
    rows_path=work/'per_query_ranks.jsonl';tmp=rows_path.with_name(rows_path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as out:
        for i,d in jobs:
            scores=score_file(work,'test',i,d); target=dataset.target('test',i,d)
            candidate=read(work/'candidates/test'/f'{i}_{d}.json'); ids=np.array([c['entity'] for c in candidate['candidates']],dtype=int)
            base=dataset.rank('test',i,d,scores,ids); ranks['M0'].append(base)
            h,q=map(int,dataset.public['test'][i,d]); valid=np.ones(len(scores),dtype=bool)
            reverse_edges=dataset.g.edges_at(h,reverse=True)
            reverse_entities=dataset.g.edge_index[0,reverse_edges[dataset.g.edge_type[reverse_edges]==q]]
            shortcut_scores=scores.copy();shortcut_scores[np.unique(reverse_entities)]+=float(np.ptp(scores))+1.
            ranks['ReverseThenM0'].append(dataset.rank('test',i,d,shortcut_scores,ids))
            reverse_flags.append(bool(target in reverse_entities));seen_flags.append((h,q) in development_keys)
            valid[list(dataset.filters['test'].get((h,q),()))]=False;valid[target]=True
            exact=np.flatnonzero(valid); exact=exact[np.lexsort((exact,-scores[exact]))][:30]
            missing=set(exact)-set(ids); filtered_diff['candidate_pools_differ']+=int(bool(missing));filtered_diff['filtered_top30_missing_entities']+=len(missing)
            row=dict(fact_index=i,direction=d,base_filtered_rank=base,target_in_candidate_pool=bool(target in ids))
            for method in METHODS:
                record=evidence(work,method,'test',i,d); cfg=Config(**selection[method]['config']); result=rerank(record,cfg)
                rank=dataset.rank('test',i,d,scores,ids,result['order']);ranks[method].append(rank);row[method]=rank
                exp=expansions[method];exp['eligible']+=int(11<=base<=30);exp['promoted']+=int(11<=base<=30 and rank<=10);exp['demoted']+=int(base<=10 and rank>10)
                ev=record['candidate_evidence']; truth=next((c for c in ev if c['entity']==target),None)
                diag=structural_diag[method]
                if truth and truth['base_rank']<=10:
                    values=[c['max_path_weight'] for c in ev[:10]];v=truth['max_path_weight']
                    diag['top10_answers']+=1;diag['top10_rr_sum']+=1/(1+sum(x>v for x in values)+.5*(sum(x==v for x in values)-1))
                if truth and 11<=truth['base_rank']<=30:
                    diag['rank11_30_answers']+=1
                    negative=[c['max_path_weight'] for c in ev[:10] if c['entity']!=target]
                    diag['strict_expansion_potential']+=int(truth['status']=='ok' and negative and truth['max_path_weight']>max(negative))
                if method=='R4' and llm is not None:
                    llm_diag['eligible']+=int(bool(trigger(record,result,cfg)))
                    path=work/'llm/test'/f'{i}_{d}.json'; response=read(path) if path.exists() else None
                    conf=confidence(response) if response else {}
                    final=rerank(record,cfg,conf,llm['strength'])
                    final_rank=dataset.rank('test',i,d,scores,ids,final['order']); ranks['R5'].append(final_rank);row['R5']=final_rank
                    ex=expansions['R5'];ex['eligible']+=int(11<=base<=30);ex['promoted']+=int(11<=base<=30 and final_rank<=10);ex['demoted']+=int(base<=10 and final_rank>10)
                    if response:
                        llm_diag['calls']+=1;llm_diag['abstain']+=int(response['decision']['abstain']);llm_diag['parse_failures']+=int(response['parse_error'] is not None)
                        llm_diag['positive_flips']+=int(rank>1 and final_rank==1);llm_diag['negative_flips']+=int(rank==1 and final_rank>1)
                        llm_diag['total_tokens']+=response['usage'].get('total_tokens',0);llm_diag['total_latency_seconds']+=response['seconds'];llm_diag['cache_hits']+=int(response['cache_hit'])
                        if response['decision']['abstain'] or response['parse_error']: assert final['order']==result['order'] and final['scores']==result['scores']
                        # Save bounded, real examples including abstentions and incorrect changes.
                        if llm_diag['calls']<=8 or rank!=final_rank:
                            dump(work/'cases'/f'{i}_{d}.json',dict(query_info=record['query_info'],candidates=record['candidate_evidence'],structural_order=result['order'],final_order=final['order'],
                                                                      response=response,structural_filtered_rank=rank,final_filtered_rank=final_rank))
                del record,result
            out.write(json.dumps(row,ensure_ascii=False)+'\n')
            del scores
    tmp.replace(rows_path)
    if llm_diag['calls']:
        llm_diag['abstain_rate']=llm_diag['abstain']/llm_diag['calls'];llm_diag['mean_latency_seconds']=llm_diag['total_latency_seconds']/llm_diag['calls']
    llm_diag['trigger_rate']=llm_diag['eligible']/len(jobs)
    output=dict(dataset=dataset.key,family=work.name,queries=len(jobs),metrics={k:metrics(v) for k,v in ranks.items()},expansion=expansions,
                llm=llm_diag if llm is not None else dict(skipped=True,reason='anonymous graph or explicit skip-llm'),filtered_candidate_difference=filtered_diff,
                structural_diagnostics=structural_diag,paired_bootstrap={} if smoke else {m:bootstrap(ranks['M0'],ranks[m]) for m in methods},
                llm_vs_R4_bootstrap={} if smoke or llm is None else bootstrap(ranks['R4'],ranks['R5']),
                paired_method_comparisons={} if smoke else {f'R4_vs_{m}':bootstrap(ranks[m],ranks['R4']) for m in ['R1','R2','R3']})
    for method,diag in structural_diag.items(): diag['top10_conditional_structural_MRR']=diag['top10_rr_sum']/diag['top10_answers'] if diag['top10_answers'] else None
    output['primary_comparisons']={}
    if not smoke:
        for new,base in [('R4','R1'),('R4','R3')]+([('R5','R4')] if llm is not None else []):
            output['primary_comparisons'][f'{new}_vs_{base}']=comparison(ranks[base],ranks[new],dataset.public['test'][selections['test']],dataset.truth['test'][selections['test']])
    output['boundary']='historically exposed repair replication; no blind or universal advantage claim'
    if work.name=='B0' and not smoke:
        old=HISTORY/'route-a-rerank-20261109/results/runs/f20ab88ff89d82d8506c'/dataset.key/'B0/ranks.npz'
        with np.load(old,allow_pickle=False) as original:
            if not np.array_equal(ranks['M0'],original['M0']):raise AssertionError('Replayed B0 baseline changed')
        output['baseline_replay_verified']=True
    output['stratified']={}
    for name,flags in [('seen_development_query',seen_flags),('same_relation_reverse_observed',reverse_flags)]:
        flags=np.asarray(flags,dtype=bool)
        output['stratified'][name]={}
        for value,mask in [('yes',flags),('no',~flags)]:
            output['stratified'][name][value]=dict(queries=int(mask.sum()),metrics={m:metrics(np.asarray(r)[mask]) for m,r in ranks.items()} if mask.any() else {})
    output['data_counts']=dict(test_facts=len(selections['test']),test_queries=len(jobs),
        unique_test_queries=len({query_key(dataset.public['test'],i,d) for i,d in jobs}),
        structure_facts=len(partition['structure']),calibration_facts=len(partition['calibration']),unique_development_queries=len(development_keys),
        test_relations=len(set(dataset.truth['test'][selections['test'],1].tolist())))
    dump(work/'summary.json',output)
    np.savez_compressed(work/'ranks.npz',**{k:np.asarray(v,dtype=np.int64) for k,v in ranks.items()})
    return output

def worker(key,family,run,smoke=False,skip_llm=False):
    work=run/key/family;work.mkdir(parents=True,exist_ok=True)
    if (work/'completion.json').exists(): return
    adapter=historical_package();g=adapter.load_observed(key);sig=adapter.load_signature(g,key)
    dataset=Dataset(key,g,work/'split_leakage_audit.json')
    partition=validation_partition(dataset.public['valid'],max_facts_per_role=5 if smoke else 512)
    dataset.configure_validation_roles(partition)
    dump(work/'validation_partition.json',partition)
    dump(work/'validation_filter_audit.json',dict(structure='observation + structure role labels only',
        calibration='observation + selected development labels',test_labels_used=False,calibration_labels_used_for_structure=False))
    selections={'valid':sorted(set(partition['structure']+partition['calibration'])),
                'test':list(range(min(5,len(dataset.public['test'])) if smoke else len(dataset.public['test'])))}
    dump(work/'selection.json',dict(indices=selections,selection='fixed seed uniform facts without inspecting labels',test_full=not smoke,smoke_queries_per_split=5 if smoke else None))
    if family=='B0':reuse_baseline_artifacts(work,key,selections)
    status(run,state='running',stage='dataset_start',dataset=key,family=family)
    infer(dataset,g,sig,family,work,selections,run)
    obtain_evidence(dataset,g,sig,work,selections,run,smoke)
    selection=select_structure(dataset,work,selections,run,smoke)
    llm=None
    if not skip_llm and key!='FBIngram_25':
        llm=calibrate_llm(dataset,work,selections,selection,run,smoke)
        cfg=Config(**selection['R4']['config']);client=Client()
        verbalizer=Verbalizer(key,dataset.names,dataset.relations,g.R)
        stats=request_llm(dataset,work,selections['test'],cfg,verbalizer,client,'test',run,smoke) if llm['strength']>0 else dict(skipped=True,reason='zero_calibrated_strength')
        dump(work/'llm_test_stats.json',stats)
    summary=evaluate(dataset,work,selections,selection,llm,run,smoke)
    del dataset,g,sig;cleanup()
    dump(work/'completion.json',dict(state='smoke_complete' if smoke else 'complete',at=now(),rss_bytes=memory(),peak_rss_bytes=peak_memory(),summary_sha256=sha(work/'summary.json')))
    status(run,state='running',stage='dataset_complete',dataset=key,family=family,metrics=summary['metrics'])

def reuse_baseline_artifacts(work,key,selections):
    old=HISTORY/'route-a-rerank-20261109/results/runs/f20ab88ff89d82d8506c'/key/'B0'
    checkpoint=HISTORY.parent/'motif-phase3-work/MOTIF/ckpts/motif_3g.pth'
    if read(old/'model_audit.json')['checkpoint_sha256']!=sha(checkpoint):raise RuntimeError('B0 reuse checkpoint mismatch')
    linked=0
    for category,suffix in [('scores','.npy'),('candidates','.json')]:
        for split,indices in selections.items():
            folder=work/category/split;folder.mkdir(parents=True,exist_ok=True)
            for i,d in selected_jobs(indices):
                source=old/category/split/f'{i}_{d}{suffix}';dest=folder/source.name
                if source.exists() and not dest.exists():
                    # Atomic outputs replace local links; no historical inode is edited.
                    os.link(source,dest);linked+=1
    dump(work/'baseline_reuse.json',dict(linked=linked,source=str(old),checkpoint_sha256=sha(checkpoint)))
