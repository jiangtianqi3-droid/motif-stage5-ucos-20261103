"""Streaming evidence, validation-only selection, calibrated LLM and offline ranks."""
import concurrent.futures
import csv
import dataclasses
import time
import numpy as np
from .util import *
from .edge_weighter import UnifiedEdgeWeighter, weights_for
from .structural_reranker import Config, grid, rerank, trigger
from .offline import Dataset, metrics, bootstrap
from .inference import infer
from .verbalizer import Verbalizer
from .llm_reranker import Client, confidence
from qc_v1.path_extractor import retrieve

def selected_jobs(indices,smoke=False):
    jobs=[(int(i),d) for i in indices for d in (0,1)]
    return jobs[:5] if smoke else jobs

def ev_path(work,method,split,i,d): return work/'evidence'/method/split/f'{i}_{d}.json'

def obtain_evidence(dataset,g,sig,work,selections,run,smoke):
    community=read(OLD/'results/runs/d48ae432a2533cdaaec0'/dataset.key/'community.json')
    labels=np.asarray(community['labels'],dtype=np.int64)
    assert labels.shape==(g.n,)
    factory=UnifiedEdgeWeighter(g,sig)
    elapsed=[];completed=0;operator_audit=[]
    jobs={}
    for split,indices in selections.items():
        for i,d in selected_jobs(indices,smoke):
            c=read(work/'candidates'/split/f'{i}_{d}.json'); q=c['query_info']['relation']
            jobs.setdefault(q,[]).append((split,i,d,c))
    # Candidate metadata are small, but never hold evidence or subgraphs for all queries.
    total=sum(map(len,jobs.values()))*4
    for q,rows in sorted(jobs.items()):
        for method in METHODS:
            w=weights_for(factory,method,q)
            if method=='R4':
                legacy=factory.compute('topology_aware',q)
                delta=w.transition-legacy.transition
                u=g.edge_index[0]
                without_sender=w.weights/(1+g.out_degree[u])**(-.25)
                mass=np.bincount(u,weights=without_sender,minlength=g.n)
                no_sender_probability=without_sender/mass[u]
                cancellation_error=float(np.max(np.abs(no_sender_probability-w.edge_probability),initial=0))
                assert cancellation_error<1e-12
                operator_audit.append(dict(relation=int(q),edge_weight_mean=float(w.weights.mean()),
                    max_forward_difference_vs_E3a=float(np.max(np.abs(delta.data),initial=0)),sender_cancellation_error=cancellation_error,
                    row_stochastic_assertions_passed=True))
                del legacy,delta,without_sender,mass,no_sender_probability
            for split,i,d,c in rows:
                path=ev_path(work,method,split,i,d)
                if not path.exists():
                    began=time.perf_counter()
                    record=retrieve(g,w,labels,c['query_info']['head'],q,c['candidates'],method,sig,max_paths=10)
                    record['query_info'].update(c['query_info'])
                    record['provenance']=dict(ground_truth_used=False,candidate_filter='observation+valid only at test; no current answer reinstatement',model=c['model'])
                    dump(path,record)
                    elapsed.append(time.perf_counter()-began)
                    del record
                completed+=1
                if completed%32==0 or completed==total:
                    cold=float(np.mean(elapsed[-128:])) if elapsed else 0.
                    status(run,state='running',stage='structural_evidence',dataset=dataset.key,family=work.name,done=completed,total=total,
                           seconds_per_query_method=cold,estimated_remaining_seconds=(total-completed)*cold)
            del w;cleanup()
    dump(work/'retrieval_completion.json',dict(records=total,mean_cold_seconds=float(np.mean(elapsed)) if elapsed else None,rss_bytes=memory(),K=30,M=10))
    dump(work/'operator_audit.json',operator_audit)
    del factory,labels,jobs;cleanup()

def score_file(work,split,i,d): return np.load(work/'scores'/split/f'{i}_{d}.npy',allow_pickle=False)

def select_structure(dataset,work,selections,run,smoke):
    frozen=work/'selected_structure.json'
    if frozen.exists(): return read(frozen)
    candidates=list(grid()); selected={}
    jobs=selected_jobs(selections['valid'],smoke)
    for method in METHODS:
        sums=np.zeros(len(candidates)); base_sum=0.
        for i,d in jobs:
            record=read(ev_path(work,method,'valid',i,d)); ids=np.array([c['entity'] for c in record['candidate_evidence']],dtype=int)
            scores=score_file(work,'valid',i,d)
            base_sum+=1/dataset.rank('valid',i,d,scores,ids)
            for j,cfg in enumerate(candidates):
                result=rerank(record,cfg)
                sums[j]+=1/dataset.rank('valid',i,d,scores,ids,result['order'])
            del record,scores
        # Grid starts with identity and ties prefer smaller residual.
        best=int(np.argmax(sums))
        selected[method]=dict(config=candidates[best].dict(),validation_MRR=float(sums[best]/len(jobs)),base_MRR=float(base_sum/len(jobs)),
                              validation_queries=len(jobs),grid_size=len(candidates),selection_split='valid',test_used=False,
                              grid=[dict(config=c.dict(),MRR=float(s/len(jobs))) for c,s in zip(candidates,sums)])
        status(run,state='running',stage='validation_selection',dataset=dataset.key,family=work.name,method=method,selected=selected[method]['config'])
    dump(frozen,selected)
    return selected

def request_llm(dataset,work,indices,cfg,verbalizer,client,split,run,smoke=False,limit=None):
    jobs=selected_jobs(indices,smoke); pending=[]; eligible=0; sent=0; cache_hits=0
    pool_limit=4
    def one(i,d):
        record=read(ev_path(work,'R4',split,i,d)); result=rerank(record,cfg); pool=trigger(record,result,cfg)
        if not pool: return None
        response=client.call(verbalizer.prompt(record,result,pool),[record['candidate_evidence'][j]['entity'] for j in pool])
        return i,d,response
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4,int(os.environ.get('LLM_MAX_CONCURRENCY','4')))) as executor:
        def collect():
            nonlocal sent,cache_hits
            for future in pending:
                value=future.result()
                if value is not None:
                    i,d,response=value
                    dump(work/'llm'/split/f'{i}_{d}.json',response)
                    cache_hits+=int(response['cache_hit']);sent+=1
            pending.clear()
        for i,d in jobs:
            record=read(ev_path(work,'R4',split,i,d)); result=rerank(record,cfg)
            active=bool(trigger(record,result,cfg)); del record
            if active:
                eligible+=1
                if limit is None or eligible<=limit: pending.append(executor.submit(one,i,d))
            if len(pending)>=pool_limit:
                collect()
                status(run,state='running',stage='llm_evidence',dataset=dataset.key,family=work.name,split=split,
                       eligible=eligible,processed=sent,network_requests=client.network_requests)
        collect()
    return dict(queries=len(jobs),eligible=eligible,processed=sent,cache_hits=cache_hits,network_requests=client.network_requests)

def calibrate_llm(dataset,work,selections,selection,run,smoke):
    frozen=work/'selected_llm.json'
    if frozen.exists(): return read(frozen)
    cfg=Config(**selection['R4']['config']); client=Client()
    verbalizer=Verbalizer(dataset.key,dataset.names,dataset.relations,dataset.g.R,work/'verbalizer_audit.json')
    stats=request_llm(dataset,work,selections['valid'],cfg,verbalizer,client,'valid',run,smoke,limit=64)
    strengths=[0.,.01,.03,.05,.1]; sums=np.zeros(len(strengths)); count=0
    for i,d in selected_jobs(selections['valid'],smoke):
        record=read(ev_path(work,'R4','valid',i,d)); ids=np.array([c['entity'] for c in record['candidate_evidence']],dtype=int)
        scores=score_file(work,'valid',i,d); path=work/'llm/valid'/f'{i}_{d}.json'
        conf=confidence(read(path)) if path.exists() else {}
        for j,lam in enumerate(strengths): sums[j]+=1/dataset.rank('valid',i,d,scores,ids,rerank(record,cfg,conf,lam)['order'])
        count+=1
        del record,scores
    chosen=int(np.argmax(sums))
    result=dict(strength=strengths[chosen],grid=[dict(strength=lam,MRR=float(s/count)) for lam,s in zip(strengths,sums)],
                validation_queries=count,calibration_calls_max=64,selection_split='valid',test_used=False,stats=stats,model=client.model)
    dump(frozen,result)
    return result

def evaluate(dataset,work,selections,selection,llm,run,smoke):
    jobs=selected_jobs(selections['test'],smoke)
    methods=METHODS+(['R5'] if llm is not None else [])
    ranks={name:[] for name in ['M0',*methods]}; expansions={m:dict(eligible=0,promoted=0,demoted=0) for m in methods}
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
            valid[list(dataset.filters['test'].get((h,q),()))]=False;valid[target]=True
            exact=np.flatnonzero(valid); exact=exact[np.lexsort((exact,-scores[exact]))][:30]
            missing=set(exact)-set(ids); filtered_diff['candidate_pools_differ']+=int(bool(missing));filtered_diff['filtered_top30_missing_entities']+=len(missing)
            row=dict(fact_index=i,direction=d,base_filtered_rank=base,target_in_candidate_pool=bool(target in ids))
            for method in METHODS:
                record=read(ev_path(work,method,'test',i,d)); cfg=Config(**selection[method]['config']); result=rerank(record,cfg)
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
    dump(work/'summary.json',output)
    np.savez_compressed(work/'ranks.npz',**{k:np.asarray(v,dtype=np.int64) for k,v in ranks.items()})
    return output

def worker(key,family,run,smoke=False,skip_llm=False):
    work=run/key/family;work.mkdir(parents=True,exist_ok=True)
    if (work/'completion.json').exists(): return
    adapter=historical_package();g=adapter.load_observed(key);sig=adapter.load_signature(g,key)
    dataset=Dataset(key,g,work/'split_leakage_audit.json')
    selections={}
    for split,array in dataset.public.items():
        n=len(array);limit=min(n,5 if smoke else (256 if split=='valid' else n))
        selections[split]=np.sort(np.random.default_rng(1024).choice(n,limit,replace=False)).tolist()
    dump(work/'selection.json',dict(indices=selections,selection='fixed seed uniform facts without inspecting labels',test_full=not smoke,smoke_queries_per_split=5 if smoke else None))
    status(run,state='running',stage='dataset_start',dataset=key,family=family)
    infer(dataset,g,sig,family,work,selections,run)
    obtain_evidence(dataset,g,sig,work,selections,run,smoke)
    selection=select_structure(dataset,work,selections,run,smoke)
    llm=None
    if not skip_llm and key!='FBIngram_25':
        llm=calibrate_llm(dataset,work,selections,selection,run,smoke)
        cfg=Config(**selection['R4']['config']);client=Client()
        verbalizer=Verbalizer(key,dataset.names,dataset.relations,g.R)
        stats=request_llm(dataset,work,selections['test'],cfg,verbalizer,client,'test',run,smoke)
        dump(work/'llm_test_stats.json',stats)
    summary=evaluate(dataset,work,selections,selection,llm,run,smoke)
    del dataset,g,sig;cleanup()
    dump(work/'completion.json',dict(state='smoke_complete' if smoke else 'complete',at=now(),rss_bytes=memory(),peak_rss_bytes=peak_memory(),summary_sha256=sha(work/'summary.json')))
    status(run,state='running',stage='dataset_complete',dataset=key,family=family,metrics=summary['metrics'])
