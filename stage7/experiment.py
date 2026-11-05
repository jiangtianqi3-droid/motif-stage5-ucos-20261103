import gc
import time
from datetime import datetime,timezone
import numpy as np
import torch
from stage6.observed_loader import load_observed,from_forward
from stage6.models import load_model,state_hash
from stage6.proxy_graph import build_graph
from stage7.util import ROOT,read,dump,sha,identity,log
from stage7.proxy_split import make_split,residual
from stage7.evaluator import safe_evaluate,metrics
from stage7.numeric_stability import install
from stage7.confirmation import paired_rr,lcb,accept
from stage7.robust_selector import choose,ALPHAS

SOURCES=['wn18rr_train','codex_medium_train','fb15k237_train']
RATES=[.05,.10,.15]


def graph_for(args,key):
    return load_observed(ROOT/'observations' if key.endswith('_new') else args.observations,key)


def save(path,value,ranks,queries,alphas):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'),ranks=ranks,queries=queries,alphas=np.asarray(alphas))
    dump(path,dict(**value,ranks_sha256=sha(path.with_suffix('.npz'))),exclusive=True)


def score(model,graph,queries,truth,alphas,batch):
    began=time.perf_counter()
    if len(queries)==0:
        return np.zeros((len(alphas),0,2),dtype=np.int64),dict(seconds=0.,peak_allocated=0,facts=0),{}
    d,t,topo=build_graph(graph,queries,truth,ROOT/'cache/topology')
    r,info,_=safe_evaluate(model,d,t,alphas,batch)
    info['wall_seconds_including_graph_and_retries']=time.perf_counter()-began
    info['requested_batch_size']=batch
    info['recovered_oom_retries']=int(np.log2(batch/info['batch_size']))
    del d,t
    gc.collect()
    torch.cuda.empty_cache()
    return r,info,topo


def load_phase(path,model,rem,queries,truth,alphas,args,extra):
    if path.exists():
        rec=read(path)
        assert sha(path.with_suffix('.npz'))==rec['ranks_sha256']
        assert rec['query_sha256']==identity(queries) and rec['checkpoint_sha256']==model.tsdc_checkpoint_sha256
        assert rec['numeric_code_sha256']==sha(ROOT/'numeric_stability.py')
        assert rec['residual_sha256']==rem.key
        if 'evaluator_sha256' in rec:
            assert rec['evaluator_sha256']==sha(ROOT/'evaluator.py')
        with np.load(path.with_suffix('.npz')) as z:
            assert np.array_equal(z['alphas'],alphas)
            assert np.array_equal(z['queries'],queries)
            assert z['ranks'].shape==(len(alphas),len(queries),2)
            assert (z['ranks']>=1).all()
            return z['ranks'].copy(),rec
    log('proxy_phase_begin',record=path.name,phase=extra,facts=len(queries),alphas=alphas)
    r,info,topo=score(model,rem,queries,truth,alphas,args.batch_size)
    rec=dict(info=info,topology=topo,query_sha256=identity(queries),residual_sha256=rem.key,
             checkpoint_sha256=model.tsdc_checkpoint_sha256,numeric_code_sha256=sha(ROOT/'numeric_stability.py'),
             evaluator_sha256=sha(ROOT/'evaluator.py'),
             phase=extra,metrics=[metrics(v) if v.size else None for v in r])
    save(path,rec,r,queries,alphas)
    log('proxy_phase_complete',record=path.name,seconds=info['seconds'])
    return r,rec


def _run(args):
    from stage7.isolation import enable
    enable()
    mechanism=args.task=='mechanism'
    if mechanism:
        keys=args.keys or SOURCES
        training=args.training_seeds or [1024]
        proxy=args.proxy_seeds or [6101,6102,6103]
    else:
        p=read(ROOT/'protocol.json')
        assert p['mechanism_passed']
        keys=args.keys or [x[1] for x in p['targets']]
        training=args.training_seeds or p['training_seeds']
        proxy=args.proxy_seeds or p['proxy_seeds']
        assert set(keys)<={row[1] for row in p['targets']}
        assert set(training)<=set(p['training_seeds']) and set(proxy)<=set(p['proxy_seeds'])
    folder=ROOT/('mechanism' if mechanism else 'adaptation')
    for key in keys:
        g=graph_for(args,key)
        truth=np.c_[g.edge_index[0],g.edge_type,g.edge_index[1]]
        for ps in proxy:
            split_began=time.perf_counter()
            splitpath=ROOT/'splits'/f'{key}__p{ps}.json'
            if splitpath.exists():
                smeta=read(splitpath)
                assert smeta['graph_sha256']==g.key
                with np.load(splitpath.with_suffix('.npz')) as z:
                    arr={k:z[k].copy() for k in z.files}
            else:
                arr,smeta,_=make_split(g,ps)
                splitpath.parent.mkdir(exist_ok=True)
                np.savez_compressed(splitpath.with_suffix('.npz'),**arr)
                smeta['split_sha256']=sha(splitpath.with_suffix('.npz'))
                smeta['shared_preparation_seconds']=time.perf_counter()-split_began
                dump(splitpath,smeta,exclusive=True)
            # Same query pool and split for all densities and environments.
            select_ids=np.unique(np.concatenate([arr[e+'_selection'] for e in 'ABCD']))
            confirm_ids=np.unique(np.concatenate([arr[e+'_confirmation'] for e in 'ABCD']))
            for seed in training:
                label=f'{key}__p{ps}__s{seed}'
                finalpath=folder/(label+'.json')
                if finalpath.exists():
                    continue
                began=time.perf_counter()
                m=install(load_model(args.stage5,args.history,args.official,'T-4D',seed))
                before=state_hash(m)
                gains=[]
                selection_refs=[]
                for di,rate in enumerate(RATES):
                    rem=residual(g,arr,di)
                    assert rem.key==smeta['mask_metadata'][di]['residual_sha256']
                    rp=folder/'ranks'/f'{label}__d{di}_selection.json'
                    r,record=load_phase(rp,m,rem,arr['facts'][select_ids],truth,ALPHAS,args,'selection')
                    rr=paired_rr(r)
                    env_records={}
                    for env in 'ABCD':
                        index=np.searchsorted(select_ids,arr[env+'_selection'])
                        gain=rr[:,index]-rr[0,index]
                        gains.append([v for v in gain])
                        env_records[env]=dict(metrics=[metrics(v[index]) if len(index) else None for v in r],
                            gains=[lcb(v) for v in gain],query_ids=arr[env+'_selection'].tolist())
                    selection_refs.append(dict(rate=rate,rank_file=str(rp.relative_to(ROOT)),record_sha256=sha(rp),environments=env_records))
                candidates,objectives=choose(gains,[len(row[0]) for row in gains])
                # Stage6 sampler baseline has its own residual and split.
                oldq=arr['legacy_queries']; cut=smeta['legacy']['cut']
                fset=set(map(tuple,oldq))
                keep=np.array([tuple(f) not in fset for f in arr['facts']])
                oldrem=from_forward(arr['facts'][keep],g.n,g.R,{'parent':g.key})
                assert oldrem.key==smeta['legacy']['metadata']['residual_sha256']
                lp=folder/'ranks'/f'{label}__legacy_selection.json'
                lr,_=load_phase(lp,m,oldrem,oldq[:cut],truth,ALPHAS,args,'legacy_selection')
                lrr=paired_rr(lr)
                lv=[float(x.mean()) if len(x) else 0. for x in lrr]
                li=int(np.argmax(lv))
                candidates['stage6_lcb']=dict(candidate_index=li,candidate_alpha=ALPHAS[li],objective=lv)
                candidates['stage6_mean']=dict(candidate_index=li,candidate_alpha=ALPHAS[li],objective=lv)
                # Seal candidate decisions BEFORE evaluating any confirmation.
                planpath=folder/'plans'/(label+'.json')
                plan=dict(candidates=candidates,objectives=objectives,selection=selection_refs,
                          split_sha256=smeta['split_sha256'],checkpoint_sha256=m.tsdc_checkpoint_sha256,
                          selection_only=True,created_utc=datetime.now(timezone.utc).isoformat())
                if planpath.exists():
                    old=read(planpath)
                    assert old['candidates']==candidates
                else:
                    dump(planpath,plan,exclusive=True)
                candidate_alphas=sorted(set([0.]+[v['candidate_alpha'] for k,v in candidates.items() if k not in ('stage6_lcb','stage6_mean','single_mean')]))
                confirmation=[]
                for di,rate in enumerate(RATES):
                    rem=residual(g,arr,di)
                    rp=folder/'ranks'/f'{label}__d{di}_confirmation.json'
                    r,_=load_phase(rp,m,rem,arr['facts'][confirm_ids],truth,candidate_alphas,args,'confirmation_candidate_union')
                    rr=paired_rr(r)
                    for env in 'ABCD':
                        index=np.searchsorted(confirm_ids,arr[env+'_confirmation'])
                        confirmation.append({a:lcb(rr[j,index]-rr[0,index]) for j,a in enumerate(candidate_alphas)})
                lc_alphas=sorted(set([0.,ALPHAS[li]]))
                lp=folder/'ranks'/f'{label}__legacy_confirmation.json'
                lcr,_=load_phase(lp,m,oldrem,oldq[cut:],truth,lc_alphas,args,'legacy_single_candidate_confirmation')
                lcrr=paired_rr(lcr)
                legacy_confirm=lcb(lcrr[lc_alphas.index(ALPHAS[li])]-lcrr[0])
                for name,dec in candidates.items():
                    a=dec['candidate_alpha']
                    if name in ('single_mean','stage6_mean','equal_mean_unconfirmed'):
                        enough=(cut>=24) if name=='stage6_mean' else all(len(row[0])>=24 for row in gains)
                        accepted=enough; checks=[]
                    elif name=='stage6_lcb':
                        checks=[legacy_confirm]; enough=cut>=24
                        accepted=enough and accept(checks)
                    else:
                        checks=[v[a] for v in confirmation]
                        enough=all(len(row[0])>=24 for row in gains)
                        if name=='single_lcb':
                            checks=[checks[4]]
                        elif name in ('four_env_lcb_10','four_env_mean_10'):
                            checks=checks[4:8]
                        elif name=='three_density_A_lcb':
                            checks=[checks[i] for i in [0,4,8]]
                        if name in ('equal_mean','sample_weighted','longtail_weighted','four_env_mean_10'):
                            weights=np.ones(4)/4 if name=='four_env_mean_10' else np.asarray(objectives['weights'][{'equal_mean':'equal','sample_weighted':'sample','longtail_weighted':'longtail'}[name]])
                            combined=dict(n=min(x['n'] for x in checks),mean=float(sum(w*x['mean'] for w,x in zip(weights,checks))),
                                se=float(sum(w*(x['se'] or 0.) for w,x in zip(weights,checks))))
                            combined['lcb']=combined['mean']-1.64*combined['se']
                            checks=[combined]
                        accepted=enough and accept(checks)
                    dec.update(final_alpha=a if accepted else 0.,accepted=bool(accepted),fallback=not accepted,
                               sample_sufficient=bool(enough),confirmation=checks)
                assert before==state_hash(m) and all(v.grad is None for v in m.parameters())
                peak=max((read(path)['info']['peak_allocated'] for path in (folder/'ranks').glob(label+'__*.json')),default=0)
                dump(finalpath,dict(key=key,proxy_seed=ps,training_seed=seed,decisions=candidates,
                    selection_plan_sha256=sha(planpath),selection=selection_refs,objectives=objectives,
                    split_sha256=smeta['split_sha256'],observation_sha256=g.key,
                    checkpoint_sha256=m.tsdc_checkpoint_sha256,seconds=time.perf_counter()-began,peak_allocated=peak,
                    no_weight_update=True,formal_labels_accessed=False),exclusive=True)
                log('adaptation_complete',key=key,proxy_seed=ps,training_seed=seed,
                    decisions={k:v['final_alpha'] for k,v in candidates.items()},seconds=time.perf_counter()-began)
                del m
                gc.collect();torch.cuda.empty_cache()


def run(args):
    if args.task=='mechanism':
        return _run(args)
    from argparse import Namespace
    p=read(ROOT/'protocol.json')
    if (ROOT/'frozen.json').exists():
        frozen=read(ROOT/'frozen.json')
        for label,rec in frozen['selections'].items():
            assert sha(ROOT/'adaptation'/(label+'.json'))==rec['sha256']
        log('already_frozen_no_new_adaptation')
        return
    assert not any((ROOT/'formal_inputs').glob('*_opened.json')),'Formal queries already opened; adaptation is locked.'
    for key in args.keys or [x[1] for x in p['targets']]:
        sub=Namespace(**vars(args));sub.keys=[key]
        try:
            _run(sub)
        except (torch.cuda.OutOfMemoryError,MemoryError) as e:
            import traceback
            failure=dict(key=key,status='resource_failure_after_batch_halving',error=repr(e),
                traceback=traceback.format_exc(),formal_labels_accessed=False)
            dump(ROOT/'failures'/(key+'.json'),failure)
            log('adaptation_resource_failure',key=key,error=repr(e))
        gc.collect();torch.cuda.empty_cache()


def run_validation(args):
    for key in SOURCES:
        g=graph_for(args,key)
        base=args.source_root
        with np.load(base/'data'/(key+'__validation.npz')) as z:
            truth=z['alltrue'].astype(np.int64)
        queries=np.load(args.history/'data'/(key+'_validation_queries.npy'))[:512]
        for seed in args.training_seeds or [1024]:
            path=ROOT/'source_validation'/f'{key}__s{seed}.json'
            if path.exists():continue
            m=install(load_model(args.stage5,args.history,args.official,'T-4D',seed))
            r,info,topo=score(m,g,queries,truth,ALPHAS,args.batch_size)
            save(path,dict(key=key,training_seed=seed,info=info,topology=topo,
                checkpoint_sha256=m.tsdc_checkpoint_sha256,metrics=[metrics(v) for v in r]),r,queries,ALPHAS)
            log('source_validation_complete',key=key,seed=seed,seconds=info['seconds'])
            del m;gc.collect();torch.cuda.empty_cache()
