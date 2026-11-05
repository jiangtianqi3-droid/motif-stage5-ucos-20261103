"""Fact-paired analysis; never imported by the selector."""
import csv
import hashlib
from collections import defaultdict,Counter
import numpy as np
from scipy.stats import pearsonr,spearmanr,rankdata
from stage7.util import ROOT,read,dump,sha,log
from stage7.confirmation import paired_rr,lcb

ALPHAS=[0.,.1,.25,.5,1.]


def write(name,rows):
    path=ROOT/'results'/name
    path.parent.mkdir(exist_ok=True)
    if not rows:
        path.write_text('',encoding='utf-8-sig');return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def metrics(r):
    return dict(mrr=float((1/r).mean()),hits_at_1=float((r<=1).mean()),
                hits_at_3=float((r<=3).mean()),hits_at_10=float((r<=10).mean()))


def bootstrap(delta,seed=7197):
    delta=np.asarray(delta,dtype=float)
    if not len(delta):return dict(delta=None,lo=None,hi=None,n=0,p=None)
    if np.ptp(delta)==0:
        d=float(delta[0])
        return dict(delta=d,lo=d,hi=d,n=len(delta),p=1. if d==0 else 1/5001)
    rng=np.random.default_rng(seed)
    draws=[]
    for at in range(0,5000,100):
        draws.extend(delta[rng.integers(len(delta),size=(min(100,5000-at),len(delta)))].mean(1))
    draws=np.asarray(draws)
    lo,hi=np.quantile(draws,[.025,.975])
    # Centered paired bootstrap two-sided null approximation, exploratory.
    p=(1+int((np.abs(draws-delta.mean())>=abs(delta.mean())).sum()))/(len(draws)+1)
    return dict(delta=float(delta.mean()),lo=float(lo),hi=float(hi),n=len(delta),p=float(p))


def holm(rows):
    ordered=sorted([r for r in rows if r.get('p') is not None],key=lambda r:r['p'])
    running=0.
    for i,r in enumerate(ordered):
        running=max(running,min(1.,r['p']*(len(ordered)-i)))
        r['holm_p']=running


def correlation(x,y):
    x,y=np.asarray(x),np.asarray(y)
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:
        return dict(n=len(x),pearson=None,spearman=None,p_pearson=None,p_spearman=None,spearman_p_method='undefined')
    p,pp=pearsonr(x,y);s,sp=spearmanr(x,y)
    method='asymptotic t approximation; exploratory'
    if len(x)<=8:
        from itertools import permutations
        from math import factorial
        rx,ry=rankdata(x),rankdata(y)
        cx,cy=rx-rx.mean(),ry-ry.mean()
        denominator=np.linalg.norm(cx)*np.linalg.norm(cy)
        extreme=sum(abs(float(np.dot(cx,permutation)/denominator))>=abs(float(s))-1e-12
                    for permutation in permutations(cy))
        sp=extreme/factorial(len(x))
        method='exact two-sided label permutation (n!); exploratory'
    return dict(n=len(x),pearson=float(p),spearman=float(s),p_pearson=float(pp),p_spearman=float(sp),spearman_p_method=method)


def random_alpha(key,seed,ps):
    code=int(hashlib.sha256(f'{key}|{seed}|{ps}|7197'.encode()).hexdigest()[:16],16)
    return ALPHAS[int(np.random.default_rng(code).integers(5))]


def run(args):
    import platform,sys,scipy
    dump(ROOT/'results/analysis_environment.json',dict(python=sys.version,platform=platform.platform(),
        numpy=np.__version__,scipy=scipy.__version__,bootstrap_generator='NumPy default_rng/PCG64',
        bootstrap_resamples=5000,bootstrap_seed=7197))
    p=read(ROOT/'protocol.json');frozen=read(ROOT/'frozen.json')
    registered_targets=list(p['targets'])
    unavailable=frozen.get('unavailable_targets',{})
    write('unavailable_targets.csv',[v['record'] for k,v in unavailable.items()])
    p['targets']=[row for row in p['targets'] if row[1] not in unavailable]
    incomplete={}
    for _,key in p['targets']:
        missing=[]
        for family,seeds in [('T-4D',p['training_seeds']),('B0',[1024]),('B1',p['training_seeds']),('B2',p['training_seeds'])]:
            missing +=[f'{family}_s{seed}' for seed in seeds if not (ROOT/'evaluation'/f'{key}__{family}_s{seed}.json').exists()]
        if missing:incomplete[key]=missing
    dump(ROOT/'results/incomplete_formal_targets.json',incomplete)
    p['targets']=[row for row in p['targets'] if row[1] not in incomplete]
    group_of={k:g for g,k in p['targets']}
    methods=p['methods']
    ranks={};raw_metrics=[];timing=[];regression=[];query_identities={}
    for path in (ROOT/'evaluation').glob('*.json'):
        rec=read(path);key=rec['key'];family=rec['family'];seed=rec['training_seed']
        with np.load(path.with_suffix('.npz')) as z:
            r=z['ranks'].copy();queries=z['queries'].copy()
        query_hash=hashlib.sha256(np.ascontiguousarray(queries).tobytes()).hexdigest()
        if key in query_identities:
            assert query_identities[key]['query_sha256']==query_hash,('Model query order mismatch',key)
        else:
            duplicate_rows=len(queries)-len(np.unique(queries,axis=0))
            # Row resampling is fact resampling only when exact triples are unique.
            assert duplicate_rows==0,('Duplicate query rows require cluster bootstrap',key,duplicate_rows)
            query_identities[key]=dict(key=key,queries=len(queries),duplicate_rows=duplicate_rows,
                query_sha256=query_hash,all_model_query_orders_identical=True)
        ranks[key,family,seed]=r
        for j,m in enumerate(rec['metrics']):raw_metrics.append(dict(key=key,group=rec['group'],family=family,training_seed=seed,
            alpha=ALPHAS[j] if family=='T-4D' else (1. if family=='B2' else 0.),**m))
        timing.append(dict(key=key,group=rec['group'],family=family,training_seed=seed,phase='formal',**rec['info']))
        old=ROOT.parent/'stage6/evaluation'/path.name
        if old.exists():
            om=read(old)
            with np.load(old.with_suffix('.npz')) as z:
                oq=z['queries'].copy();orr=z['ranks'].copy()
            assert np.array_equal(queries,oq)
            for j in range(len(r)):
                a=ALPHAS[j] if family=='T-4D' else (1. if family=='B2' else 0.)
                oi=om['configs'].index([a,0.,'conditional'])
                delta=paired_rr(r[j])-paired_rr(orr[oi])
                regression.append(dict(key=key,family=family,training_seed=seed,alpha=a,
                    changed_directions=int((r[j]!=orr[oi]).sum()),old_mrr=float((1/orr[oi]).mean()),
                    new_mrr=float((1/r[j]).mean()),delta_mrr=float(delta.mean()),maximum_fact_rr_change=float(abs(delta).max())))
    write('all_fixed_grid_metrics.csv',raw_metrics);write('numeric_baseline_regression.csv',regression)
    write('query_identity_audit.csv',list(query_identities.values()))
    # Selected methods are literal lookups of immutable choices, no target optimization.
    vectors={};selected=[];decisions=[];stability=[]
    for group,key in p['targets']:
        if (key,'T-4D',1024) not in ranks:continue
        for ps in p['proxy_seeds']:
            sm=read(ROOT/'splits'/f'{key}__p{ps}.json')
            timing.append(dict(key=key,group=group,proxy_seed=ps,phase='shared_proxy_preparation',seconds=sm.get('shared_preparation_seconds')))
        pre=read(ROOT/'results/new_graph_preflight.json').get(key)
        if pre:
            timing.append(dict(key=key,group=group,phase='shared_observation_preflight',seconds=sum(v.get('seconds',0) for v in pre.get('attempts',[])),
                peak_cpu_rss=max(v.get('peak_rss_bytes',0) for v in pre.get('attempts',[]))))
        for seed in p['training_seeds']:
            r=ranks[key,'T-4D',seed]
            for ps in p['proxy_seeds']:
                rec=read(ROOT/'adaptation'/f'{key}__p{ps}__s{seed}.json')
                for method in methods:
                    a=0. if method=='reference' else .25 if method=='fixed025' else random_alpha(key,seed,ps) if method=='random' else rec['decisions'][method]['final_alpha']
                    v=r[ALPHAS.index(a)]
                    vectors[key,method,seed,ps]=paired_rr(v)
                    selected.append(dict(key=key,group=group,method=method,training_seed=seed,proxy_seed=ps,alpha=a,**metrics(v)))
                for name,d in rec['decisions'].items():
                    reason=('insufficient_samples' if not d['sample_sufficient'] else
                        'reference_won_selection' if d['candidate_alpha']==0 else
                        'confirmation_rejected' if d['fallback'] else
                        'enabled_without_confirmation' if not d['confirmation'] else 'enabled_after_confirmation')
                    decisions.append(dict(key=key,group=group,training_seed=seed,proxy_seed=ps,method=name,
                        candidate_alpha=d['candidate_alpha'],final_alpha=d['final_alpha'],fallback=d['fallback'],
                        accepted=d['accepted'],sample_sufficient=d['sample_sufficient'],decision_reason=reason,
                        worst_confirmation_lcb=min([v['lcb'] for v in d['confirmation'] if v['lcb'] is not None],default=None)))
                timing.append(dict(key=key,group=group,family='T-4D',training_seed=seed,proxy_seed=ps,phase='adaptation',seconds=rec['seconds'],peak_allocated=rec.get('peak_allocated')))
            for method in methods:
                aa=[v['alpha'] for v in selected if v['key']==key and v['training_seed']==seed and v['method']==method]
                counts=Counter(aa)
                stability.append(dict(key=key,group=group,training_seed=seed,method=method,distinct_alphas=len(counts),
                    modal_fraction=max(counts.values())/len(aa),selected_alphas=str(aa)))
        for family in ['B0','B1','B2']:
            for seed in p['training_seeds']:
                v=ranks[key,family,1024 if family=='B0' else seed][0]
                for ps in p['proxy_seeds']:
                    vectors[key,family,seed,ps]=paired_rr(v)
                if family!='B0' or seed==1024:
                    selected.append(dict(key=key,group=group,method=family,training_seed=seed,proxy_seed=6101,alpha=None,**metrics(v)))
    write('selection_decisions.csv',decisions);write('selection_stability.csv',stability)
    write('per_seed_metrics.csv',selected);write('time_memory.csv',timing)
    proxy_seed_metrics=[];proxy_seed_macro=[]
    for group,key in p['targets']:
        for method in methods:
            for ps in p['proxy_seeds']:
                rows=[v for v in selected if v['key']==key and v['method']==method and v['proxy_seed']==ps]
                if not rows:continue
                proxy_seed_metrics.append(dict(key=key,group=group,method=method,proxy_seed=ps,
                    mrr=float(np.mean([v['mrr'] for v in rows])),mrr_training_sd=float(np.std([v['mrr'] for v in rows],ddof=1))))
    for group in list(dict.fromkeys(group_of.values()))+['historical_all11']:
        for method in methods:
            for ps in p['proxy_seeds']:
                rows=[v for v in proxy_seed_metrics if v['method']==method and v['proxy_seed']==ps and
                    (v['group']==group or group=='historical_all11' and v['group']!='new_prospective')]
                if rows:proxy_seed_macro.append(dict(group=group,method=method,proxy_seed=ps,graphs=len(rows),
                    mrr=float(np.mean([v['mrr'] for v in rows]))))
    write('proxy_seed_metrics.csv',proxy_seed_metrics);write('macro_proxy_seed_metrics.csv',proxy_seed_macro)
    graph_metrics=[];comparisons=[];graph_vectors={}
    all_methods=methods+['B0','B1','B2']
    for group,key in p['targets']:
        if (key,'reference',1024,6101) not in vectors:continue
        for method in all_methods:
            v=np.stack([vectors[key,method,s,6101] for s in p['training_seeds']])
            graph_vectors[key,method]=v.mean(0)
            rows=[r for r in selected if r['key']==key and r['method']==method and r['proxy_seed']==6101]
            metric={name:float(np.mean([r[name] for r in rows])) for name in ['mrr','hits_at_1','hits_at_3','hits_at_10']}
            metric['mrr_sd']=float(np.std([r['mrr'] for r in rows],ddof=1)) if len(rows)>1 else None
            differences={b:float(v.mean()-np.mean([vectors[key,b,s,6101].mean() for s in p['training_seeds']])) for b in ['B0','B1','B2']}
            graph_metrics.append(dict(key=key,group=group,method=method,training_seeds=len(rows),**metric,
                **{'delta_'+b:d for b,d in differences.items()}))
            if method not in ['B0','B1','B2']:
                references=['reference','stage6_lcb','single_lcb','fixed025','random','B0','B1','B2'] if method=='robust_lcb' else ['reference']
                for reference in references:
                    if method==reference:continue
                    ref=np.stack([vectors[key,reference,s,6101] for s in p['training_seeds']]).mean(0)
                    comparisons.append(dict(key=key,group=group,method=method,reference=reference,
                        primary=method=='robust_lcb' and reference=='stage6_lcb',**bootstrap(v.mean(0)-ref)))
    holm([r for r in comparisons if r['primary']])
    write('P0_metrics.csv',graph_metrics);write('paired_bootstrap.csv',comparisons)
    macro=[];macro_ci=[]
    new_meta=read(ROOT/'observations/manifest.json')
    groups=list(dict.fromkeys(group_of.values()))+['historical_all11','new_official_inductive_or_ookb','new_external_zero_shot']
    for group in groups:
        keys=[k for k,g in group_of.items() if (g==group or group=='historical_all11' and g!='new_prospective' or
              group=='new_official_inductive_or_ookb' and g=='new_prospective' and new_meta[k]['type'].startswith('inductive') or
              group=='new_external_zero_shot' and g=='new_prospective' and new_meta[k]['type'].startswith('cross_graph')) and (k,'reference') in graph_vectors]
        for method in all_methods:
            rows=[r for r in graph_metrics if r['key'] in keys and r['method']==method]
            if not rows:continue
            byseed=[]
            for s in p['training_seeds']:
                byseed.append(np.mean([vectors[k,method,s,6101].mean() for k in keys]))
            macro.append(dict(group=group,method=method,graphs=len(keys),mrr=float(np.mean([r['mrr'] for r in rows])),
                mrr_training_sd=float(np.std(byseed,ddof=1)) if method!='B0' else None,hits_at_1=float(np.mean([r['hits_at_1'] for r in rows])),
                hits_at_3=float(np.mean([r['hits_at_3'] for r in rows])),hits_at_10=float(np.mean([r['hits_at_10'] for r in rows]))))
        for reference in ['reference','stage6_lcb','single_lcb','fixed025','random','B0','B1','B2']:
            if not keys:continue
            ds=[graph_vectors[k,'robust_lcb']-graph_vectors[k,reference] for k in keys]
            rng=np.random.default_rng(7197);draw=np.zeros(5000)
            for d in ds:
                for at in range(0,5000,100):
                    draw[at:at+100]+=d[rng.integers(len(d),size=(100,len(d)))].mean(1)/len(ds)
            lo,hi=np.quantile(draw,[.025,.975])
            macro_ci.append(dict(group=group,method='robust_lcb',reference=reference,graphs=len(keys),
                 delta=float(np.mean([d.mean() for d in ds])),lo=float(lo),hi=float(hi),bootstrap='5000 graph-stratified fact resamples; seeds averaged'))
    write('macro_metrics.csv',macro);write('macro_paired_bootstrap.csv',macro_ci)
    analyze_proxies(p,graph_vectors)
    analyze_selected_predictions(p)
    analyze_distribution(args,p)
    analyze_groups(p,graph_vectors)
    analyze_new_quality(p,graph_vectors)
    analyze_old_selection_replay(p)
    analyze_alias_repeats(p)
    analyze_oracle_ceiling(p,graph_vectors)
    from stage7.constraint_bias import run as audit_constraints
    audit_constraints(args.observations)
    dump(ROOT/'results/analysis_status.json',dict(status='complete',graphs=len({r['key'] for r in graph_metrics}),
         registered_graphs=len(registered_targets),incomplete_formal_targets=incomplete,unavailable_targets=list(unavailable),
         formal_runs=len(ranks),frozen_sha256=sha(ROOT/'frozen.json'),primary_proxy_seed=6101,
         no_score_ensemble=True,numpy=np.__version__,bootstrap_resamples=5000,bootstrap_seed=7197,
         interval_scope='Conditional on graphs, trained checkpoints and frozen primary-proxy choices; not adaptation-seed or independent-KB-source uncertainty'))
    log('analysis_complete',graphs=len(group_of),formal_runs=len(ranks))


def analyze_proxies(p,graph_vectors):
    proxy=[];density=[];dominance=[];preferences=[];predictions=[];confirmation=[]
    store=defaultdict(list);legacy=defaultdict(list);physical_graph_counts={}
    for group,key in p['targets']:
        for seed in p['training_seeds']:
            for ps in p['proxy_seeds']:
                path=ROOT/'adaptation'/f'{key}__p{ps}__s{seed}.json'
                if not path.exists():continue
                rec=read(path)
                with np.load(ROOT/'adaptation/ranks'/f'{key}__p{ps}__s{seed}__legacy_selection.npz') as z:
                    lrr=paired_rr(z['ranks'])
                for j,a in enumerate(ALPHAS):legacy[key,a].append(float((lrr[j]-lrr[0]).mean()) if lrr.shape[1] else 0.)
                sm=read(ROOT/'splits'/f'{key}__p{ps}.json')
                physical_graph_counts[key,ps]=len({v['residual_sha256'] for v in sm['mask_metadata']})
                for di,s in enumerate(rec['selection']):
                    for env,e in s['environments'].items():
                        for j,(a,gain,met) in enumerate(zip(ALPHAS,e['gains'],e['metrics'])):
                            row=dict(key=key,group=group,training_seed=seed,proxy_seed=ps,density=s['rate'],environment=env,alpha=a,
                                actual_density=sm['mask_metadata'][di]['actual_rate'],hidden=sm['mask_metadata'][di]['hidden'],
                                selection_n=gain['n'],gain=gain['mean'],gain_se=gain['se'],gain_lcb=gain['lcb'],**(met or {}))
                            proxy.append(row);store[key,env,di,a].append(gain['mean'])
                        preferences.append(dict(key=key,training_seed=seed,proxy_seed=ps,density=s['rate'],environment=env,
                            best_alpha=ALPHAS[int(np.argmax([v['mean'] for v in e['gains']]))]))
                    cp=ROOT/'adaptation/ranks'/f'{key}__p{ps}__s{seed}__d{di}_confirmation.npz'
                    with np.load(cp) as z:
                        cr=z['ranks'].copy();cq=z['queries'].copy();ca=z['alphas'].copy()
                    with np.load(ROOT/'splits'/f'{key}__p{ps}.npz') as z:
                        facts=z['facts'].copy();ids={e:z[e+'_confirmation'].copy() for e in 'ABCD'}
                    lookup={tuple(q):i for i,q in enumerate(cq)}
                    crr=paired_rr(cr)
                    for env in 'ABCD':
                        order=np.asarray([lookup[tuple(q)] for q in facts[ids[env]]],dtype=int)
                        for j,a in enumerate(ca):
                            confirmation.append(dict(key=key,group=group,training_seed=seed,proxy_seed=ps,density=s['rate'],
                                environment=env,alpha=a,phase='confirmation_frozen_candidate_union',
                                **(metrics(cr[j,order]) if len(order) else {}),
                                **lcb(crr[j,order]-crr[0,order])))
                lower=np.array(rec['objectives']['environment_lcbs'])
                ci=rec['decisions']['robust_lcb']['candidate_index']
                nonzero=int(np.argmax(lower[:,1:].min(0)))+1
                winners=np.argmax(lower.min(0))
                leave=[ALPHAS[int(np.argmax(np.delete(lower,at,axis=0).min(0)))] for at in range(12)]
                dominance.append(dict(key=key,training_seed=seed,proxy_seed=ps,candidate_alpha=ALPHAS[ci],
                    worst_environment_index=int(np.argmin(lower[:,ci])) if ci else None,
                    best_nonzero_alpha=ALPHAS[nonzero],best_nonzero_worst_lcb=float(lower[:,nonzero].min()),
                    nonzero_blocking_environments=int((lower[:,nonzero]<=0).sum()),
                    nonzero_worst_environment_index=int(np.argmin(lower[:,nonzero])),
                    leave_one_environment_changes=sum(a!=ALPHAS[ci] for a in leave),leave_one_out_alphas=str(leave)))
        # Paired density comparison on the same seed6101 fact queries; average training seeds first.
        for env in 'ABCD':
            values=[]
            for di in range(3):
                seed_values=[]
                for seed in p['training_seeds']:
                    path=ROOT/'adaptation/ranks'/f'{key}__p6101__s{seed}__d{di}_selection.npz'
                    if not path.exists():continue
                    with np.load(path) as z:r=z['ranks'].copy();q=z['queries'].copy()
                    with np.load(ROOT/'splits'/f'{key}__p6101.npz') as z:
                        facts=z['facts'].copy();ix=z[env+'_selection'].copy()
                    lookup={tuple(v):i for i,v in enumerate(q)}
                    order=np.array([lookup[tuple(v)] for v in facts[ix]],dtype=int)
                    rr=paired_rr(r)
                    seed_values.append(rr[2,order]-rr[0,order])
                values.append(np.mean(seed_values,axis=0) if seed_values else np.array([]))
            if all(len(v) for v in values):
                for di in [1,2]:density.append(dict(key=key,group=group,environment=env,density=[.05,.10,.15][di],reference_density=.05,
                    alpha=.25,**bootstrap(values[di]-values[0])))
        if (key,'reference') in graph_vectors:
            for a in ALPHAS[1:]:
                formal=[]
                for seed in p['training_seeds']:
                    with np.load(ROOT/'evaluation'/f'{key}__T-4D_s{seed}.npz') as z:
                        r=z['ranks'].copy()
                    rr=paired_rr(r)
                    formal.append((rr[ALPHAS.index(a)]-rr[0]).mean())
                val=float(np.mean(formal))
                predictions.append(dict(key=key,group=group,alpha=a,estimator='stage6_legacy',density='capped15%',
                    proxy_gain=float(np.mean(legacy[key,a])),formal_gain=val))
                for di,rate in enumerate([.05,.10,.15]):
                    for env in 'ABCD':
                        predictions.append(dict(key=key,group=group,alpha=a,estimator=env,density=rate,
                            proxy_gain=float(np.mean(store[key,env,di,a])),formal_gain=val))
                for estimator in ['equal_mean','worst_mean','worst_lcb']:
                    gg=[float(np.mean(store[key,e,di,a])) for di in range(3) for e in 'ABCD']
                    if estimator=='equal_mean':estimate=float(np.mean(gg))
                    elif estimator=='worst_mean':estimate=float(np.min(gg))
                    else:
                        worst=[]
                        for seed in p['training_seeds']:
                            for ps in p['proxy_seeds']:
                                rec=read(ROOT/'adaptation'/f'{key}__p{ps}__s{seed}.json')
                                vv=np.asarray(rec['objectives']['environment_lcbs'])[:,ALPHAS.index(a)]
                                if vv.min()>-1e20:worst.append(float(vv.min()))
                        estimate=float(np.mean(worst)) if worst else 0.
                    predictions.append(dict(key=key,group=group,alpha=a,estimator=estimator,density='all',proxy_gain=estimate,formal_gain=val))
    holm(density)
    write('proxy_metrics.csv',proxy);write('proxy_confirmation_metrics.csv',confirmation)
    write('density_sensitivity.csv',density);write('environment_preferences.csv',preferences)
    write('environment_dominance.csv',dominance);write('proxy_formal_prediction.csv',predictions)
    # Descriptive disagreements, not independent repeated hypothesis tests.
    from itertools import combinations
    env_groups=defaultdict(list);density_groups=defaultdict(list)
    gain_lookup={(r['key'],r['training_seed'],r['proxy_seed'],r['environment'],r['density']):r['gain']
                 for r in proxy if r['alpha']==.25}
    for r in preferences:
        env_groups[r['key'],r['training_seed'],r['proxy_seed'],r['density']].append(r)
        density_groups[r['key'],r['training_seed'],r['proxy_seed'],r['environment']].append(r)
    agreement=[];density_agreement=[]
    for (key,seed,ps,rate),data in env_groups.items():
        data=sorted(data,key=lambda r:r['environment']);aa=[r['best_alpha'] for r in data]
        assert len(data)==4
        agreement.append(dict(key=key,training_seed=seed,proxy_seed=ps,density=rate,
            **{'alpha_'+r['environment']:r['best_alpha'] for r in data},distinct_preferences=len(set(aa)),
            pairwise_agreement=float(np.mean([a==b for a,b in combinations(aa,2)]))))
    for (key,seed,ps,env),data in density_groups.items():
        data=sorted(data,key=lambda r:r['density']);aa=[r['best_alpha'] for r in data]
        gg=[gain_lookup[key,seed,ps,env,r['density']] for r in data]
        assert len(data)==3
        density_agreement.append(dict(key=key,training_seed=seed,proxy_seed=ps,environment=env,
            **{'alpha_'+str(r['density']):r['best_alpha'] for r in data},distinct_preferences=len(set(aa)),
            pairwise_agreement=float(np.mean([a==b for a,b in combinations(aa,2)])),
            alpha025_gain_min=float(min(gg)),alpha025_gain_max=float(max(gg)),
            distinct_residual_graphs=physical_graph_counts[key,ps],
            alpha025_sign_changes=min(gg)<0<max(gg)))
    write('environment_parameter_agreement.csv',agreement)
    write('density_parameter_agreement.csv',density_agreement)
    correlations=[]
    for estimator,dens in list(dict.fromkeys((r['estimator'],r['density']) for r in predictions)):
        for scope in ['historical_all11','new_prospective','all','historical_without_ConceptNet','all_without_ConceptNet','all_without_UMLS','all_without_ConceptNet_and_UMLS']:
            rows=[r for r in predictions if r['alpha']==.25 and r['estimator']==estimator and r['density']==dens and
                (scope=='all' or scope=='new_prospective' and r['group']=='new_prospective' or
                 scope=='historical_all11' and r['group']!='new_prospective' or
                 scope=='historical_without_ConceptNet' and r['group']!='new_prospective' and r['key']!='ConceptNet100k_external' or
                 scope=='all_without_ConceptNet' and r['key']!='ConceptNet100k_external' or
                 scope=='all_without_UMLS' and r['key']!='UMLS_external_new' or
                 scope=='all_without_ConceptNet_and_UMLS' and r['key'] not in ['ConceptNet100k_external','UMLS_external_new'])]
            correlations.append(dict(scope=scope,estimator=estimator,density=dens,alpha=.25,
                **correlation([r['proxy_gain'] for r in rows],[r['formal_gain'] for r in rows])))
    write('correlations.csv',correlations)
    influence=[]
    for estimator,dens in list(dict.fromkeys((r['estimator'],r['density']) for r in predictions)):
        data=[r for r in predictions if r['alpha']==.25 and r['estimator']==estimator and r['density']==dens]
        full=correlation([r['proxy_gain'] for r in data],[r['formal_gain'] for r in data])
        for omitted in data:
            subset=[r for r in data if r['key']!=omitted['key']]
            result=correlation([r['proxy_gain'] for r in subset],[r['formal_gain'] for r in subset])
            influence.append(dict(estimator=estimator,density=dens,alpha=.25,omitted_key=omitted['key'],
                full_pearson=full['pearson'],full_spearman=full['spearman'],**result))
    write('correlation_influence.csv',influence)


def analyze_distribution(args,p):
    from scipy.spatial.distance import jensenshannon
    from stage6.observed_loader import load_observed
    rows=[];overlap=[];masks=[];topology=[];strata=[]
    for group,key in p['targets']:
        file=ROOT/'evaluation'/f'{key}__T-4D_s1024.npz'
        if not file.exists():continue
        graph=load_observed(ROOT/'observations' if key.endswith('_new') else args.observations,key)
        facts=graph.forward
        with np.load(file) as z:formal=z['queries'].copy()
        rf=np.bincount(facts[:,1],minlength=graph.R//2)
        fr=np.bincount(formal[:,1],minlength=graph.R//2)
        deg=np.bincount(facts[:,[0,2]].ravel(),minlength=graph.n)
        sampling=read(ROOT/'splits'/f'{key}__p6101.json')['description']
        motif=np.asarray(sampling['motif_support'])
        assert len(motif)==graph.R//2
        degree_bins=np.searchsorted(sampling['degree_bounds'],deg,side='left')
        motif_bins=np.searchsorted(sampling['motif_bounds'],motif,side='left')
        formal_degree=np.bincount(degree_bins[formal[:,[0,2]]].ravel(),minlength=3)
        formal_motif=np.bincount(motif_bins[formal[:,1]],minlength=3)
        def features(q):
            if not len(q):return dict(relation_coverage=0.,mean_log_endpoint_degree=None,mean_log_relation_frequency=None,js_to_observed=None,js_to_formal=None)
            counts=np.bincount(q[:,1],minlength=graph.R//2)
            return dict(relation_coverage=float(((counts>0)&(rf>0)).sum()/max((rf>0).sum(),1)),
                queries_on_unobserved_relation_fraction=float(counts[rf==0].sum()/counts.sum()),
                mean_log_endpoint_degree=float(np.log1p(deg[q[:,[0,2]]]).mean()),
                mean_log_relation_frequency=float(np.log1p(rf[q[:,1]]).mean()),
                mean_log_relation_motif_support=float(np.log1p(motif[q[:,1]]).mean()),
                low_motif_support_fraction=float((motif_bins[q[:,1]]==0).mean()),
                high_motif_support_fraction=float((motif_bins[q[:,1]]==2).mean()),
                zero_motif_support_fraction=float((motif[q[:,1]]==0).mean()),
                js_endpoint_degree_to_formal=float(jensenshannon(np.bincount(degree_bins[q[:,[0,2]]].ravel(),minlength=3),formal_degree)),
                js_motif_support_to_formal=float(jensenshannon(np.bincount(motif_bins[q[:,1]],minlength=3),formal_motif)),
                js_to_observed=float(jensenshannon(counts,rf)),js_to_formal=float(jensenshannon(counts,fr)))
        for ps in p['proxy_seeds']:
            with np.load(ROOT/'splits'/f'{key}__p{ps}.npz') as z:arr={k:z[k].copy() for k in z.files}
            sm=read(ROOT/'splits'/f'{key}__p{ps}.json')
            for env,stats in sm['description'].get('stratum_counts',{}).items():
                nominal={'A':1,'B':int((rf>0).sum()),'C':9,'D':3}[env]
                strata.append(dict(key=key,proxy_seed=ps,environment=env,nominal_strata=nominal,
                    available_strata=len(stats['available']),selection_strata=len(stats['selection']),confirmation_strata=len(stats['confirmation']),
                    available_counts=str(stats['available']),selection_counts=str(stats['selection']),confirmation_counts=str(stats['confirmation'])))
            for name,q in [('observed',facts),('formal',formal),('physical5_hidden',facts[arr['hidden0']]),('stage6_legacy',arr['legacy_queries'])]+[(e,np.r_[facts[arr[e+'_selection']],facts[arr[e+'_confirmation']]]) for e in 'ABCD']:
                rows.append(dict(key=key,group=group,proxy_seed=ps,distribution=name,n=len(q),**features(q)))
            for a in 'ABCD':
                for b in 'ABCD':
                    if a>=b:continue
                    sa=set(arr[a+'_selection']);sb=set(arr[b+'_selection'])
                    overlap.append(dict(key=key,proxy_seed=ps,environment_a=a,environment_b=b,
                        selection_jaccard=len(sa&sb)/max(len(sa|sb),1),shared_facts=len(sa&sb),size_a=len(sa),size_b=len(sb)))
            for di,meta in enumerate(sm['mask_metadata']):
                masks.append(dict(key=key,group=group,proxy_seed=ps,density=meta['rate'],desired=meta['desired'],hidden=meta['hidden'],
                    actual_rate=meta['actual_rate'],added=meta['added'],hidden_sha256=meta['hidden_sha256'],
                    residual_sha256=meta['residual_sha256'],selection_per_environment=sm['description']['selection_per_environment'],
                    confirmation_per_environment=sm['description']['confirmation_per_environment']))
                record=read(ROOT/'adaptation/ranks'/f'{key}__p{ps}__s1024__d{di}_selection.json')
                info=record.get('topology',{})
                if info:
                    with np.load(ROOT/'cache/topology'/(info['topology_key']+'.npz')) as z:
                        raw=z['raw'].copy();ht=z['edge_type'].copy()
                    topology.append(dict(key=key,proxy_seed=ps,density=meta['rate'],actual_rate=meta['actual_rate'],
                        hyperedges=len(ht),mean_log_motif_count=float(raw[:,0].mean()),mean_rw_support=float(raw[:,2].mean()),
                        mean_exclusivity=float(raw[:,3].mean()),topology_key=info['topology_key']))
    write('proxy_formal_distribution.csv',rows);write('environment_overlap.csv',overlap)
    write('mask_rates.csv',masks);write('topology_density_drift.csv',topology)
    write('environment_strata_coverage.csv',strata)


def analyze_groups(p,graph_vectors):
    rows=[]
    for group,key in p['targets']:
        maskpath=ROOT.parent/'analysis'/f'query_masks_{key}.npz'
        if not maskpath.exists() or (key,'robust_lcb') not in graph_vectors:continue
        with np.load(maskpath) as z:masks={name:z[name].copy().astype(bool) for name in ['high_degree_hub','low_relation_frequency','low_incident_motif_support']}
        for name,mask in masks.items():
            # Archived masks may be directional or one-per-forward-fact.
            if mask.ndim==2:
                factmask=mask.any(axis=1)
            elif len(mask)==2*len(graph_vectors[key,'reference']):
                factmask=mask.reshape(-1,2).any(1)
            else:factmask=mask
            assert len(factmask)==len(graph_vectors[key,'reference'])
            for method in ['robust_lcb','stage6_lcb','single_lcb','equal_mean','fixed025']:
                delta=graph_vectors[key,method][factmask]-graph_vectors[key,'reference'][factmask]
                rows.append(dict(key=key,group=group,subgroup=name,definition='Archived Stage5 mask; fact union if directional; final evaluation only',method=method,
                    mrr=float(graph_vectors[key,method][factmask].mean()) if factmask.any() else None,**bootstrap(delta)))
    write('archived_query_groups.csv',rows)


def analyze_selected_predictions(p):
    rows=[]
    for group,key in p['targets']:
        for seed in p['training_seeds']:
            file=ROOT/'evaluation'/f'{key}__T-4D_s{seed}.npz'
            if not file.exists():continue
            with np.load(file) as z:rr=paired_rr(z['ranks'])
            for ps in p['proxy_seeds']:
                rec=read(ROOT/'adaptation'/f'{key}__p{ps}__s{seed}.json')
                for method,d in rec['decisions'].items():
                    ci=ALPHAS.index(d['candidate_alpha']);fi=ALPHAS.index(d['final_alpha'])
                    estimate=d['objective'][ci]-d['objective'][0]
                    rows.append(dict(key=key,group=group,training_seed=seed,proxy_seed=ps,method=method,
                        candidate_alpha=d['candidate_alpha'],final_alpha=d['final_alpha'],sample_sufficient=d['sample_sufficient'],
                        proxy_candidate_objective_gain=estimate,formal_candidate_gain=float((rr[ci]-rr[0]).mean()),
                        formal_final_gain=float((rr[fi]-rr[0]).mean())))
    write('selected_prediction.csv',rows)
    graphrows=[];cs=[]
    for key in dict.fromkeys(r['key'] for r in rows):
        for method in dict.fromkeys(r['method'] for r in rows):
            data=[r for r in rows if r['key']==key and r['method']==method]
            if not data:continue
            graphrows.append(dict(key=key,group=data[0]['group'],method=method,
                nonzero_candidate_fraction=float(np.mean([r['candidate_alpha']!=0 for r in data])),
                nonzero_final_fraction=float(np.mean([r['final_alpha']!=0 for r in data])),
                proxy_candidate_objective_gain=float(np.mean([r['proxy_candidate_objective_gain'] for r in data])),
                formal_candidate_gain=float(np.mean([r['formal_candidate_gain'] for r in data])),
                formal_final_gain=float(np.mean([r['formal_final_gain'] for r in data])),
                sample_sufficient_fraction=float(np.mean([r['sample_sufficient'] for r in data]))))
    for method in dict.fromkeys(r['method'] for r in graphrows):
        for scope in ['all','historical_all11','new_prospective','historical_without_ConceptNet']:
            data=[r for r in graphrows if r['method']==method and (scope=='all' or
                scope=='historical_all11' and r['group']!='new_prospective' or scope=='new_prospective' and r['group']=='new_prospective' or
                scope=='historical_without_ConceptNet' and r['group']!='new_prospective' and r['key']!='ConceptNet100k_external')]
            cs.append(dict(method=method,scope=scope,nonzero_candidate_graphs=sum(r['nonzero_candidate_fraction']>0 for r in data),
                **correlation([r['proxy_candidate_objective_gain'] for r in data],[r['formal_candidate_gain'] for r in data])))
    write('selected_graph_prediction.csv',graphrows);write('selected_correlations.csv',cs)


def analyze_new_quality(p,graph_vectors):
    from stage7.named_overlap import canon,parse
    from stage7.util import runtime_path
    from stage6.observed_loader import load_observed
    source=set()
    for rec in read(ROOT/'results/pretraining_observation_overlap.json')['sources']:
        file=runtime_path(rec['path'])
        assert sha(file)==rec['sha256']
        source |=parse(file.read_text(encoding='utf8'))
    rows=[];subgroups=[]
    for group,key in p['targets']:
        if group!='new_prospective' or (key,'reference') not in graph_vectors:continue
        graph=load_observed(ROOT/'observations',key)
        vocab=read(ROOT/'observations'/(key+'_vocab.json'))
        entities={v:k for k,v in vocab['entities'].items()};relations={v:k for k,v in vocab['relations'].items()}
        with np.load(ROOT/'evaluation'/f'{key}__T-4D_s1024.npz') as z:q=z['queries'].copy()
        named=[canon((entities[h],relations[r],entities[t])) for h,r,t in q]
        overlap=np.asarray([v in source for v in named],dtype=bool)
        obs=set(map(tuple,graph.forward));pairs={(h,t) for h,r,t in obs}
        rows.append(dict(key=key,test_facts=len(q),duplicate_test_facts=len(q)-len(set(map(tuple,q))),
            direct_observation_matches=sum(tuple(v) in obs for v in q),
            reverse_same_predicate_observation_matches=sum((t,r,h) in obs for h,r,t in q),
            reverse_any_predicate_observation_matches=sum((t,h) in pairs for h,r,t in q),
            syntactic_pretraining_test_fact_matches=int(overlap.sum()),
            status='Post-freeze quality audit; no rows removed or parameters changed',
            caveat='Syntactic overlap only; opaque IDs and semantic KB alignments cannot prove independence.'))
        for name,mask in [('not_syntactically_pretrained',~overlap),('syntactically_pretrained',overlap)]:
            for method in ['reference','fixed025','stage6_lcb','single_lcb','robust_lcb']:
                delta=graph_vectors[key,method][mask]-graph_vectors[key,'reference'][mask]
                subgroups.append(dict(key=key,subgroup=name,method=method,**bootstrap(delta)))
    write('new_graph_quality.csv',rows);write('new_pretraining_overlap_subgroups.csv',subgroups)


def analyze_old_selection_replay(p):
    rows=[]
    for group,key in p['targets']:
        if group=='new_prospective':continue
        for seed in p['training_seeds']:
            file=ROOT/'evaluation'/f'{key}__T-4D_s{seed}.npz'
            if not file.exists():continue
            with np.load(file) as z:r=z['ranks'].copy()
            for ps in p['proxy_seeds']:
                old=ROOT.parent/'stage6/adaptation'/f'{key}__rate0.15__p{ps}__s{seed}.json'
                rec=read(old)
                old_decision=rec['decisions']['topology_lcb']
                a=old_decision['selected'][0]
                current=read(ROOT/'adaptation'/f'{key}__p{ps}__s{seed}.json')['decisions']['stage6_lcb']
                rows.append(dict(key=key,group=group,training_seed=seed,proxy_seed=ps,old_alpha=a,
                    recomputed_alpha=current['final_alpha'],selection_changed=a!=current['final_alpha'],
                    old_frozen_alpha_new_numeric_mrr=float((1/r[ALPHAS.index(a)]).mean()),
                    recomputed_selection_new_numeric_mrr=float((1/r[ALPHAS.index(current['final_alpha'])]).mean()),
                    old_record_sha256=sha(old)))
    write('stage6_frozen_selection_replay.csv',rows)


def analyze_alias_repeats(p):
    rows=[]
    for _,key in p['targets']:
        for ps in p['proxy_seeds']:
            meta=read(ROOT/'splits'/f'{key}__p{ps}.json')['mask_metadata']
            for a in range(3):
                for b in range(a+1,3):
                    if meta[a]['residual_sha256']!=meta[b]['residual_sha256']:continue
                    for seed in p['training_seeds']:
                        prefix=ROOT/'adaptation/ranks'/f'{key}__p{ps}__s{seed}'
                        with np.load(str(prefix)+f'__d{a}_selection.npz') as z:ra=z['ranks'].copy();qa=z['queries'].copy()
                        with np.load(str(prefix)+f'__d{b}_selection.npz') as z:rb=z['ranks'].copy();qb=z['queries'].copy()
                        assert np.array_equal(qa,qb)
                        for j,alpha in enumerate(ALPHAS):
                            rows.append(dict(key=key,proxy_seed=ps,training_seed=seed,alpha=alpha,
                                density_a=[.05,.10,.15][a],density_b=[.05,.10,.15][b],
                                identical_residual_sha256=meta[a]['residual_sha256'],directional_rank_changes=int((ra[j]!=rb[j]).sum()),
                                delta_mrr=float((1/ra[j]-1/rb[j]).mean()) if ra[j].size else 0.))
    write('aliased_graph_repeatability.csv',rows)


def analyze_oracle_ceiling(p,graph_vectors):
    rows=[]
    for group,key in p['targets']:
        for seed in p['training_seeds']:
            path=ROOT/'evaluation'/f'{key}__T-4D_s{seed}.npz'
            if not path.exists():continue
            with np.load(path) as z:rr=paired_rr(z['ranks'])
            means=rr.mean(1);best=int(np.argmax(means))
            frozen=read(ROOT/'adaptation'/f'{key}__p6101__s{seed}.json')['decisions']
            ra=ALPHAS.index(frozen['robust_lcb']['final_alpha']);sa=ALPHAS.index(frozen['stage6_lcb']['final_alpha'])
            rows.append(dict(key=key,group=group,training_seed=seed,posthoc_oracle_alpha=ALPHAS[best],
                posthoc_oracle_mrr=float(means[best]),reference_mrr=float(means[0]),
                oracle_gain_over_reference=float(means[best]-means[0]),
                robust_selection_regret=float(means[best]-means[ra]),stage6_selection_regret=float(means[best]-means[sa]),
                warning='Uses formal labels after freeze: optimistic diagnostic upper bound, never a deployable selector or retuned parameter.'))
    write('posthoc_grid_ceiling.csv',rows)
