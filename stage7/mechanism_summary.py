import numpy as np
from stage7.util import ROOT,read,dump,log


def run():
    records=[read(p) for p in (ROOT/'mechanism').glob('*.json')]
    assert len(records)==9,len(records)
    rows=[]
    for r in records:
        support={env:[] for env in 'ABCD'}
        for row in r['selection']:
            for env in support:
                support[env].append([x['mean'] for x in row['environments'][env]['gains']])
        rows.append(dict(key=r['key'],proxy_seed=r['proxy_seed'],
            preferred_alpha_by_environment={env:[float([0,.1,.25,.5,1][np.argmax(v)]) for v in vals] for env,vals in support.items()},
            decisions={k:v['final_alpha'] for k,v in r['decisions'].items()}))
    validation={p.stem:read(p) for p in (ROOT/'source_validation').glob('*.json')}
    assert len(validation)>=3
    predictive=[]
    source_outcomes=[]
    source_metrics=[]
    for r in records:
        vm=validation[r['key']+'__s1024']['metrics']
        from stage7.analysis import random_alpha
        alphas=[0.,.1,.25,.5,1.]
        selected={name:d['final_alpha'] for name,d in r['decisions'].items()}
        selected.update(reference=0.,fixed025=.25,random=random_alpha(r['key'],1024,r['proxy_seed']))
        for name,a in selected.items():
            source_outcomes.append(dict(key=r['key'],proxy_seed=r['proxy_seed'],training_seed=1024,method=name,alpha=a,
                validation_gain=vm[alphas.index(a)]['mrr']-vm[0]['mrr'],**vm[alphas.index(a)]))
        for env in 'ABCD':
            for di,selection in enumerate(r['selection']):
                for a,gain,met in zip(alphas,selection['environments'][env]['gains'],selection['environments'][env]['metrics']):
                    source_metrics.append(dict(key=r['key'],proxy_seed=r['proxy_seed'],training_seed=1024,
                        environment=env,density=[.05,.10,.15][di],alpha=a,proxy_gain=gain['mean'],proxy_lcb=gain['lcb'],
                        facts=gain['n'],**(met or {})))
                gain=selection['environments'][env]['gains'][2]['mean']
                predictive.append(dict(key=r['key'],proxy_seed=r['proxy_seed'],environment=env,density=[.05,.10,.15][di],
                    proxy_alpha025_gain=gain,validation_alpha025_gain=vm[2]['mrr']-vm[0]['mrr']))
    summary=dict(status='passed',records=9,training_seed=1024,source_graphs=3,proxy_seeds=3,
        source_predictive_records=predictive,
        source_outcomes=source_outcomes,
        rows=rows,no_rule_change_from_source_mrr=True,
        caveat='Passing mechanisms validates isolation/numerics, not predictive usefulness. The predeclared target study can falsify the hypothesis.')
    dump(ROOT/'results/mechanism_summary.json',summary)
    from stage7.analysis import write
    from stage7.analysis import bootstrap,holm
    from stage7.confirmation import paired_rr
    validation_ci=[]
    for key in ['wn18rr_train','codex_medium_train','fb15k237_train']:
        with np.load(ROOT/'source_validation'/f'{key}__s1024.npz') as z:rr=paired_rr(z['ranks'])
        for j,a in enumerate([0.,.1,.25,.5,1.]):
            if j:validation_ci.append(dict(key=key,alpha=a,training_seed=1024,**bootstrap(rr[j]-rr[0])))
    holm(validation_ci)
    write('source_validation_methods.csv',source_outcomes)
    write('source_proxy_metrics.csv',source_metrics)
    write('source_proxy_validation_predictions.csv',predictive)
    write('source_validation_bootstrap.csv',validation_ci)
    log('mechanism_summary_complete',records=9)


if __name__=='__main__':run()
