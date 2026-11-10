"""Independent checks of completed route-A rank arrays and recorded metrics."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/d/motif-phase4-work/route-a-rerank-20261109')
RUN=ROOT/'results/runs/f20ab88ff89d82d8506c'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

result={'passed':True,'groups':[],'query_evaluations':0,'max_peak_rss_bytes':0}
assert read(RUN/'completion.json')['state']=='complete'
for key in ['FB15k237Inductive_v1','WN18RRInductive_v1','NELLInductive_v1','FBIngram_25']:
    for family in ['B0','T-4D']:
        work=RUN/key/family
        summary=read(work/'summary.json'); done=read(work/'completion.json')
        audit=read(work/'model_audit.json'); split=read(work/'split_leakage_audit.json')
        assert done['state']=='complete'
        assert hashlib.sha256((work/'summary.json').read_bytes()).hexdigest()==done['summary_sha256']
        assert audit['parameters_before']==audit['parameters_after']
        assert split['direct_inverse_leakage']==0
        expected=2*split['test_facts']
        assert summary['queries']==expected
        result['max_peak_rss_bytes']=max(result['max_peak_rss_bytes'],done['peak_rss_bytes'])
        with np.load(work/'ranks.npz',allow_pickle=False) as ranks:
            assert set(ranks.files)==set(summary['metrics'])
            for method in ranks.files:
                r=ranks[method].astype(float)
                assert r.shape==(expected,) and np.isfinite(r).all() and (r>=1).all()
                recomputed={'MRR':float(np.sum(1/r)/expected),'Hits1':float(np.count_nonzero(r<=1)/expected),
                            'Hits3':float(np.count_nonzero(r<=3)/expected),'Hits10':float(np.count_nonzero(r<=10)/expected)}
                for metric,value in recomputed.items():
                    assert abs(summary['metrics'][method][metric]-value)<1e-12,(key,family,method,metric)
                if method!='M0':
                    for metric,ci in summary['paired_bootstrap'][method].items():
                        assert ci['clusters']==split['test_facts'] and ci['replicates']==2000
                        assert abs(ci['delta']-(recomputed[metric]-summary['metrics']['M0'][metric]))<1e-12
            for method,expansion in summary['expansion'].items():
                base=ranks['M0'];end=ranks[method]
                assert expansion['promoted']==int(np.count_nonzero((base>=11)&(base<=30)&(end<=10)))
                assert expansion['demoted']==int(np.count_nonzero((base<=10)&(end>10)))
            llm=work/'selected_llm.json'
            if llm.exists() and read(llm)['strength']==0:
                assert np.array_equal(ranks['R4'],ranks['R5'])
        seen=set()
        with (work/'per_query_ranks.jsonl').open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line);identity=(row['fact_index'],row['direction'])
                assert identity not in seen
                seen.add(identity)
        assert len(seen)==expected
        for fact_index in range(split['test_facts']):
            assert (fact_index,0) in seen and (fact_index,1) in seen
        result['groups'].append(dict(dataset=key,family=family,queries=expected,passed=True))
        result['query_evaluations']+=expected
assert result['max_peak_rss_bytes']<1_800_000_000
result['historical_verification']=read(RUN/'historical_verification.json')
assert result['historical_verification']['passed']
(RUN/'final_delivery_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
