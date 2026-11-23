"""Full 100-query repair acceptance, including the eight previously failing queries."""
from pathlib import Path
import numpy as np
from v21.common import OUTPUT,read,dump,verify,sha
from p2.assets import Target,p1_root
from p1.assets import Evaluator

state=read(OUTPUT/'status.json')
folder=Path(state['run_directory'])
view=Target('FBIngram_25',p1_root(),OUTPUT/'repair_diagnostic')
with np.load(folder/'full_scores.npz',allow_pickle=False) as z:
    indices=z['indices']
    scores=z['scores']
verify(len(indices)==100,'acceptance requires all original 100 full-graph queries')
evaluator=Evaluator(view.asset)
checks=[]
for (i,d),row in zip(indices,scores):
    i,d=int(i),int(d)
    old=view.raw[i][d]
    ordered=np.sort(row)
    rank=len(row)-np.searchsorted(ordered,row,side='left')
    proxy=1/(60+rank.astype(np.float32))
    h,r=map(int,view.asset.queries[i,d])
    target,excluded,expected=evaluator.query(i,d,h,r,old)
    actual=evaluator.rank(row,target,excluded)
    verify(np.array_equal(proxy,old),f'{i}:{d}: remaining unfiltered rank discrepancy')
    verify(actual==expected,f'{i}:{d}: remaining filtered rank discrepancy')
    checks.append(dict(query_id=f'{view.key}:{i}:{d}',proxy_differences=0,rank=actual,expected_rank=expected))
old_folder=OUTPUT/'581dae1db50ccae320b5ce5163e1c2fe3ea494647cc16a2b64a93583c252c8f8'
same_sample=read(old_folder/'provenance.json')['selection_sha256']==read(folder/'provenance.json')['selection_sha256']
verify(same_sample,'restart changed the fixed sample')
problem={f'FBIngram_25:{i}:{d}' for i in (248,1681,2607,3172) for d in (0,1)}
verify(problem<={r['query_id'] for r in checks},'previous failures absent from sample')
result=dict(queries=len(checks),all_exact=True,previously_failing_queries=len(problem),same_original_sample=same_sample,
            full_scores_sha256=sha(folder/'full_scores.npz'),
            relation_schedule_sha256=sha(folder/'relation_cache_schedule.json'),checks=checks)
dump(folder/'repair_acceptance.json',result)
dump(OUTPUT/'repair_diagnostic/repair_acceptance.json',result)
print({k:v for k,v in result.items() if k!='checks'})
