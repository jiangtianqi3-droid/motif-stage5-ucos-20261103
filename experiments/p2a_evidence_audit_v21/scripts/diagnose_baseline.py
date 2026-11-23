"""Read-only comparison of failed cached full logits and archived rank proxies."""
from pathlib import Path
import numpy as np
from v21.common import OUTPUT,read,dump
from p2.assets import Target,p1_root

folder=OUTPUT/'581dae1db50ccae320b5ce5163e1c2fe3ea494647cc16a2b64a93583c252c8f8'
v=Target('FBIngram_25',p1_root(),OUTPUT/'repair_diagnostic')
data=np.load(folder/'full_scores.npz',allow_pickle=False)
rows=[]
for (i,d),score in zip(data['indices'],data['scores']):
    i,d=int(i),int(d)
    old=v.raw[i][d]
    rank=len(score)-np.searchsorted(np.sort(score),score,side='left')
    old_rank=np.rint(1/old-60).astype(np.int64)
    reconstructed=1/(60+rank.astype(np.float32))
    diff=np.flatnonzero(reconstructed!=old)
    if len(diff):
        ids=diff[:20]
        rows.append(dict(query_id=f'{v.key}:{i}:{d}',proxy_differences=len(diff),
                         actual_rank_differences=int(np.count_nonzero(rank!=old_rank)),
                         max_rank_delta=int(np.max(np.abs(rank-old_rank))),
                         examples=[dict(entity=int(x),new_rank=int(rank[x]),old_rank=int(old_rank[x]),
                           old_proxy=float(old[x]),new_proxy=float(reconstructed[x]),score=float(score[x])) for x in ids]))
dump(OUTPUT/'repair_diagnostic/cached_score_comparison.json',dict(rows=rows,queries=len(data['indices'])))
print(dict(queries=len(data['indices']),mismatched_queries=len(rows),
           rank_mismatch_queries=sum(r['actual_rank_differences']>0 for r in rows)))
for r in rows[:4]:print(r)
