"""Compare eager relation precompute against the exact archived first-miss schedule."""
from pathlib import Path
import numpy as np
import torch
from p1.assets import open_model,save_npz
from p2.assets import Target,p1_root
from v21.common import OUTPUT,read,dump

old=OUTPUT/'581dae1db50ccae320b5ce5163e1c2fe3ea494647cc16a2b64a93583c252c8f8'
v=Target('FBIngram_25',p1_root(),OUTPUT/'repair_diagnostic')
with np.load(old/'full_relation_embeddings.npz',allow_pickle=False) as z:
    cached=dict(zip(map(int,z['relations']),z['representations']))
network,data=open_model(v.asset)
pending=set(cached)
seen=set()
groups=[]
representation={}
with torch.no_grad():
    for at in range(0,len(v.asset.queries),v.asset.meta['batch']):
        for direction in (0,1):
            wanted=set(map(int,v.asset.queries[at:at+v.asset.meta['batch'],direction,1]))
            missing=sorted(wanted-seen)
            for offset in range(0,len(missing),2):
                group=missing[offset:offset+2]
                groups.append(group)
                if pending.intersection(group):
                    actual=network.relation_model(data.relation_hypergraph,torch.tensor(group,device='cuda'))
                    for j,q in enumerate(group):
                        if q in pending:
                            representation[q]=actual[j].cpu().numpy()
                            pending.remove(q)
                seen.update(group)
        if not pending:break
rows=[dict(relation=q,historical_group=next(g for g in groups if q in g),
           differs=not np.array_equal(cached[q],representation[q]),
           max_abs=float(np.max(np.abs(cached[q]-representation[q]))))
      for q in sorted(cached)]
dump(OUTPUT/'repair_diagnostic/historical_relation_comparison.json',dict(rows=rows,groups=groups))
save_npz(OUTPUT/'repair_diagnostic/historical_relation_embeddings.npz',
         relations=np.array(sorted(representation)),representations=np.stack([representation[q] for q in sorted(representation)]))
print(dict(total_relations=len(rows),differing=sum(r['differs'] for r in rows)))
print([r for r in rows if r['differs']])
