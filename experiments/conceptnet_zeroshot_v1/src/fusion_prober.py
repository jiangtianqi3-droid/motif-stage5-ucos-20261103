import json
import numpy as np
import torch
from evaluator import fuse,ranks,atomic_json

PRIOR=[1.,.8,1.2,.6,.8,.5]

def objective(shards,weights,truth,degree):
    totals=np.zeros(len(weights));n=0
    for path in shards:
        with np.load(path) as f:
            features=torch.tensor(f['features'],device='cuda');queries=f['queries']
        for j,w in enumerate(weights):totals[j]+=(1/ranks(fuse(features,degree[queries[:,0]],w),queries,truth)).sum()
        n+=len(queries)
        del features
    return totals/n

def calibrate(shards,truth,degree,output,log,smoke=False):
    if output.exists():return json.loads(output.read_text())['weights']
    w=list(PRIOR);history=[]
    # Nonnegative bounded coordinate ascent; weights never optimized on test.
    for round_,step in enumerate([.5,.25] if not smoke else [.5]):
        for coordinate in range(6):
            proposals=[list(w)]
            for delta in [-step,step]:
                p=list(w);p[coordinate]=float(np.clip(p[coordinate]+delta,0,3));proposals.append(p)
            scores=objective(shards,proposals,truth,degree)
            chosen=int(np.argmax(scores));w=proposals[chosen]
            history.append(dict(round=round_,coordinate=coordinate,scores=scores.tolist(),weights=list(w)))
            log('calibration',protocol=output.stem,round=round_,coordinate=coordinate,objective=float(scores[chosen]))
    atomic_json(output,dict(weights=w,history=history,algorithm='nonnegative coordinate grid ascent',bound=[0,3]))
    return w
