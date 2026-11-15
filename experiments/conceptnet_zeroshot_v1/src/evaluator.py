import os,json,time,resource
from pathlib import Path
import numpy as np
import torch

def safe_z(x):
    return ((x-x.mean(-1,keepdim=True))/(x.std(-1,unbiased=False,keepdim=True)+1e-6)).clamp(-5,5)

def resources():
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    gpu=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    if rss>=3.5*1024**3 or gpu>=4*1024**3:raise MemoryError(f'Hard budget exceeded: RSS={rss}, GPU reserved={gpu}')
    return dict(peak_rss_bytes=rss,peak_gpu_reserved_bytes=gpu)

def truth_index(facts,R=34):
    from dataset_inspector import bidirectional
    result={}
    for h,r,t in bidirectional(facts,R):result.setdefault((int(h),int(r)),set()).add(int(t))
    return {k:np.array(sorted(v),dtype=np.int64) for k,v in result.items()}

def ranks(scores,queries,truth):
    rows=torch.arange(len(queries),device=scores.device);target=torch.tensor(queries[:,2],device=scores.device)
    answer=scores[rows,target].clone();masked=scores.clone()
    for i,(h,r,t) in enumerate(queries):
        ids=truth.get((int(h),int(r)),[])
        masked[i,ids]=-torch.inf
    masked[rows,target]=-torch.inf
    # Match historical pessimistic ties. Never alter features using truth.
    return (1+(masked>=answer[:,None]).sum(1)).cpu().numpy()

def metrics(r):
    r=np.asarray(r)
    return dict(MRR=float((1/r).mean()),Hits1=float((r<=1).mean()),Hits3=float((r<=3).mean()),Hits10=float((r<=10).mean()))

def bootstrap(r,reference,seed=20261113):
    # Bind two directions of the same fact in each of 2000 paired resamples.
    r=np.asarray(r).reshape(-1,2);ref=np.asarray(reference).reshape(-1,2)
    vals=np.stack([1/r,r<=1,r<=3,r<=10],axis=-1).mean(1)
    base=np.stack([1/ref,ref<=1,ref<=3,ref<=10],axis=-1).mean(1)
    rng=np.random.default_rng(seed);own=[];delta=[]
    for _ in range(2000):
        ix=rng.integers(len(r),size=len(r));own.append(vals[ix].mean(0));delta.append((vals-base)[ix].mean(0))
    return dict(metric_ci95=np.quantile(own,[.025,.975],axis=0).T.tolist(),paired_delta_ci95=np.quantile(delta,[.025,.975],axis=0).T.tolist(),order=['MRR','Hits1','Hits3','Hits10'],unit='fact with both directions',resamples=2000)

def fuse(features,degree,weights):
    w=features.new_tensor(weights)
    eta=1+w[5]*(1+features.new_tensor(degree)).rsqrt()
    return w[0]*features[:,0]+w[1]*features[:,1]+eta[:,None]*(w[2]*features[:,2]+w[3]*features[:,3])+w[4]*features[:,4]

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)
