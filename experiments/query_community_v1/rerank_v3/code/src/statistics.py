"""Predeclared paired comparisons, dependence sensitivity, multiplicity control."""
import itertools
from collections import defaultdict
import numpy as np

def sign_flip(delta,seed=20261112,repetitions=10000):
    x=np.asarray(delta,float);x=x[np.abs(x)>1e-15]
    if not len(x):return dict(p=1.,nonzero_units=0,exact=True)
    threshold=abs(x.sum())-1e-12
    if len(x)<=16:
        p=sum(abs(np.dot(s,x))>=threshold for s in itertools.product([-1,1],repeat=len(x)))/(2**len(x))
        return dict(p=float(p),nonzero_units=len(x),exact=True)
    rng=np.random.default_rng(seed);extreme=0
    for at in range(0,repetitions,32):
        signs=2*rng.integers(2,size=(min(32,repetitions-at),len(x)))-1
        extreme+=int(np.count_nonzero(np.abs(signs@x)>=threshold))
    return dict(p=(extreme+1)/(repetitions+1),nonzero_units=len(x),exact=False,repetitions=repetitions)

def group_interval(delta,keys,seed=20261112,repetitions=2000):
    groups=defaultdict(list)
    for d,k in zip(delta,keys):groups[tuple(k) if isinstance(k,(list,np.ndarray)) else k].append(d)
    sums=np.array([sum(x) for x in groups.values()]);counts=np.array([len(x) for x in groups.values()]);n=len(sums)
    if not n:return dict(groups=0,lo=None,hi=None)
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(repetitions):
        ix=rng.integers(n,size=n);draws.append(float(sums[ix].sum()/counts[ix].sum()))
    return dict(groups=n,delta=float(sums.sum()/counts.sum()),lo=float(np.quantile(draws,.025)),hi=float(np.quantile(draws,.975)),
                few_groups=n<20,interpretation='dependence sensitivity, not independent confirmation')

def holm(pvalues):
    p=np.asarray(pvalues,float);order=np.argsort(p);out=np.ones(len(p));running=0.
    for j,i in enumerate(order):
        running=max(running,(len(p)-j)*p[i]);out[i]=min(1.,running)
    return out.tolist()

def comparison(base,target,public,triples):
    base=np.asarray(base,float);target=np.asarray(target,float);delta=1/target-1/base
    return dict(delta_MRR=float(delta.mean()),delta_Hits10=float(np.mean(target<=10)-np.mean(base<=10)),
                fact_sign_flip=sign_flip(delta.reshape(-1,2).mean(1)),
                query_clusters=group_interval(delta,[tuple(q) for q in public.reshape(-1,2)]),
                relation_clusters=group_interval(delta.reshape(-1,2).mean(1),[int(t[1]) for t in triples]),
                changed_facts=int(np.count_nonzero(np.any(base.reshape(-1,2)!=target.reshape(-1,2),axis=1))))
