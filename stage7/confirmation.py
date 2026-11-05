import numpy as np


def paired_rr(ranks):
    r=np.asarray(ranks)
    assert r.shape[-1]==2 and (r>=1).all()
    return (1/r.astype(np.float64)).mean(axis=-1)


def lcb(delta,kappa=1.64):
    delta=np.asarray(delta,dtype=float)
    if not len(delta):
        return dict(n=0,mean=0.,se=None,lcb=None)
    mean=float(delta.mean())
    se=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None
    return dict(n=len(delta),mean=mean,se=se,lcb=mean-kappa*se if se is not None else None)


def accept(records,min_n=16):
    return bool(records and all(r['n']>=min_n and r['lcb'] is not None and r['lcb']>0 for r in records))
