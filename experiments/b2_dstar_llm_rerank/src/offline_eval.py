"""The only stage allowed to apply held-out truth masks or inspect answers."""
import numpy as np
from common_io import M5RUN, S7, SEEDS, read, sha

def truth_index(facts, R):
    result = {}
    for h,r,t in facts:
        result.setdefault((int(h),int(r)),set()).add(int(t))
        result.setdefault((int(t),int(r)+R),set()).add(int(h))
    return result

def baseline_rank(score, h, r, answer, truth):
    mask = score >= score[answer]
    mask[list(truth.get((h,r),set()))] = False
    mask[answer] = False
    return int(1+mask.sum())

def changed_rank(score, h, r, answer, truth, top30, updated):
    original = baseline_rank(score,h,r,answer,truth)
    if top30 == updated or answer not in top30: return original
    blocked = truth.get((h,r),set()) | {answer}
    # Preserve original pessimistic tie contributions outside the changed prefix.
    old_before = sum(x not in blocked and score[x]>=score[answer] for x in top30)
    new_before = sum(x not in blocked for x in updated[:updated.index(answer)])
    return original - old_before + new_before

def metrics(ranks):
    a=np.asarray(ranks,dtype=float)
    if a.size==0: return None
    return dict(MRR=float((1/a).mean()),Hits1=float((a<=1).mean()),Hits3=float((a<=3).mean()),Hits10=float((a<=10).mean()))

def paired_bootstrap(a,b,draws=2000,seed=20261116):
    # Average seeds within each fact, bind both directions in every resample.
    a=np.asarray(a).reshape(len(SEEDS),-1,2); b=np.asarray(b).reshape(len(SEEDS),-1,2)
    vals=lambda x:np.stack([1/x,x<=1,x<=3,x<=10],axis=-1).mean(axis=(0,2))
    va,vb=vals(a),vals(b); rng=np.random.default_rng(seed)
    own=[]; delta=[]
    for _ in range(draws):
        ix=rng.integers(len(va),size=len(va)); own.append(va[ix].mean(0)); delta.append((va-vb)[ix].mean(0))
    return dict(order=['MRR','Hits1','Hits3','Hits10'],metric_ci95=np.quantile(own,[.025,.975],axis=0).T.tolist(),
        paired_delta_ci95=np.quantile(delta,[.025,.975],axis=0).T.tolist(),resamples=draws,unit='fact, both directions and three seeds bound')

def historical(key, queries):
    folder=M5RUN/'per_graph'/key
    assert sha(folder/'rankings.npz')==read(folder/'complete.json')['ranking_sha256']
    with np.load(folder/'rankings.npz') as z:
        assert np.array_equal(z['queries'],queries)
        base=z['B2/M2'].copy(); m5=z['B2/M5'].copy()
    for j,seed in enumerate(SEEDS):
        path=S7/'evaluation'/key/(f'B2_s{seed}+Dstar.npz')
        assert sha(path)==read(path.with_suffix('.json'))['npz_sha256']
        with np.load(path) as z:
            assert np.array_equal(z['ranks'].reshape(-1),base[j])
    return base,m5
