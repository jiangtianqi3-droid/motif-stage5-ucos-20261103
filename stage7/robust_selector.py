"""Selection consumes only selection RR; confirmation cannot change candidate."""
import numpy as np
from stage7.confirmation import lcb

ALPHAS=[0.,.1,.25,.5,1.]


def choose(gains,counts):
    # Environments are ordered density-major A,B,C,D: 12 distributions.
    means=np.array([[lcb(x)['mean'] for x in row] for row in gains])
    lower=np.array([[lcb(x)['lcb'] if lcb(x)['lcb'] is not None else -1e30 for x in row] for row in gains])
    # shapes: [environment, alpha], deterministic ties favor smaller alpha.
    equal=means.mean(0)
    counts=np.asarray(counts,dtype=float)
    weighted=np.average(means,axis=0,weights=counts) if counts.sum()>0 else equal
    longtail=np.tile([1.,2.,1.,1.],len(gains)//4)
    objectives={'single_mean':means[4], 'single_lcb':means[4],
                'equal_mean':equal,'equal_mean_unconfirmed':equal,'sample_weighted':weighted,
                'longtail_weighted':np.average(means,axis=0,weights=longtail),
                'worst_mean':means.min(0),'robust_lcb':lower.min(0),
                'four_env_lcb_10':lower[4:8].min(0),
                'three_density_A_lcb':lower[[0,4,8]].min(0),
                'four_env_mean_10':means[4:8].mean(0)}
    decisions={name:dict(candidate_index=int(np.argmax(values)),candidate_alpha=ALPHAS[int(np.argmax(values))],
                   objective=values.tolist()) for name,values in objectives.items()}
    return decisions,dict(environment_means=means.tolist(),environment_lcbs=lower.tolist(),
             weights=dict(equal=[1/len(gains)]*len(gains),sample=(counts/max(counts.sum(),1)).tolist(),
                          longtail=(longtail/longtail.sum()).tolist()))
