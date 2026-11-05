import numpy as np


def paired_bootstrap(delta, replicates=5000, seed=6713):
    """One value per original triple (directions and training seeds already averaged)."""
    delta = np.asarray(delta, dtype=np.float64)
    if not len(delta):
        return dict(delta=None, lo=None, hi=None, facts=0), None
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    for at in range(0, replicates, 32):
        size = min(32, replicates - at)
        ix = rng.integers(len(delta), size=(size, len(delta)))
        samples[at:at + size] = delta[ix].mean(1)
    lo, hi = np.quantile(samples, [.025, .975])
    return dict(delta=float(delta.mean()), lo=float(lo), hi=float(hi), facts=len(delta)), samples
