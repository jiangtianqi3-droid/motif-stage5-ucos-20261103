"""Observation-only fold construction, anchor-preserving fusion and calibration."""
import numpy as np

PRIOR = [.15, .15, .20, .10, .30]

def fuse(raw, auxiliary, degree, weights, text_free=False):
    w = np.asarray(weights, dtype=np.float64).copy()
    if text_free:
        w[2] = 0.
    # Bypass arithmetic completely, including constant-score and adjacent-float cases.
    if not np.any(w[:4]):
        return raw
    raw64 = raw.astype(np.float64)
    base = (raw64 - raw64.mean(axis=-1, keepdims=True)) / (raw64.std(axis=-1, keepdims=True) + 1e-6)
    eta = 1 + w[4] / np.sqrt(1 + np.asarray(degree))
    return base + w[0] * auxiliary[:, 0] + eta[:, None] * (w[1] * auxiliary[:, 1] + w[2] * auxiliary[:, 2]) + w[3] * auxiliary[:, 3 if text_free else 4]

def micro_folds(facts, n, seed=20261114):
    rng = np.random.default_rng(seed)
    degree = np.bincount(facts[:, [0, 2]].ravel(), minlength=n)
    eligible = rng.permutation(np.flatnonzero((degree[facts[:, 0]] > 1) & (degree[facts[:, 2]] > 1)))
    size = min(64, int(.05 * len(facts)))
    used = set()
    folds = []
    for _ in range(5):
        left = degree.copy()
        selected = []
        for i in eligible:
            h, _, t = facts[i]
            if int(i) in used or h == t or left[h] <= 1 or left[t] <= 1:
                continue
            selected.append(int(i))
            used.add(int(i))
            left[h] -= 1
            left[t] -= 1
            if len(selected) == size:
                break
        if size == 0:
            selected = []
        folds.append(np.asarray(selected, dtype=np.int64))
    return folds

def filtered_ranks(scores, queries, truth):
    answer = scores[np.arange(len(queries)), queries[:, 2]]
    mask = scores >= answer[:, None]
    for i, (h, r, t) in enumerate(queries):
        mask[i, truth.get((int(h), int(r)), [])] = False
        mask[i, t] = False
    return 1 + mask.sum(axis=1)

def calibrate(evaluate, text_free=False, minimum_gain=.0005, guard=True, reference=None):
    """evaluate(w) returns per-fold MRR, computed from masked-observation scores."""
    zero = [0.] * 5
    memo = {}
    original_evaluate = evaluate
    def evaluate(w):
        key = tuple(w)
        if key not in memo:
            memo[key] = np.asarray(original_evaluate(w))
        return memo[key]
    baseline = np.asarray(evaluate(zero))
    best = zero.copy()
    scores = baseline.copy()
    # Deterministic tie rule retains incumbent, favoring the zero anchor.
    for _ in range(2):
        changed = False
        for axis in range(5):
            if text_free and axis == 2:
                continue
            for value in [0., .05, .10, .15, .20, .30, .50, 1., 2.]:
                candidate = best.copy()
                candidate[axis] = value
                trial = np.asarray(evaluate(candidate))
                if trial.mean() > scores.mean() + 1e-12:
                    best, scores, changed = candidate, trial, True
        if not changed:
            break
    if reference is not None:
        baseline = np.asarray(reference)
    accepted = (len(baseline) == 5 and scores.mean() - baseline.mean() >= minimum_gain and np.sum(scores >= baseline) >= 3) if guard else scores.mean() >= baseline.mean()
    return dict(weights=best if accepted else zero, candidate_weights=best,
                baseline_fold_mrr=baseline.tolist(), candidate_fold_mrr=scores.tolist(),
                self_gain=float(scores.mean() - baseline.mean()), accepted=bool(accepted),
                guard_scope='Self-supervision only; no guarantee of nonnegative test gain')
