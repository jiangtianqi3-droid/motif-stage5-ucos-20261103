"""Pure paired-fact selection. Confirmation is consulted once per predeclared arm."""
import numpy as np


def rr(ranks):
    return (1 / np.asarray(ranks, dtype=np.float64)).mean(axis=-1)


def choose(ranks, configs, cut, allowed=None, kappa=1.64, min_select=24, min_confirm=16):
    # ranks = candidate x original-fact x direction. No formal metrics accepted.
    ref = configs.index([0., 0., 'conditional'])
    allowed = list(range(len(configs))) if allowed is None else list(allowed)
    values = rr(ranks)
    means = values[allowed, :cut].mean(1) if cut else np.zeros(len(allowed))
    # Fixed complexity-first tie breaking, never use confirmation to break a tie.
    best = sorted(zip(means, allowed), key=lambda v: (-v[0], abs(configs[v[1]][0]), abs(configs[v[1]][1]), v[1]))[0][1]
    delta = values[best, cut:] - values[ref, cut:]
    se = float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else None
    mean = float(delta.mean()) if len(delta) else 0.
    lcb = mean - kappa * se if se is not None else None
    sufficient = cut >= min_select and len(delta) >= min_confirm
    accept = bool(sufficient and lcb is not None and lcb > 0)
    return dict(candidate=configs[best], selected=configs[best if accept else ref],
                candidate_index=best, final_index=best if accept else ref,
                reference=configs[ref], reference_index=ref, fallback=not accept,
                selection_mrr=float(values[best, :cut].mean()) if cut else None,
                reference_selection_mrr=float(values[ref, :cut].mean()) if cut else None,
                confirmation_delta=mean, confirmation_se=se, confirmation_lcb=lcb,
                confirmation_mrr=float(values[best, cut:].mean()) if len(delta) else None,
                sufficient=sufficient, selection_facts=cut, confirmation_facts=len(delta))
