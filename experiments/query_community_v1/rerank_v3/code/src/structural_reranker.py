"""Answer-free score normalization, expansion gate and rank-one protection."""
from dataclasses import dataclass, asdict
import itertools
import numpy as np

@dataclass(frozen=True)
class Config:
    strength: float = 0.
    beta_w: float = .5
    beta_p: float = .25
    expansion_quantile: float = .8
    lock_margin: float = .1
    lock_support: float = .8
    margin_trigger: float = .05
    def dict(self): return asdict(self)

def normalize(values, supported=None, logarithmic=False):
    x = np.asarray(values, dtype=float)
    mask = np.ones(len(x), dtype=bool) if supported is None else np.asarray(supported, dtype=bool)
    out = np.zeros_like(x)
    if not mask.any(): return out
    y = np.log(np.maximum(x[mask], 1e-300)) if logarithmic else x[mask]
    span = np.ptp(y)
    # Equal nonempty evidence is equally supported; absent evidence remains zero.
    out[mask] = (y - y.min()) / span if span > 1e-12 else 1.
    return out

def community_scores(evidence, cfg):
    supported = [c['status'] == 'ok' and bool(c['evidence_paths']) for c in evidence]
    w = [c['max_path_weight'] for c in evidence]
    p = [max((p['transition_path_probability'] for p in c['evidence_paths']), default=0.) for c in evidence]
    a = [c['ppr_affinity_score'] for c in evidence]
    score = cfg.beta_w * normalize(w, supported, True) + cfg.beta_p * normalize(p, supported, True) + (1-cfg.beta_w-cfg.beta_p) * normalize(a, supported, True)
    # Anchor the strongest composite support at one, without inventing support.
    if score.max(initial=0) > 0: score /= score.max()
    assert np.all((score >= 0) & (score <= 1 + 1e-10))
    return score

def enforce(order, gate, locked):
    # Ineligible expansion candidates fill only ranks > 10 in original order.
    eligible = [int(i) for i in order if gate[i]]
    top = eligible[:min(10, len(eligible))]
    if locked and 0 not in top: top = [0] + top[:9]
    if locked: top = [0] + [i for i in top if i != 0]
    rest_eligible = [i for i in eligible if i not in top]
    blocked = [i for i in range(len(gate)) if not gate[i]]
    out = top + rest_eligible + blocked
    assert len(set(out)) == len(gate)
    assert all(gate[i] for i in out[:min(10, len(gate))])
    return out

def rerank(record, cfg, confidence=None, llm_strength=0.):
    ev = record['candidate_evidence']; n = len(ev)
    if not n: return dict(order=[], scores=[], support=[], gate=[], locked=False)
    base = normalize([c['base_logit'] for c in ev])
    support = community_scores(ev, cfg)
    threshold = float(np.quantile(support[:min(10, n)], cfg.expansion_quantile))
    gate = np.arange(n) < 10
    gate |= (support > threshold) & (support > 0)
    locked = bool(n > 1 and support[0] >= cfg.lock_support and base[0]-base[1] > cfg.lock_margin)
    scores = base + cfg.strength * gate * support
    if confidence:
        scores += llm_strength * gate * np.array([confidence.get(c['entity'], 0.) for c in ev])
    order = np.lexsort((np.arange(n), -scores))
    order = enforce(order, gate, locked)
    # Zero residual must preserve the exact frozen candidate order, including ties.
    if cfg.strength == 0 and not (confidence and llm_strength): order = list(range(n))
    return dict(order=order, scores=scores.tolist(), support=support.tolist(), gate=gate.tolist(), locked=locked)

def trigger(record, result, cfg):
    ev = record['candidate_evidence']
    pool = [i for i in result['order'] if result['gate'][i] and ev[i]['status'] == 'ok']
    if len(pool) < 2: return []
    order = result['order']; scores = result['scores']; s = result['support']
    ambiguous = scores[order[0]] - scores[order[1]] <= cfg.margin_trigger
    challenge = any(s[i] > s[0] + .2 for i in pool if i != 0)
    if result['locked'] or not (ambiguous or challenge): return []
    first = [i for i in range(min(5, len(ev))) if i in pool]
    expansion = sorted([i for i in pool if i not in first], key=lambda i: (-s[i], i))
    return (first + expansion)[:8]

def grid():
    # Repair protocol: only residual strength is tuned. Identity wins uncertainty ties.
    for strength in (0., .15, .3):
        yield Config(strength=strength)
