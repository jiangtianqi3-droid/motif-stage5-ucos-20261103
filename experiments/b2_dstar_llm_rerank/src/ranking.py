"""Label-free candidate selection, strict response validation and stable insertion."""
import json
import math
import numpy as np

def candidate_pool(scores, observed_tails, k=30):
    """Only OBSERVED positives may be supplied. Scores remain untouched."""
    order = np.argsort(-np.asarray(scores), kind='stable')
    blocked = set(map(int, observed_tails))
    return [int(x) for x in order if int(x) not in blocked][:k]

def decision_pool(top30, evidence):
    first = list(top30[:5])
    strong = [x for x in top30[5:] if evidence[x]['paths']]
    if strong:
        strong.sort(key=lambda x: (-sum(p['weight'] for p in evidence[x]['paths']), top30.index(x)))
        return first + strong[:3]
    return list(top30[:8])

def validate(content, allowed):
    try:
        obj = json.loads(content)
        if not isinstance(obj, dict) or type(obj.get('abstain')) is not bool: raise ValueError()
        conf = obj.get('confidence')
        if type(conf) not in (float, int) or not math.isfinite(conf) or not 0 <= conf <= 1: raise ValueError()
        if not isinstance(obj.get('evidence_summary'), str) or len(obj['evidence_summary']) > 50: raise ValueError()
        ids = obj.get('ranked_candidate_ids')
        if not isinstance(ids, list) or any(type(x) is not int for x in ids): raise ValueError()
        if not obj['abstain'] and (len(ids) != len(allowed) or len(set(ids)) != len(ids) or set(ids) != set(allowed)):
            raise ValueError()
        return obj, None
    except (ValueError, TypeError):
        return None, 'parse_or_validation_fallback'

def writeback(top30, pool, ranked):
    """Nonselected candidates retain relative order; remote losers stay put."""
    if set(ranked) != set(pool) or len(ranked) != len(set(ranked)):
        raise ValueError('Not a bijection')
    if set(pool) == set(top30[:len(pool)]):
        return list(ranked) + list(top30[len(pool):])
    front = set(top30[:5])
    promoted = [x for x in ranked[:5] if x not in front]
    movable = front | set(promoted)
    # Reorder original front-five; insert only successful challengers. Never
    # move an unsuccessful challenger to a different tail slot deliberately.
    prefix = [x for x in ranked if x in movable]
    result = prefix + [x for x in top30 if x not in movable]
    assert len(result) == len(top30) and set(result) == set(top30)
    return result

def apply_response(top30, pool, evidence, content=None, exception=None):
    if not any(evidence[x]['paths'] for x in pool): return list(top30), 'no_evidence'
    if exception is not None: return list(top30), 'parse_or_validation_fallback'
    obj, error = validate(content, pool)
    if error: return list(top30), error
    if obj['abstain']: return list(top30), 'abstain'
    return writeback(top30, pool, obj['ranked_candidate_ids']), 'reranked'

def grouped_validation(facts, R, seed=20261116):
    """Connected components of facts sharing EITHER public directional query."""
    parent = list(range(len(facts)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i, (h, r, t) in enumerate(facts):
        for q in [(int(h), int(r), 'forward'), (int(t), int(r), 'inverse')]:
            if q in seen: parent[find(i)] = find(seen[q])
            else: seen[q] = i
    groups = {}
    for i in range(len(facts)): groups.setdefault(find(i), []).append(i)
    keys = sorted(groups); np.random.default_rng(seed).shuffle(keys)
    selected, confirmed = [], []
    for k in keys:
        (selected if len(selected) <= len(confirmed) else confirmed).extend(groups[k])
    return sorted(selected), sorted(confirmed)
