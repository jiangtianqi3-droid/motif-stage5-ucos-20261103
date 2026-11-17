"""Validated, stable writeback of a fixed candidate set."""

def writeback(full_ranking, candidates, llm_order):
    if len(set(full_ranking)) != len(full_ranking):
        raise ValueError('full ranking has duplicates')
    if len(set(candidates)) != len(candidates) or len(set(llm_order)) != len(llm_order):
        raise ValueError('candidate duplicates')
    if set(candidates) != set(llm_order) or not set(candidates).issubset(full_ranking):
        raise ValueError('candidate set mismatch')
    order = iter(llm_order)
    fixed = set(candidates)
    return [next(order) if x in fixed else x for x in full_ranking]


def validate_response(response, candidates, evidence_ids):
    if not isinstance(response, dict) or not isinstance(response.get('abstain'), bool):
        return False
    order, used = response.get('ranked_candidate_ids'), response.get('used_evidence_ids')
    return (isinstance(order, list) and all(isinstance(x, str) for x in order)
            and len(order) == len(candidates) and len(set(order)) == len(order)
            and set(order) == set(candidates) and isinstance(used, list)
            and all(isinstance(x, str) for x in used) and set(used).issubset(evidence_ids))
