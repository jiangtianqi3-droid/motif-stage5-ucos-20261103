"""Offline filtered evaluation. This module is the only consumer of labels at inference time."""
from collections import defaultdict

import numpy as np

from src.io_utils import STAGE7, HISTORY, PRIMARY, read, snapshot, sha
from src.recovery import load
from src.ranking_fusion import writeback


class OfflineEvaluation:
    def __init__(self, key, ranking_model=PRIMARY):
        entry = next(r for r in read(STAGE7 / 'inputs.json') if r['key'] == key)
        path = HISTORY / 'motif-stage6-entity-degree-p0p1-v2' / entry['file']
        if sha(path) != entry['sha256']:
            raise ValueError('filter file checksum changed')
        self.answers = load(snapshot(key, 'offline_targets_ONLY.npz'))['queries']
        self.known = defaultdict(set)
        with np.load(path, allow_pickle=False) as z:
            half = int(z['R']) // 2
            for h, r, t in z['alltrue']:
                self.known[(h, r)].add(int(t))
                self.known[(t, r+half)].add(int(h))
        self.reference = load(STAGE7 / 'evaluation' / key / (ranking_model + '.npz'))['ranks']

    @staticmethod
    def filtered_rank(scores, target, excluded):
        ahead = scores >= scores[target]
        ahead[list(excluded | {target})] = False
        return 1 + int(ahead.sum())

    def row(self, job, scores, arm, result, fusion):
        index, direction = job['fact_index'], job['direction']
        target = int(self.answers[index, 2 if direction == 0 else 0])
        excluded = self.known[(job['head'],job['relation'])]
        original = self.filtered_rank(scores, target, excluded)
        if original != self.reference[index, direction]:
            raise ValueError('baseline rank no longer matches Stage-7 filter')
        response = result.get('response')
        candidates = job['candidate_ids']
        changed = original
        order = candidates
        if result['status'] == 'valid':
            evidence = set(job['retrieval'].get(arm, {}).get('supported_ids', []))
            qualified = fusion == 'direct' or bool(evidence.intersection(response['used_evidence_ids']))
            if qualified:
                order = [job['aliases'][x] for x in response['ranked_candidate_ids']]
                full = np.lexsort((np.arange(len(scores)), -scores)).tolist()
                written = writeback(full, candidates, order)
                if written[:len(candidates)] != order or written[len(candidates):] != full[len(candidates):]:
                    raise ValueError('candidate writeback changed other entities')
                # Move the fixed score tiers with their slots. Preserve pessimistic ties and
                # the original filtered protocol, including ties that cross the Top-12 boundary.
                assigned = scores.copy()
                assigned[order] = scores[candidates]
                changed = self.filtered_rank(assigned, target, excluded)
        return dict(dataset=job['dataset'], query_id=job['query_id'], fact_index=index,
                    direction='tail' if direction == 0 else 'head', arm=arm, fusion=fusion,
                    original_rank=original, new_rank=changed, target=target, top12=target in candidates,
                    top30=target in job['top30'], positive=changed < original, negative=changed > original,
                    status=result['status'], input_tokens=result.get('input_tokens',0),
                    output_tokens=result.get('output_tokens',0), seconds=result.get('seconds',0),
                    estimated=result.get('estimated',False), cache_hit=result.get('cache_hit',False),
                    written_candidate_ids=order)
