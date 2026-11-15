"""Read-only Stage-7 adapter; no historical runner or output writer is invoked."""
import gc
import numpy as np
import torch
from engine import model, RelationCache, state_hash
from common import make_data
from calibration import install

def bidirectional(facts, R):
    return np.stack([facts, np.c_[facts[:, 2], facts[:, 1] + R, facts[:, 0]]], axis=1).reshape(-1, 3)

def truth_index(facts, R):
    out = {}
    for h, r, t in facts:
        out.setdefault((int(h), int(r)), set()).add(int(t))
        inverse = int(r + R if r < R else r - R)
        out.setdefault((int(t), inverse), set()).add(int(h))
    return {k: np.asarray(sorted(v), dtype=np.int64) for k, v in out.items()}

class Backbone:
    def __init__(self, z, family='B2', seed=1024, calibrated=True, adaptive=False):
        extreme = adaptive and z['n'] / (z['R'] / 2) > 500
        if extreme:
            from structural_calibrator import sparsify
            z = sparsify(z)
        self.z = z
        self.data, _ = make_data(dict(z, queries=np.empty((0, 3), np.int64), alltrue=np.empty((0, 3), np.int64)))
        self.model = model(family, seed)
        if extreme:
            from adaptive_backbone import install_adaptive
            install_adaptive(self.model, self.data, family)
        elif calibrated:
            install(self.model, self.data, dict(kind='D-Sender', gamma=.75, alpha=.20, anchor=True))
        self.cache = RelationCache(self.model)
        self.before = state_hash(self.model)

    @torch.inference_mode()
    def scores(self, facts):
        # Preserve Stage-7 batching: 16 forward queries then 16 inverse queries.
        outputs = []
        micro = 2 if self.z['n'] > 30000 else 16
        for at in range(0, len(facts), micro):
            part = facts[at:at + micro]
            both = bidirectional(part, self.z['R'] // 2).reshape(-1, 2, 3)
            directions = []
            for direction in range(2):
                q = torch.as_tensor(both[:, direction], device='cuda')
                batch = torch.empty(len(q), self.z['n'], 3, dtype=torch.long, device='cuda')
                batch[:, :, 0] = q[:, 0, None]
                batch[:, :, 1] = torch.arange(self.z['n'], device='cuda')
                batch[:, :, 2] = q[:, 1, None]
                score = self.model(self.data, batch)
                assert torch.isfinite(score).all()
                directions.append(score.cpu().numpy())
                from evaluator import resources
                resources()
            outputs.append(np.stack(directions, axis=1).reshape(-1, self.z['n']))
        return np.concatenate(outputs)

    def close(self):
        assert self.before == state_hash(self.model), 'Frozen parameter mutation'
        self.model.relation_model.forward = self.cache.original
        del self.model, self.data, self.cache
        gc.collect()
        torch.cuda.empty_cache()

def masked_observation(z, facts, removed, cache):
    from topology import extract
    R = z['R'] // 2
    held = set(map(tuple, facts[removed]))
    keep = np.asarray([tuple(f) not in held for f in facts])
    remainder = facts[keep]
    assert not held.intersection(map(tuple, remainder))
    edge = np.concatenate([remainder[:, [0, 2]], remainder[:, [2, 0]]]).T
    relation = np.r_[remainder[:, 1], remainder[:, 1] + R]
    signature = extract(edge, relation, z['n'], z['R'], cache=cache)
    result = dict(n=z['n'], R=z['R'], edge_index=edge, edge_type=relation,
                hyper_edge=signature['edge_index'], hyper_type=signature['edge_type'],
                z=signature['z'], raw=signature['raw'])
    if '_lex_rel' in z:
        result['_lex_rel'] = z['_lex_rel']
    return result
