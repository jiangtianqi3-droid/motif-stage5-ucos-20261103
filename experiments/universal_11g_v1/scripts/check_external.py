"""Prelaunch large-graph microbatch alignment and adaptive operator smoke."""
import gc
import json
import numpy as np
import torch
from regime_inspector import ROOT, STAGE7, records, observation, save_json
from protocol_data import final_labels, validation
from backbone import Backbone, bidirectional, truth_index
from micro_probe_calibrator import filtered_ranks, fuse
from context_features import Context
from lexical import embeddings
from evaluator import resources

torch.set_num_threads(4)
rec = next(r for r in records() if r['group'] == 'external')
z = observation(rec)
lex = embeddings(rec, z)
context = Context(z, lex)
test, true = final_labels(rec)
test = test[:16]
q = bidirectional(test, z['R'] // 2)
truth = truth_index(true, z['R'] // 2)
aux = context.score(q[:, :2])
degree = context.degree[q[:, 0]]
result = {}
for family, seed in [('B0', 1024), ('B2', 1024), ('B2', 2026), ('B2', 3407)]:
    model = Backbone(z, family, seed)
    raw = model.scores(test)
    ranks = filtered_ranks(raw, q, truth)
    assert fuse(raw, aux, degree, [0] * 5) is raw
    tag = 'B0+Dstar' if family == 'B0' else f'B2_s{seed}+Dstar'
    with np.load(STAGE7 / 'evaluation' / rec['key'] / (tag + '.npz')) as old:
        expected = old['ranks'][:16].reshape(-1)
    assert np.array_equal(ranks, expected), (tag, ranks.tolist(), expected.tolist())
    model.close()
    adaptive = Backbone(z, family, seed, adaptive=True)
    scores = adaptive.scores(test[:2])
    assert np.isfinite(scores).all()
    adaptive.close()
    result[tag] = dict(exact_baseline_ranks=True, queries=32, adaptive_finite=True, **resources())
    print(json.dumps({tag: result[tag]}), flush=True)
val_z, val, _, _, protocol = validation(rec, z, ROOT / 'cache/external_smoke_validation')
assert len(val) > 0 and protocol == 'target_graph_valid_disjoint_from_test'
save_json(ROOT / 'results/external_smoke.json', dict(passed=True, models=result, validation_facts=len(val), **resources()))
print('External smoke passed', flush=True)
