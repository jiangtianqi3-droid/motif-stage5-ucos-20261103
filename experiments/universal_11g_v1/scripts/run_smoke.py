"""Two real graphs, all frozen B2 seeds, five masked folds, 32 final queries."""
import gc
import json
import os
import time
import traceback
import numpy as np
import torch
from regime_inspector import ROOT, STAGE6, STAGE7, records, observation, inspect, save_json
from context_features import Context
from lexical import embeddings
from micro_probe_calibrator import fuse, filtered_ranks, micro_folds, calibrate
from backbone import Backbone, bidirectional, truth_index, masked_observation
from evaluator import resources

START = time.time()

def log(stage, **kw):
    state = dict(stage=stage, pid=os.getpid(), elapsed_seconds=time.time() - START, **kw)
    save_json(ROOT / 'results/smoke_status.json', state)
    print(json.dumps(state), flush=True)

def main():
    torch.set_num_threads(4)
    result = {}
    for rec in records():
        if rec['key'] not in ['NELLInductive_v1', 'WKIngram_25']:
            continue
        key = rec['key']
        log('observation', graph=key)
        z = observation(rec)
        R = z['R'] // 2
        idx = np.flatnonzero(z['edge_type'] < R)
        facts = np.c_[z['edge_index'][0, idx], z['edge_type'][idx], z['edge_index'][1, idx]]
        lex = embeddings(rec, z)
        assert (lex is not None) == (key == 'NELLInductive_v1')
        masks = micro_folds(facts, z['n'])
        save_json(ROOT / 'results/smoke' / key / 'masks.json', [m.tolist() for m in masks])
        probe_records = []
        for fold, ids in enumerate(masks):
            assert len(ids) == min(64, int(.05 * len(facts)))
            log('probe', graph=key, fold=fold)
            masked = masked_observation(z, facts, ids, ROOT / 'cache/smoke' / key / str(fold))
            # Smoke limits query volume, not the specified number of removed edges.
            queries = bidirectional(facts[ids[:4]], R)
            context = Context(masked, lex)
            aux = context.score(queries[:, :2])
            if lex is None:
                assert not aux[:, 2].any()
            backbone = Backbone(masked)
            raw = backbone.scores(facts[ids[:4]])
            backbone.close()
            probe_records.append((raw, aux, context.degree[queries[:, 0]], queries, truth_index(facts, R)))
            del context, masked
            gc.collect()
            resources()
        calibrations = {}
        for pure in [True, False]:
            def evaluate(w):
                return [float(np.mean(1 / filtered_ranks(fuse(s, a, d, w, text_free=pure), q, truth))) for s, a, d, q, truth in probe_records]
            calibrations['pure' if pure else 'full'] = calibrate(evaluate, text_free=pure)
        save_json(ROOT / 'results/smoke' / key / 'calibration.json', calibrations)
        # Target labels are opened only after observation-only probes are frozen.
        with np.load(STAGE6 / rec['file']) as frozen:
            evaluation_facts = frozen['queries'][:16].copy()
            truth = truth_index(frozen['alltrue'], R)
        q = bidirectional(evaluation_facts, R)
        context = Context(z, lex)
        aux = context.score(q[:, :2])
        anchors = {}
        for family, seed in [('B0', 1024), ('B2', 1024), ('B2', 2026), ('B2', 3407)]:
            log('baseline_alignment', graph=key, family=family, seed=seed)
            backbone = Backbone(z, family, seed)
            raw = backbone.scores(evaluation_facts)
            zero = fuse(raw, aux, context.degree[q[:, 0]], [0] * 5)
            assert np.array_equal(raw, zero)
            rank = filtered_ranks(zero, q, truth)
            tag = 'B0+Dstar' if family == 'B0' else f'B2_s{seed}+Dstar'
            with np.load(STAGE7 / 'evaluation' / key / (tag + '.npz')) as old:
                assert np.array_equal(old['triples'][:16][:, [0, 2, 1]], evaluation_facts)
                expected = old['ranks'][:16].reshape(-1)
            assert np.array_equal(rank, expected), (key, tag, rank.tolist(), expected.tolist())
            anchors[tag] = dict(exact_score_identity=True, exact_historical_ranks=True, queries=len(q))
            backbone.close()
        result[key] = dict(passed=True, anchors=anchors, modality=inspect(rec, z), fold_sizes=list(map(len, masks)),
                           probe_queries_per_fold=8, p0_locked_before_test_open=True, calibration=calibrations, **resources())
        save_json(ROOT / 'results/smoke_result.json', result)
        del context, probe_records, z, lex
        gc.collect()
        torch.cuda.empty_cache()
    assert len(result) == 2
    log('smoke_passed', graphs=list(result), **resources())

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        log('failed', error=repr(error))
        traceback.print_exc()
        raise
