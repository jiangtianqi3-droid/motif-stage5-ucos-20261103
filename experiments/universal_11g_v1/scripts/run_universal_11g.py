#!/usr/bin/env python
"""Resumable isolated evaluation; launch through scripts/run.sh."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PARSER = argparse.ArgumentParser()
PARSER.add_argument('--smoke', action='store_true')
PARSER.add_argument('--resume-from', help='Reuse the completed graphs and frozen calibration of an earlier full run')
ARGS = PARSER.parse_args()
if ARGS.resume_from and (ARGS.smoke or len(ARGS.resume_from) != 20 or not all(x in '0123456789abcdef' for x in ARGS.resume_from)):
    PARSER.error('--resume-from requires a full-run fingerprint')
LOCK = (ROOT / 'results/run.lock').open('a')
fcntl.flock(LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
START = time.time()
STATE = dict(pid=os.getpid(), mode='smoke' if ARGS.smoke else 'full', completed_graphs=[])

def log(stage, **kw):
    STATE.update(stage=stage, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), elapsed_seconds=time.time() - START, **kw)
    path = ROOT / 'results/status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(STATE, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)
    print(json.dumps(dict(stage=stage, **kw), ensure_ascii=False), flush=True)

# Startup acknowledgement precedes slow torch imports, but follows the exclusive lock.
log('starting', full_run_started=not ARGS.smoke)

import numpy as np
import torch
from regime_inspector import records, observation, inspect, digest, save_json, STAGE7
from protocol_data import forward_facts, validation, final_labels
from lexical import embeddings, encode_names, attach_relations
from backbone import bidirectional, truth_index, masked_observation
from micro_probe_calibrator import micro_folds, calibrate, PRIOR
from streaming import feature_shards, backbone_shards, evaluate_shards, fold_evaluator
from evaluator import resources
from reporting import build_report

ZERO = [0.] * 5
SEEDS = [('B0', 1024), ('B2', 1024), ('B2', 2026), ('B2', 3407)]

def historical(rec, tag, facts):
    path = STAGE7 / 'evaluation' / rec['key'] / (tag + '.npz')
    meta = json.loads(path.with_suffix('.json').read_text())
    assert digest(path) == meta['npz_sha256']
    with np.load(path) as old:
        assert np.array_equal(old['triples'][:len(facts)][:, [0, 2, 1]], facts)
        return old['ranks'][:len(facts)].reshape(-1).copy()

def search(folds, tag, pure, reference=None, validation_mode=False):
    evaluate = fold_evaluator(folds, tag, pure)
    count = 0
    last = 0
    def progress(w):
        nonlocal count, last
        count += 1
        if time.time() - last > 20:
            log('weight_search', backbone=tag, pure=pure, validation=validation_mode, candidates=count)
            last = time.time()
        return evaluate(w)
    return calibrate(progress, text_free=pure, guard=not validation_mode, reference=reference)

def prepare_fold(z, facts, ids, lex, directory, smoke):
    masked = masked_observation(z, facts, ids, directory / 'topology')
    probes = facts[ids[:4]] if smoke else facts[ids]
    assert len(probes) > 0, 'No eligible observation probe edges'
    shards = feature_shards(masked, probes, lex, directory / 'features', log)
    for family, seed in SEEDS:
        backbone_shards(masked, shards, family, seed, log)
        if z['n'] / (z['R'] / 2) > 500:
            backbone_shards(masked, shards, family, seed, log, adaptive=True)
    result = dict(shards=shards, truth=truth_index(facts, z['R'] // 2),
                  degree=np.bincount(masked['edge_index'][0], minlength=z['n']))
    del masked
    gc.collect()
    torch.cuda.empty_cache()
    return result

def graph(rec, run, cache, prior=None):
    key = rec['key']
    out = run / 'per_graph' / key
    out.mkdir(parents=True, exist_ok=True)
    cache = cache / key
    z = observation(rec)
    facts = forward_facts(z)
    regime = inspect(rec, z)
    save_json(out / 'regime.json', regime)
    extreme = regime['regime'] == 'Extreme-Commonsense'
    lex = embeddings(rec, z)
    degree = np.bincount(z['edge_index'][0], minlength=z['n'])
    masks = micro_folds(facts, z['n'])
    save_json(out / 'probe_masks.json', dict(indices=[x.tolist() for x in masks],
              mask_counts=[len(x) for x in masks], seed=20261114,
              external_rule='Also use five disjoint micro-folds capped at 64 for a shared 3-of-5 fallback audit'))
    if prior is not None:
        prior_out = prior / 'per_graph' / key
        for filename in ['p0_frozen.json', 'p1_protocol.json'] + [f'{family}_{seed}_{phase}.json' for family, seed in SEEDS for phase in ['p0', 'p1']]:
            shutil.copyfile(prior_out / filename, out / filename)
        locked = json.loads((out / 'p0_frozen.json').read_text())['weights']
        p1 = {f'{family}_{seed}': json.loads((out / f'{family}_{seed}_p1.json').read_text()) for family, seed in SEEDS}
        assert all(locked[tag] == json.loads((out / (tag + '_p0.json')).read_text()) for tag in p1)
        log('calibration_restored', graph=key)
    else:
        locked, p1 = calibrate_graph(rec, z, facts, masks, lex, cache, out, extreme)
    finish_graph(rec, run, z, lex, cache, out, locked, p1, extreme, degree)

def calibrate_graph(rec, z, facts, masks, lex, cache, out, extreme):
    key = rec['key']
    folds = [prepare_fold(z, facts, ids, lex, cache / f'fold_{i}', ARGS.smoke) for i, ids in enumerate(masks)]
    locked = {}
    for family, seed in SEEDS:
        tag = f'{family}_{seed}'
        path = out / (tag + '_p0.json')
        if path.exists():
            locked[tag] = json.loads(path.read_text())
            continue
        result = dict(pure=search(folds, tag, True), full=search(folds, tag, False))
        if extreme:
            reference = fold_evaluator(folds, tag, True)(ZERO)
            result['adaptive_pure'] = search(folds, tag + '_adaptive', True, reference=reference)
            result['adaptive_full'] = search(folds, tag + '_adaptive', False, reference=reference)
        locked[tag] = result
        save_json(path, result)
    save_json(out / 'p0_frozen.json', dict(weights=locked, frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              target_valid_or_test_labels_loaded=False, pure_consensus_has_no_lexical_input=True))
    del folds
    gc.collect()
    # Authorized P1: GraIL uses its independent training graph, never valid_ind.
    log('p1_validation', graph=key)
    val_z, val_facts, val_true, val_names, val_protocol = validation(rec, z, cache / 'validation_topology')
    if ARGS.smoke:
        val_facts = val_facts[:16]
    val_lex = lex if val_z is z else encode_names(val_names)
    if val_z is not z and val_lex is not None:
        attach_relations(val_z, val_z['_relation_names'])
    val_shards = feature_shards(val_z, val_facts, val_lex, cache / 'validation', log)
    val_fold = [dict(shards=val_shards, truth=truth_index(val_true, val_z['R'] // 2),
                    degree=np.bincount(val_z['edge_index'][0], minlength=val_z['n']))]
    p1 = {}
    for family, seed in SEEDS:
        tag = backbone_shards(val_z, val_shards, family, seed, log)
        path = out / (tag + '_p1.json')
        if path.exists():
            p1[tag] = json.loads(path.read_text())
        else:
            p1[tag] = search(val_fold, tag, False, validation_mode=True)
            save_json(path, p1[tag])
    save_json(out / 'p1_protocol.json', dict(protocol=val_protocol, queries=len(val_facts)*2,
              locked_before_final_test_open=True, target_test_labels_used=False))
    del val_z, val_lex, val_fold, val_shards, val_true
    gc.collect()
    torch.cuda.empty_cache()
    return locked, p1

def finish_graph(rec, run, z, lex, cache, out, locked, p1, extreme, degree):
    key = rec['key']
    # Only evaluator code reads final target labels and filtering truth.
    test, alltrue = final_labels(rec)
    if ARGS.smoke:
        test = test[:16]
    q = bidirectional(test, z['R'] // 2)
    truth = truth_index(alltrue, z['R'] // 2)
    del alltrue
    shards = feature_shards(z, test, lex, cache / 'test', log)
    ranks = {}
    b0 = historical(rec, 'B0', test)
    b1 = historical(rec, 'B0+Dstar', test)
    b2 = [historical(rec, f'B2_s{s}+Dstar', test) for s in [1024, 2026, 3407]]
    for family in ['B0', 'B2']:
        ranks[family + '/M0'] = [b0]
        ranks[family + '/M1'] = [b1]
        ranks[family + '/M2'] = b2
    alignment = {}
    for family, seed in SEEDS:
        tag = f'{family}_{seed}'
        scores_tag = backbone_shards(z, shards, family, seed, log, replay=extreme)
        anchor = evaluate_shards(shards, scores_tag, truth, degree, ZERO)
        expected = b1 if family == 'B0' else b2[[1024, 2026, 3407].index(seed)]
        mismatches = np.flatnonzero(anchor != expected)
        alignment[tag] = dict(exact_match=not len(mismatches), mismatch_count=len(mismatches),
                              query_count=len(q), first_mismatches=mismatches[:20].tolist())
        save_json(out / 'baseline_alignment.json', alignment)
        assert not len(mismatches), f'{key}/{tag}: baseline mismatch; no test-driven parameter adjustment'
        methods = dict(M3=(locked[tag]['pure']['weights'], True), M4=(PRIOR, False),
                       M5=(locked[tag]['full']['weights'], False), M6=(p1[tag]['weights'], False))
        current = {}
        for name, (weights, pure) in methods.items():
            current[name] = evaluate_shards(shards, scores_tag, truth, degree, weights, pure)
            ranks.setdefault(family + '/' + name, []).append(current[name])
        if extreme:
            adaptive_tag = backbone_shards(z, shards, family, seed, log, adaptive=True)
            adaptive_anchor = evaluate_shards(shards, adaptive_tag, truth, degree, ZERO)
            ranks.setdefault(family + '/A0', []).append(adaptive_anchor)
            for name, label, pure in [('A3', 'adaptive_pure', True), ('A5', 'adaptive_full', False)]:
                choice = locked[tag][label]
                value = evaluate_shards(shards, adaptive_tag, truth, degree, choice['weights'], pure) if choice['accepted'] else anchor.copy()
                ranks.setdefault(family + '/' + name, []).append(value)
        else:
            for name, value in [('A0', anchor), ('A3', current['M3']), ('A5', current['M5'])]:
                ranks.setdefault(family + '/' + name, []).append(value)
        log('family_complete', graph=key, family=family, seed=seed, **resources())
    temporary = out / 'rankings.tmp'
    with temporary.open('wb') as f:
        np.savez_compressed(f, queries=q, **{k: np.asarray(v) for k, v in ranks.items()})
    temporary.replace(out / 'rankings.npz')
    save_json(out / 'complete.json', dict(key=key, queries=len(q), completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              ranking_sha256=digest(out / 'rankings.npz'), **resources()))
    if not ARGS.smoke:
        save_json(ROOT / 'results/per_graph' / key / 'complete.json',
                  dict(run=str(run), artifacts=str(out), **json.loads((out / 'complete.json').read_text())))
    del z, lex, shards, ranks, q, test, truth
    gc.collect()
    torch.cuda.empty_cache()

def main():
    torch.set_num_threads(4)
    torch.manual_seed(1024)
    np.random.seed(1024)
    own = {str(p.relative_to(ROOT)): digest(p) for folder in ['src', 'scripts'] for p in (ROOT / folder).glob('*') if p.is_file()}
    protected = json.loads((ROOT / 'results/historical_hashes.json').read_text())
    audit = json.loads((ROOT / 'results/asset_audit.json').read_text())
    protected.update({v['path']: v['sha256'] for v in audit['checkpoints'].values()})
    assert all(digest(p) == h for p, h in protected.items()), 'Historical asset changed'
    fingerprint = hashlib.sha256(json.dumps([own, protected, ARGS.smoke, ARGS.resume_from], sort_keys=True).encode()).hexdigest()[:20]
    run = ROOT / 'results' / ('smoke_integration' if ARGS.smoke else 'runs') / fingerprint
    cache = ROOT / 'cache' / fingerprint
    prior = None
    if ARGS.resume_from:
        prior = ROOT / 'results/runs' / ARGS.resume_from
        old = json.loads((prior / 'provenance.json').read_text())
        assert old['historical_hashes'] == protected, 'Historical assets must match the original run'
        changes = {name for name in set(old['own_code']) | set(own) if old['own_code'].get(name) != own.get(name)}
        recovery_files = {'scripts/run_universal_11g.py', 'scripts/background_entry.sh', 'scripts/launch_background.ps1',
                          'src/reporting.py', 'src/streaming.py', 'src/stage7_replay.py'}
        assert changes <= recovery_files, 'Calibration dependencies changed: ' + str(changes - recovery_files)
        cache = ROOT / 'cache' / ARGS.resume_from
    run.mkdir(parents=True, exist_ok=True)
    log('preparing', run=str(run), fingerprint=fingerprint)
    save_json(run / 'provenance.json', dict(own_code=own, historical_hashes=protected, resumed_from=ARGS.resume_from))
    save_json(run / 'protocol.json', dict(p1_grail='independent_training_graph_valid', original_anchor_preserved=True,
              adaptive_columns=['A0', 'A3', 'A5'], family_seeds=SEEDS, logical_query_chunk=256,
              extreme_query_micro_chunk=32, extreme_gnn_micro_chunk=2, extreme_anchor_test_batch=16, bootstrap=2000,
              rss_limit_gib=3.5, gpu_reserved_limit_gib=4., test_exposure='all graphs historically evaluated',
              text_branch='frozen BGE-small; observation empirical translation plus local/global lexical prototypes',
              fallback_scope='Only five self-supervised folds; cannot guarantee test gain'))
    recs = records()
    if ARGS.smoke:
        recs = [r for r in recs if r['key'] in ['NELLInductive_v1', 'WKIngram_25']]
    regimes = {r['key']: inspect(r, observation(r)) for r in records()}
    save_json(ROOT / 'results/graph_regimes.json', regimes)
    for rec in recs:
        key = rec['key']
        complete = run / 'per_graph' / key / 'complete.json'
        prior_graph = prior / 'per_graph' / key if prior is not None else None
        if prior_graph is not None and (prior_graph / 'complete.json').exists() and not complete.exists():
            original = json.loads((prior_graph / 'complete.json').read_text())
            assert original['ranking_sha256'] == digest(prior_graph / 'rankings.npz')
            shutil.copytree(prior_graph, complete.parent, dirs_exist_ok=True, ignore=shutil.ignore_patterns('complete.json'))
            save_json(complete, original)
        if complete.exists():
            done = json.loads(complete.read_text())
            assert done['ranking_sha256'] == digest(complete.parent / 'rankings.npz')
        else:
            log('graph_start', graph=key)
            graph(rec, run, cache, prior=prior if prior_graph is not None and (prior_graph / 'p1_protocol.json').exists() else None)
        STATE['completed_graphs'].append(key)
        log('graph_complete', graph=key, **resources())
    assert all(digest(p) == h for p, h in protected.items()), 'Historical asset changed'
    report = build_report(run, recs, smoke=ARGS.smoke)
    save_json(run / 'verification.json', dict(historical_unchanged=True, all_graphs_completed=True, **resources()))
    log('smoke_passed' if ARGS.smoke else 'completed', report=str(report), **resources())

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        log('failed', error=repr(exc))
        traceback.print_exc()
        raise
