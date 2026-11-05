"""Formal query assets are imported only in a NEW process after the global freeze."""
import gc
import hashlib
import numpy as np
import torch
from stage6.util import ROOT, read, dump, sha, runtime_path, log
from stage6.observed_loader import load_observed
from stage6.proxy_graph import build_graph
from stage6.models import load_model, state_hash
from stage6.evaluator import safe_evaluate, metrics


def load_queries(args, key, graph):
    if key == 'ConceptNet100k_external':
        p = args.history / 'data/external/ConceptNet100k/observed.npz'
        with np.load(p) as z:
            return z['queries'].astype(np.int64), z['alltrue'].astype(np.int64)
    meta = read(args.history / 'semantic/target_mapping_provenance.json')[key]
    vocab = read(args.history / 'semantic/relation_mappings.json')[key]
    rels = {r['original_name']: r['id'] for r in vocab[:graph.R // 2]}
    raw = runtime_path(graph.provenance['source']).parent
    entities = {}
    def parse(path):
        rows = []
        for line in path.read_text(encoding='utf-8').splitlines():
            h, q, t = line.split()
            for token in (h, t):
                if token not in entities:
                    entities[token] = len(entities)
            rows.append((entities[h], rels[q], entities[t]))
        return np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    parse(raw / meta['fact_file'])
    if meta['kind'] == 'grail':
        valid, test = parse(raw / 'valid_ind.txt'), parse(raw / 'test_ind.txt')
        queries = np.r_[valid, test]
        extra = queries
    else:
        valid, test = parse(raw / 'inf_valid.txt'), parse(raw / 'inf_test.txt')
        queries, extra = test, np.r_[valid, test]
    assert len(entities) == graph.n
    truth = np.r_[np.c_[graph.edge_index[0], graph.edge_type, graph.edge_index[1]], extra]
    return queries, truth


def save_result(path, value, ranks, queries, evidence=None):
    np.savez_compressed(path.with_suffix('.npz'), ranks=ranks, queries=queries,
                        triples=queries[:, [0, 2, 1]], evidence=evidence if evidence is not None else np.empty((0, 12)))
    value['ranks_sha256'] = sha(path.with_suffix('.npz'))
    dump(path, value, exclusive=True)


def run(args):
    frozen = read(ROOT / 'frozen.json')
    p = read(ROOT / 'protocol.json')
    assert frozen['protocol_sha256'] == sha(ROOT / 'protocol.json')
    assert frozen['observation_manifest_sha256'] == sha(args.observations / 'manifest.json')
    assert frozen['model_manifest_sha256'] == sha(ROOT / 'model_manifest.json')
    assert frozen['matched_global_protocol_sha256'] == sha(ROOT / 'matched_global_protocol.json')
    # Adaptation-relevant code must match the freeze. Reporting code may be added later.
    for name, expected in frozen['code_hashes'].items():
        assert sha(ROOT / name) == expected, name
    for label, rec in frozen['selections'].items():
        assert sha(ROOT / 'adaptation' / (label + '.json')) == rec['sha256']
    fh = sha(ROOT / 'frozen.json')
    folder = ROOT / 'evaluation'
    folder.mkdir(exist_ok=True)
    expected = {r['key']: r for r in read(args.stage5 / 'analysis/target_inputs.json')}
    for group, key in p['targets']:
        graph = load_observed(args.observations, key)
        queries, truth = load_queries(args, key, graph)
        trip = queries[:, [0, 2, 1]].copy()
        assert hashlib.sha256(trip.tobytes()).hexdigest() == expected[key]['query_sha256']
        e = truth[:, [0, 2]].T.copy()
        r = truth[:, 1].copy()
        assert hashlib.sha256(e.tobytes() + r.tobytes()).hexdigest() == expected[key]['filtered_sha256']
        assert hashlib.sha256(graph.edge_index.tobytes() + graph.edge_type.tobytes()).hexdigest() == expected[key]['observed_sha256']
        data, filtered, topo = build_graph(graph, queries, truth, ROOT / 'cache/topology')
        he = data.relation_hypergraph.edge_index.cpu().numpy()
        ht = data.relation_hypergraph.edge_type.cpu().numpy()
        assert hashlib.sha256(he.tobytes() + ht.tobytes()).hexdigest() == expected[key]['hypergraph_sha256']
        for family, seeds in [('T-4D', p['training_seeds']), ('B0', [1024]), ('B1', p['training_seeds']), ('B2', p['training_seeds'])]:
            for seed in seeds:
                path = folder / f'{key}__{family}_s{seed}.json'
                if path.exists():
                    assert read(path)['frozen_sha256'] == fh
                    continue
                m = load_model(args.stage5, args.history, args.official, family, seed)
                before = state_hash(m)
                if family == 'T-4D':
                    recs = [read(ROOT / 'adaptation' / f'{key}__rate{p["target_rate"]}__p{ps}__s{seed}.json') for ps in p['proxy_seeds']]
                    cfgs = recs[0]['configs']
                    assert all(rec['configs'] == cfgs for rec in recs)
                else:
                    native = 1. if family == 'B2' else 0.
                    # Post-freeze explanatory ablation only. These are not selected using formal results.
                    cfgs = [[native, g, 'conditional'] for g in p['gamma_grid']]
                    cfgs += [[native, g, 'global'] for g in p['global_gamma_grid'] if g != 0]
                from stage6.matched_control import append as append_matched_control
                cfgs = append_matched_control(cfgs, p['alpha_grid'] if family == 'T-4D' else [native])
                log('formal_begin', key=key, family=family, seed=seed, facts=len(queries))
                ranks, info, evidence = safe_evaluate(m, data, filtered, graph, cfgs, args.batch_size,
                                                      collect=family == 'T-4D' or (family == 'B0' and key == 'NELLInductive_v1'))
                assert before == state_hash(m)
                save_result(path, dict(key=key, group=group, family=family, training_seed=seed, configs=cfgs,
                    info=info, topology=topo, checkpoint_sha256=m.tsdc_checkpoint_sha256, frozen_sha256=fh,
                    query_order_sha256=expected[key]['query_sha256'], filtered_sha256=expected[key]['filtered_sha256'],
                    metrics=[metrics(r) for r in ranks], no_weight_update=True), ranks, queries, evidence)
                if family == 'B0' and key == 'NELLInductive_v1':
                    # Isolate repeat / batch dependence without selecting a favorable run.
                    base_ix = cfgs.index([0., 0., 'conditional'])
                    repeats = [ranks[base_ix]]
                    for bs in [args.batch_size, 1, 4]:
                        rr, ii, ee = safe_evaluate(m, data, filtered, graph, [[0., 0., 'conditional']], bs, True)
                        repeats.append(rr[0])
                        np.savez_compressed(folder / f'B0_repeat_NELLv1_b{bs}_{len(repeats)}.npz', ranks=rr[0], evidence=ee)
                    rows = [dict(batch=bs, rank_changes=int((r != repeats[0]).sum()), mrr=metrics(r)['mrr'],
                                 delta_mrr=float((1 / r - 1 / repeats[0]).mean()))
                            for bs, r in zip([args.batch_size, args.batch_size, 1, 4], repeats)]
                    dump(ROOT / 'results/b0_repeat.json', dict(rows=rows, compared_to='first current B0; no favorable run selection'))
                log('formal_complete', key=key, family=family, seed=seed, seconds=info['seconds'])
                del m
                gc.collect()
                torch.cuda.empty_cache()
        del data, filtered
        gc.collect()
        torch.cuda.empty_cache()
    dump(ROOT / 'results/formal_status.json', dict(status='complete', graphs=11, model_runs=110, frozen_sha256=fh))
