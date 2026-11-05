import gc
import hashlib
import time
import numpy as np
import torch
from stage6.util import ROOT, read, dump, sha, identity, log
from stage6.observed_loader import load_observed
from stage6.proxy_split import make_split
from stage6.proxy_graph import build_graph
from stage6.models import load_model, state_hash
from stage6.evaluator import safe_evaluate, metrics
from stage6.protocol import configs
from stage6.select import choose, rr


def decisions(ranks, cfgs, cut, seed, split_seed, key, p):
    kw = dict(kappa=p['lcb_kappa'], min_select=p['min_selection_facts'], min_confirm=p['min_confirmation_facts'])
    cond = [i for i, c in enumerate(cfgs) if c[2] == 'conditional' and c[0] in p['alpha_grid']]
    top = [i for i in cond if cfgs[i][1] == 0]
    ent = [i for i in cond if cfgs[i][0] == 0]
    glob = [i for i, c in enumerate(cfgs) if c[0] == 0 and (c[2] == 'global' or c == [0., 0., 'conditional'])]
    jglob = [i for i, c in enumerate(cfgs) if c[0] in p['alpha_grid'] and (c[2] == 'global' or c[1] == 0)]
    arms = {name: choose(ranks, cfgs, cut, allowed, **kw) for name, allowed in
            [('joint_lcb', cond), ('topology_lcb', top), ('entity_lcb', ent), ('global_lcb', glob), ('joint_global_lcb', jglob)]}
    arms['joint_mean'] = dict(arms['joint_lcb'], selected=arms['joint_lcb']['candidate'],
                              final_index=arms['joint_lcb']['candidate_index'], fallback=False)
    ref = cfgs.index([0., 0., 'conditional'])
    rngseed = int(hashlib.sha256(repr((key, seed, split_seed, 6629)).encode()).hexdigest()[:16], 16)
    rnd = int(np.random.default_rng(rngseed).choice(cond))
    for name, ix in [('reference', ref), ('fixed', cfgs.index([.25, 0., 'conditional'])), ('cv', len(cfgs) - 1), ('random', rnd)]:
        arms[name] = dict(selected=cfgs[ix], final_index=ix, fallback=False, reference_index=ref)
    return arms


def run_one(args, key, rate, split_seed, source=False):
    graph_started = time.perf_counter()
    if not source:
        assert len(list((ROOT / 'source_validation').glob('*.json'))) == 9, 'source reliability checks must finish before target adaptation'
        assert len(list((ROOT / 'source_density').glob('*.json'))) == 9, 'source density checks must finish before target adaptation'
        assert read(ROOT / 'results/numeric_tests.json')['status'] == 'passed'
    p = read(ROOT / 'protocol.json')
    graph = load_observed(args.observations, key)
    rem, queries, cut, split = make_split(graph, split_seed, rate, p['max_proxy_facts'], p['min_endpoint_remaining_degree'])
    label = f'{key}__rate{rate}__p{split_seed}'
    folder = ROOT / ('source_diagnostics' if source else 'adaptation')
    folder.mkdir(exist_ok=True)
    split_path = folder / (label + '__split.npz')
    if not split_path.exists():
        np.savez_compressed(split_path, queries=queries, cut=cut, rem_edge_index=rem.edge_index, rem_edge_type=rem.edge_type)
    # Closed data boundary: truth consists only of the observation graph, incl. held-out facts.
    truth = np.c_[graph.edge_index[0], graph.edge_type, graph.edge_index[1]]
    data, filtered, topo = build_graph(rem, queries, truth, ROOT / 'cache/topology')
    # CV control is defined on the complete observed graph, as in stage five.
    # Only this predeclared scalar enters the control; proxy gate signatures stay residual-only.
    from topology import extract
    fullsig = extract(graph.edge_index, graph.edge_type, graph.n, graph.R, cache=ROOT / 'cache/topology')
    raw = fullsig['raw'][:, 3]
    cv = float(raw.std() / max(abs(raw.mean()), 1e-6)) if len(raw) else 0.
    cfgs = configs(p, .15 + .10 * cv / (1 + cv))
    del fullsig
    graph_preparation_seconds = time.perf_counter() - graph_started
    for seed in p['training_seeds']:
        dest = folder / (label + f'__s{seed}.json')
        if dest.exists():
            rec = read(dest)
            assert rec['protocol_sha256'] == sha(ROOT / 'protocol.json') and rec['split'] == split and rec['confirmation_protocol_version'] == 2
            assert sha(dest.with_suffix('.npz')) == rec['ranks_sha256']
            continue
        started = time.perf_counter()
        m = load_model(args.stage5, args.history, args.official, 'T-4D', seed)
        before = state_hash(m)
        log('proxy_begin', key=key, seed=seed, proxy_seed=split_seed, rate=rate, facts=len(queries))
        reused = None
        # A rate cap can produce exactly the same split. Reuse only after checking
        # every semantic identity, not filename/rate similarity.
        if source:
            for previous in folder.glob(f'{key}__rate*__p{split_seed}__s{seed}.json'):
                old = read(previous)
                if (old['model_sha256'] == m.tsdc_checkpoint_sha256 and old['configs'] == cfgs and
                    old['split']['graph_sha256'] == graph.key and old['split']['residual_sha256'] == rem.key and
                    old['split']['hidden_sha256'] == split['hidden_sha256'] and
                    old['split']['selection_sha256'] == split['selection_sha256'] and
                    old['protocol_sha256'] == sha(ROOT / 'protocol.json') and old.get('confirmation_protocol_version') == 2):
                    assert sha(previous.with_suffix('.npz')) == old['ranks_sha256']
                    with np.load(previous.with_suffix('.npz')) as z:
                        assert np.array_equal(z['queries'], queries) and int(z['cut']) == cut
                        ranks = z['ranks']
                    info = dict(old['info'], seconds=0., reused_from=str(previous),
                                original_evaluation_seconds=old['info'].get('original_evaluation_seconds', old['info']['seconds']))
                    reused = previous
                    break
        if reused is None:
            # First pass: NO confirmation queries in the model input at all.
            all_index, all_type = data.target_edge_index, data.target_edge_type
            data.target_edge_index = all_index[:, :cut]
            data.target_edge_type = all_type[:cut]
            selection_ranks, selection_info, _ = safe_evaluate(m, data, filtered, rem, cfgs, args.batch_size)
            initial = decisions(selection_ranks, cfgs, cut, seed, split_seed, key, p)
            selected_indices = sorted({cfgs.index([0., 0., 'conditional'])} |
                {initial[name]['candidate_index'] for name in ['joint_lcb', 'topology_lcb', 'entity_lcb', 'global_lcb', 'joint_global_lcb']})
            confirmation_configs = [cfgs[i] for i in selected_indices]
            # Second pass: one winner per independently predeclared arm + reference.
            data.target_edge_index = all_index[:, cut:]
            data.target_edge_type = all_type[cut:]
            confirmation_ranks, confirmation_info, _ = safe_evaluate(m, data, filtered, rem, confirmation_configs, args.batch_size)
            data.target_edge_index, data.target_edge_type = all_index, all_type
            ranks = np.full((len(cfgs), len(queries), 2), np.nan)
            ranks[:, :cut] = selection_ranks
            for i, actual in enumerate(selected_indices):
                ranks[actual, cut:] = confirmation_ranks[i]
            info = dict(selection_info, seconds=selection_info['seconds'] + confirmation_info['seconds'],
                peak_allocated=max(selection_info['peak_allocated'], confirmation_info['peak_allocated']),
                selection_seconds=selection_info['seconds'], confirmation_seconds=confirmation_info['seconds'],
                confirmation_evaluated_indices=selected_indices, confirmation_configs=confirmation_configs)
        assert state_hash(m) == before and all(t.grad is None for t in m.parameters())
        arms = decisions(ranks, cfgs, cut, seed, split_seed, key, p)
        np.savez_compressed(dest.with_suffix('.npz'), ranks=ranks, queries=queries,
                            configs=np.asarray([str(c) for c in cfgs]), cut=cut)
        rec = dict(key=key, training_seed=seed, proxy_seed=split_seed, rate=rate, split=split, topology=topo,
                   configs=cfgs, decisions=arms, model_sha256=m.tsdc_checkpoint_sha256,
                   state_sha256=before, no_weight_update=True, info=info, total_seconds=time.perf_counter() - started,
                   graph_preparation_seconds_shared=graph_preparation_seconds,
                   ranks_sha256=sha(dest.with_suffix('.npz')), protocol_sha256=sha(ROOT / 'protocol.json'),
                   selection_metrics=[metrics(r[:, :]) if cut else None for r in ranks[:, :cut]],
                   confirmation_protocol_version=2,
                   confirmation_metrics=[metrics(r) if len(queries) > cut and np.isfinite(r).all() else None for r in ranks[:, cut:]])
        dump(dest, rec, exclusive=True)
        log('proxy_complete', key=key, seed=seed, proxy_seed=split_seed, rate=rate,
            theta=arms['joint_lcb']['selected'], fallback=arms['joint_lcb']['fallback'], seconds=rec['total_seconds'])
        del m
        gc.collect()
        torch.cuda.empty_cache()
    del data, filtered
    gc.collect()
    torch.cuda.empty_cache()


def freeze(args):
    from stage6.matched_control import register as register_matched_control
    matched_hash = register_matched_control()
    p = read(ROOT / 'protocol.json')
    records = {}
    for _, key in p['targets']:
        for split_seed in p['proxy_seeds']:
            for seed in p['training_seeds']:
                label = f'{key}__rate{p["target_rate"]}__p{split_seed}__s{seed}'
                path = ROOT / 'adaptation' / (label + '.json')
                rec = read(path)
                assert rec['protocol_sha256'] == sha(ROOT / 'protocol.json')
                records[label] = dict(sha256=sha(path), model_sha256=rec['model_sha256'],
                    observed_sha256=rec['split']['graph_sha256'], residual_sha256=rec['split']['residual_sha256'],
                    decisions=rec['decisions'], training_seed=seed, proxy_seed=split_seed, key=key)
    report_code = {'analysis.py', 'report_writer.py', 'plot.py'}
    code = {str(x.relative_to(ROOT)): sha(x) for x in ROOT.rglob('*.py') if x.name not in report_code}
    code.update({name: sha(ROOT / name) for name in ['run.sh', 'workflow.sh']})
    authoring = {name: sha(ROOT / name) for name in sorted(report_code)}
    source_checks = {}
    for x in (ROOT / 'source_density').glob('*.json'):
        source_checks[str(x.relative_to(ROOT))] = sha(x)
    assert len(source_checks) == 9, 'source density diagnostic incomplete'
    assert read(ROOT / 'results/legacy_adapter_tests.json')['status'] == 'passed'
    for key in ['wn18rr_train', 'fb15k237_train', 'codex_medium_train']:
        for seed in p['training_seeds']:
            v = ROOT / 'source_validation' / f'{key}__s{seed}.json'
            assert read(v)['protocol_sha256'] == sha(ROOT / 'protocol.json')
            source_checks[str(v.relative_to(ROOT))] = sha(v)
            for rate in p['source_rates']:
                for ps in p['proxy_seeds']:
                    x = ROOT / 'source_diagnostics' / f'{key}__rate{rate}__p{ps}__s{seed}.json'
                    assert read(x)['confirmation_protocol_version'] == 2
                    source_checks[str(x.relative_to(ROOT))] = sha(x)
    value = dict(protocol_sha256=sha(ROOT / 'protocol.json'), observation_manifest_sha256=sha(args.observations / 'manifest.json'),
                 selections=records, code_hashes=code, initial_report_code_hashes=authoring, source_diagnostic_hashes=source_checks,
                 stage5_protected_manifest_sha256=sha(ROOT / 'protected_stage5.json'),
                 history_protected_manifest_sha256=sha(ROOT / 'protected_history.json'),
                 environment_sha256=sha(ROOT / 'results/environment.json'),
                 model_manifest_sha256=sha(ROOT / 'model_manifest.json'),
                 source_density_protocol_sha256=sha(ROOT / 'source_density_protocol.json'),
                 matched_global_protocol_sha256=matched_hash,
                 status='all target selections frozen before formal query loading')
    path = ROOT / 'frozen.json'
    if path.exists():
        assert read(path) == value
    else:
        dump(path, value, exclusive=True)
    log('all_selections_frozen', sha256=sha(path), records=len(records))
