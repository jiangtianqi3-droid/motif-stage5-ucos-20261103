"""Post-freeze analysis only. Never writes protocol or selection artifacts."""
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
from stage6.util import ROOT, read, dump, sha, log
from stage6.evaluator import metrics
from stage6.select import rr
from stage6.statistics import paired_bootstrap
from stage6.observed_loader import load_observed


def csv_write(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return dict(pearson=None, spearman=None, n=len(x))
    return dict(pearson=float(stats.pearsonr(x, y)[0]), spearman=float(stats.spearmanr(x, y)[0]), n=len(x))


def mdtable(rows, columns, fmt=None):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
    for row in rows:
        cells = []
        for k in columns:
            v = row.get(k)
            if v is None:
                cells.append('NA')
            elif isinstance(v, float):
                cells.append(f'{v:.6f}')
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def selection_tables(p):
    proxy_rows, selection_rows, source_rows = [], [], []
    for folder, dest in [('adaptation', proxy_rows), ('source_diagnostics', source_rows)]:
        for path in sorted((ROOT / folder).glob('*.json')):
            rec = read(path)
            for i, cfg in enumerate(rec['configs']):
                for part, key in [('selection', 'selection_metrics'), ('confirmation', 'confirmation_metrics')]:
                    met = rec[key][i]
                    dest.append(dict(key=rec['key'], training_seed=rec['training_seed'], proxy_seed=rec['proxy_seed'],
                        rate=rec['rate'], partition=part, alpha=cfg[0], gamma=cfg[1], mode=cfg[2],
                        facts=rec['split'][part], observed_sha256=rec['split']['graph_sha256'],
                        residual_sha256=rec['split']['residual_sha256'], **(met or {})))
            if folder == 'adaptation':
                for method, decision in rec['decisions'].items():
                    a, g, mode = decision['selected']
                    selection_rows.append(dict(key=rec['key'], training_seed=rec['training_seed'], proxy_seed=rec['proxy_seed'],
                        method=method, alpha=a, gamma=g, mode=mode, fallback=decision['fallback'],
                        confirmation_lcb=decision.get('confirmation_lcb'), candidate=str(decision.get('candidate')),
                        candidate_is_reference=decision.get('candidate') == [0., 0., 'conditional'],
                        fallback_reason=('none' if not decision.get('fallback') else
                            'insufficient_samples' if decision.get('sufficient') is False else
                            'reference_won_selection' if decision.get('candidate') == [0., 0., 'conditional'] else
                            'nonpositive_confirmation_lcb' if decision.get('fallback') else 'none'),
                        confirmation_delta=decision.get('confirmation_delta'),
                        selection_facts=rec['split']['selection'], confirmation_facts=rec['split']['confirmation'],
                        adaptation_seconds=rec['total_seconds'], peak_allocated=rec['info']['peak_allocated'],
                        checkpoint_sha256=rec['model_sha256']))
    csv_write(ROOT / 'results/proxy_metrics.csv', proxy_rows)
    csv_write(ROOT / 'results/source_proxy_metrics.csv', source_rows)
    csv_write(ROOT / 'results/selection_decisions.csv', selection_rows)
    return proxy_rows, selection_rows


def load_results(args, p):
    per_fact, rank_arrays, rows, all_splits, baseline_repeats, grid_rows = {}, {}, [], [], [], []
    for group, key in p['targets']:
        per_fact[key] = defaultdict(list)
        rank_arrays[key] = defaultdict(list)
        for seed in p['training_seeds']:
            path = ROOT / 'evaluation' / f'{key}__T-4D_s{seed}.json'
            rec = read(path)
            for config, metric in zip(rec['configs'], rec['metrics']):
                grid_rows.append(dict(key=key, group=group, family='T-4D', training_seed=seed,
                    alpha=config[0], gamma=config[1], mode=config[2], **metric))
            with np.load(path.with_suffix('.npz')) as z:
                ranks = z['ranks']
            for ps in p['proxy_seeds']:
                adapt = read(ROOT / 'adaptation' / f'{key}__rate{p["target_rate"]}__p{ps}__s{seed}.json')
                for method, dec in adapt['decisions'].items():
                    cfg = dec['selected']
                    ix = rec['configs'].index(cfg)
                    met = metrics(ranks[ix])
                    row = dict(key=key, group=group, method=method, training_seed=seed, proxy_seed=ps,
                               alpha=cfg[0], gamma=cfg[1], mode=cfg[2], **met)
                    all_splits.append(row)
                    if ps == p['primary_proxy_seed']:
                        rows.append(row)
                        per_fact[key][method].append(rr(ranks[ix]))
                        rank_arrays[key][method].append(ranks[ix])
        for family, seeds in [('B0', [1024]), ('B1', p['training_seeds']), ('B2', p['training_seeds'])]:
            for seed in seeds:
                path = ROOT / 'evaluation' / f'{key}__{family}_s{seed}.json'
                rec = read(path)
                for config, metric in zip(rec['configs'], rec['metrics']):
                    grid_rows.append(dict(key=key, group=group, family=family, training_seed=seed,
                        alpha=config[0], gamma=config[1], mode=config[2], **metric))
                with np.load(path.with_suffix('.npz')) as z:
                    ranks, triples = z['ranks'], z['triples']
                native = 1. if family == 'B2' else 0.
                ix = rec['configs'].index([native, 0., 'conditional'])
                rows.append(dict(key=key, group=group, method=family, training_seed=seed, proxy_seed=None,
                                 alpha=native, gamma=0., mode='conditional', **metrics(ranks[ix])))
                per_fact[key][family].append(rr(ranks[ix]))
                rank_arrays[key][family].append(ranks[ix])
                legacy = args.stage5 / 'evaluation' / (f'{key}__B0.npz' if family == 'B0' else f'{key}__{family}_s{seed}.npz')
                with np.load(legacy) as z:
                    assert np.array_equal(z['triples'], triples)
                    old = z['ranks']
                baseline_repeats.append(dict(key=key, family=family, training_seed=seed,
                    old_mrr=metrics(old)['mrr'], current_mrr=metrics(ranks[ix])['mrr'],
                    delta_mrr=metrics(ranks[ix])['mrr'] - metrics(old)['mrr'],
                    changed_directional_ranks=int((ranks[ix] != old).sum()), total_directions=old.size,
                    legacy_sha256=sha(legacy), current_sha256=sha(path.with_suffix('.npz'))))
    for key, methods in per_fact.items():
        for method in methods:
            methods[method] = np.mean(methods[method], axis=0)
    csv_write(ROOT / 'results/P0_metrics.csv', rows)
    csv_write(ROOT / 'results/ablation.csv', all_splits)
    csv_write(ROOT / 'results/baseline_reproducibility.csv', baseline_repeats)
    csv_write(ROOT / 'results/all_fixed_grid_metrics.csv', grid_rows)
    return per_fact, rank_arrays, rows, all_splits, baseline_repeats


def summarize(p, per_fact, ranks, rows):
    methods = list(per_fact[p['targets'][0][1]])
    metric_names = ['mrr', 'hits_at_1', 'hits_at_3', 'hits_at_10']
    graph_rows, macro_rows, boots, groups = [], [], [], []
    comparisons = [('joint_lcb', b) for b in ['reference', 'fixed', 'random', 'joint_mean', 'B0', 'B1', 'B2', 'topology_lcb', 'entity_lcb', 'joint_global_lcb']]
    comparisons += [(m, 'reference') for m in ['entity_lcb', 'topology_lcb', 'global_lcb', 'joint_global_lcb', 'joint_mean']]
    comparisons += [('joint_mean', 'fixed'), ('joint_mean', 'random'), ('entity_lcb', 'global_lcb')]
    for group, key in p['targets']:
        for method in methods:
            subset = [r for r in rows if r['key'] == key and r['method'] == method]
            graph_rows.append(dict(key=key, group=group, method=method,
                **{m: float(np.mean([r[m] for r in subset])) for m in metric_names},
                mrr_seed_sd=float(np.std([r['mrr'] for r in subset], ddof=1)) if len(subset) > 1 else None,
                delta_B0=float((per_fact[key][method] - per_fact[key]['B0']).mean()),
                delta_B1=float((per_fact[key][method] - per_fact[key]['B1']).mean()),
                delta_B2=float((per_fact[key][method] - per_fact[key]['B2']).mean())))
        for method, base in comparisons:
            delta = per_fact[key][method] - per_fact[key][base]
            record, samples = paired_bootstrap(delta, p['bootstrap'], p['bootstrap_seed'])
            boots.append(dict(key=key, group=group, method=method, comparator=base, **record))
        maskpath = ROOT.parent / 'analysis' / f'query_masks_{key}.npz'
        with np.load(maskpath) as z:
            masks = {k: z[k] for k in ['high_degree_hub', 'low_relation_frequency', 'low_incident_motif_support']}
        for name, mask in masks.items():
            for base in ['reference', 'B0', 'B1', 'B2']:
                record, _ = paired_bootstrap((per_fact[key]['joint_lcb'] - per_fact[key][base])[mask], p['bootstrap'], p['bootstrap_seed'])
                groups.append(dict(key=key, group=group, query_group=name, method='joint_lcb', comparator=base, **record))
    for group in ['locked', 'historical', 'external']:
        keys = [key for g, key in p['targets'] if g == group]
        for method in methods:
            seed_values = []
            seeds = [1024] if method == 'B0' else p['training_seeds']
            for seed in seeds:
                seed_values.append(float(np.mean([r['mrr'] for r in rows if r['key'] in keys and r['method'] == method and r['training_seed'] == seed])))
            chosen = [r for r in graph_rows if r['key'] in keys and r['method'] == method]
            macro_rows.append(dict(group=group, method=method, mrr=float(np.mean([r['mrr'] for r in chosen])),
                                  mrr_seed_sd=float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else None,
                                  delta_B0=float(np.mean([r['delta_B0'] for r in chosen])),
                                  delta_B1=float(np.mean([r['delta_B1'] for r in chosen])), delta_B2=float(np.mean([r['delta_B2'] for r in chosen]))))
        for method, base in comparisons:
            # Use graph-specific independent resampling streams for the macro CI.
            # Same deterministic seed would correlate index streams between graphs.
            independent = []
            for j, key in enumerate(keys):
                _, bs = paired_bootstrap(per_fact[key][method] - per_fact[key][base], p['bootstrap'], p['bootstrap_seed'] + 101 * (j + 1))
                independent.append(bs)
            samples = np.mean(independent, axis=0)
            lo, hi = np.quantile(samples, [.025, .975])
            boots.append(dict(key='macro_' + group, group=group, method=method, comparator=base,
                delta=float(np.mean([(per_fact[key][method] - per_fact[key][base]).mean() for key in keys])),
                lo=float(lo), hi=float(hi), facts=sum(len(per_fact[key]['joint_lcb']) for key in keys),
                interval_scope='conditional within-graph triple bootstrap; graph collection fixed'))
    csv_write(ROOT / 'results/graph_metrics.csv', graph_rows)
    csv_write(ROOT / 'results/macro_metrics.csv', macro_rows)
    csv_write(ROOT / 'results/paired_bootstrap.csv', boots)
    csv_write(ROOT / 'results/query_groups.csv', groups)
    return graph_rows, macro_rows, boots, groups


def reliability(p):
    points, within, stability, source_points, source_splits, selected_topology = [], [], [], [], [], []
    for source in [False, True]:
        keys = [k for _, k in p['targets']] if not source else ['fb15k237_train', 'wn18rr_train', 'codex_medium_train']
        for key in keys:
            val_folder = 'source_validation' if source else 'evaluation'
            adapter_folder = 'source_diagnostics' if source else 'adaptation'
            source_rates = p['source_rates'] if source else [p['target_rate']]
            for rate in source_rates:
                px, ty = [], []
                for alpha in p['alpha_grid']:
                    if alpha == 0:
                        continue
                    proxy, target = [], []
                    for seed in p['training_seeds']:
                        file = f'{key}__s{seed}' if source else f'{key}__T-4D_s{seed}'
                        vr = read(ROOT / val_folder / (file + '.json'))
                        vi = vr['configs'].index([alpha, 0., 'conditional'])
                        v0 = vr['configs'].index([0., 0., 'conditional'])
                        target.append(vr['metrics'][vi]['mrr'] - vr['metrics'][v0]['mrr'])
                        for ps in p['proxy_seeds']:
                            pr = read(ROOT / adapter_folder / f'{key}__rate{rate}__p{ps}__s{seed}.json')
                            pi = pr['configs'].index([alpha, 0., 'conditional'])
                            p0 = pr['configs'].index([0., 0., 'conditional'])
                            proxy.append(pr['selection_metrics'][pi]['mrr'] - pr['selection_metrics'][p0]['mrr'])
                    x, y = float(np.mean(proxy)), float(np.mean(target))
                    px.append(x)
                    ty.append(y)
                    row = dict(key=key, rate=rate, alpha=alpha, proxy_gain=x, formal_gain=y)
                    (source_points if source else points).append(row)
                within.append(dict(key=key, source=source, rate=rate, **corr(px, ty)))
                for seed in p['training_seeds']:
                    selected, candidate = [], []
                    valfile = f'{key}__s{seed}' if source else f'{key}__T-4D_s{seed}'
                    validation = read(ROOT / val_folder / (valfile + '.json'))
                    v0 = validation['configs'].index([0., 0., 'conditional'])
                    for ps in p['proxy_seeds']:
                        rec = read(ROOT / adapter_folder / f'{key}__rate{rate}__p{ps}__s{seed}.json')
                        d = rec['decisions']['joint_lcb']
                        selected.append(str(d['selected']))
                        candidate.append(str(d['candidate']))
                        topo = rec['decisions']['topology_lcb']
                        vi = validation['configs'].index(topo['candidate'])
                        gain = validation['metrics'][vi]['mrr'] - validation['metrics'][v0]['mrr']
                        selected_topology.append(dict(key=key, source=source, rate=rate, training_seed=seed, proxy_seed=ps,
                            candidate_alpha=topo['candidate'][0], proxy_selection_gain=topo['selection_mrr'] - topo['reference_selection_mrr'],
                            proxy_confirmation_gain=topo['confirmation_delta'], fallback=topo['fallback'],
                            formal_candidate_gain=gain, formal_final_gain=0. if topo['fallback'] else gain))
                        if source:
                            source_splits.append(dict(key=key, rate=rate, proxy_seed=ps, training_seed=seed,
                                requested=rec['split']['requested'], actual=rec['split']['actual'],
                                unique_observed_facts=rec['split']['unique_observed_facts'],
                                actual_rate=rec['split']['actual'] / rec['split']['unique_observed_facts'],
                                residual_sha256=rec['split']['residual_sha256'], fallback=d['fallback'], selected=str(d['selected'])))
                    stability.append(dict(key=key, source=source, rate=rate, training_seed=seed,
                        distinct_final=len(set(selected)), modal_final_fraction=max(Counter(selected).values()) / len(selected),
                        distinct_candidate=len(set(candidate)), selected=';'.join(selected)))
    csv_write(ROOT / 'results/proxy_formal_correlation.csv', points)
    csv_write(ROOT / 'results/source_proxy_validation_correlation.csv', source_points)
    csv_write(ROOT / 'results/within_graph_correlation.csv', within)
    csv_write(ROOT / 'results/selection_stability.csv', stability)
    csv_write(ROOT / 'results/source_mask_rates.csv', source_splits)
    csv_write(ROOT / 'results/selected_topology_prediction.csv', selected_topology)
    fixed = [r for r in points if r['alpha'] == .25]
    source_fixed = [r for r in source_points if r['alpha'] == .25 and r['rate'] == .15]
    summary = dict(graph_level_alpha025=corr([r['proxy_gain'] for r in fixed], [r['formal_gain'] for r in fixed]),
                   source_graph_level_alpha025=corr([r['proxy_gain'] for r in source_fixed], [r['formal_gain'] for r in source_fixed]))
    summary['fixed_alpha025_sign_agreement_graphs'] = int(sum(np.sign(r['proxy_gain']) == np.sign(r['formal_gain']) for r in fixed))
    graph_predictions = []
    for _, key in p['targets']:
        rs = [r for r in selected_topology if r['key'] == key and not r['source']]
        graph_predictions.append(dict(key=key, proxy_gain=float(np.mean([r['proxy_selection_gain'] for r in rs])),
            formal_gain=float(np.mean([r['formal_candidate_gain'] for r in rs])),
            positive_formal_candidates=sum(r['formal_candidate_gain'] > 0 for r in rs), attempts=len(rs)))
    summary['selected_topology_graph_level'] = corr([r['proxy_gain'] for r in graph_predictions], [r['formal_gain'] for r in graph_predictions])
    csv_write(ROOT / 'results/selected_topology_graph_prediction.csv', graph_predictions)
    dump(ROOT / 'results/correlation_summary.json', summary)
    return points, within, stability, source_splits, summary


def density_analysis(p):
    rows, differences = [], []
    for key in ['fb15k237_train', 'wn18rr_train', 'codex_medium_train']:
        arrays = {}
        hashes = set()
        for rate in p['source_rates']:
            path = ROOT / 'source_density' / f'{key}__rate{rate}.json'
            rec = read(path)
            hashes.add(rec['query_sha256'])
            with np.load(path.with_suffix('.npz')) as z:
                ranks = z['ranks']
            ix0 = rec['configs'].index([0., 0., 'conditional'])
            ix1 = rec['configs'].index([.25, 0., 'conditional'])
            arrays[rate] = rr(ranks[ix1]) - rr(ranks[ix0])
            for cfg, met in zip(rec['configs'], rec['metrics']):
                rows.append(dict(key=key, nominal_rate=rate, actual_rate=rec['actual_rate'],
                    queries=len(ranks[0]), alpha=cfg[0], query_sha256=rec['query_sha256'], **met))
        assert len(hashes) == 1
        for rate in [.10, .15]:
            ci, _ = paired_bootstrap(arrays[rate] - arrays[.05], p['bootstrap'], p['bootstrap_seed'])
            differences.append(dict(key=key, contrast=f'alpha025-minus-alpha0 gain at {rate} minus at .05', **ci))
    csv_write(ROOT / 'results/source_density_metrics.csv', rows)
    csv_write(ROOT / 'results/source_density_sensitivity.csv', differences)


def distribution_analysis(args, p):
    from scipy.spatial.distance import jensenshannon
    from stage6.observed_loader import ObservedGraph
    rows = []
    def features(graph, q):
        degree = np.bincount(graph.edge_index[1], minlength=graph.n)
        freq = np.bincount(graph.edge_type, minlength=graph.R)
        local = np.bincount(graph.edge_type * graph.n + graph.edge_index[1], minlength=graph.R * graph.n)
        relations = np.c_[q[:, 1], q[:, 1] + graph.R // 2].ravel()
        candidates = np.c_[q[:, 2], q[:, 0]].ravel()
        anchors = np.c_[q[:, 0], q[:, 2]].ravel()
        support = local[relations * graph.n + candidates]
        return dict(mean_log_anchor_degree=float(np.log1p(degree[anchors]).mean()),
            mean_log_true_candidate_degree=float(np.log1p(degree[candidates]).mean()),
            zero_anchor_degree_fraction=float((degree[anchors] == 0).mean()),
            zero_true_relation_support_fraction=float((support == 0).mean()),
            mean_log_query_relation_frequency=float(np.log1p(freq[relations]).mean()))
    for group, key in p['targets']:
        graph = load_observed(args.observations, key)
        with np.load(ROOT / 'evaluation' / f'{key}__T-4D_s1024.npz') as z:
            formal_queries = z['queries']
        formal = features(graph, formal_queries)
        ft = np.bincount(formal_queries[:, 1], minlength=graph.R // 2).astype(float)
        for ps in p['proxy_seeds']:
            path = ROOT / 'adaptation' / f'{key}__rate{p["target_rate"]}__p{ps}__split.npz'
            with np.load(path) as z:
                rem = ObservedGraph(z['rem_edge_index'], z['rem_edge_type'], graph.n, graph.R, {})
                q = z['queries']
            proxy = features(rem, q)
            pr = np.bincount(q[:, 1], minlength=graph.R // 2).astype(float)
            js = float(jensenshannon(pr + 1e-12, ft + 1e-12, base=2))
            rows.append(dict(key=key, group=group, proxy_seed=ps, relation_distribution_js_distance=js,
                **{'proxy_' + name: value for name, value in proxy.items()},
                **{'formal_' + name: value for name, value in formal.items()}))
    csv_write(ROOT / 'results/proxy_formal_distribution.csv', rows)


def hub_analysis(args, p):
    records, detail_groups = [], []
    for group, key in p['targets']:
        graph = load_observed(args.observations, key)
        degree = np.bincount(graph.edge_index[1], minlength=graph.n)
        threshold = float(np.quantile(degree[degree > 0], .9))
        seed_rows = []
        for seed in p['training_seeds']:
            path = ROOT / 'evaluation' / f'{key}__T-4D_s{seed}.json'
            rec = read(path)
            with np.load(path.with_suffix('.npz')) as z:
                evidence, queries, ranks = z['evidence'], z['queries'], z['ranks']
            e = evidence[evidence[:, 0] == 0.]
            # evaluator appends tail and head batches; reorder to original triple/direction.
            e = e[np.lexsort((e[:, 2], e[:, 1]))]
            assert len(e) == 2 * len(queries)
            true_gl, true_cl, wrong_gl, wrong_cl = e[:, 5], e[:, 6], e[:, 7], e[:, 8]
            outrank = e[:, 10] >= e[:, 9]
            true_high, wrong_high = np.expm1(true_gl) >= threshold - 1e-4, np.expm1(wrong_gl) >= threshold - 1e-4
            ix0 = rec['configs'].index([0., 0., 'conditional'])
            ix1 = rec['configs'].index([0., .03, 'conditional'])
            delta = (1 / ranks[ix1] - 1 / ranks[ix0]).reshape(-1)
            anchors = np.c_[queries[:, 0], queries[:, 2]].ravel()
            anchor_high = degree[anchors] >= threshold
            q = np.c_[queries[:, 1], queries[:, 1] + graph.R // 2].ravel()
            # Exhaustive observed relation support for true and highest-scoring wrong candidates.
            support_truth_corr = corr(np.r_[true_cl, wrong_cl], np.r_[np.ones(len(e)), np.zeros(len(e))])
            row = dict(key=key, group=group, training_seed=seed, candidate_degree_threshold=threshold,
                high_query_entity_fraction=float(anchor_high.mean()), high_true_candidate_fraction=float(true_high.mean()),
                high_wrong_candidate_fraction=float(wrong_high.mean()), wrong_outrank_fraction=float(outrank.mean()),
                high_hub_outrank_fraction=float((outrank & wrong_high).mean()),
                true_conditional_log_mean=float(true_cl.mean()), wrong_conditional_log_mean=float(wrong_cl.mean()),
                true_global_log_mean=float(true_gl.mean()), wrong_global_log_mean=float(wrong_gl.mean()),
                conditional_log_vs_correct_spearman=support_truth_corr['spearman'],
                relation_support_vs_calibration_gain_spearman=corr(true_cl, delta)['spearman'],
                gamma003_delta_mrr=float(delta.mean()),
                gamma003_improved_directions=int((delta > 0).sum()), gamma003_harmed_directions=int((delta < 0).sum()),
                exact_score_ties_mean=float(e[:, 11].mean()))
            records.append(row)
            for name, mask in [('high_query_entity', anchor_high), ('high_true_candidate', true_high),
                               ('wrong_hub_outranks_true', wrong_high & outrank)]:
                # Keep only eligible directions *inside* each fact before averaging.
                # Taking both directions whenever either is high would make the
                # anchor-high and true-candidate-high groups identical by symmetry.
                direction_mask = mask.reshape(-1, 2)
                triplemask = direction_mask.any(1)
                directional_delta = 1 / ranks[ix1] - 1 / ranks[ix0]
                per_fact = (directional_delta * direction_mask).sum(1) / direction_mask.sum(1).clip(min=1)
                diff = per_fact[triplemask]
                ci, _ = paired_bootstrap(diff, p['bootstrap'], p['bootstrap_seed'])
                detail_groups.append(dict(key=key, training_seed=seed, query_group=name, comparator='alpha0 gamma0',
                                           method='alpha0 gamma0.03 conditional diagnostic',
                                           eligible_directions=int(direction_mask.sum()),
                                           paired_unit='fact; mean over eligible directions within each fact', **ci))
    csv_write(ROOT / 'results/hub_diagnostics.csv', records)
    csv_write(ROOT / 'results/hub_candidate_groups.csv', detail_groups)
    return records


def numerical_analysis():
    folder = ROOT / 'evaluation'
    with np.load(folder / 'NELLInductive_v1__B0_s1024.npz') as z:
        base = z['evidence']
        rec = read(folder / 'NELLInductive_v1__B0_s1024.json')
        ix = rec['configs'].index([0., 0., 'conditional'])
        base_ranks = z['ranks'][ix]
    base = base[np.lexsort((base[:, 2], base[:, 1]))]
    rows = []
    for path in sorted(folder.glob('B0_repeat_NELLv1_*.npz')):
        with np.load(path) as z:
            e, ranks = z['evidence'], z['ranks']
        e = e[np.lexsort((e[:, 2], e[:, 1]))]
        changed = (ranks != base_ranks).reshape(-1)
        assert np.array_equal(base[:, 1:4], e[:, 1:4])
        rows.append(dict(file=path.name, changed_directional_ranks=int(changed.sum()),
            max_true_score_abs_change=float(np.max(np.abs(e[:, 9] - base[:, 9]))),
            max_top_wrong_score_abs_change=float(np.max(np.abs(e[:, 10] - base[:, 10]))),
            changed_exact_tie_counts=int((e[:, 11] != base[:, 11]).sum()),
            base_mean_exact_tie_count=float(base[:, 11].mean()),
            mean_top_wrong_margin_on_changed=float((e[changed, 10] - e[changed, 9]).mean()) if changed.any() else None,
            delta_mrr=float((1 / ranks - 1 / base_ranks).mean())))
    csv_write(ROOT / 'results/b0_score_reproducibility.csv', rows)


def matched_degree_analysis(p):
    from stage6.matched_control import CONTROL
    rows, macro_rows = [], []
    macro_samples = defaultdict(list)
    for gi, (group, key) in enumerate(p['targets']):
        for fi, family in enumerate(['T-4D', 'B0', 'B1', 'B2']):
            seeds = [1024] if family == 'B0' else p['training_seeds']
            alpha = 1. if family == 'B2' else 0.
            cache = []
            for seed in seeds:
                path = ROOT / 'evaluation' / f'{key}__{family}_s{seed}.json'
                rec = read(path)
                with np.load(path.with_suffix('.npz')) as z:
                    cache.append((rec['configs'], z['ranks']))
            for li, gamma in enumerate(CONTROL['gamma_grid']):
                if gamma == 0:
                    continue
                conditional, glob, raw = [], [], []
                for cfgs, ranks in cache:
                    conditional.append(rr(ranks[cfgs.index([alpha, gamma, 'conditional'])]))
                    glob.append(rr(ranks[cfgs.index([alpha, gamma, 'global'])]))
                    raw.append(rr(ranks[cfgs.index([alpha, 0., 'conditional'])]))
                conditional, glob, raw = np.mean(conditional, 0), np.mean(glob, 0), np.mean(raw, 0)
                ci, samples = paired_bootstrap(conditional - glob, p['bootstrap'], p['bootstrap_seed'] + gi * 100000 + fi * 1000 + li)
                rows.append(dict(key=key, group=group, family=family, gamma=gamma, alpha=alpha,
                    conditional_mrr=float(conditional.mean()), global_mrr=float(glob.mean()), raw_mrr=float(raw.mean()),
                    conditional_gain=float((conditional - raw).mean()), global_gain=float((glob - raw).mean()), **ci))
                macro_samples[group, family, gamma].append(samples)
    for (group, family, gamma), samples in macro_samples.items():
        rs = [r for r in rows if r['group'] == group and r['family'] == family and r['gamma'] == gamma]
        bs = np.mean(samples, 0)
        lo, hi = np.quantile(bs, [.025, .975])
        macro_rows.append(dict(group=group, family=family, gamma=gamma,
            conditional_mrr=float(np.mean([r['conditional_mrr'] for r in rs])),
            global_mrr=float(np.mean([r['global_mrr'] for r in rs])),
            delta=float(np.mean([r['delta'] for r in rs])), lo=float(lo), hi=float(hi)))
    csv_write(ROOT / 'results/degree_matched_control.csv', rows)
    csv_write(ROOT / 'results/degree_matched_macro.csv', macro_rows)


def b0_cause_diagnostic(args):
    """Explanatory numerical experiment; never replaces any main B0 ranking."""
    import gc
    import types
    import torch
    from stage6.formal import load_queries
    from stage6.proxy_graph import build_graph
    from stage6.models import load_model, state_hash
    from stage6.evaluator import safe_evaluate
    from stage6.util import identity
    dest = ROOT / 'results/b0_fixed_relation_diagnostic.json'
    artifact = ROOT / 'diagnostics/b0_fixed_relation.npz'
    if dest.exists():
        assert sha(artifact) == read(dest)['artifact_sha256']
        return
    graph = load_observed(args.observations, 'NELLInductive_v1')
    queries, truths = load_queries(args, 'NELLInductive_v1', graph)
    d, f, _ = build_graph(graph, queries, truths, ROOT / 'cache/topology')
    m = load_model(args.stage5, args.history, args.official, 'B0', 1024)
    before = state_hash(m)
    native = m.relation_model.forward
    ids = sorted(set(queries[:, 1].tolist()) | set((queries[:, 1] + graph.R // 2).tolist()))
    def representations(batch):
        outputs = {}
        with torch.no_grad():
            for at in range(0, len(ids), batch):
                qs = ids[at:at + batch]
                out = native(d.relation_hypergraph, torch.tensor(qs, device='cuda'))
                for i, key in enumerate(qs):
                    outputs[key] = out[i].clone()
        return outputs
    pair_a, pair_b, single = representations(2), representations(2), representations(1)
    def packed(memory):
        return torch.stack([memory[key] for key in ids]).detach().cpu().numpy()
    a, b, s = packed(pair_a), packed(pair_b), packed(single)
    def fixed(self, hyper, query):
        return torch.stack([pair_a[key] for key in query.cpu().tolist()])
    fixed_method = types.MethodType(fixed, m.relation_model)
    m.tsdc_native_forward = fixed_method
    m.relation_model.forward = fixed_method
    ranks, evidence, rows = [], [], []
    for batch in [8, 8, 1, 4]:
        r, info, ev = safe_evaluate(m, d, f, graph, [[0., 0., 'conditional']], batch, True)
        ranks.append(r[0])
        evidence.append(ev)
        ev = ev[np.lexsort((ev[:, 2], ev[:, 1]))]
        if len(rows) == 0:
            base_scores = ev[:, 9].copy()
        rows.append(dict(batch=batch, rank_changes_vs_first=int((r[0] != ranks[0]).sum()),
            mrr=metrics(r[0])['mrr'], delta_mrr_vs_first=float((1 / r[0] - 1 / ranks[0]).mean()),
            max_true_score_change=float(np.max(np.abs(ev[:, 9] - base_scores)))))
    assert state_hash(m) == before and identity(packed(pair_a)) == identity(a)
    artifact.parent.mkdir(exist_ok=True)
    np.savez_compressed(artifact, ranks=np.asarray(ranks), relation_pair_a=a, relation_pair_b=b,
                        relation_single=s, relation_ids=np.asarray(ids), **{f'evidence_{i}': e for i, e in enumerate(evidence)})
    dump(dest, dict(setting='post-freeze explanatory diagnostic only; main B0 is never replaced',
        checkpoint_sha256=m.tsdc_checkpoint_sha256, graph_sha256=graph.key, no_weight_update=True,
        same_batch_relation_repeat_max_error=float(np.max(np.abs(a - b))),
        relation_batch1_vs_batch2_max_error=float(np.max(np.abs(a - s))),
        fixed_relation_representation_sha256=identity(a), entity_batch_comparison=rows,
        artifact_sha256=sha(artifact)), exclusive=True)
    del d, f, m, pair_a, pair_b, single
    gc.collect()
    torch.cuda.empty_cache()


def b0_operator_trace(args):
    """Trace a recorded changed query with fixed relations; explanatory only."""
    import gc
    import types
    import torch
    from motif import tasks
    from stage6.formal import load_queries
    from stage6.proxy_graph import build_graph
    from stage6.models import load_model, state_hash
    dest = ROOT / 'results/b0_operator_trace.json'
    artifact = ROOT / 'diagnostics/b0_operator_trace.npz'
    if dest.exists():
        assert sha(artifact) == read(dest)['artifact_sha256']
        return
    with np.load(ROOT / 'diagnostics/b0_fixed_relation.npz') as z:
        changed = np.argwhere(z['ranks'][2] != z['ranks'][0])
        index, direction = map(int, changed[0]) if len(changed) else (0, 0)
        memory = {int(k): torch.tensor(v, device='cuda') for k, v in zip(z['relation_ids'], z['relation_pair_a'])}
    graph = load_observed(args.observations, 'NELLInductive_v1')
    queries, truths = load_queries(args, 'NELLInductive_v1', graph)
    d, f, _ = build_graph(graph, queries, truths, ROOT / 'cache/topology')
    model = load_model(args.stage5, args.history, args.official, 'B0', 1024)
    before = state_hash(model)
    def fixed(self, hyper, query):
        return torch.stack([memory[k] for k in query.cpu().tolist()])
    model.relation_model.forward = types.MethodType(fixed, model.relation_model)
    triples = torch.cat([d.target_edge_index, d.target_edge_type[None]]).T
    captures, scores = [], []
    for batch_limit in [8, 1]:
        start = (index // batch_limit) * batch_limit
        batch = triples[start:start + batch_limit]
        position, size = index - start, len(batch)
        events, handles = [], []
        def select(t):
            if not isinstance(t, torch.Tensor) or t.ndim == 0:
                return None
            if t.shape[0] == size:
                return t[position].detach().cpu().numpy().copy()
            if t.ndim > 1 and t.shape[1] == size:
                return t[:, position].detach().cpu().numpy().copy()
            return None
        def capture(name):
            def hook(module, inputs, output):
                events.append((name, select(inputs[0]), select(output)))
            return hook
        for name, module in model.named_modules():
            if name.startswith('entity_model.') and isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                handles.append(module.register_forward_hook(capture(name)))
        try:
            with torch.no_grad():
                candidates = tasks.all_negative(d, batch)[direction]
                output = model(d, candidates)
                scores.append(output[position].cpu().numpy())
        finally:
            for handle in handles:
                handle.remove()
        captures.append(events)
    assert before == state_hash(model)
    rows, arrays = [], dict(score_batch8=scores[0], score_batch1=scores[1])
    for i, (a, b) in enumerate(zip(*captures)):
        assert a[0] == b[0]
        row = dict(module=a[0])
        for field, aa, bb in [('input', a[1], b[1]), ('output', a[2], b[2])]:
            if aa is not None and bb is not None and aa.shape == bb.shape:
                row[field + '_max_abs_diff'] = float(np.max(np.abs(aa - bb)))
                arrays[f'event{i}_{field}_batch8'] = aa
                arrays[f'event{i}_{field}_batch1'] = bb
            else:
                row[field + '_max_abs_diff'] = None
        rows.append(row)
    first = next((r for r in rows if (r['input_max_abs_diff'] or 0) > 0 or (r['output_max_abs_diff'] or 0) > 0), None)
    np.savez_compressed(artifact, **arrays)
    dump(dest, dict(setting='post-hoc localization; no main metric replacement', original_query_index=index,
        direction='tail' if direction == 0 else 'head', relation_representations_fixed=True,
        no_weight_update=True, first_divergence=first, module_trace=rows,
        score_vector_max_abs_diff=float(np.max(np.abs(scores[0] - scores[1]))),
        artifact_sha256=sha(artifact)), exclusive=True)
    del model, d, f, memory
    gc.collect()
    torch.cuda.empty_cache()


def factorial_interaction(p):
    """Post-freeze rank interaction at the actual selected theta; no test argmax."""
    rows, group_draws, group_values = [], defaultdict(list), defaultdict(list)
    for gi, (group, key) in enumerate(p['targets']):
        values = []
        for seed in p['training_seeds']:
            rec = read(ROOT / 'evaluation' / f'{key}__T-4D_s{seed}.json')
            choice = read(ROOT / 'adaptation' / f'{key}__rate0.15__p6101__s{seed}.json')['decisions']['joint_lcb']['selected']
            a, gamma, _ = choice
            with np.load(ROOT / 'evaluation' / f'{key}__T-4D_s{seed}.npz') as z:
                ranks = z['ranks']
            if a == 0 or gamma == 0:
                delta = np.zeros(len(ranks[0]))
            else:
                index = lambda alpha, g: rec['configs'].index([alpha, g, 'conditional'])
                delta = rr(ranks[index(a, gamma)]) - rr(ranks[index(a, 0.)]) - rr(ranks[index(0., gamma)]) + rr(ranks[index(0., 0.)])
            values.append(delta)
        value = np.mean(values, 0)
        ci, samples = paired_bootstrap(value, p['bootstrap'], p['bootstrap_seed'] + 101 * (gi + 1))
        rows.append(dict(key=key, group=group, contrast='RR(a,g)-RR(a,0)-RR(0,g)+RR(0,0) at frozen joint theta', **ci))
        group_draws[group].append(samples)
        group_values[group].append(ci['delta'])
    for group, draws in group_draws.items():
        samples = np.mean(draws, 0)
        lo, hi = np.quantile(samples, [.025, .975])
        rows.append(dict(key='macro_' + group, group=group, delta=float(np.mean(group_values[group])),
                         lo=float(lo), hi=float(hi), contrast='equal-graph conditional rank interaction'))
    csv_write(ROOT / 'results/factorial_interaction.csv', rows)


def figures(args, points, macros, boots):
    import os
    import subprocess
    folder = ROOT / 'figures'
    folder.mkdir(exist_ok=True)
    path = folder / 'inputs.json'
    dump(path, dict(points=points, macros=macros, bootstrap=boots))
    def native(p):
        s = str(p)
        if str(args.plot_python).endswith('.exe') and s.startswith('/mnt/'):
            return s[5].upper() + ':' + s[6:]
        return s
    # The ML environment is untouched. Use the user's existing plotting interpreter.
    renderer_environment = dict(os.environ)
    if str(args.plot_python).lower().endswith('.exe'):
        renderer_environment.pop('PYTHONPATH', None)
    subprocess.run([str(args.plot_python), native(ROOT / 'plot.py'), native(path)],
                   env=renderer_environment, check=True)


def verify(args):
    checks = {}
    protected = read(ROOT / 'protected_stage5.json')
    changed = [name for name, expected in protected.items() if sha(args.stage5 / name.replace('\\', '/')) != expected]
    assert not changed, changed
    checks['stage5_original_files_unchanged'] = len(protected)
    # Also verify the independent working copy's entire tracked stage-five files.
    copy_changed = [name for name, expected in protected.items() if sha(ROOT.parent / name.replace('\\', '/')) != expected]
    assert not copy_changed, copy_changed
    checks['stage5_working_copy_unchanged'] = True
    historical = read(ROOT / 'protected_history.json')
    from stage6.util import runtime_path
    assert all(sha(runtime_path(name)) == expected for name, expected in historical.items())
    checks['historical_assets_unchanged'] = len(historical)
    assert read(ROOT / 'results/formal_status.json')['status'] == 'complete'
    frozen = read(ROOT / 'frozen.json')
    assert sha(ROOT / 'protected_stage5.json') == frozen['stage5_protected_manifest_sha256']
    assert sha(ROOT / 'protected_history.json') == frozen['history_protected_manifest_sha256']
    assert sha(ROOT / 'model_manifest.json') == frozen['model_manifest_sha256']
    assert sha(args.observations / 'manifest.json') == frozen['observation_manifest_sha256']
    assert sha(ROOT / 'results/environment.json') == frozen['environment_sha256']
    for name, digest in frozen['code_hashes'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in frozen['source_diagnostic_hashes'].items():
        assert sha(ROOT / name) == digest, name
    assert sha(ROOT / 'source_density_protocol.json') == frozen['source_density_protocol_sha256']
    assert sha(ROOT / 'matched_global_protocol.json') == frozen['matched_global_protocol_sha256']
    count = 0
    for folder in ['adaptation', 'source_diagnostics', 'source_density', 'source_validation', 'evaluation']:
        for path in (ROOT / folder).glob('*.json'):
            rec = read(path)
            assert sha(path.with_suffix('.npz')) == rec['ranks_sha256']
            count += 1
    checks['rank_artifacts_hash_verified'] = count
    tests = read(ROOT / 'results/core_tests.json')
    assert tests['failures'] == tests['errors'] == 0
    assert read(ROOT / 'results/numeric_tests.json')['status'] == 'passed'
    checks.update(status='passed', core_tests=tests, frozen_sha256=sha(ROOT / 'frozen.json'),
        model_updates=0, target_selection_records=len(frozen['selections']),
        formal_graphs=11, formal_model_runs=110, no_ai_handoff=True)
    checks['final_report_code_hashes'] = {name: sha(ROOT / name) for name in frozen['initial_report_code_hashes']}
    checks['report_authoring_files_changed_after_freeze'] = [name for name, old in frozen['initial_report_code_hashes'].items()
        if checks['final_report_code_hashes'][name] != old]
    checks['report_authoring_note'] = 'Only report aggregation/rendering may be repaired after freeze; selector, model, formal inference, statistics code and all frozen parameters remain hash-locked.'
    numerical = read(ROOT / 'results/b0_fixed_relation_diagnostic.json')
    assert sha(ROOT / 'diagnostics/b0_fixed_relation.npz') == numerical['artifact_sha256']
    checks['post_freeze_numerical_diagnostic_artifact_verified'] = True
    trace = read(ROOT / 'results/b0_operator_trace.json')
    assert sha(ROOT / 'diagnostics/b0_operator_trace.npz') == trace['artifact_sha256']
    checks['entity_operator_trace_artifact_verified'] = True
    dump(ROOT / 'results/verification.json', checks)
    return checks


def run(args):
    p = read(ROOT / 'protocol.json')
    proxy, selections = selection_tables(p)
    per_fact, ranks, rows, all_splits, repeat = load_results(args, p)
    graphs, macros, boots, groups = summarize(p, per_fact, ranks, rows)
    points, within, stability, source_splits, correlation = reliability(p)
    density_analysis(p)
    distribution_analysis(args, p)
    hub = hub_analysis(args, p)
    numerical_analysis()
    b0_cause_diagnostic(args)
    b0_operator_trace(args)
    matched_degree_analysis(p)
    factorial_interaction(p)
    figures(args, points, macros, boots)
    checks = verify(args)
    write_report(args, p, graphs, macros, boots, groups, selections, points, within,
                 stability, source_splits, correlation, hub, repeat, checks)
    log('analysis_complete', report=str(ROOT / 'report.md'), verification=checks['status'])


def write_report(args, p, graphs, macros, boots, groups, selections, points, within,
                 stability, source_splits, correlation, hub, repeat, checks):
    # Filled from actual outputs only; no expected gains or target-driven tuning.
    from stage6.report_writer import write
    write(args, p, graphs, macros, boots, groups, selections, points, within,
          stability, source_splits, correlation, hub, repeat, checks)
