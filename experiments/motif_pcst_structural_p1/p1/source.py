from statistics import mean

import numpy as np

from p1.assets import Assets, Evaluator, prepare
from p1.paths import counts_from_rows, source_probes
from p1.retrieval import Retriever
from p1.scoring import evidence, filtered_after, permutations, raw_order
from src.io_utils import dump, read, write_csv


def source_calibration(cfg, folder):
    cache = folder/'source_calibration.json'
    if cache.is_file():
        return read(cache)
    sc = cfg['source']
    asset = Assets(sc['graph'],folder)
    graph, removed, samples = source_probes(asset.graph,cfg['random_seed'],sc['mask_fit'],sc['mask_probe'])
    table = counts_from_rows(row for row in samples if row['fold']=='fit')
    scored = []
    for row in samples:
        if row['fold'] != 'probe':
            continue
        p = row['path_ids']
        from p1.paths import reliability
        prob = reliability(table,graph,row['relation'],tuple(p)) if p else .5
        scored.append(dict(row,support=prob))
    positive = [r for r in scored if r['observed_positive']]
    contrast = [r for r in scored if not r['observed_positive']]
    write_csv(folder/'source_masked_probe.csv',scored)
    dump(folder/'source_mask_manifest.json',dict(masked_graph=graph.key,source_graph=asset.graph.key,
         removed_observed_facts=removed,fit_facts=sc['mask_fit'],probe_facts=sc['mask_probe']))
    pos_mean = mean(r['support'] for r in positive)
    neg_mean = mean(r['support'] for r in contrast)
    null = mean(r['observed_positive'] for r in scored)
    # A probability-like statistic is used solely for a descriptive discrimination check.
    diagnose = dict(positives=len(positive),nonedges=len(contrast),positive_path_rate=mean(r['has_path'] for r in positive),
                    nonedge_path_rate=mean(r['has_path'] for r in contrast),mean_positive_support=pos_mean,
                    mean_nonedge_support=neg_mean,proxy_brier=mean((r['support']-r['observed_positive'])**2 for r in scored),
                    rate_baseline_brier=mean((null-r['observed_positive'])**2 for r in scored))
    rng = np.random.default_rng(cfg['random_seed']+1)
    indices = sorted(rng.choice(len(asset.queries),sc['tuning_facts'],replace=False).tolist())
    scores,gates = prepare(asset,indices)
    retriever = Retriever(asset.graph,gates,table,cfg['retrieval'])
    queries = []
    for i in indices:
        for d,(h,r) in enumerate(asset.queries[i]):
            raw = scores[i][d]
            candidates = raw_order(raw)[:cfg['retrieval']['candidate_topk']]
            selected = retriever.query(int(h),int(r),candidates,cfg['evidence_budget']['primary_max_facts'],('R5',))['R5']
            values = evidence(asset.graph,int(r),candidates,selected,table)
            queries.append((i,d,int(h),int(r),raw,candidates,values))
    evalr = Evaluator(asset)
    stats = {str(l):[] for l in sc['lambdas']}
    for i,d,h,r,raw,candidates,values in queries:
        target,excluded,_ = evalr.query(i,d,h,r,raw)
        for lam in sc['lambdas']:
            order = permutations(raw,candidates,values,lam,sc['gate_margin'])['fusion']
            rank = filtered_after(raw,candidates,order,target,excluded,evalr.rank)
            stats[str(lam)].append(1/rank)
    grid = {k:mean(v) for k,v in stats.items()}
    chosen = min(sc['lambdas'],key=lambda lam:(-grid[str(lam)],lam))
    data = dict(table=table,mask_diagnostics=diagnose,lambda_selected=chosen,source_grid=grid,
                tuning_queries=len(queries),source=asset.key,validation_indices=indices)
    dump(cache,data)
    return data
