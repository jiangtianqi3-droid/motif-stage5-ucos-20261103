"""Secondary source-only density sensitivity with FIXED queries across mask rates.

Main 256-fact protocol remains unchanged. Here the graph masks 5/10/15 percent
while scoring the same <=256 facts chosen from the 5-percent masked set. No
parameter selection is performed, so there is no confirmation reuse.
"""
import gc
import numpy as np
import torch
from stage6.util import ROOT, read, dump, sha, identity, log
from stage6.observed_loader import load_observed
from stage6.proxy_split import make_split
from stage6.proxy_graph import build_graph
from stage6.models import load_model
from stage6.evaluator import safe_evaluate, metrics


def run(args):
    protocol = dict(type='secondary source-only density diagnostic; does not select target rules',
        training_seed=1024, proxy_seed=6101, rates=[.05, .10, .15],
        query_cap=256, queries='fixed ordered subset of lowest-rate hidden facts shared across all rates',
        mask_cap='floor(rate * unique observation facts), subject to same connectivity/coverage constraints',
        candidates='five registered topology alpha values, gamma=0',
        limitation='one training seed and one graph-mask seed; main protocol has three seeds')
    path = ROOT / 'source_density_protocol.json'
    if path.exists():
        assert read(path) == protocol
    else:
        dump(path, protocol, exclusive=True)
    folder = ROOT / 'source_density'
    folder.mkdir(exist_ok=True)
    p = read(ROOT / 'protocol.json')
    cfgs = [[a, 0., 'conditional'] for a in p['alpha_grid']]
    for key in ['wn18rr_train', 'codex_medium_train', 'fb15k237_train']:
        graph = load_observed(args.observations, key)
        base_rem, hidden, cut, meta = make_split(graph, 6101, .05, cap=len(graph.forward))
        queries = hidden[:256]
        truth = np.c_[graph.edge_index[0], graph.edge_type, graph.edge_index[1]]
        for rate in protocol['rates']:
            out = folder / f'{key}__rate{rate}.json'
            if out.exists():
                assert read(out)['protocol_sha256'] == sha(path)
                continue
            rem, pool, _, split = make_split(graph, 6101, rate, cap=len(graph.forward))
            assert set(map(tuple, queries.tolist())) <= set(map(tuple, pool.tolist()))
            d, f, topo = build_graph(rem, queries, truth, ROOT / 'cache/topology')
            m = load_model(args.stage5, args.history, args.official, 'T-4D', 1024)
            log('density_begin', key=key, rate=rate, queries=len(queries), masked=split['actual'])
            ranks, info, _ = safe_evaluate(m, d, f, rem, cfgs, args.batch_size)
            np.savez_compressed(out.with_suffix('.npz'), ranks=ranks, queries=queries)
            dump(out, dict(key=key, rate=rate, actual_rate=split['actual'] / len(graph.forward), split=split,
                query_sha256=identity(queries), configs=cfgs, metrics=[metrics(r) for r in ranks], info=info,
                topology=topo, model_sha256=m.tsdc_checkpoint_sha256, protocol_sha256=sha(path),
                ranks_sha256=sha(out.with_suffix('.npz'))), exclusive=True)
            log('density_complete', key=key, rate=rate, seconds=info['seconds'])
            del d, f, m
            gc.collect()
            torch.cuda.empty_cache()
