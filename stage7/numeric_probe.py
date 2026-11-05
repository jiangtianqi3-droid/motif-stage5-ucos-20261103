import gc
import numpy as np
import torch
from stage6.models import load_model, state_hash
from stage6.observed_loader import load_observed
from stage6.proxy_graph import build_graph
from stage6.formal import load_queries
from stage7.util import ROOT, dump, log
from stage7.numeric_stability import install, configure
from stage7.evaluator import safe_evaluate


def run(args):
    g = load_observed(args.observations, 'NELLInductive_v1')
    q, truth = load_queries(args, 'NELLInductive_v1', g)
    # Historical diagnostic only. No new target labels accessed here.
    ix = np.unique(np.r_[np.arange(32), [13, 48, 64, 128, 256, 400, 500]])
    ix = ix[ix < len(q)]
    d, t, topo = build_graph(g, q[ix], truth, ROOT/'cache/topology')
    rows = []
    arrays = {}
    for family in ['B0', 'T-4D', 'B1', 'B2']:
        m = load_model(args.stage5, args.history, args.official, family, 1024)
        before = state_hash(m)
        alpha = .25 if family == 'T-4D' else (1. if family == 'B2' else 0.)
        for path in ['native_fp32', 'querywise_fp32']:
            if path == 'querywise_fp32':
                install(m)
            configure(False)
            ref = None
            for batch in [1, 2, 4, 8, 8]:
                r, info, s = safe_evaluate(m, d, t, [alpha], batch, True)
                s = s[0]
                if ref is None:
                    ref = (r.copy(), s.copy())
                diff = np.abs(s-ref[1])
                rows.append(dict(family=family, path=path, tf32=False, batch=batch,
                    score_max_error=float(diff.max()), score_differences=int((diff>0).sum()),
                    rank_changes=int((r != ref[0]).sum()), compared_to='same path batch1',
                    seconds=info['seconds'], peak_allocated=info['peak_allocated']))
                log('numeric_probe', **rows[-1])
                arrays[f'{family}_{path}_{batch}'] = r
            if family == 'B0':
                # TF32 sensitivity is diagnostic, never a performance-based choice.
                configure(True)
                r, info, s = safe_evaluate(m, d, t, [alpha], 4, True)
                rows.append(dict(family=family, path=path, tf32=True, batch=4,
                    score_max_error=float(np.abs(s[0]-ref[1]).max()),
                    rank_changes=int((r != ref[0]).sum()), compared_to='FP32 same path batch1'))
                configure(False)
        assert before == state_hash(m)
        del m
        gc.collect()
        torch.cuda.empty_cache()
    np.savez_compressed(ROOT/'results/numeric_probe.npz', **arrays)
    dump(ROOT/'results/numeric_probe.json', dict(rows=rows, query_indices=ix.tolist(),
         topology=topo, no_weight_update=True))
