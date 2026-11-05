import copy
import gc
import types
import numpy as np
import torch
from stage6.util import ROOT, read, dump, log
from stage6.models import load_model, set_alpha, state_hash
from stage6.observed_loader import load_observed
from stage6.proxy_split import make_split
from stage6.proxy_graph import build_graph
from stage6.evaluator import safe_evaluate


def run(args):
    graph = load_observed(args.observations, 'wn18rr_train')
    rem, queries, cut, meta = make_split(graph, 6101)
    truth = np.c_[graph.edge_index[0], graph.edge_type, graph.edge_index[1]]
    d, f, _ = build_graph(rem, queries[:16], truth, ROOT / 'cache/topology')
    m = load_model(args.stage5, args.history, args.official, 'T-4D', 1024)
    before = state_hash(m)
    q = torch.tensor([0, 1, graph.R // 2], device='cuda')
    rel = m.relation_model
    cfg = copy.deepcopy(rel.stage5_config)
    outputs = {}
    layer_values = {}
    with torch.no_grad():
        for a in [0., .1, .25, .5, 1.]:
            set_alpha(m, a)
            captures = []
            hook = rel.layers[0].linear.register_forward_hook(lambda _, inp, out: captures.append(out.detach().clone()))
            outputs[a] = rel(d.relation_hypergraph, q).clone()
            layer_values[a] = captures[0]
            hook.remove()
            assert torch.isfinite(outputs[a]).all()
        rel.stage5_config['method'] = 'plain'
        # Explicitly remove every gate from this SAME checkpoint, preserving weights.
        gates = [layer.topology_gate for layer in rel.layers]
        for layer in rel.layers:
            del layer.topology_gate
        plain = rel(d.relation_hypergraph, q)
        for layer, gate in zip(rel.layers, gates):
            layer.topology_gate = gate
        error = float((plain - outputs[0.]).abs().max())
        assert error == 0., error
        internal = {str(a): float((layer_values[a] - layer_values[0.]).abs().max()) for a in outputs}
        assert all(internal[str(a)] > 0 for a in [.1, .25, .5, 1.])
    rel.stage5_config = cfg
    assert before == state_hash(m)
    ranks, info, _ = safe_evaluate(m, d, f, rem, [[0., 0., 'conditional'], [.25, 0., 'conditional']], 4)
    from evaluate import evaluate as stage5_evaluate
    set_alpha(m, .25)
    original_info, standard_ranks, standard_queries = stage5_evaluate(m, d, f, 4, (0.,))
    assert np.array_equal(ranks[1], standard_ranks[0]), 'proxy evaluator differs from standard stage5 ranking'
    assert all(t.grad is None for t in m.parameters()) and before == state_hash(m)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    # Repeat native B0 on the identical source inputs with different entity batches.
    m = load_model(args.stage5, args.history, args.official, 'B0', 1024)
    repeats = []
    for batch in [4, 4, 1, 8]:
        r, inf, _ = safe_evaluate(m, d, f, rem, [[0., 0., 'conditional']], batch)
        repeats.append(r[0])
    rows = [dict(batch=b, rank_changes=int((r != repeats[0]).sum()),
                 mrr=float((1 / r).mean()), delta_mrr=float((1 / r - 1 / repeats[0]).mean()))
            for b, r in zip([4, 4, 1, 8], repeats)]
    dump(ROOT / 'results/numeric_tests.json', dict(status='passed', source='wn18rr_train',
         alpha_zero_same_checkpoint_plain_max_error=error, alpha_internal_message_change=internal,
         finite_alpha_grid=True, no_gradient_update=True, standard_stage5_proxy_ranking_exact=True,
         b0_source_repeats=rows, info=info))
    log('numeric_tests_passed', alpha0_error=error, internal=internal, b0_source_repeats=rows)
