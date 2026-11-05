"""Source-only standard validation; never called inside target adaptation."""
import gc
import numpy as np
import torch
from stage6.util import ROOT, read, dump, sha, log
from stage6.observed_loader import load_observed
from stage6.proxy_graph import build_graph
from stage6.models import load_model
from stage6.protocol import configs
from stage6.evaluator import safe_evaluate, metrics


def verify_legacy(args, data):
    """Compare adapters to the actual recovered historical forward implementation."""
    import importlib.util
    import sys
    dest = ROOT / 'results/legacy_adapter_tests.json'
    if dest.exists():
        assert read(dest)['status'] == 'passed'
        return
    sys.path.insert(0, str(args.history))
    try:
        spec = importlib.util.spec_from_file_location('tsdc_historical_stable_gate', args.history / 'stable_gate.py')
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
    finally:
        sys.path.pop(0)
    rows = []
    for family in ['B1', 'B2']:
        record = read(args.history / 'frozen.json')['records'][family + '_s1024']
        m = load_model(args.stage5, args.history, args.official, family, 1024)
        q = torch.tensor([0, 1, int(data.num_relations) // 2], device='cuda')
        with torch.no_grad():
            reference = m.relation_model(data.relation_hypergraph, q).clone()
            legacy.install(m, record['config'])
            m.load_state_dict(torch.load(args.history / record['checkpoint'], map_location='cpu')['model'], strict=True)
            m.eval()
            original = m.relation_model(data.relation_hypergraph, q)
            error = float((original - reference).abs().max())
        assert error == 0., (family, error)
        rows.append(dict(family=family, max_relation_representation_error=error,
                         legacy_implementation_sha256=sha(args.history / 'stable_gate.py')))
        del m
        gc.collect()
        torch.cuda.empty_cache()
    dump(dest, dict(status='passed', source='wn18rr_train', rows=rows))


def run(args):
    from stage6.source_density import run as density_run
    density_run(args)
    p = read(ROOT / 'protocol.json')
    folder = ROOT / 'source_validation'
    folder.mkdir(exist_ok=True)
    for key in ['wn18rr_train', 'codex_medium_train', 'fb15k237_train']:
        graph = load_observed(args.observations, key)
        with np.load(args.source_root / 'data' / (key + '__validation.npz')) as z:
            truth = z['alltrue'].astype(np.int64)
        queries = np.load(args.history / 'data' / (key + '_validation_queries.npy'))[:512]
        d, f, topo = build_graph(graph, queries, truth, ROOT / 'cache/topology')
        if key == 'wn18rr_train':
            verify_legacy(args, d)
        # Standard-source reliability check concerns H1's topology grid. The full
        # degree grid is already evaluated on the source proxy and unit tested.
        cfgs = [[alpha, 0., 'conditional'] for alpha in p['alpha_grid']]
        for seed in p['training_seeds']:
            path = folder / f'{key}__s{seed}.json'
            if path.exists():
                assert read(path)['protocol_sha256'] == sha(ROOT / 'protocol.json')
                continue
            log('source_validation_begin', key=key, seed=seed)
            m = load_model(args.stage5, args.history, args.official, 'T-4D', seed)
            ranks, info, _ = safe_evaluate(m, d, f, graph, cfgs, args.batch_size)
            np.savez_compressed(path.with_suffix('.npz'), ranks=ranks, queries=queries)
            dump(path, dict(key=key, training_seed=seed, configs=cfgs, metrics=[metrics(r) for r in ranks],
                            info=info, topology=topo, protocol_sha256=sha(ROOT / 'protocol.json'),
                            ranks_sha256=sha(path.with_suffix('.npz'))), exclusive=True)
            log('source_validation_complete', key=key, seed=seed, seconds=info['seconds'])
            del m
            gc.collect()
            torch.cuda.empty_cache()
        del d, f
        gc.collect()
        torch.cuda.empty_cache()
