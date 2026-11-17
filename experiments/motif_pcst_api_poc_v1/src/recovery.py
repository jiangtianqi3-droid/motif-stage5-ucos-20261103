"""Replay frozen inference from answer-free directional queries, using Stage-7 operators."""
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from src.io_utils import (HISTORY, KEYS, METHODS, RESULTS, STAGE7, dump, read, sha,
                          snapshot, write_csv)

OUT = RESULTS / 'recovery'
S6 = HISTORY / 'motif-stage6-entity-degree-p0p1-v2'


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audited_inputs(key):
    from s7util import arrsha
    rec = next(r for r in read(STAGE7 / 'inputs.json') if r['key'] == key)
    bundle = S6 / rec['file']
    require(sha(bundle) == rec['sha256'], f'{key}: Stage-6 bundle checksum')
    observed = load(snapshot(key, 'observed.npz'))
    with np.load(bundle, allow_pickle=False) as source:
        arrays = {k: source[k] for k in ('edge_index', 'edge_type', 'n', 'R', 'hyper_edge', 'hyper_type', 'raw', 'z')}
    for name in ('edge_index', 'edge_type', 'n', 'R'):
        require(np.array_equal(observed[name], arrays[name]), f'{key}: observed {name} differs')
    e, r = arrays['edge_index'], arrays['edge_type']
    require(arrsha(e, r) == rec['observed_sha256'], f'{key}: observation identity')
    require(arrsha(arrays['hyper_edge'], arrays['hyper_type']) == rec['hypergraph_sha256'],
            f'{key}: hypergraph identity')
    return rec, bundle, arrays


def prepare_queries(key, rec, bundle, arrays):
    """Offline isolation gate. Scoring receives only the two directional projections."""
    from collections import Counter
    from s7util import arrsha
    targets = load(snapshot(key, 'offline_targets_ONLY.npz'))['queries']
    with np.load(bundle, allow_pickle=False) as source:
        truth = source['alltrue']
        require(np.array_equal(source['queries'], targets), f'{key}: query indexing')
    require(arrsha(targets[:, [0, 2, 1]]) == rec['query_sha256'], f'{key}: query identity')
    require(arrsha(truth[:, [0, 2]].T, truth[:, 1]) == rec['filtered_sha256'], f'{key}: filtering identity')
    n, relations = int(arrays['n']), int(arrays['R'])
    require(relations > 0 and relations % 2 == 0 and n >= 30, f'{key}: vocabulary')
    half = relations // 2
    e, r = arrays['edge_index'], arrays['edge_type']
    require(np.all((0 <= e) & (e < n)) and np.all((0 <= r) & (r < relations)),
            f'{key}: observation vocabulary')
    obs = Counter(zip(e[0], r, e[1]))
    require(all(obs[(v, (rel + half) % relations, u)] == count
                for (u, rel, v), count in obs.items()), f'{key}: inverse pairs')
    protocol = next(r['protocol'] for r in read(snapshot(key, 'observed.npz').parents[1] / 'manifest.json')['datasets'] if r['dataset'] == key)
    observed_direct = {triple for triple in obs if triple[1] < half}
    # GraIL filter includes both held-out splits (merged); Ingram includes separate valid.
    hidden = set(map(tuple, truth)) - set(obs)
    require(set(map(tuple, targets)).issubset(hidden), f'{key}: target overlaps observations or is absent from filtering')
    require(all((v, (rel + half) % relations, u) not in obs for u, rel, v in hidden),
            f'{key}: hidden inverse overlaps observations')
    if protocol == 'grail':
        require(hidden == set(map(tuple, targets)), f'{key}: GraIL merged split mismatch')
    require(np.all((0 <= targets[:, [0, 2]]) & (targets[:, [0, 2]] < n))
            and np.all((0 <= targets[:, 1]) & (targets[:, 1] < half)), f'{key}: query vocabulary')
    projections = np.stack((targets[:, [0, 1]], targets[:, [2, 1]] + [0, half]), axis=1)
    dump(OUT / key / 'isolation.json', dict(protocol=protocol, observed_facts=len(obs),
         observed_direct=len(observed_direct), hidden=len(hidden), query_facts=len(targets), checked=True))
    return projections


def verify_runtime():
    run = read(STAGE7 / 'run_identity.json')
    relevant = {'common.py', 'engine.py', 'calibration.py', 'entity_normalization.py',
                'gated_model.py', 'plans.py', 'weighted.py', 's7util.py'}
    verified = {}
    for name, expected in run['code'].items():
        if name in relevant or name.startswith('runtime/'):
            require(sha(STAGE7 / name) == expected, f'Stage-7 source hash changed: {name}')
            verified[name] = expected
    require(relevant <= verified.keys(), 'Stage-7 runtime manifest incomplete')
    from common import CKPT
    expected = read(STAGE7 / 'results/official_checkpoint_audit.json')['expected_stage6_sha256']
    require(sha(CKPT) == expected, 'official B0 checkpoint checksum')
    return verified


def reference(key, method, rec, projections):
    from s7util import arrsha
    base = STAGE7 / 'evaluation' / key
    path = base / (method + '.npz')
    meta = read(base / (method + '.json'))
    require(sha(path) == meta['npz_sha256'], f'{key}/{method}: historical artifact checksum')
    require(meta['query_sha256'] == rec['query_sha256'] and meta['batch'] == 16, 'query/batch metadata')
    ref = load(path)
    require(arrsha(ref['triples']) == rec['query_sha256'], 'reference query checksum')
    require(ref['ranks'].shape == projections.shape[:2], 'reference ranks shape')
    return ref, meta


def inference(arrays, projections, method, meta, path):
    import torch
    from calibration import install
    from common import make_data
    from engine import RelationCache, model, state_hash

    n = int(arrays['n'])
    # make_data also makes target/truth fields. Pass empty placeholders exclusively.
    empty = np.empty((0, 3), dtype=np.int64)
    d, _ = make_data(dict(arrays, alltrue=empty), q=empty)
    del d.target_edge_index, d.target_edge_type
    net = model('B0' if method == 'B0' else 'B2', 1024)
    before = state_hash(net)
    require(before == meta['parameter_sha256_before'], f'{method}: model parameter hash')
    install(net, d, meta['config'])
    RelationCache(net)
    start = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tmp = path.with_suffix('.partial')
    feature = np.lib.format.open_memmap(tmp, mode='w+', dtype='float32', shape=(len(projections), 2, n))
    candidates = torch.arange(n, device='cuda')
    with torch.inference_mode():
        for at in range(0, len(projections), 16):
            for direction in range(2):
                query = torch.as_tensor(projections[at:at+16, direction], device='cuda')
                src, rel = [x[:, None].expand(-1, n) for x in query.T]
                batch = torch.stack((src, candidates.expand_as(src), rel), -1)
                score = net(d, batch)
                require(bool(torch.isfinite(score).all()), 'nonfinite scores')
                sorted_score = score.sort(dim=1).values.contiguous()
                rank = n - torch.searchsorted(sorted_score, score.contiguous(), right=False)
                feature[at:at+len(query), direction] = (1. / (60 + rank.float())).cpu().numpy()
    torch.cuda.synchronize()
    require(state_hash(net) == before, 'model changed during inference')
    peak = torch.cuda.max_memory_allocated()
    feature.flush()
    del feature, net, d
    torch.cuda.empty_cache()
    return tmp, peak, perf_counter() - start


def verify_offline(path, bundle, ref, projections, arrays):
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    n = int(arrays['n'])
    require(values.shape == (*projections.shape[:2], n) and values.dtype == np.float32, 'cache schema')
    with np.load(bundle, allow_pickle=False) as source:
        known = {}
        for h, r, t in source['alltrue']:
            known.setdefault((h, r), set()).add(t)
            known.setdefault((t, r + int(arrays['R']) // 2), set()).add(h)
    ids = np.arange(n)
    for i, (h, t, _) in enumerate(ref['triples']):
        for direction, target in enumerate((t, h)):
            scores = values[i, direction]
            order = np.lexsort((ids, -scores))[:30]
            require(np.all(np.isfinite(scores)) and len(set(order)) == len(order) == 30, 'invalid complete candidates')
            hits = scores >= scores[target]
            hits[list(known[tuple(projections[i, direction])] | {target})] = False
            require(1 + np.count_nonzero(hits) == ref['ranks'][i, direction], 'historical filtered rank mismatch')


def run():
    verified = verify_runtime()
    audit = []
    inventory = []
    def inventory_row(path, graph, model_name, role):
        if path.exists():
            stat = path.stat()
            inventory.append(dict(path=str(path), bytes=stat.st_size, modified_ns=stat.st_mtime_ns,
                 sha256=sha(path), model=model_name, graph=graph, source=role))
    from common import CKPT
    inventory_row(CKPT, 'both', 'B0', 'official checkpoint verified by Stage-7 audit')
    model_meta = read(STAGE7 / 'model_manifest.json')['B2_s1024']
    inventory_row(HISTORY / 'topology-semantic-stage4-20261102' / model_meta['checkpoint'],
                  'both', METHODS[1], 'Stage-7 model manifest')
    archive = STAGE7 / 'stage7_report_and_data.zip'
    with ZipFile(archive) as z:
        zip_rank_files = [name for name in z.namelist() if name.endswith('.rrf.npy')]
    for key in KEYS:
        rec, bundle, arrays = audited_inputs(key)
        projections = prepare_queries(key, rec, bundle, arrays)
        for name in ('observed.npz', 'offline_targets_ONLY.npz', 'canonical_topology.npz'):
            inventory_row(snapshot(key, name), key, 'shared inputs', 'query-community archive manifest')
        inventory_row(bundle, key, 'shared inputs', 'Stage-7 input manifest')
        for method in METHODS:
            ref, meta = reference(key, method, rec, projections)
            old = STAGE7 / 'evaluation' / key / (method + '.rrf.npy')
            inventory_row(old.with_suffix('').with_suffix('.npz'), key, method, 'historical filtered ranks and triples')
            dest = OUT / key / (method + '.rrf.npy')
            dest.parent.mkdir(parents=True, exist_ok=True)
            peak, seconds = None, None
            if dest.exists():
                checked, origin = dest, 'new experiment cache'
            elif old.exists():
                checked, origin = old, 'historical cache'
            else:
                print(f'Replaying {key} {method}', flush=True)
                checked, peak, seconds = inference(arrays, projections, method, meta, dest)
                origin = 'frozen checkpoint replay'
            actual = sha(checked)
            require(actual == meta['feature_sha256'], f'{key} {method}: full unfiltered cache SHA mismatch ({actual})')
            verify_offline(checked, bundle, ref, projections, arrays)
            if checked.suffix == '.partial':
                checked.replace(dest)
                checked = dest
            inventory_row(checked, key, method, origin)
            audit.append(dict(dataset=key, model=method, source=origin, path=str(checked),
                 facts=len(projections), directional_queries=2*len(projections), entities=int(arrays['n']),
                 checkpoint_parameters_sha256=meta['parameter_sha256_before'], observation_sha256=rec['observed_sha256'],
                 cache_sha256=actual, exact_cache_match=True, exact_filtered_ranks=True,
                 historical_npz_schema={k: [list(a.shape), str(a.dtype)] for k,a in ref.items()},
                 gpu_peak_bytes=peak, seconds=seconds))
            dump(OUT / 'recovery.json', dict(records=audit, verified_code=verified, archive_rrf_entries=zip_rank_files))
            write_csv(OUT / 'asset_inventory.csv', inventory)
        dump(OUT / key / 'queries.json', projections.tolist())
        print(f'{key}: candidate caches and isolation verified', flush=True)


if __name__ == '__main__':
    run()
