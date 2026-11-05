"""Sanitize observation assets, accessing named NPZ members only (numpy is lazy).

Candidate counts are public metadata from observation topology-cache degree length;
no valid/test file or formal ranking is opened. Namespace order comes from the
already published relation vocabulary; entity order comes from observation text.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from topology import graph_key
from stage6.observed_loader import ObservedGraph
from stage6.util import ROOT, dump, read, sha, runtime_path


def write_bundle(out, key, e, r, n, R, provenance):
    g = ObservedGraph(np.asarray(e, dtype=np.int64), np.asarray(r, dtype=np.int64), n, R, provenance)
    path = out / (key + '.npz')
    if not path.exists():
        np.savez_compressed(path, edge_index=g.edge_index, edge_type=g.edge_type, n=n, R=R)
    with np.load(path) as z:
        assert np.array_equal(z['edge_index'], g.edge_index) and np.array_equal(z['edge_type'], g.edge_type)
    return dict(file=path.name, sha256=sha(path), observed_sha256=g.key, n=n, R=R, **provenance)


def prepare(stage5, history, source_root, out, raw_root=None):
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    vocab = read(history / 'semantic/relation_mappings.json')
    provenance = read(history / 'semantic/target_mapping_provenance.json')
    caches = list((history / 'cache/downstream').glob('*.npz'))
    candidates = {}
    for path in caches:
        with np.load(path) as z:
            candidates[path.stem] = len(z['degree'])
    # This file's query metrics/masks are NOT passed to the selector.
    # Only the pre-existing observation-edge digest is used for an equality audit.
    expected = {r['key']: r['observed_sha256'] for r in read(stage5 / 'analysis/target_inputs.json')}
    for key, meta in provenance.items():
        raw = runtime_path(meta['raw'])
        if raw_root is not None:
            suffix = str(meta['raw']).replace('\\', '/').split('/data/', 1)[1]
            raw = raw_root / suffix
        p = raw / meta['fact_file']
        rels = {r['original_name']: r['id'] for r in vocab[key] if not r.get('inverse', False)}
        R = len(vocab[key])
        # The first half always denotes forward relations in this published mapping.
        rels = {r['original_name']: r['id'] for r in vocab[key][:R // 2]}
        entities, facts = {}, []
        for line in p.read_text(encoding='utf-8').splitlines():
            h, q, t = line.split()
            for token in (h, t):
                if token not in entities:
                    entities[token] = len(entities)
            facts.append((entities[h], rels[q], entities[t]))
        f = np.asarray(facts, dtype=np.int64)
        e = np.c_[f[:, [0, 2]].T, f[:, [2, 0]].T]
        r = np.r_[f[:, 1], f[:, 1] + R // 2]
        ns = [n for key2, n in candidates.items() if graph_key(e, r, n, R) == key2]
        assert len(ns) == 1, (key, ns, len(entities), R)
        assert hashlib.sha256(e.tobytes() + r.tobytes()).hexdigest() == expected[key]
        manifest[key] = write_bundle(out, key, e, r, ns[0], R,
            dict(source=str(p), source_sha256=sha(p), namespace='observation-first entity order; published relation IDs',
                 public_metadata='candidate count from pre-existing observation-only topology cache',
                 observed_entities=len(entities), isolated_metadata_candidates=ns[0] - len(entities),
                 stage5_observed_digest_matched=True, kind=meta['kind']))
    p = history / 'data/external/ConceptNet100k/observed.npz'
    with np.load(p, allow_pickle=False) as z:
        manifest['ConceptNet100k_external'] = write_bundle(out, 'ConceptNet100k_external',
            z['edge_index'], z['edge_type'], int(z['n']), int(z['R']),
            dict(source=str(p), source_sha256=sha(p), kind='external',
                 namespace='existing public integer namespace; read only edge_index/edge_type/n/R NPZ members'))
    for source in ['fb15k237_train', 'wn18rr_train', 'codex_medium_train']:
        p = source_root / 'data' / (source + '__validation.npz')
        with np.load(p, allow_pickle=False) as z:
            manifest[source] = write_bundle(out, source, z['edge_index'], z['edge_type'], int(z['n']), int(z['R']),
                dict(source=str(p), source_sha256=sha(p), kind='source',
                     namespace='source public IDs; read only observation members, no validation queries'))
    dest = out / 'manifest.json'
    if dest.exists():
        assert read(dest) == manifest
    else:
        dump(dest, manifest, exclusive=True)
    print('SANITIZED', [(k, r['n'], r['R']) for k, r in manifest.items()], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for arg in ['stage5', 'history', 'source-root']:
        p.add_argument('--' + arg, type=runtime_path, required=True)
    p.add_argument('--out', type=runtime_path, default=ROOT / 'observations')
    p.add_argument('--raw-root', type=runtime_path, help='Optional relocated stage-one data root containing grail/ and ingram/')
    a = p.parse_args()
    prepare(a.stage5, a.history, a.source_root, a.out, a.raw_root)
