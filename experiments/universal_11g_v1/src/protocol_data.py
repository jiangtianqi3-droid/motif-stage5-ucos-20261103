"""P0 observation loading is separate from P1 and final evaluator label access."""
import numpy as np
from regime_inspector import ROOT, STAGE6, raw_observation_path, vocabulary

def forward_facts(z):
    ix = z['edge_type'] < z['R'] // 2
    return np.c_[z['edge_index'][0, ix], z['edge_type'][ix], z['edge_index'][1, ix]]

def rows(path):
    return [line.split('\t') if '\t' in line else line.split() for line in path.read_text().splitlines() if line.strip()]

def build_observation(facts, n, R, cache):
    from topology import extract
    edge = np.concatenate([facts[:, [0, 2]], facts[:, [2, 0]]]).T
    rel = np.r_[facts[:, 1], facts[:, 1] + R]
    s = extract(edge, rel, n, 2 * R, cache=cache)
    return dict(n=n, R=2 * R, edge_index=edge, edge_type=rel, hyper_edge=s['edge_index'], hyper_type=s['edge_type'], z=s['z'], raw=s['raw'])

def validation(rec, test_z, cache):
    """Called only after P0 weights have been persisted. No final-test reads."""
    directory = raw_observation_path(rec['key']).parent
    if 'Inductive' in rec['key']:
        entities, relations = {}, {}
        def convert(raw):
            result = []
            for h, r, t in raw:
                for name in (h, t):
                    if name not in entities:
                        entities[name] = len(entities)
                if r not in relations:
                    relations[r] = len(relations)
                result.append((entities[h], relations[r], entities[t]))
            return np.asarray(result, dtype=np.int64)
        observed = convert(rows(directory / 'train.txt'))
        queries = convert(rows(directory / 'valid.txt'))
        z = build_observation(observed, len(entities), len(relations), cache)
        z['_relation_names'] = list(relations)
        return z, queries, np.concatenate([observed, queries]), list(entities), 'independent_training_graph_valid'
    names, relations, _ = vocabulary(rec, test_z)
    emap = {name: i for i, name in enumerate(names) if name is not None}
    rmap = {name: i for i, name in enumerate(relations) if name is not None}
    path = directory / ('valid.txt' if rec['group'] == 'external' else 'inf_valid.txt')
    queries = np.asarray([(emap[h], rmap[r], emap[t]) for h, r, t in rows(path)], dtype=np.int64)
    return test_z, queries, np.concatenate([forward_facts(test_z), queries]), names, 'target_graph_valid_disjoint_from_test'

def final_labels(rec):
    with np.load(STAGE6 / rec['file']) as data:
        return data['queries'].copy(), data['alltrue'].copy()
