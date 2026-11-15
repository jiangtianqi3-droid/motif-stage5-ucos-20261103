"""Read observation arrays and align raw observation tokens, never read held-out rows."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGE7 = Path('/mnt/d/motif-phase4-work/motif-stage7-asymmetric-20261106')
STAGE6 = STAGE7.parent / 'motif-stage6-entity-degree-p0p1-v2'
HISTORY = STAGE7.parent / 'topology-semantic-stage4-20261102'
PREVIOUS = ROOT.parent / 'conceptnet_zeroshot_v1'

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def records():
    return [r for r in json.loads((STAGE7 / 'inputs.json').read_text()) if not r['source']]

def observation(rec):
    path = STAGE6 / rec['file']
    assert digest(path) == rec['sha256'], str(path)
    # npz is lazy: do not materialize queries or alltrue here.
    with np.load(path) as f:
        z = {k: f[k].copy() for k in ['n', 'R', 'edge_index', 'edge_type', 'hyper_edge', 'hyper_type', 'z', 'raw']}
    z['n'], z['R'] = int(z['n']), int(z['R'])
    return z

def raw_observation_path(key):
    if key == 'ConceptNet100k_external':
        return HISTORY / 'data/external/ConceptNet100k/train.txt'
    name, version = key.rsplit('_', 1)
    if 'Inductive' in name:
        family = {'NELLInductive': 'IndNELL', 'FB15k237Inductive': 'IndFB15k237', 'WN18RRInductive': 'IndWN18RR'}[name]
        return HISTORY / 'data/downstream/grail' / family / version / 'raw/train_ind.txt'
    return HISTORY / 'data/downstream/ingram' / name[:2].lower() / version / 'raw/inference_graph.txt'

def readable(token):
    if token is None or re.fullmatch(r'[\d\W_]+', token):
        return False
    # Freebase machine IDs and WordNet offsets are identifiers, not words.
    if re.fullmatch(r'/?[mg][./][\w]+', token) or re.fullmatch(r'Q\d+|P\d+', token):
        return False
    return bool(re.search('[a-zA-Z]', token))

def verbalize(token):
    if not readable(token):
        return None
    if token.startswith('/c/en/'):
        token = token.split('/')[3]
    if token.startswith('concept:'):
        token = token[len('concept:'):]
    return token.replace('_', ' ').replace(':', ': ').strip()

def vocabulary(rec, z):
    path = raw_observation_path(rec['key'])
    rows = [line.split('\t') if '\t' in line else line.split() for line in path.read_text().splitlines() if line.strip()]
    R = z['R'] // 2
    ids = np.flatnonzero(z['edge_type'] < R)
    assert len(rows) == len(ids), (rec['key'], len(rows), len(ids))
    entity, relation = [None] * z['n'], [None] * R
    for (h, r, t), i in zip(rows, ids):
        u, v = z['edge_index'][:, i]
        for names, index, value in [(entity, u, h), (entity, v, t), (relation, z['edge_type'][i], r)]:
            assert names[index] in (None, value), 'Raw observation order differs from frozen IDs'
            names[index] = value
    if rec['key'] == 'ConceptNet100k_external':
        public = json.loads((path.parent / 'vocab.json').read_text())
        for names, mapping in [(entity, public['entities']), (relation, public['relations'])]:
            assert len(names) == len(mapping)
            for token, index in mapping.items():
                assert names[index] in (None, token)
                names[index] = token
    return entity, relation, path

def inspect(rec, z):
    entities, relations, path = vocabulary(rec, z)
    R = z['R'] // 2
    E = int((z['edge_type'] < R).sum())
    kappa = z['n'] / R
    extreme = kappa > 500
    return dict(key=rec['key'], group=rec['group'], entities=z['n'], forward_relations=R,
                observed_forward_edges=E, kappa=kappa, mean_forward_out_degree=E / z['n'],
                regime='Extreme-Commonsense' if extreme else 'Standard-Inductive',
                beta_tri=.15 if extreme else 1., p_tri=.30 if extreme else 1.,
                alpha=.90 if extreme else .20, gamma_out=.75, gamma_rel=.50 if extreme else 0.,
                has_transparent_entity_text=any(map(readable, entities)),
                has_transparent_relation_text=any(map(readable, relations)),
                readable_entities=sum(map(readable, entities)), readable_relations=sum(map(readable, relations)),
                missing_observation_entity_names=sum(x is None for x in entities),
                entity_examples=[x for x in entities if x is not None][:5],
                relation_examples=[x for x in relations if x is not None][:5],
                mapping_source=str(path), mapping_sha256=digest(path),
                text_policy='Only readable original observation names; unknown/anonymous rows stay zero; no ID verbalization',
                test_exposure='Historically evaluated; locked denotes legacy group, not first blind test')

def main():
    result = {}
    for rec in records():
        z = observation(rec)
        result[rec['key']] = inspect(rec, z)
    save_json(ROOT / 'results/graph_regimes.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
