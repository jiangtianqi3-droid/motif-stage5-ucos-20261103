"""Only observation facts enter feature builders; vocabulary is public metadata."""
import json, re, hashlib
from pathlib import Path
import numpy as np

ASSETS = Path('/mnt/d/motif-phase4-work/topology-semantic-stage4-20261102')
DATA = ASSETS / 'data/external/ConceptNet100k'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def verbalize(name):
    if name.startswith('/c/en/'): name = name.split('/')[3]
    return name.replace('_', ' ').strip()

def load_vocab():
    v = json.loads((DATA/'vocab.json').read_text())
    entities = sorted(v['entities'], key=v['entities'].get)
    relations = sorted(v['relations'], key=v['relations'].get)
    assert list(sorted(v['entities'].values())) == list(range(78334))
    assert len(relations) == 34
    texts = [verbalize(x) for x in entities]
    # Numeric-looking *original concepts* are legitimate; ID-to-text fallback is prohibited.
    assert all(texts) and len(set(texts)) > 78000
    return v, texts, relations

def read_split(split, vocab):
    rows = [line.split('\t') for line in (DATA/(split+'.txt')).read_text().splitlines() if line]
    return np.array([(vocab['entities'][h],vocab['relations'][r],vocab['entities'][t]) for h,r,t in rows], dtype=np.int64)

def observed(train, n, R, cache=None, historical=True):
    edge=np.concatenate([train[:,[0,2]],train[:,[2,0]]]).T
    rel=np.concatenate([train[:,1],train[:,1]+R])
    if historical:
        with np.load(DATA/'observed.npz') as f:
            assert np.array_equal(edge,f['edge_index']) and np.array_equal(rel,f['edge_type'])
            sig={k:f[k].copy() for k in ['hyper_edge','hyper_type','z','raw']}
    else:
        from topology import extract
        s=extract(edge,rel,n,2*R,cache=cache)
        sig=dict(hyper_edge=s['edge_index'],hyper_type=s['edge_type'],z=s['z'],raw=s['raw'])
    return dict(n=n,R=2*R,edge_index=edge,edge_type=rel,**sig)

def bidirectional(facts,R=34):
    return np.stack([facts,np.c_[facts[:,2],facts[:,1]+R,facts[:,0]]],axis=1).reshape(-1,3)

def relation_texts(names):
    overrides={'CapableOf':'is capable of','AtLocation':'is at location','IsA':'is a','HasA':'has a','UsedFor':'is used for','PartOf':'is part of','RelatedTo':'is related to','ReceivesAction':'receives action'}
    base=[overrides.get(r,re.sub(r'([a-z])([A-Z])',r'\1 \2',r).lower()) for r in names]
    return base+['inverse of '+r for r in base]
