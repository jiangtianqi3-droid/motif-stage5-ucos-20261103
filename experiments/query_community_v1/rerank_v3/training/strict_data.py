"""Role-separated label stores. Training never opens validation/test triples."""
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def encode_file(path, entities, relations):
    rows=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            h,r,t=line.split();rows.append((entities[h],relations[r],entities[t]))
    return np.asarray(rows,dtype=np.int64).reshape(-1,3)

def truth_for_role(train, valid=None, *, role):
    train=np.asarray(train,dtype=np.int64).reshape(-1,3)
    if role=='train':return np.unique(train,axis=0)
    if role=='validation':
        if valid is None:raise ValueError('Validation truths required for validation only')
        return np.unique(np.concatenate([train,np.asarray(valid).reshape(-1,3)]),axis=0)
    raise ValueError('Only explicit train and validation roles are supported')

def load_truth(root,source,role):
    if role not in ('train','validation'):raise ValueError(role)
    root=Path(root);manifest=json.loads((root/'data/strict_manifest.json').read_text())
    entry=manifest[source][role];path=root/entry['path']
    if sha(path)!=entry['sha256']:raise RuntimeError('Label-store fingerprint changed')
    return np.load(path,allow_pickle=False)

def prepare(root,history):
    root=Path(root);history=Path(history)
    mappings=history/'topology-stage2-20261102/analysis/source_mappings.json'
    manifest={}
    for m in json.loads(mappings.read_text()):
        source=m['source'];train_path=history/'topology-stage1-20261101/raw/pretraining'/f'{source}.txt'
        valid_path=history/'topology-stage2-20261102/data'/f'{source}_valid.txt'
        train=encode_file(train_path,m['entities'],m['relations'])
        valid=encode_file(valid_path,m['entities'],m['relations'])
        train_set=set(map(tuple,train.tolist()));valid_set=set(map(tuple,valid.tolist()))
        if train_set&valid_set:raise ValueError('Source train/validation duplicate triples')
        row=dict(train_input_sha256=sha(train_path),validation_input_sha256=sha(valid_path),
                 vocabulary_sha256=sha(mappings),source_test_files_opened=False,fold_checks=[])
        for role in ('train','validation'):
            a=truth_for_role(train,valid if role=='validation' else None,role=role)
            path=root/'data'/f'{source}_{role}_truth.npy';np.save(path,a,allow_pickle=False)
            row[role]=dict(path=str(path.relative_to(root)),sha256=sha(path),facts=len(a))
        for fold in range(5):
            path=history/'topology-stage2-20261102/data'/f'{source}__fold{fold}.npz'
            with np.load(path,allow_pickle=False) as z:
                q=z['queries'];edges=z['edge_index'];r=z['edge_type'];R=int(z['R'])
            obs=np.c_[edges[0,r<R//2],r[r<R//2],edges[1,r<R//2]]
            observed=set(map(tuple,obs.tolist()));supervised=set(map(tuple,q.tolist()))
            if observed&supervised or (observed|supervised)!=train_set:
                raise ValueError('Fold observations/supervision do not partition source train')
            row['fold_checks'].append(dict(fold=fold,observed=len(observed),supervised=len(supervised),passed=True))
        manifest[source]=row
    path=root/'data/strict_manifest.json'
    text=json.dumps(manifest,indent=2,sort_keys=True)
    if path.exists() and path.read_text()!=text:raise RuntimeError('Prepared inputs changed')
    path.write_text(text,encoding='utf-8')
    return manifest
