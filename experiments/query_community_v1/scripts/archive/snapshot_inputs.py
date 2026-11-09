"""Read-only copy of the exact public graph inputs and offline labels."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HISTORY=Path('D:/motif-phase4-work')
RUN='d48ae432a2533cdaaec0'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def native(path):return Path(str(path).replace('/mnt/d/','D:/').replace('\\','/'))

def main():
    output=ROOT/'artifacts/inputs';output.mkdir(parents=True,exist_ok=True)
    vocab=json.loads((HISTORY/'topology-semantic-stage4-20261102/semantic/relation_mappings.json').read_text(encoding='utf-8'))
    provenance=json.loads((HISTORY/'topology-semantic-stage4-20261102/semantic/target_mapping_provenance.json').read_text(encoding='utf-8'))
    observations=HISTORY/'tsdc-stage6-20261103/stage6/observations'
    original=json.loads((observations/'manifest.json').read_text(encoding='utf-8'))
    protocol=json.loads((ROOT/'results/runs'/RUN/'protocol.json').read_text(encoding='utf-8'))
    records=[]
    def preserve(source,target):
        source=Path(source);target=output/target;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==sha(source),'Input snapshot differs'
        else:shutil.copy2(source,target)
        assert sha(target)==sha(source)
        assert target.stat().st_size<90*1024**2
        return dict(file=target.relative_to(ROOT).as_posix(),bytes=target.stat().st_size,sha256=sha(target),original_path=str(source))
    for key in protocol['config']['datasets']:
        preprocessing=json.loads((ROOT/'results/runs'/RUN/key/'preprocessing.json').read_text(encoding='utf-8'))
        audit=json.loads((ROOT/'cache/signatures'/(preprocessing['graph_key']+'_audit.json')).read_text(encoding='utf-8'))
        source=observations/(key+'.npz');assert sha(source)==original[key]['sha256']
        files=[preserve(source,Path(key)/'observed.npz'),
               preserve(native(audit['canonical_source']),Path(key)/'canonical_topology.npz')]
        raw=native(provenance[key]['raw'])
        names=[provenance[key]['fact_file']]+(['valid_ind.txt','test_ind.txt'] if provenance[key]['kind']=='grail' else ['inf_valid.txt','inf_test.txt'])
        for name in names:files.append(preserve(raw/name,Path(key)/'raw'/name))
        target=output/key/'relation_vocabulary.json'
        target.write_text(json.dumps(vocab[key],ensure_ascii=False,indent=2),encoding='utf-8')
        files.append(dict(file=target.relative_to(ROOT).as_posix(),bytes=target.stat().st_size,sha256=sha(target),original_source='published frozen relation vocabulary'))
        source=HISTORY/'motif-stage6-entity-degree-p0p1-v2/data'/(key+'.npz')
        with np.load(source,allow_pickle=False) as z:
            queries=z['queries'].copy()
        target=output/key/'offline_targets_ONLY.npz'
        np.savez_compressed(target,queries=queries)
        files.append(dict(file=target.relative_to(ROOT).as_posix(),bytes=target.stat().st_size,sha256=sha(target),
            original_path=str(source),original_sha256=sha(source),source_member='queries',role='offline evaluation only; never a retrieval seed'))
        records.append(dict(dataset=key,graph_key=preprocessing['graph_key'],canonical_key=audit['canonical_key'],
            queries=len(queries),protocol=provenance[key]['kind'],files=files))
    manifest=dict(run=RUN,observed_and_offline_labels_separate=True,original_files_modified=False,datasets=records)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(datasets=len(records),files=sum(len(r['files']) for r in records),
        bytes=sum(f['bytes'] for r in records for f in r['files'])),indent=2))

if __name__=='__main__':main()
