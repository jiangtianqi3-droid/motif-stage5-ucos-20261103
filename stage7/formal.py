"""Separate formal-evaluation process. Imports labels only after all choices freeze."""
import gc
import hashlib
import zipfile
from datetime import datetime,timezone
import numpy as np
import torch
from stage7.util import ROOT,read,dump,sha,identity,log
from stage7.experiment import graph_for,score,save
from stage7.numeric_stability import install
from stage6.models import load_model,state_hash
from stage7.robust_selector import ALPHAS
from stage7.prepare import fetch
from stage7.evaluator import metrics


def prospective_queries(key,graph):
    rec=read(ROOT/'observations/manifest.json')[key]
    vocab=read(ROOT/'observations'/(key+'_vocab.json'))
    entities,relations=vocab['entities'],vocab['relations']
    def text(which):
        if 'archive' in rec:
            archive=ROOT/rec['download_file'].replace('\\','/')
            assert sha(archive)==rec['download_sha256']
            with zipfile.ZipFile(archive) as z:
                member=rec[which+'_member']
                if member not in z.namelist():
                    matches=[m for m in z.namelist() if m.endswith('/'+member)]
                    assert len(matches)==1
                    member=matches[0]
                return z.read(member).decode('utf8')
        path=ROOT/'formal_inputs'/(key+'_'+which+'.txt')
        fetch(rec[which],path)
        return path.read_text(encoding='utf8')
    def parse(value):
        facts=[]
        for line in value.splitlines():
            if not line.strip():continue
            h,r,t=line.split('\t') if '\t' in line else line.split()
            for x in (h,t):
                if x not in entities:
                    entities[x]=len(entities)
            if r not in relations:
                raise ValueError('Held-out relation outside observation vocabulary: '+r)
            facts.append((entities[h],relations[r],entities[t]))
        assert len(entities)<=graph.n,('published candidate metadata insufficient',key,len(entities),graph.n)
        return np.asarray(facts,dtype=np.int64)
    valid=parse(text('valid')) if 'valid_member' in rec or 'valid' in rec else np.empty((0,3),dtype=np.int64)
    test=parse(text('test'))
    truth=np.r_[np.c_[graph.edge_index[0],graph.edge_type,graph.edge_index[1]],valid,test]
    return test,truth


def run(args):
    frozen=read(ROOT/'frozen.json');p=read(ROOT/'protocol.json')
    assert frozen['protocol_sha256']==sha(ROOT/'protocol.json')
    assert frozen['model_manifest_sha256']==sha(ROOT.parent/'stage6/model_manifest.json')
    assert frozen['preserved_manifest_sha256']==sha(ROOT/'preserved_assets.json')
    for name,h in read(ROOT/'preserved_assets.json').items():
        assert sha(ROOT.parent/name)==h,name
    for name,h in frozen['code_hashes'].items():assert sha(ROOT/name)==h,name
    for label,r in frozen['selections'].items():assert sha(ROOT/'adaptation'/(label+'.json'))==r['sha256']
    assert frozen['new_observation_manifest_sha256']==sha(ROOT/'observations/manifest.json')
    assert frozen['old_observation_manifest_sha256']==sha(args.observations/'manifest.json')
    fh=sha(ROOT/'frozen.json')
    expected={r['key']:r for r in read(args.stage5/'analysis/target_inputs.json')}
    for group,key in p['targets']:
        if key in frozen.get('unavailable_targets',{}):
            log('formal_skipped_unavailable',key=key)
            continue
        if args.keys and key not in args.keys:continue
        graph=graph_for(args,key)
        marker=ROOT/'formal_inputs'/(key+'_opened.json')
        if not marker.exists():dump(marker,dict(opened_utc=datetime.now(timezone.utc).isoformat(),frozen_sha256=fh),exclusive=True)
        if key.endswith('_new'):
            queries,truth=prospective_queries(key,graph)
        else:
            from stage6.formal import load_queries
            queries,truth=load_queries(args,key,graph)
            trip=queries[:,[0,2,1]].copy()
            assert hashlib.sha256(trip.tobytes()).hexdigest()==expected[key]['query_sha256']
            e,r=truth[:,[0,2]].T.copy(),truth[:,1].copy()
            assert hashlib.sha256(e.tobytes()+r.tobytes()).hexdigest()==expected[key]['filtered_sha256']
            assert hashlib.sha256(graph.edge_index.tobytes()+graph.edge_type.tobytes()).hexdigest()==expected[key]['observed_sha256']
            from topology import extract
            sig=extract(graph.edge_index,graph.edge_type,graph.n,graph.R,cache=ROOT/'cache/topology')
            he=sig['edge_index'].astype(np.int64);ht=sig['edge_type'].astype(np.int64)
            assert hashlib.sha256(he.tobytes()+ht.tobytes()).hexdigest()==expected[key]['hypergraph_sha256']
            del sig,he,ht
        for family,seeds in [('T-4D',p['training_seeds']),('B0',[1024]),('B1',p['training_seeds']),('B2',p['training_seeds'])]:
            for seed in seeds:
                path=ROOT/'evaluation'/f'{key}__{family}_s{seed}.json'
                if path.exists():
                    assert read(path)['frozen_sha256']==fh
                    assert read(path)['ranks_sha256']==sha(path.with_suffix('.npz'))
                    continue
                log('formal_begin',key=key,family=family,seed=seed,facts=len(queries))
                m=install(load_model(args.stage5,args.history,args.official,family,seed));before=state_hash(m)
                alphas=ALPHAS if family=='T-4D' else ([1.] if family=='B2' else [0.])
                try:
                    ranks,info,topo=score(m,graph,queries,truth,alphas,args.batch_size)
                except (torch.cuda.OutOfMemoryError,MemoryError) as e:
                    import traceback
                    dump(ROOT/'formal_failures'/f'{key}__{family}_s{seed}.json',dict(key=key,family=family,seed=seed,
                        status='resource_failure_after_batch_halving',error=repr(e),traceback=traceback.format_exc(),frozen_sha256=fh))
                    log('formal_resource_failure',key=key,family=family,seed=seed,error=repr(e))
                    del m;gc.collect();torch.cuda.empty_cache()
                    continue
                assert before==state_hash(m)
                save(path,dict(key=key,group=group,family=family,training_seed=seed,frozen_sha256=fh,
                     checkpoint_sha256=m.tsdc_checkpoint_sha256,observation_sha256=graph.key,query_sha256=identity(queries),
                     filtered_sha256=identity(truth),info=info,topology=topo,metrics=[metrics(v) for v in ranks],
                     no_weight_update=True,numeric_policy=p['numeric']),ranks,queries,alphas)
                log('formal_complete',key=key,family=family,seed=seed,seconds=info['seconds'])
                del m;gc.collect();torch.cuda.empty_cache()
    expected=len(p['targets'])*10
    completed=len(list((ROOT/'evaluation').glob('*.json')))
    dump(ROOT/'results/formal_status.json',dict(status='complete' if completed==expected else 'incomplete',
        registered_graphs=len(p['targets']),successful_model_runs=completed,expected_model_runs=expected,
        unavailable_targets=list(frozen.get('unavailable_targets',{})),frozen_sha256=fh))
