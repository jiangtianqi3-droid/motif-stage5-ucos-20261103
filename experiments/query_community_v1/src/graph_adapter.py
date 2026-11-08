"""Read-only adapters. Retrieval receives observations and (head, relation), never answers."""
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]

def runtime_path(path):
    text = str(path).replace('\\', '/')
    if os.name != 'nt' and len(text)>2 and text[1]==':':
        text = '/mnt/'+text[0].lower()+text[2:]
    return Path(text)

HISTORY = runtime_path('D:/motif-phase4-work')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def digest(*arrays):
    h=hashlib.sha256()
    for x in arrays:
        a=np.ascontiguousarray(x)
        h.update(str((a.shape,str(a.dtype))).encode()); h.update(a.tobytes())
    return h.hexdigest()

def dump(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

@dataclass
class ObservedGraph:
    edge_index: np.ndarray
    edge_type: np.ndarray
    n: int
    R: int

    def __post_init__(self):
        self.edge_index=np.array(self.edge_index,dtype=np.int64,copy=True)
        self.edge_type=np.array(self.edge_type,dtype=np.int64,copy=True)
        assert self.n>=0 and self.R>0 and self.R%2==0
        assert self.edge_index.shape==(2,len(self.edge_type))
        if len(self.edge_type):
            assert self.edge_index.min()>=0 and self.edge_index.max()<self.n
            assert self.edge_type.min()>=0 and self.edge_type.max()<self.R
        self.out_degree=np.bincount(self.edge_index[0],minlength=self.n)
        self.in_degree=np.bincount(self.edge_index[1],minlength=self.n)
        _, inverse, counts=np.unique(self.edge_index[0]*self.R+self.edge_type,return_inverse=True,return_counts=True)
        self.edge_rel_out=counts[inverse]
        # CSR positions retain all parallel, duplicate and inverse edge IDs.
        self.out_order=np.argsort(self.edge_index[0],kind='stable')
        self.in_order=np.argsort(self.edge_index[1],kind='stable')
        self.out_ptr=np.r_[0,np.cumsum(self.out_degree)]
        self.in_ptr=np.r_[0,np.cumsum(self.in_degree)]
        self.key=digest(self.edge_index,self.edge_type,np.array([self.n,self.R]))
        self.edge_index.flags.writeable=False; self.edge_type.flags.writeable=False

    def inverse(self,r): return (int(r)+self.R//2)%self.R

    def edges_at(self,v,reverse=False):
        ptr,order=(self.in_ptr,self.in_order) if reverse else (self.out_ptr,self.out_order)
        return order[ptr[v]:ptr[v+1]]

    def assert_inverse_pairs(self):
        from collections import Counter
        counts=Counter(zip(self.edge_index[0].tolist(),self.edge_type.tolist(),self.edge_index[1].tolist()))
        assert all(counts[(v,self.inverse(r),u)]==c for (u,r,v),c in counts.items())

    def assert_hidden_absent(self,hidden):
        facts=set(zip(self.edge_index[0].tolist(),self.edge_type.tolist(),self.edge_index[1].tolist()))
        for h,r,t in hidden:
            assert (int(h),int(r),int(t)) not in facts, 'Hidden target in observations'
            assert (int(t),self.inverse(r),int(h)) not in facts, 'Hidden inverse in observations'

def load_observed(key):
    path=HISTORY/'tsdc-stage6-20261103/stage6/observations'/f'{key}.npz'
    manifest=json.loads((path.parent/'manifest.json').read_text(encoding='utf-8-sig'))
    assert sha(path)==manifest[key]['sha256']
    with np.load(path,allow_pickle=False) as z:
        g=ObservedGraph(z['edge_index'],z['edge_type'],int(z['n']),int(z['R']))
    g.assert_inverse_pairs()
    return g

def load_signature(graph,key):
    """Verify graph identity before reading historical observation-only features."""
    path=HISTORY/'motif-stage6-entity-degree-p0p1-v2/data'/f'{key}.npz'
    with np.load(path,allow_pickle=False) as z:
        assert np.array_equal(z['edge_index'],graph.edge_index)
        assert np.array_equal(z['edge_type'],graph.edge_type)
        expected_he=z['hyper_edge'].copy();expected_ht=z['hyper_type'].copy();expected_z=z['z'].copy()
        bundle_raw_placeholder=bool(np.all(z['raw']==0))
    # Stage-6 observation bundles intentionally stored zero raw placeholders.
    # Recover the canonical complete signatures, not synthetic counts from z.
    topology_file=HISTORY/'topology-ucos-stage5-20261103/topology.py'
    spec=importlib.util.spec_from_file_location('qc_frozen_topology',topology_file)
    topology=importlib.util.module_from_spec(spec)
    # Execute the trusted frozen source without creating __pycache__ beside it.
    exec(compile(topology_file.read_text(encoding='utf-8'),str(topology_file),'exec'),topology.__dict__)
    canonical_key=topology.graph_key(graph.edge_index,graph.edge_type,graph.n,graph.R)
    source=HISTORY/'topology-semantic-stage4-20261102/cache/downstream'/f'{canonical_key}.npz'
    if source.exists():
        with np.load(source,allow_pickle=False) as z: sig={k:z[k].copy() for k in z.files}
    else:
        sig=topology.extract(graph.edge_index,graph.edge_type,graph.n,graph.R,cache=ROOT/'cache/signatures')
    order=topology.align(sig,expected_he,expected_ht,graph.R)
    for field in ['raw','z','count']:
        sig[field]=sig[field][order]
    sig['edge_index']=expected_he;sig['edge_type']=expected_ht
    assert np.allclose(sig['z'],expected_z,atol=2e-5,rtol=2e-5),'Historical standardized signature mismatch'
    assert np.all(sig['raw'][:,0]>0),'Raw frequency placeholders forbidden'
    rebuilt,_,_=topology.standardize(sig['raw'],sig['edge_type'])
    assert np.allclose(rebuilt,sig['z'],atol=1e-8,rtol=1e-8)
    dump(ROOT/'cache/signatures'/f'{graph.key}_audit.json',dict(canonical_source=str(source),
        graph_key=graph.key,canonical_key=canonical_key,bundle_raw_was_placeholder=bundle_raw_placeholder,
        z_max_abs_difference=float(np.max(np.abs(sig['z']-expected_z))),raw_recovered=True))
    assert sig['raw'].shape==sig['z'].shape==(len(sig['edge_type']),4)
    assert np.isfinite(sig['z']).all()
    return sig

def load_query_inputs(key):
    """Only public query inputs are returned. Answers stay in offline evaluation."""
    path=HISTORY/'motif-stage6-entity-degree-p0p1-v2/data'/f'{key}.npz'
    with np.load(path,allow_pickle=False) as z:
        queries=z['queries']
        return queries[:,[0,1]].copy(),queries[:,[2,1]].copy()

class CandidateProvider:
    """Stage-7 B0 unfiltered reciprocal-rank records, not target-specific ranks.

    Scores are 1/(60+raw pessimistic rank); true logits were not retained.
    Equal scores are resolved by entity ID; no ground-truth filters are read.
    """
    def __init__(self,key,n):
        self.path=HISTORY/'motif-stage7-asymmetric-20261106/evaluation'/key/'B0.rrf.npy'
        meta=json.loads(self.path.with_name('B0.json').read_text(encoding='utf-8-sig'))
        assert sha(self.path)==meta['feature_sha256']
        self.scores=np.load(self.path,mmap_mode='r',allow_pickle=False)
        assert self.scores.shape[1:]==(2,n)
        self.source_sha256=meta['feature_sha256']
        self.query_sha256=meta['query_sha256']

    def get(self,index,direction,k=30):
        score=self.scores[index,direction]
        assert np.isfinite(score).all()
        order=np.lexsort((np.arange(len(score)),-score))[:k]
        return [dict(entity=int(t),base_score=float(score[t]),base_rank=i+1,
                     raw_tie_rank=int(round(1/float(score[t])-60)),score_kind='reciprocal_raw_rank')
                for i,t in enumerate(order)]

def extract_frozen_gates(graph,sig,relations,cache,on_progress=None):
    """Exact frozen T-4D relation forward, capturing six raw gate tensors.

    Imports the historical model read-only; CUDA compilation and all writes are
    redirected to this phase by run.sh. No entity model forward or training.
    """
    import torch
    sys.dont_write_bytecode=True
    torch.set_num_threads(4)
    old=HISTORY/'topology-ucos-stage5-20261103'
    sys.path[:0]=[str(old/'runtime'),str(old)]
    from common import base_model,make_data
    from model import install
    frozen=json.loads((old/'frozen.json').read_text())['formal']['T-4D']['1024']
    checkpoint=old/frozen['checkpoint']
    assert sha(checkpoint)==frozen['checkpoint_sha256']
    model=install(base_model(),frozen['config']).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['model'],strict=True)
    for p in model.parameters(): p.requires_grad_(False)
    before=digest(*[v.detach().cpu().numpy() for v in model.state_dict().values()])
    # Dummy empty targets are structural placeholders, never real answers.
    z=dict(n=graph.n,R=graph.R,edge_index=graph.edge_index.copy(),edge_type=graph.edge_type.copy(),
        queries=np.empty((0,3),dtype=np.int64),alltrue=np.empty((0,3),dtype=np.int64),
        hyper_edge=sig['edge_index'],hyper_type=sig['edge_type'],raw=sig['raw'],z=sig['z'])
    data,_=make_data(z,device='cuda')
    capture=[]
    handles=[layer.topology_gate.register_forward_hook(lambda m,i,o: capture.append(o.detach()))
             for layer in model.relation_model.layers]
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    results={}
    try:
        with torch.inference_mode():
            ordered=sorted(set(map(int,relations)))
            for position,q in enumerate(ordered):
                path=cache/f'{graph.key}_{frozen["checkpoint_sha256"]}_{q}.npy'
                if path.exists():
                    a=np.load(path,allow_pickle=False)
                else:
                    capture.clear()
                    model.relation_model(data.relation_hypergraph,torch.tensor([q],device='cuda'))
                    assert len(capture)==len(model.relation_model.layers)==6
                    a=torch.stack(capture).mean(0)[0].cpu().numpy()
                    tmp=path.with_suffix('.tmp.npy');np.save(tmp,a);tmp.replace(path)
                assert a.shape==(len(sig['edge_type']),) and np.isfinite(a).all() and (a>0).all()
                results[q]=a
                if on_progress is not None: on_progress(position+1,len(ordered),q)
    finally:
        for handle in handles: handle.remove()
    after=digest(*[v.detach().cpu().numpy() for v in model.state_dict().values()])
    assert before==after
    dump(cache/f'{graph.key}_gate_audit.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=frozen['checkpoint_sha256'],
        model_parameters_sha256_before=before,model_parameters_sha256_after=after,layers=6,
        gate_definition='arithmetic layer mean of actual TopologyGate outputs before incidence normalization',
        relations=sorted(results),training=False,entity_forward_called=False))
    del model,data
    torch.cuda.empty_cache()
    return results
