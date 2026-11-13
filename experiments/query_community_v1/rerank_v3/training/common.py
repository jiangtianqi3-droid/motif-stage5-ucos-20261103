import gc,hashlib,json,random,shutil,os
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from motif.models import MOTIF
from plans import to_device,build_plan
ROOT=Path(__file__).resolve().parent
HISTORY=Path('D:/motif-phase4-work' if os.name=='nt' else '/mnt/d/motif-phase4-work')
OLD=HISTORY/'topology-semantic-stage4-20261102'
S2=HISTORY/'topology-stage2-20261102'
SOURCES=['fb15k237_train','wn18rr_train','codex_medium_train']
CKPT=HISTORY.parent/'motif-phase3-work/MOTIF/ckpts/motif_3g.pth'
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def log(event,**kw):print(json.dumps(dict(event=event,**kw)),flush=True)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def array_sha(*arrays):
    h=hashlib.sha256()
    for x in arrays:h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()
def base_model(seed=1024):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    m=MOTIF(rel_model_cfg=dict(input_dim=64,hidden_dims=[64]*6,short_cut=True,use_triton=True,aggregate_func='sum'),entity_model_cfg=dict(input_dim=64,hidden_dims=[64]*6,message_func='distmult',aggregate_func='sum',short_cut=True,layer_norm=True,use_triton=True))
    m.load_state_dict(torch.load(CKPT,map_location='cpu')['model'],strict=True);return m
def make_data(z,q=None,plan=None,device='cuda'):
    n,R=int(z['n']),int(z['R']);he=z['hyper_edge'];ht=z['hyper_type'];q=z['queries'] if q is None else q
    h=Data(edge_index=torch.from_numpy(he.astype(np.int64)).to(device),edge_type=torch.from_numpy(ht.astype(np.int64)).to(device),num_nodes=R,num_relations=7)
    h.plan=to_device(plan if plan is not None else build_plan(he,ht,R),device)
    h.topology_z=torch.from_numpy(z['z'].astype(np.float32)).to(device)
    raw=z['raw'][:,3].astype(float);cv=float(raw.std()/max(abs(raw.mean()),1e-6)) if len(raw) else 0.
    h.alpha_adaptive=.15+.10*cv/(1+cv);h.topology_cv=cv
    h.type_bounds=np.r_[0,np.cumsum(np.bincount(ht,minlength=7))].tolist()
    qt=torch.from_numpy(q.astype(np.int64)).to(device)
    d=Data(edge_index=torch.from_numpy(z['edge_index'].astype(np.int64)).to(device),edge_type=torch.from_numpy(z['edge_type'].astype(np.int64)).to(device),num_nodes=n,num_relations=R,target_edge_index=qt[:,[0,2]].T,target_edge_type=qt[:,1],relation_hypergraph=h)
    t=torch.from_numpy(z['alltrue'].astype(np.int64)).to(device)
    truth=Data(edge_index=t[:,[0,2]].T,edge_type=t[:,1],num_nodes=n)
    d.hub_log_degree=torch.bincount(d.edge_index[1],minlength=n).float().log1p()
    return d,truth
def load_cached(source,label,device='cuda'):
    # Never deserialize the legacy alltrue field: it contains held-out labels.
    with np.load(S2/'data'/f'{source}__{label}.npz',allow_pickle=False) as zz:
        z={k:zz[k] for k in zz.files if k!='alltrue'}
    from strict_data import load_truth
    z['alltrue']=load_truth(ROOT,source,'validation' if label=='validation' else 'train')
    q=np.load(OLD/'data'/f'{source}_validation_queries.npy')[:512] if label=='validation' else z['queries']
    with np.load(S2/'plans'/f'{source}__{label}.npz') as p:plan={k:p[k] for k in p.files}
    return make_data(z,q,plan,device)
def downstream(name,version,device='cuda'):
    if name=='ConceptNet100k_external':
        with np.load(OLD/'data/external/ConceptNet100k/observed.npz') as zz:return make_data({k:zz[k] for k in zz.files},device=device)
    from motif import datasets
    from topology import graph_key
    for raw in (OLD/'data/downstream').rglob('raw'):
        dest=ROOT/'data/downstream'/raw.relative_to(OLD/'data/downstream')
        if not dest.exists():shutil.copytree(raw,dest)
    if name.endswith('Inductive'):
        old=getattr(datasets,name);cls=type('Raw'+name,(datasets.GrailInductiveDataset,),dict(name=old.name,urls=old.urls))
        ds=cls(root=str(ROOT/'data/downstream'),version=version,device='cpu',pre_transform=None)
    else:ds=getattr(datasets,name)(root=str(ROOT/'data/downstream'),version=version,device='cpu',pre_transform=None)
    train,valid,test=[ds[i] for i in range(3)]
    true_edges=torch.cat([test.edge_index,valid.target_edge_index,test.target_edge_index],1) if 'Ingram' in name else torch.cat([test.edge_index,test.target_edge_index],1)
    true_rel=torch.cat([test.edge_type,valid.target_edge_type,test.target_edge_type]) if 'Ingram' in name else torch.cat([test.edge_type,test.target_edge_type])
    key=graph_key(test.edge_index.numpy(),test.edge_type.numpy(),test.num_nodes,int(test.num_relations));path=OLD/'cache/downstream'/(key+'.npz');assert path.exists(),path
    with np.load(path) as zz:sig={k:zz[k] for k in zz.files}
    z=dict(n=test.num_nodes,R=test.num_relations,edge_index=test.edge_index.numpy(),edge_type=test.edge_type.numpy(),queries=np.c_[test.target_edge_index[0],test.target_edge_type,test.target_edge_index[1]],alltrue=np.c_[true_edges[0],true_rel,true_edges[1]],hyper_edge=sig['edge_index'],hyper_type=sig['edge_type'],z=sig['z'],raw=sig['raw'])
    return make_data(z,device=device)
