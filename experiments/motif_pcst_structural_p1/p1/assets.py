"""Read-only frozen assets and answer-free graph/rank adapters."""
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from p1.config import ROOT, RESULTS
from src.io_utils import HISTORY, STAGE7, PRIMARY, SNAPSHOT, dump, read, sha, snapshot
from src.graph_loader import ObservedGraph
from src.recovery import load, verify_runtime

POC = ROOT.parent / 'motif_pcst_api_poc_v1'
BUNDLE = HISTORY / 'motif-stage6-entity-degree-p0p1-v2'
M5 = HISTORY / 'conceptnet-zeroshot-boost/experiments/universal_11g_v1/results/runs/5443b773ab1db4353493/per_graph'


def verify(condition, msg):
    if not condition:
        raise ValueError(msg)


class Assets:
    def __init__(self, key, folder):
        from s7util import arrsha

        self.key = key
        self.folder = folder / 'assets' / key
        self.folder.mkdir(parents=True, exist_ok=True)
        self.entry = next(r for r in read(STAGE7/'inputs.json') if r['key']==key)
        self.source = self.entry['source']
        self.bundle = BUNDLE / self.entry['file']
        verify(sha(self.bundle)==self.entry['sha256'],f'{key} bundle hash mismatch')
        self.history = STAGE7/'evaluation'/key
        self.meta = read(self.history/(PRIMARY+'.json'))
        verify(self.meta['query_sha256']==self.entry['query_sha256'],f'{key} query identity')
        self.ref = self.history/(PRIMARY+'.npz')
        verify(sha(self.ref)==self.meta['npz_sha256'],f'{key} frozen rank hash')
        self.arr = {}
        with np.load(self.bundle,allow_pickle=False) as z:
            for name in ('n','R','edge_index','edge_type','hyper_edge','hyper_type','z','raw'):
                self.arr[name]=z[name]
        a = self.arr
        self.graph = ObservedGraph(a['edge_index'],a['edge_type'],int(a['n']),int(a['R']))
        self.graph.assert_inverse_pairs()
        verify(arrsha(a['edge_index'],a['edge_type'])==self.entry['observed_sha256'], f'{key} observation hash')
        verify(arrsha(a['hyper_edge'],a['hyper_type'])==self.entry['hypergraph_sha256'], f'{key} motif hash')
        self.queries = self.isolated_queries()

    def isolated_queries(self):
        from s7util import arrsha
        with np.load(self.bundle,allow_pickle=False) as z:
            triples, truth = z['queries'], z['alltrue']
        g = self.graph
        if not self.source:
            observed = load(snapshot(self.key,'observed.npz'))
            verify(np.array_equal(observed['edge_index'],g.edge_index) and np.array_equal(observed['edge_type'],g.edge_type)
                   and np.array_equal(load(snapshot(self.key,'offline_targets_ONLY.npz'))['queries'],triples), 'published snapshot mismatch')
        verify(arrsha(triples[:,[0,2,1]])==self.entry['query_sha256'],'query identity')
        verify(arrsha(truth[:,[0,2]].T,truth[:,1])==self.entry['filtered_sha256'],'filter identity')
        observed = set(zip(g.edge_index[0],g.edge_type,g.edge_index[1]))
        held_out = set(map(tuple,truth))-observed
        verify(set(map(tuple,triples)).issubset(held_out),'hidden query in observation')
        g.assert_hidden_absent(held_out)
        return np.stack((triples[:,[0,1]],triples[:,[2,1]]+[0,g.R//2]),axis=1)

    def open_previous(self):
        previous = POC/'results/recovery/recovery.json'
        if previous.exists():
            proof = next((r for r in read(previous)['records'] if r['dataset']==self.key and r['model']==PRIMARY),None)
            if proof:
                file = Path(proof['path'])
                verify(proof['cache_sha256']==self.meta['feature_sha256'] and
                       proof['observation_sha256']==self.entry['observed_sha256'] and
                       sha(file)==self.meta['feature_sha256'],'previous candidate cache identity')
                return file
        return None

    def m5_ranks(self):
        from s7util import arrsha
        base = M5/self.key
        if not (base/'complete.json').is_file():
            return None
        path = base/'rankings.npz'
        verify(sha(path)==read(base/'complete.json')['ranking_sha256'],'M5 hash')
        with np.load(path,allow_pickle=False) as z:
            m0 = z['B0/M0']
            verify(m0.size == len(self.queries)*2, 'M5 row count')
            expected = read(self.history/'B0.json')['npz_sha256']
            baseline_file = self.history/'B0.npz'
            verify(sha(baseline_file)==expected, 'B0 reference checksum')
            baseline = load(baseline_file)
            h,t,r = baseline['triples'].T
            expected_queries = np.stack((np.c_[h,r,t], np.c_[t,r+self.graph.R//2,h]),axis=1).reshape(-1,3)
            verify(arrsha(baseline['triples'])==self.entry['query_sha256'] and
                   np.array_equal(m0.reshape(-1,2),baseline['ranks']) and
                   np.array_equal(z['queries'],expected_queries), 'M5 row order vs B0 archive')
            return z['B2/M5'].reshape(3,-1,2)


def open_model(asset):
    import torch
    from calibration import install
    from common import make_data
    from engine import model, state_hash
    empty = np.empty((0,3),dtype=np.int64)
    d,_ = make_data(dict(asset.arr,alltrue=empty),q=empty)
    del d.target_edge_index, d.target_edge_type
    net = model('B2',1024)
    verify(state_hash(net)==asset.meta['parameter_sha256_before'],'B2 parameter hash')
    install(net,d,asset.meta['config'])
    torch.cuda.reset_peak_memory_stats()
    return net,d


def predict(asset, net, data, indices):
    import torch
    from engine import RelationCache
    RelationCache(net)
    n = asset.graph.n
    batch_size = asset.meta['batch']
    all_batches = sorted({int(i)//batch_size for i in indices})
    requested = {int(idx):i for i,idx in enumerate(indices)}
    result = np.empty((len(indices),2,n),np.float32)
    entities = torch.arange(n,device='cuda')

    def forward(q):
        src,rel = [x[:,None].expand(-1,n) for x in q.T]
        return net(data,torch.stack((src,entities.expand_as(src),rel),-1))

    with torch.inference_mode():
        for b in all_batches:
            first = b*batch_size
            for direction in (0,1):
                queries = torch.as_tensor(asset.queries[first:first+batch_size,direction],device='cuda')
                micro = len(queries)
                while True:
                    try:
                        scores = torch.cat([forward(q) for q in queries.split(micro)])
                        break
                    except torch.cuda.OutOfMemoryError:
                        if micro == 1:
                            raise
                    torch.cuda.empty_cache()
                    micro = max(1,micro//2)
                verify(bool(torch.isfinite(scores).all()),'nonfinite backbone score')
                ordered = scores.sort(dim=1).values.contiguous()
                rr = 1./(60+(n-torch.searchsorted(ordered,scores.contiguous(),right=False)).float())
                rr = rr.cpu().numpy()
                for at, row in enumerate(rr):
                    if first+at in requested:
                        result[requested[first+at],direction] = row
    return result


def gates(asset, net, data, indices, folder):
    """Collapse verified B2 per-layer query gates by q-active relation motifs."""
    import torch
    from scipy.special import expit
    from scipy.sparse import coo_matrix

    a = asset.arr
    he = a['hyper_edge']
    demand = sorted(set(map(int,asset.queries[indices,:,1].ravel())))
    slots, cols = np.nonzero(he>=0)
    inc = coo_matrix((np.ones(len(cols)),(he[slots,cols],cols)),shape=(asset.graph.R,he.shape[1])).tocsr()
    inc.data[:] = 1
    active_layers = []
    handles = [l.topology_gate.register_forward_hook(lambda m,i,o:active_layers.append(o.detach()[0]))
               for l in net.relation_model.layers]
    try:
        with torch.inference_mode():
            for q in demand:
                path = folder/f'{q}.npz'
                if path.exists():
                    continue
                active = inc.getrow(q).indices
                active_layers.clear()
                net.relation_model(data.relation_hypergraph,torch.tensor([q],device='cuda'))
                verify(len(active_layers)==6,'expected six B2 layer gates')
                selected = torch.as_tensor(active.copy(),device='cuda')
                raw_gate = torch.stack([v[selected] for v in active_layers]).cpu().numpy()
                gate_value = expit(np.log(raw_gate))
                projection = inc[:,active]
                support = projection @ np.ones(len(active))
                # Low-count shrinkage gives missing relations zero prior, without claiming probability.
                layers = (projection @ gate_value.T).T/(support[None,:]+1)
                save_npz(path,layers=layers,support=support,
                         exclusivity=(projection @ expit(a['z'][active,3]))/(support+1))
    finally:
        for h in handles:
            h.remove()


def save_npz(path, **data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.partial')
    with temp.open('wb') as stream:
        np.savez_compressed(stream,**data)
    temp.replace(path)


def prepare(asset, indices):
    from src.evaluation import OfflineEvaluation
    import torch
    from engine import state_hash
    previous = asset.open_previous()
    full = None if previous is None else np.load(previous,mmap_mode='r',allow_pickle=False)
    folder = asset.folder
    gate_dir = folder/'gates'
    score_path = folder/'scores.npz'
    local = None
    if full is None and score_path.is_file():
        proof = read(folder/'scores_hash.json')
        verify(proof['sha256']==sha(score_path) and proof['graph']==asset.entry['observed_sha256'] and
               proof['checkpoint']==asset.meta['parameter_sha256_before'],'P1 candidate cache manifest')
        local = load(score_path)
    needed = sorted(set(indices)-set(local['indices'] if local is not None else ()))
    demand = set(map(int,asset.queries[indices,:,1].ravel()))
    gpu = None
    if (full is None and needed) or any(not (gate_dir/f'{q}.npz').is_file() for q in demand):
        net, data = open_model(asset)
        # Capture real gates before attaching the historical relation cache wrapper.
        gates(asset,net,data,indices,gate_dir)
        if full is None and needed:
            vals = predict(asset,net,data,needed)
            if local is not None:
                vals = np.concatenate((local['scores'],vals))
                needed = local['indices'].tolist()+needed
            local = dict(indices=np.array(needed),scores=vals)
            save_npz(score_path,**local)
            dump(folder/'scores_hash.json',dict(sha256=sha(score_path),model=PRIMARY,checkpoint=asset.meta['parameter_sha256_before'],
                 graph=asset.entry['observed_sha256']))
        verify(state_hash(net)==asset.meta['parameter_sha256_before'],'frozen B2 changed')
        gpu = torch.cuda.max_memory_allocated()
        del net,data
        torch.cuda.empty_cache()
    g = asset.graph
    if full is not None:
        verify(full.shape==(*asset.queries.shape[:2],g.n),'historical candidate vocabulary')
        scores = {i:full[i] for i in indices}
    else:
        proof = read(folder/'scores_hash.json')
        verify(proof['sha256']==sha(score_path) and proof['graph']==asset.entry['observed_sha256'] and
               proof['checkpoint']==asset.meta['parameter_sha256_before'],'P1 candidate cache manifest')
        lookup = {int(idx):i for i,idx in enumerate(local['indices'])}
        scores = {i:local['scores'][lookup[i]] for i in indices}
    gate_vals = {q:load(gate_dir/f'{q}.npz') for q in demand}
    old_gpu = read(folder/'prepared.json').get('peak_allocated') if (folder/'prepared.json').exists() else None
    dump(folder/'prepared.json',dict(dataset=asset.key, model=PRIMARY, checkpoint=asset.meta['parameter_sha256_before'],
         observation=asset.entry['observed_sha256'], coverage=len(scores), peak_allocated=max(gpu or 0,old_gpu or 0),
         candidate_source=str(previous if previous else score_path), relation_count=len(demand), gate_layers=6))
    return scores,gate_vals


class Evaluator:
    def __init__(self,asset):
        from src.evaluation import OfflineEvaluation
        self.rank = OfflineEvaluation.filtered_rank
        with np.load(asset.bundle,allow_pickle=False) as z:
            self.answers = z['queries']
            self.true = defaultdict(set)
            half = asset.graph.R//2
            for h,r,t in z['alltrue']:
                self.true[h,r].add(int(t))
                if r<half:
                    self.true[t,r+half].add(int(h))
        self.ref = load(asset.ref)['ranks']
        b0 = asset.history/'B0.npz'
        verify(sha(b0)==read(asset.history/'B0.json')['npz_sha256'],'B0 reference hash')
        self.b0 = load(b0)['ranks']
        self.m5 = asset.m5_ranks()

    def query(self,index,direction,h,r,scores):
        target = int(self.answers[index,2 if direction==0 else 0])
        excluded = self.true[h,r]
        before = self.rank(scores,target,excluded)
        verify(before==self.ref[index,direction],f'{index}/{direction}: Stage-7 filtered rank mismatch')
        return target,excluded,before
