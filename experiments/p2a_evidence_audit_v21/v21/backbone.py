"""Freeze the full-graph relation output and D* buffers; vary entity edges only."""
from copy import copy
from pathlib import Path
import types
import time

import numpy as np
import torch

from p1.assets import open_model, save_npz, verify
from engine import RelationCache, state_hash
from s7util import arrsha
from calibration import factors
from src.io_utils import dump, sha, identity


class FrozenBackbone:
    def __init__(self, asset, demands, directory, cfg):
        self.asset=asset
        self.cfg=cfg
        self.directory=directory
        self.net,self.data=open_model(asset)
        self.graph=asset.graph
        self.parameter_before=state_hash(self.net)
        self.relation_calls=0
        self.calls=0
        self.adjustments=[]
        verify(not self.net.training and not any(p.requires_grad for p in self.net.parameters()),'backbone must be frozen')
        config=asset.meta['config']
        verify(config==dict(kind='D-Sender',gamma=.75,alpha=.2,anchor=True),'unexpected Dstar')
        layers=self.net.entity_model.layers
        self.buffer_before=self.buffer_hash()
        a,b,z,stats=factors(self.data.edge_index,self.graph.n,config)
        verify(all(torch.equal(l.s7_a,a) and torch.equal(l.s7_b,b) for l in layers),'full Dstar installed incorrectly')
        verify(torch.equal(self.net.entity_model.s7_z,z),'query stats differ')
        self.stats=stats
        self.full_ids=np.arange(len(self.graph.edge_type),dtype=np.int64)
        save_npz(directory/'full_dstar.npz',
                 sender_degree=np.bincount(self.graph.edge_index[1],minlength=self.graph.n),
                 archived_receiver_degree=np.bincount(self.graph.edge_index[0],minlength=self.graph.n),
                 a=a.cpu().numpy(),b=b.cpu().numpy(),z=z.cpu().numpy(),
                 raw_mean=np.array(stats['raw_edge_mean']),anchor_float32=(1/b[0]).cpu().numpy())
        rel=self.net.relation_model
        original=rel.forward
        demand=sorted(set(map(int,demands)))
        def count_forward(hyper,query):
            self.relation_calls+=1
            return original(hyper,query)
        rel.forward=count_forward
        relation_cache=RelationCache(self.net)
        with torch.no_grad():
            for at in range(0,len(demand),2):
                rel(self.data.relation_hypergraph,torch.tensor(demand[at:at+2],device='cuda'))
        self.reps={q:relation_cache.memory[q].detach().clone() for q in demand}
        save_npz(directory/'full_relation_embeddings.npz',relations=np.array(demand),
                 representations=torch.stack([self.reps[q] for q in demand]).cpu().numpy())
        relation_digest=sha(directory/'full_relation_embeddings.npz')
        self.relation_precompute_calls=self.relation_calls
        def frozen_lookup(hyper,query):
            verify(hyper is self.data.relation_hypergraph,'relation hypergraph replaced')
            qs=query.detach().cpu().tolist()
            verify(set(qs)<=self.reps.keys(),'uncached full-graph relation')
            return torch.stack([self.reps[q] for q in qs])
        rel.forward=frozen_lookup
        self.static_fingerprint=identity(dict(checkpoint=self.parameter_before,
              observation=asset.entry['observed_sha256'],relations=relation_digest,buffers=self.buffer_before))
        self.precompute_peak_allocated=torch.cuda.max_memory_allocated()
        self.precompute_peak_reserved=torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()

    def buffer_hash(self):
        entity=self.net.entity_model
        return arrsha(entity.s7_z.cpu().numpy(),
              *[x.cpu().numpy() for l in entity.layers for x in (l.s7_a,l.s7_b)])

    def check(self):
        verify(self.relation_calls==self.relation_precompute_calls,'relation encoder ran on counterfactual')
        verify(self.buffer_hash()==self.buffer_before,'full graph Dstar statistics changed')
        verify(self.data.num_nodes==self.graph.n,'entity vocabulary shrunk')

    def score(self, queries, ids, allow_split=True):
        """One shared observation graph per call, with only (anchor, relation) inputs."""
        if not queries:
            return np.empty((0,self.graph.n),np.float32)
        edge_ids=np.asarray(ids,dtype=np.int64)
        verify(edge_ids.ndim==1 and len(set(edge_ids))==len(edge_ids),'invalid directed graph IDs')
        verify(not len(edge_ids) or (edge_ids.min()>=0 and edge_ids.max()<len(self.graph.edge_type)),'edge ID out of bounds')
        g=copy(self.data)
        ix=torch.from_numpy(edge_ids).to('cuda')
        g.edge_index=self.data.edge_index[:,ix]
        g.edge_type=self.data.edge_type[ix]
        verify(int(g.num_nodes)==self.graph.n and int(g.num_relations)==self.graph.R,'vocabulary changed')
        n=self.graph.n
        q=torch.tensor(queries,device='cuda',dtype=torch.long)
        entities=torch.arange(n,device='cuda')
        # Everything below enters the original MOTIF/Stage-7 forward.
        def evaluate(batch):
            u,r=[t[:,None].expand(-1,n) for t in batch.T]
            candidates=entities.expand_as(u)
            return self.net(g,torch.stack((u,candidates,r),dim=-1))
        batch=len(queries)
        while True:
            try:
                with torch.no_grad():
                    score=torch.cat([evaluate(part) for part in q.split(batch)])
                    verify(score.shape==(len(queries),n) and bool(torch.isfinite(score).all()),'full entity scores invalid')
                    value=score.detach().cpu().numpy()
                break
            except torch.cuda.OutOfMemoryError:
                if batch==1 or not allow_split:
                    raise
                self.adjustments.append(dict(before=batch,after=max(1,batch//2),reason='CUDA OOM'))
                batch=max(1,batch//2)
                torch.cuda.empty_cache()
        self.calls+=1
        self.check()
        return value

    def close(self):
        self.check()
        verify(state_hash(self.net)==self.parameter_before,'frozen weights changed')
        info=dict(parameter_sha_before=self.parameter_before,parameter_sha_after=state_hash(self.net),
          relation_embeddings_sha256=sha(self.directory/'full_relation_embeddings.npz'),
          dstar_cache_sha256=sha(self.directory/'full_dstar.npz'),buffer_hash=self.buffer_before,
          dstar_stats=self.stats,entity_space=self.graph.n,relation_precompute_calls=self.relation_precompute_calls,
          counterfactual_relation_encoder_calls=self.relation_calls-self.relation_precompute_calls,
          full_statistics_fixed=True,batch_adjustments=self.adjustments,forward_calls=self.calls,
          peak_allocated=max(self.precompute_peak_allocated,torch.cuda.max_memory_allocated()),
          peak_reserved=max(self.precompute_peak_reserved,torch.cuda.max_memory_reserved()))
        dump(self.directory/'freeze_proof.json',info)
        return info
