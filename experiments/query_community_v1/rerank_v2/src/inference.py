"""Frozen all-entity scoring; candidates see only public anchors and allowed masks."""
import types
import numpy as np
from .util import STAGE5, ROOT, read, dump, sha, cleanup, memory, model_environment

def infer(dataset,g,sig,family,work,selections,run):
    model_environment()
    import torch
    from common import base_model,make_data,CKPT
    from model import install
    from .util import status
    torch.set_num_threads(4)
    frozen=read(STAGE5/'frozen.json')['formal']['T-4D']['1024']
    if family=='B0':
        model=base_model().cuda().eval(); checkpoint=CKPT
    else:
        checkpoint=STAGE5/frozen['checkpoint']; assert sha(checkpoint)==frozen['checkpoint_sha256']
        model=install(base_model(),frozen['config']).cuda().eval()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu')['model'],strict=True)
    for p in model.parameters(): p.requires_grad_(False)
    def state_hash():
        from qc_v1.graph_adapter import digest
        return digest(*[v.detach().cpu().numpy() for v in model.state_dict().values()])
    before=state_hash()
    z=dict(n=g.n,R=g.R,edge_index=g.edge_index.copy(),edge_type=g.edge_type.copy(),queries=np.empty((0,3),dtype=np.int64),
           alltrue=np.empty((0,3),dtype=np.int64),hyper_edge=sig['edge_index'],hyper_type=sig['edge_type'],raw=sig['raw'],z=sig['z'])
    data,_=make_data(z,device='cuda'); del z
    original=model.relation_model.forward; cached={}
    def forward(self,hyper,query):
        relation=int(query[0]); assert bool((query==relation).all())
        if relation not in cached:
            cached.clear()
            cached[relation]=original(hyper,query[:1]).detach()
        return cached[relation].expand(len(query),-1,-1)
    model.relation_model.forward=types.MethodType(forward,model.relation_model)
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            for split,indices in selections.items():
                jobs={}
                for i in indices:
                    for d in (0,1):
                        path=work/'scores'/split/f'{i}_{d}.npy'
                        cp=work/'candidates'/split/f'{i}_{d}.json'
                        if path.exists() and cp.exists(): continue
                        h,q=map(int,dataset.public[split][i,d]);jobs.setdefault(q,[]).append((int(i),d,h))
                done=0;total=sum(map(len,jobs.values()))
                for q,rows in sorted(jobs.items()):
                    for at in range(0,len(rows),16):
                        group=rows[at:at+16]
                        batch=torch.empty((len(group),g.n,3),dtype=torch.long,device='cuda')
                        batch[:,:,0]=torch.tensor([h for i,d,h in group],device='cuda')[:,None]
                        batch[:,:,1]=torch.arange(g.n,device='cuda')[None]
                        batch[:,:,2]=q
                        scores=model(data,batch).cpu().numpy()
                        assert np.isfinite(scores).all()
                        for j,(i,d,h) in enumerate(group):
                            s=scores[j]; eligible=np.ones(g.n,dtype=bool)
                            eligible[list(dataset.candidate_masks[split].get((h,q),()))]=False
                            ids=np.flatnonzero(eligible); ids=ids[np.lexsort((ids,-s[ids]))][:30]
                            path=work/'scores'/split/f'{i}_{d}.npy';path.parent.mkdir(parents=True,exist_ok=True)
                            tmp=path.with_name(path.name+'.tmp')
                            with tmp.open('wb') as f: np.save(f,s,allow_pickle=False)
                            tmp.replace(path)
                            dump(work/'candidates'/split/f'{i}_{d}.json',dict(query_info=dict(head=h,relation=q,fact_index=i,direction=d),
                                 candidates=[dict(entity=int(t),base_logit=float(s[t]),base_rank=k+1,base_candidate_rank=k+1,score_kind='frozen_logit') for k,t in enumerate(ids)],
                                 public_filter_only=True,model=family))
                        done+=len(group)
                        if done%128<16 or done==total: status(run,state='running',stage='frozen_inference',dataset=dataset.key,family=family,split=split,done=done,total=total)
                        del batch,scores
                        memory()
                    cached.clear();cleanup()
        after=state_hash(); assert before==after
        dump(work/'model_audit.json',dict(family=family,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),parameters_before=before,parameters_after=after,
                                        peak_cuda_bytes=torch.cuda.max_memory_allocated(),training=False,relation_cache_max=1,batch=16,all_entity_logits=True))
    finally:
        model.relation_model.forward=original
        del model,data,cached,original
        cleanup()
