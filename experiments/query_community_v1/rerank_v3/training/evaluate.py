import gc,time,types
import numpy as np
import torch
from motif import tasks
def metrics(ranks):
    r=np.asarray(ranks,dtype=float)
    return dict(mrr=float((1/r).mean()),hits_at_1=float((r<=1).mean()),hits_at_3=float((r<=3).mean()),hits_at_10=float((r<=10).mean()))
@torch.no_grad()
def evaluate(model,data,filtered,batch_size=4,lambdas=(0.,)):
    model.eval();triples=torch.cat([data.target_edge_index,data.target_edge_type[None]]).T
    ranks=np.zeros((len(lambdas),len(triples),2),dtype=np.int64)
    original=model.relation_model.forward;memory={};began=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def cached_forward(self,hyper,query):
        missing=sorted(set(query.cpu().tolist())-set(memory))
        for at in range(0,len(missing),2):
            ix=missing[at:at+2];rep=original(hyper,torch.tensor(ix,device=query.device))
            for i,k in enumerate(ix):memory[k]=rep[i].clone()
        return torch.stack([memory[k] for k in query.cpu().tolist()])
    model.relation_model.forward=types.MethodType(cached_forward,model.relation_model)
    try:
        for at in range(0,len(triples),batch_size):
            batch=triples[at:at+batch_size];tail,head=tasks.all_negative(data,batch)
            t,h=model(data,tail),model(data,head)
            assert torch.isfinite(t).all() and torch.isfinite(h).all()
            tm,hm=tasks.strict_negative_mask(filtered,batch)
            for li,lam in enumerate(lambdas):
                tc=t-lam*data.hub_log_degree[None] if lam else t
                hc=h-lam*data.hub_log_degree[None] if lam else h
                ranks[li,at:at+len(batch),0]=tasks.compute_ranking(tc,batch[:,1],tm).cpu().numpy()
                ranks[li,at:at+len(batch),1]=tasks.compute_ranking(hc,batch[:,0],hm).cpu().numpy()
    finally:model.relation_model.forward=original
    torch.cuda.synchronize()
    return dict(seconds=time.perf_counter()-began,peak_allocated=torch.cuda.max_memory_allocated(),queries=len(triples),batch_size=batch_size),ranks,triples.cpu().numpy()
def safe_evaluate(model,data,truth,batch_size=16,lambdas=(0.,)):
    while True:
        try:return evaluate(model,data,truth,batch_size,lambdas)
        except torch.cuda.OutOfMemoryError:
            if batch_size==1:raise
            batch_size=max(1,batch_size//2);gc.collect();torch.cuda.empty_cache()
