import torch
from motif import tasks
from common import ROOT,load_cached,base_model,dump
from cosine import attach
from model import install
from train import config
d,t=load_cached('fb15k237_train','fold0');attach(d.relation_hypergraph,'fb15k237_train')
m=install(base_model().cuda(),config(.2,'gate_only',semantic=True));m.train();m.entity_model.eval()
triples=torch.cat([d.target_edge_index,d.target_edge_type[None]]).T[:2]
batch=tasks.negative_sampling(t,triples,64,strict=True)
torch.cuda.reset_peak_memory_stats();scores=m(d,batch);loss=scores.square().mean();loss.backward()
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
dump(ROOT/'results/memory_preflight.json',dict(status='passed',peak_allocated=torch.cuda.max_memory_allocated(),semantic_gradients={n:float(p.grad.norm()) for n,p in m.named_parameters() if 'cos_gate' in n and p.grad is not None}))
print('SEMANTIC MEMORY PREFLIGHT PASSED',torch.cuda.max_memory_allocated(),flush=True)
