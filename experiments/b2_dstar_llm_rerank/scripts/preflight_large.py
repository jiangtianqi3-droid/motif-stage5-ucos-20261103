"""Check full historical batch on the largest graph without modifying math."""
import numpy as np
import torch
from common_io import ROOT,read,dump
from regime_inspector import records,observation
from protocol_data import final_labels
from backbone import bidirectional
from experiment import score_graph,score_rows
from offline_eval import baseline_rank,truth_index,historical
torch.set_num_threads(4)
rec=next(r for r in records() if r['key']=='ConceptNet100k_external')
z=observation(rec);facts,alltrue=final_labels(rec);q=bidirectional(facts,z['R']//2)
base,_=historical(rec['key'],q)
directory=ROOT/'results/large_preflight'
def progress(stage,**kw):print(stage,kw,flush=True)
info=score_graph(z,q[:32,:2],1024,directory,progress,verify=True)
truth=truth_index(alltrue,z['R']//2)
for i,score in score_rows(directory):
    h,r,t=map(int,q[i]);assert baseline_rank(score,h,r,t,truth)==base[0,i]
dump(ROOT/'results/large_preflight.json',dict(passed=True,entities=z['n'],batch_facts=16,
    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),backbone=info))
