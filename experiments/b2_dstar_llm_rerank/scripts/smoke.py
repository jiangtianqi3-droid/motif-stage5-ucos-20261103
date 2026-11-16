import time
import numpy as np
import torch
from common_io import ROOT, SEEDS, MODES, read, dump, sha, identity
from regime_inspector import records, observation, vocabulary
from protocol_data import final_labels
from backbone import bidirectional
from experiment import score_graph, public_inference, lexicon, score_rows
from evidence import EvidenceGraph
from offline_eval import truth_index, baseline_rank, historical

torch.set_num_threads(4)
began=time.monotonic(); config=read(ROOT/'config/experiment.json')
rec=next(x for x in records() if x['key']=='NELLInductive_v1')
z=observation(rec); facts,alltrue=final_labels(rec)
q=bidirectional(facts,z['R']//2); base,m5=historical(rec['key'],q)
# Score a complete historical batch; dispatch exactly eight unique public queries.
pairs=q[:32,:2].copy()
smokeid=identity({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').rglob('*.py')})[:16]
directory=ROOT/'results/smoke'/smokeid
def progress(stage,**kw):
    dump(ROOT/'status.json',dict(state='smoke',stage=stage,**kw))
    print(stage,kw,flush=True)
infos=[]
for j,seed in enumerate(SEEDS):
    info=score_graph(z,pairs,seed,directory/'scores'/str(seed),progress,verify=True)
    truth=truth_index(alltrue,z['R']//2)
    for i,score in score_rows(directory/'scores'/str(seed)):
        h,r,t=map(int,q[i]); assert baseline_rank(score,h,r,t,truth)==base[j,i]
    infos.append(info)
graph=EvidenceGraph(z,config);names,rels,_=vocabulary(rec,z);lex=lexicon(names,rels,graph)
stats=public_inference(graph,lex,rec['key'],'smoke',SEEDS[0],pairs,
    directory/'scores'/str(SEEDS[0]),directory/'queries',config,progress,limit=8)
statuses={}; api_accepted=0; api_attempted=0
for path in (directory/'queries').glob('*.json'):
    row=read(path)
    for item in row.get('modes',{}).values():
        statuses[item['status']]=statuses.get(item['status'],0)+1
        api_attempted+=item.get('attempted',0)
        api_accepted+=int(item['status'] in ('reranked','abstain'))
assert api_attempted>0 and api_accepted>0, 'No successfully validated real LLM response; cannot launch formal run'
repeat=public_inference(graph,lex,rec['key'],'smoke',SEEDS[0],pairs,
    directory/'scores'/str(SEEDS[0]),directory/'queries',config,progress,limit=8)
assert repeat['api_calls']==0
result=dict(passed=True,directory=str(directory),query_count=8,scored_queries_per_seed=32,backbones=infos,statuses=statuses,
    llm=stats,cache_replay_api_calls=0,seconds=time.monotonic()-began,
    raw_score_comparison='bitwise vs independent historical Stage-7 all_negative computation; historical raw arrays unavailable',
    historical_filtered_rank_alignment=True)
dump(ROOT/'results/smoke_verification.json',result)
manifest=read(ROOT/'freeze_manifest.json');manifest.update(model_actually_loaded=True,
    actual_loaded_proof=str(ROOT/'results/smoke_verification.json'),actual_family='B2+D*')
dump(ROOT/'freeze_manifest.json',manifest)
print(result,flush=True)
