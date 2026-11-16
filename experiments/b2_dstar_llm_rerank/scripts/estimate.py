"""Planning uses public query counts, never answer-dependent success rates."""
import numpy as np
from common_io import ROOT,SEEDS,MODES,read,dump
from regime_inspector import records,raw_observation_path
from protocol_data import final_labels,rows
smoke=read(ROOT/'results/smoke_verification.json')
counts=[]
for rec in records():
    facts,_=final_labels(rec)
    test=len({(int(h),int(r),'f') for h,r,t in facts}|{(int(t),int(r),'i') for h,r,t in facts})
    raw=raw_observation_path(rec['key']).parent
    filename='valid.txt' if 'Inductive' in rec['key'] or rec['group']=='external' else 'inf_valid.txt'
    validrows=rows(raw/filename)
    valid=len({(h,r,'f') for h,r,t in validrows}|{(t,r,'i') for h,r,t in validrows})
    counts.append(dict(dataset=rec['key'],test_unique_queries=test,valid_unique_queries=valid))
maxcalls=sum(x['test_unique_queries']+x['valid_unique_queries'] for x in counts)*len(SEEDS)*len(MODES)
stats=smoke['llm']; observed_seconds=maxcalls*stats['seconds']/stats['query_modes']
result=dict(per_graph=counts,max_api_calls_before_gating=maxcalls,
    observed_smoke_rate_query_modes_per_second=stats['query_modes']/stats['seconds'],
    naive_seconds_including_observed_timeouts=observed_seconds,
    no_timeout_lower_planning_seconds=maxcalls*.5,
    input_tokens_estimate=maxcalls*stats['prompt_tokens']/stats['query_modes'],
    output_tokens_estimate=maxcalls*stats['completion_tokens']/stats['query_modes'],
    qualifications=['Tiny NELL smoke is not representative of larger graphs.',
        'Includes validation, all test queries, three seeds and four LLM ablations.',
        'No-evidence and token gates reduce calls; no automatic paid retries.',
        'Retrieval/scoring time and variable provider latency are additional uncertainty.'])
result['peak_cache_miss_usd_upper_estimate']=(result['input_tokens_estimate']*.3+result['output_tokens_estimate']*1.2)/1e6
result['pricing_source']='https://api-docs.deepseek.com/quick_start/pricing/'
dump(ROOT/'results/resource_estimate.json',result)
print({k:v for k,v in result.items() if k!='per_graph'})
