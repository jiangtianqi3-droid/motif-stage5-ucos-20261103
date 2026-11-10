"""Five real validation prompts, then prove exact zero-request cache replay."""
import concurrent.futures
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.util import *
from src.offline import Dataset
from src.structural_reranker import Config, rerank, trigger
from src.verbalizer import Verbalizer
from src.llm_reranker import Client

run=ROOT/'results/runs'/sys.argv[1]
cases=[];datasets={};verbalizers={}
for path in sorted(run.glob('*/*/llm/valid/*.json')):
    previous=read(path)
    if previous['parse_error']!='reasoning missing':continue
    work=path.parents[2];key=work.parent.name
    if key not in datasets:
        g=historical_package().load_observed(key)
        datasets[key]=Dataset(key,g,run/'contract_recheck'/f'{key}_split_audit.json')
        ds=datasets[key];verbalizers[key]=Verbalizer(key,ds.names,ds.relations,g.R)
    record=read(work/'evidence/R4/valid'/path.name)
    cfg=Config(**read(work/'selected_structure.json')['R4']['config'])
    result=rerank(record,cfg);pool=trigger(record,result,cfg)
    cases.append((verbalizers[key].prompt(record,result,pool),[record['candidate_evidence'][i]['entity'] for i in pool]))
    if len(cases)==5:break
assert len(cases)==5
client=Client()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    responses=list(executor.map(lambda args:client.call(*args),cases))
errors=[r['parse_error'] for r in responses]
assert not any(errors),errors
requests_before=client.network_requests
for case in cases:
    response=client.call(*case);assert response['cache_hit']
assert client.network_requests==requests_before
output=dict(passed=True,real_validation_queries=5,network_requests=requests_before,replay_network_requests=0,parse_failures=0,
            peak_rss_bytes=peak_memory(),tokens=sum(r['usage'].get('total_tokens',0) for r in responses))
dump(run/'contract_recheck/audit.json',output)
print(json.dumps(output,indent=2))
