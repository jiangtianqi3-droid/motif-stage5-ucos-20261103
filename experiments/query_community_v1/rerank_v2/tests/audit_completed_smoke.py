"""Independent completion audit and historical B0 ranking regression."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.util import ROOT, HISTORY, DATASETS, read, dump, sha, SOURCE_ROOT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args()
    run=ROOT/'results/runs'/args.run
    assert read(run/'completion.json')['state']=='smoke_complete'
    assert read(run/'api_smoke.json')['replay_network_requests']==0
    assert read(run/'historical_verification.json')['passed']
    output=dict(passed=True,graphs={},historical_B0_compared=0,historical_B0_rank_differences=[],llm_test_responses=0,llm_schema_fallbacks=0,max_checkpoint_rss_bytes=0)
    for key in DATASETS:
        for family in ['B0','T-4D']:
            work=run/key/family;summary=read(work/'summary.json');audit=read(work/'model_audit.json')
            assert audit['parameters_before']==audit['parameters_after']
            assert summary['queries']==5
            output['max_checkpoint_rss_bytes']=max(output['max_checkpoint_rss_bytes'],read(work/'completion.json')['rss_bytes'],read(work/'retrieval_completion.json')['rss_bytes'])
            output['graphs'][key+'/'+family]=dict(queries=5,model_identity=True,hidden_edge_audit=read(work/'split_leakage_audit.json')['direct_inverse_leakage'])
            assert output['graphs'][key+'/'+family]['hidden_edge_audit']==0
            if not summary['llm'].get('skipped'):
                output['llm_test_responses']+=summary['llm']['calls'];output['llm_schema_fallbacks']+=summary['llm']['parse_failures']
            else: assert key=='FBIngram_25'
            if family=='B0':
                audit=read(work/'split_leakage_audit.json')
                offset=audit['valid_facts'] if key!='FBIngram_25' else 0
                with np.load(HISTORY/'motif-stage7-asymmetric-20261106/evaluation'/key/'B0.npz') as z: ranks=z['ranks']
                with (work/'per_query_ranks.jsonl').open(encoding='utf-8') as f:
                    for line in f:
                        row=json.loads(line);old=int(ranks[offset+row['fact_index'],row['direction']]);output['historical_B0_compared']+=1
                        if old!=row['base_filtered_rank']:output['historical_B0_rank_differences'].append(dict(dataset=key,old=old,new=row['base_filtered_rank']))
    assert not output['historical_B0_rank_differences'],output['historical_B0_rank_differences']
    assert output['max_checkpoint_rss_bytes']<1_800_000_000
    assert output['llm_test_responses']>0
    dump(run/'smoke_audit.json',output)
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
