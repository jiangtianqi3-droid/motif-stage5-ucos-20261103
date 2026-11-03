import json
from pathlib import Path
import numpy as np
import torch
from common import ROOT,OLD,S2,CKPT,SOURCES,sha,dump,array_sha
from train import code_hashes
def runtime_path(path):
    text=str(path).replace('\\','/')
    if len(text)>2 and text[1]==':':text='/mnt/'+text[0].lower()+text[2:]
    return Path(text)
def main():
    frozen=json.loads((ROOT/'frozen.json').read_text());checks={}
    assert json.loads((ROOT/'results/operator_tests.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'results/statistical_tests.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'results/name_mapping_audit.json').read_text())['status']=='passed'
    for name in ['code_hashes','semantic_hashes','checkpoint_hashes','group_hashes']:
        for path,h in frozen[name].items():assert sha(ROOT/path)==h,(name,path)
        checks[name]=len(frozen[name])
    for path,h in frozen['input_hashes'].items():assert sha(Path(path))==h,path
    checks['input_hashes']=len(frozen['input_hashes']);assert sha(CKPT)==frozen['official_checkpoint_sha256']
    protected=json.loads((ROOT/'analysis/readonly_inputs_sha256.json').read_text())
    for path,h in protected.items():assert sha(runtime_path(path))==h,path
    checks['historical_protected_references_unchanged']=len(protected)
    assert frozen['semantic_hashes']==json.loads((ROOT/'analysis/semantic_pretraining_sha256.json').read_text())
    original=torch.load(CKPT,map_location='cpu')['model'];parameter_audit=[]
    for family,records in frozen['formal'].items():
        for seed,rec in records.items():
            saved=torch.load(ROOT/rec['checkpoint'],map_location='cpu')['model']
            assert rec['code_hashes']==code_hashes()
            entity=[k for k in original if k.startswith('entity_model.')];assert all(torch.equal(saved[k],original[k]) for k in entity)
            rel=[k for k in original if k.startswith('relation_model.')]
            relsame=all(torch.equal(saved[k],original[k]) for k in rel)
            if rec['config']['freedom']=='gate_only':assert relsame
            assert rec['semantic_parameters']==(12 if rec['config']['semantic'] else 0)
            # Both selected and final unselected checkpoints must preserve frozen parameters.
            for step in [0,100,250,500]:
                pp=ROOT/'checkpoints'/f"{rec['tag']}_step{step}.pth";ss=torch.load(pp,map_location='cpu')['model']
                assert all(torch.equal(ss[k],original[k]) for k in entity)
                if rec['config']['freedom']=='gate_only':assert all(torch.equal(ss[k],original[k]) for k in rel)
            delta2=sum(float((saved[k].double()-original[k].double()).square().sum()) for k in rel)
            base2=sum(float(original[k].double().square().sum()) for k in rel)
            parameter_audit.append(dict(family=family,seed=int(seed),entity_bitwise_unchanged=True,relation_bitwise_unchanged=relsame,relation_parameter_drift_l2=delta2**.5,relation_parameter_relative_drift=(delta2/max(base2,1e-30))**.5,semantic_theta={k:saved[k].tolist() for k in saved if 'cos_gate.theta' in k},selected_step=rec['best_step']))
    checks['frozen_parameter_audit']=parameter_audit
    streams={}
    for path in (ROOT/'results').glob('training_*_s*.json'):
        r=json.loads(path.read_text());streams.setdefault(r['seed'],set()).add(r['data_stream_sha256'])
    assert all(len(v)==1 for v in streams.values());checks['matched_data_streams']=True
    all_runs=list((ROOT/'results').glob('training_*_s*.json'));assert len(all_runs)==21
    checks['training_trajectories']=len(all_runs);checks['total_optimizer_steps']=sum(json.loads(p.read_text())['steps'] for p in all_runs)
    # Audit the source-screen candidates too, rather than only promoted models.
    for path in all_runs:
        rec=json.loads(path.read_text())
        for step in [0,100,250,500]:
            pp=ROOT/'checkpoints'/f"{rec['tag']}_step{step}.pth";ss=torch.load(pp,map_location='cpu')['model']
            assert all(torch.equal(ss[k],original[k]) for k in original if k.startswith('entity_model.'))
            if rec['config']['freedom']=='gate_only':assert all(torch.equal(ss[k],original[k]) for k in original if k.startswith('relation_model.'))
    # Audit source query leakage against each masked observed graph.
    leakage=[]
    for source in SOURCES:
        for label in ['validation']+['fold'+str(i) for i in range(5)]:
            with np.load(S2/'data'/f'{source}__{label}.npz') as z:
                n,R=int(z['n']),int(z['R']);q=np.load(OLD/'data'/f'{source}_validation_queries.npy')[:512] if label=='validation' else z['queries'];e=z['edge_index'];rel=z['edge_type']
                encoded=(e[0].astype(np.int64)*n+e[1])*R+rel
                direct=(q[:,0].astype(np.int64)*n+q[:,2])*R+q[:,1]
                inverse=(q[:,2].astype(np.int64)*n+q[:,0])*R+q[:,1]+R//2
                assert not np.isin(direct,encoded).any() and not np.isin(inverse,encoded).any()
                leakage.append(dict(source=source,label=label,queries=len(q),direct_leaks=0,inverse_leaks=0))
    checks['source_leakage_audit']=leakage
    expected=0;query_checks=[]
    for target in frozen['targets']:
        key=target['key'];seen={}
        for path in (ROOT/'evaluation').glob(key+'__*.npz'):
            a=np.load(path);rr,tt=a['ranks'],a['triples'];assert array_sha(tt)==target['query_sha256'];assert rr.shape==(len(tt),2) and (rr>0).all()
            j=json.loads(path.with_suffix('.json').read_text());assert abs(float((1/rr).mean())-j['mrr'])<1e-12
            seen[path.stem.split('__')[1]]=True;expected+=1
        assert len(seen)==25,(key,len(seen))
        if frozen['calibration']['lambda_hub']==0:
            for seed in [1024,2026,3407]:
                a=np.load(ROOT/'evaluation'/f'{key}__T-4D_s{seed}.npz')['ranks']
                b=np.load(ROOT/'evaluation'/f'{key}__T-4D+HubCal_s{seed}.npz')['ranks']
                assert np.array_equal(a,b)
        query_checks.append(dict(key=key,models=len(seen),queries=target['queries'],query_order_matches=True))
    checks['target_rank_audit']=query_checks;checks['target_evaluations_including_reused']=expected
    assert expected==275
    files=['report.md','results/P0_metrics.csv','results/paired_bootstrap.csv','results/query_stratified_metrics.csv','results/query_stratified_bootstrap.csv','results/summary.json']
    assert all((ROOT/p).is_file() and (ROOT/p).stat().st_size>0 for p in files)
    checks['status']='passed';dump(ROOT/'results/verification.json',checks)
    dump(ROOT/'results/completion.json',dict(status='complete',report='report.md',verification='results/verification.json',frozen_sha256=sha(ROOT/'frozen.json')))
    print('FINAL VERIFICATION PASSED',expected,flush=True)
if __name__=='__main__':main()
