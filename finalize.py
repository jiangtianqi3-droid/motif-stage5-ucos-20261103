import gc,json,time,shutil
import numpy as np
import torch
from common import ROOT,OLD,S2,SOURCES,CKPT,base_model,load_cached,downstream,dump,sha,array_sha,log
from cosine import attach
from model import install
from evaluate import evaluate,safe_evaluate,metrics
def cleanup():gc.collect();torch.cuda.empty_cache()
def load_model(rec):
    m=install(base_model(rec['seed']).cuda(),rec['config']);m.load_state_dict(torch.load(ROOT/rec['checkpoint'],map_location='cpu')['model'],strict=True);return m.eval()
def calibration(formal):
    out=ROOT/'results/hub_calibration.json'
    model_hashes={seed:rec['checkpoint_sha256'] for seed,rec in formal['T-4D'].items()}
    if out.exists():
        old=json.loads(out.read_text());assert old['model_checkpoint_hashes']==model_hashes and old['protocol_sha256']==sha(ROOT/'protocol.json');return old
    grid=json.loads((ROOT/'protocol.json').read_text())['hub_lambda_grid'];rows=[]
    for seed,rec in formal['T-4D'].items():
        m=load_model(rec)
        for source in SOURCES:
            d,t=load_cached(source,'validation');info,ranks,triples=evaluate(m,d,t,4,grid)
            np.savez_compressed(ROOT/'curves'/f'HubCal_s{seed}_{source}.npz',ranks=ranks,triples=triples,lambdas=grid)
            for li,lam in enumerate(grid):rows.append(dict(seed=int(seed),source=source,lambda_hub=lam,**metrics(ranks[li])))
            del d,t;cleanup()
        del m;cleanup()
    means=[float(np.mean([r['mrr'] for r in rows if r['lambda_hub']==lam])) for lam in grid]
    ix=int(np.argmax(means));result=dict(lambda_hub=grid[ix],selection='single shared scalar maximizing three-source three-seed validation macroMRR; ties choose smallest lambda',grid=grid,macro_mrr=means,rows=rows,model_checkpoint_hashes=model_hashes,protocol_sha256=sha(ROOT/'protocol.json'))
    dump(out,result);return result
def prepare_targets(specs):
    rows=[]
    for group,name,version in specs:
        key=name if version is None else name+'_'+version;d,t=downstream(name,version)
        triple=torch.cat([d.target_edge_index,d.target_edge_type[None]]).T.cpu().numpy()
        he=d.relation_hypergraph.edge_index.cpu().numpy();ht=d.relation_hypergraph.edge_type.cpu().numpy()
        old=OLD/'analysis'/f'query_masks_{key}.npz';qm=np.load(old)
        assert qm['query_order_sha256'].item()==array_sha(triple)
        masks={k:qm[k] for k in qm.files};maskpath=ROOT/'analysis'/f'query_masks_{key}.npz';np.savez_compressed(maskpath,**masks)
        z=d.relation_hypergraph.topology_z;alpha=.2*torch.sigmoid(-2*torch.maximum(z[:,0],z[:,3])-.5)
        attach(d.relation_hypergraph,key,False)
        masked_alpha=d.relation_hypergraph.sem_alpha
        rows.append(dict(group=group,key=key,name=name,version=version,queries=len(triple),query_sha256=array_sha(triple),hypergraph_sha256=array_sha(he,ht),observed_sha256=array_sha(d.edge_index.cpu().numpy(),d.edge_type.cpu().numpy()),filtered_sha256=array_sha(t.edge_index.cpu().numpy(),t.edge_type.cpu().numpy()),alpha_adaptive=d.relation_hypergraph.alpha_adaptive,s_pmi_cv=d.relation_hypergraph.topology_cv,semantic_alpha_unmasked_mean=float(alpha.mean()),semantic_alpha_unmasked_max=float(alpha.max()),semantic_alpha_positive_support_mean=float(alpha[torch.maximum(z[:,0],z[:,3])>0].mean()),query_masks_sha256=sha(maskpath),query_group_counts={k:int(qm[k].sum()) for k in qm.files if qm[k].dtype==bool}))
        rows[-1].update(semantic_alpha_masked_mean=float(masked_alpha.mean()),semantic_alpha_masked_max=float(masked_alpha.max()),opaque_hyperedges=int((~d.relation_hypergraph.cos_edge_trans).sum()))
        log('target_inputs_fixed',key=key);del d,t;cleanup()
    dump(ROOT/'analysis/target_inputs.json',rows);return rows
def freeze(formal,cal,targets):
    out=ROOT/'frozen.json'
    code={str(p.relative_to(ROOT)):sha(p) for p in list(ROOT.glob('*.py'))+[ROOT/'protocol.json',ROOT/'run.sh']+list((ROOT/'runtime').rglob('*.py'))}
    semantic={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'semantic').glob('*') if p.is_file()}
    assert semantic==json.loads((ROOT/'analysis/semantic_pretraining_sha256.json').read_text())
    checkpoints={r['checkpoint']:r['checkpoint_sha256'] for fam in formal.values() for r in fam.values()}
    inputs={str(p):sha(p) for p in list((S2/'data').glob('*__*.npz'))+list((S2/'plans').glob('*.npz'))+list((OLD/'data').glob('*_validation_queries.npy'))}
    inputs.update({str(p):sha(p) for p in (OLD/'cache/downstream').glob('*.npz')})
    inputs.update({str(p):sha(p) for p in (ROOT/'data/downstream').rglob('*') if p.is_file() and 'raw' in p.parts})
    for baseline in ['B1','B2']:
        inputs.update({str(p):sha(p) for p in (OLD/'evaluation').glob('*__'+baseline+'_s*.*')})
    inputs[str(OLD/'data/external/ConceptNet100k/observed.npz')]=sha(OLD/'data/external/ConceptNet100k/observed.npz')
    inputs[str(ROOT/'analysis/environment.json')]=sha(ROOT/'analysis/environment.json')
    inputs[str(ROOT/'analysis/execution_parallelism.json')]=sha(ROOT/'analysis/execution_parallelism.json')
    inputs[str(ROOT/'analysis/source_grid_choice.json')]=sha(ROOT/'analysis/source_grid_choice.json')
    for testname in ['operator_tests','statistical_tests','memory_preflight','name_mapping_audit']:
        inputs[str(ROOT/'results'/(testname+'.json'))]=sha(ROOT/'results'/(testname+'.json'))
    payload=dict(created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),formal=formal,calibration=cal,targets=targets,code_hashes=code,semantic_hashes=semantic,checkpoint_hashes=checkpoints,input_hashes=inputs,official_checkpoint_sha256=sha(CKPT),source_choice_sha256=sha(ROOT/'analysis/source_choices.json'),calibration_sha256=sha(ROOT/'results/hub_calibration.json'),group_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'analysis').glob('query_masks_*.npz')})
    if out.exists():
        old=json.loads(out.read_text())
        for k in payload:
            if k!='created_at_utc':assert old[k]==payload[k],k
        return old
    dump(out,payload);return payload
def target_eval(frozen):
    fh=sha(ROOT/'frozen.json');formal=frozen['formal'];lam=frozen['calibration']['lambda_hub']
    for target in frozen['targets']:
        key=target['key'];d,t=downstream(target['name'],target['version']);trip=torch.cat([d.target_edge_index,d.target_edge_type[None]]).T.cpu().numpy()
        assert array_sha(trip)==target['query_sha256']
        assert array_sha(d.edge_index.cpu().numpy(),d.edge_type.cpu().numpy())==target['observed_sha256']
        assert array_sha(t.edge_index.cpu().numpy(),t.edge_type.cpu().numpy())==target['filtered_sha256']
        assert array_sha(d.relation_hypergraph.edge_index.cpu().numpy(),d.relation_hypergraph.edge_type.cpu().numpy())==target['hypergraph_sha256']
        jobs=[('B0',1024,None)]+[(family,int(seed),rec) for family,rows in formal.items() for seed,rec in rows.items()]
        for family,seed,rec in jobs:
            tag='B0' if rec is None else family+'_s'+str(seed);dest=ROOT/'evaluation'/f'{key}__{tag}.json'
            if dest.exists():assert json.loads(dest.read_text())['frozen_sha256']==fh;continue
            m=base_model().cuda().eval() if rec is None else load_model(rec)
            if rec and rec['config']['semantic']:attach(d.relation_hypergraph,key,rec['config']['shuffle'])
            grid=(0.,lam) if family=='T-4D' else (0.,)
            log('P0_begin',key=key,tag=tag)
            info,ranks,triples=safe_evaluate(m,d,t,16,grid);assert np.array_equal(triples,trip)
            for li,ff in enumerate([family,'T-4D+HubCal'] if family=='T-4D' else [family]):
                tt='B0' if rec is None else ff+'_s'+str(seed);path=ROOT/'evaluation'/f'{key}__{tt}.json'
                np.savez_compressed(path.with_suffix('.npz'),ranks=ranks[li],triples=triples)
                row=dict(key=key,group=target['group'],family=ff,tag=tt,seed=seed,selected_step=0 if rec is None else rec['best_step'],checkpoint_sha256=frozen['official_checkpoint_sha256'] if rec is None else rec['checkpoint_sha256'],lambda_hub=lam if ff=='T-4D+HubCal' else 0.,frozen_sha256=fh,**info,**metrics(ranks[li]))
                dump(path,row);log('P0_complete',key=key,tag=tt,mrr=row['mrr'])
            del m;cleanup()
        # Historical controls are read only AFTER all source choices and freeze.
        for family in ['B1','B2']:
            for seed in [1024,2026,3407]:
                old=OLD/'evaluation'/f'{key}__{family}_s{seed}.npz';a=np.load(old);assert np.array_equal(a['triples'],trip)
                shutil.copy2(old,ROOT/'evaluation'/old.name)
                row=json.loads(old.with_suffix('.json').read_text());row.update(family=family,frozen_sha256=fh,reused_from=str(old),reused_npz_sha256=sha(old))
                dump(ROOT/'evaluation'/old.with_suffix('.json').name,row)
        del d,t;cleanup()
    dump(ROOT/'results/target_status.json',dict(status='complete',graphs=len(frozen['targets']),frozen_sha256=fh))
def main():
    formal=json.loads((ROOT/'analysis/source_choices.json').read_text())['formal']
    assert formal['T-4D']['1024']['config']==json.loads((ROOT/'analysis/source_grid_choice.json').read_text())['main_config']
    cal=calibration(formal)
    targets=prepare_targets(json.loads((ROOT/'protocol.json').read_text())['targets'])
    frozen=freeze(formal,cal,targets);target_eval(frozen)
    from summarize import main as summarize
    summarize()
    from verify import main as verify
    verify()
if __name__=='__main__':main()
