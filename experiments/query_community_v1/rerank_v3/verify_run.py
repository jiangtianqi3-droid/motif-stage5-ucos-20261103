"""Independent final checks. Completion is written only if every check passes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'training'))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(value,message):
    if not value:raise AssertionError(message)
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)

def verify_training():
    from strict_data import load_truth,encode_file
    from src.util import HISTORY
    training=ROOT/'training';frozen=read(training/'frozen.json');manifest=read(training/'data/strict_manifest.json')
    require(frozen['source_test_used'] is False,'Training protocol uses source test')
    require(frozen['strict_manifest_sha256']==sha(training/'data/strict_manifest.json'),'Training data fingerprint changed')
    require(frozen['protocol_sha256']==sha(training/'protocol.json'),'Training protocol changed')
    maps=read(HISTORY/'topology-stage2-20261102/analysis/source_mappings.json')
    for m in maps:
        s=m['source'];train=encode_file(HISTORY/'topology-stage1-20261101/raw/pretraining'/f'{s}.txt',m['entities'],m['relations'])
        valid=encode_file(HISTORY/'topology-stage2-20261102/data'/f'{s}_valid.txt',m['entities'],m['relations'])
        require(np.array_equal(load_truth(training,s,'train'),np.unique(train,axis=0)),f'Training truth contamination: {s}')
        require(np.array_equal(load_truth(training,s,'validation'),np.unique(np.concatenate([train,valid]),axis=0)),f'Validation truth contamination: {s}')
    require(set(frozen['formal']['T-4D'])=={'1024','2026','3407'},'Missing training seed')
    for seed,rec in frozen['formal']['T-4D'].items():
        require(sha(training/rec['checkpoint'])==rec['checkpoint_sha256'],'Checkpoint fingerprint mismatch')
        for name,h in rec['code_hashes'].items():require(sha(training/name)==h,f'Training source mismatch: {name}')
        losses=read(training/'curves'/f"{rec['tag']}_loss.json")
        require([r['step'] for r in losses]==list(range(1,501)),f'Incomplete training steps for seed {seed}')
        require(all(np.isfinite(r['loss']) for r in losses),'Nonfinite training loss')
        require(rec['best_step'] in [0,100,250,500],'Unexpected checkpoint selection')
    result=dict(passed=True,seeds=[1024,2026,3407],steps_each=500,training_truth='source train only',validation_truth='source train+valid only',source_test_files_read=False)
    save(ROOT/'results/training_verification.json',result)
    return frozen

def verify_results(frozen):
    from src.util import historical_package,verify_snapshot,source_identity
    from src.offline import Dataset,metrics
    from src.structural_reranker import Config,rerank
    from src.pipeline import evidence
    from src.llm_reranker import confidence
    from src.statistics import holm
    run=Path(read(ROOT/'results/active_run.json')['run']);protocol=read(run/'protocol.json')
    require(not protocol['smoke'],'A smoke run cannot satisfy full completion')
    require(read(run/'completion.json')['state']=='complete','Formal run incomplete')
    require(protocol['source']==source_identity(),'Evaluation source changed')
    verify_snapshot(read(run/'dependency_hashes.json'))
    adapter=historical_package();checks=[];pvalues=[];metric_count=0
    for key in protocol['datasets']:
        graph=adapter.load_observed(key);dataset=Dataset(key,graph,run/'verification'/f'{key}_splits.json')
        for family in protocol['families']:
            work=run/key/family;done=read(work/'completion.json');summary=read(work/'summary.json');audit=read(work/'model_audit.json')
            require(done['state']=='complete' and done['summary_sha256']==sha(work/'summary.json'),'Group completion mismatch')
            require(audit['parameters_before']==audit['parameters_after'] and audit['training'] is False,'Model changed during inference')
            require(done['peak_rss_bytes']<1_800_000_000,'Inference memory limit exceeded')
            if family!='B0':
                seed=family.rsplit('-s',1)[1]
                require(audit['checkpoint_sha256']==frozen['formal']['T-4D'][seed]['checkpoint_sha256'],'Wrong repaired checkpoint used')
            partition=read(work/'validation_partition.json');dataset.configure_validation_roles(partition)
            sets={role:{tuple(q) for i in partition[role] for q in dataset.public['valid'][i]} for role in ['structure','calibration']}
            require(not sets['structure']&sets['calibration'],'Shared public query across validation roles')
            require(not set(partition['structure'])&set(partition['calibration']),'Shared fact across validation roles')
            config=read(work/'selected_structure.json')
            for selected in config.values():
                require(selected['config']['strength'] in [0.,.15,.3] and selected['grid_size']==3,'Unexpected parameter grid')
                require(not selected['test_used'],'Test used for structural selection')
            llm=read(work/'selected_llm.json') if (work/'selected_llm.json').exists() else None
            if llm:
                require(llm['selection_split']=='calibration' and not llm['test_used'],'LLM split violation')
                if llm['strength']>0:require(llm['unique_prompts']>=32 and llm['selection_audit']['groups']>=32,'Insufficient LLM calibration evidence')
            with np.load(work/'ranks.npz',allow_pickle=False) as z:ranks={name:z[name].copy() for name in z.files}
            expected=2*len(dataset.truth['test']);require(summary['queries']==expected,'Incomplete test coverage')
            for name,r in ranks.items():
                require(r.shape==(expected,) and np.isfinite(r).all() and (r>=1).all(),'Invalid ranks')
                for m,v in metrics(r).items():require(abs(v-summary['metrics'][name][m])<1e-12,'Metric mismatch')
                metric_count+=1
            if llm and llm['strength']==0:require(np.array_equal(ranks['R4'],ranks['R5']),'Zero LLM residual changed ranks')
            observed=set(zip(graph.edge_index[0].tolist(),graph.edge_type.tolist(),graph.edge_index[1].tolist()))
            inspected=0
            for qi in np.random.default_rng(20261112).choice(expected,min(16,expected),replace=False):
                i,d=divmod(int(qi),2);scores=np.load(work/'scores/test'/f'{i}_{d}.npy',allow_pickle=False)
                c=read(work/'candidates/test'/f'{i}_{d}.json');ids=np.array([x['entity'] for x in c['candidates']])
                h,q=map(int,dataset.public['test'][i,d]);mask=np.ones(graph.n,bool);mask[list(dataset.candidate_masks['test'].get((h,q),()))]=False
                eligible=np.flatnonzero(mask);correct=eligible[np.lexsort((eligible,-scores[eligible]))][:30]
                require(np.array_equal(ids,correct),'Candidate pool mismatch')
                require(dataset.rank('test',i,d,scores,ids)==ranks['M0'][qi],'Baseline rank mismatch')
                for method in ['R1','R2','R3','R4']:
                    ev=evidence(work,method,'test',i,d);cfg=Config(**config[method]['config']);order=rerank(ev,cfg)['order']
                    require(dataset.rank('test',i,d,scores,ids,order)==ranks[method][qi],'Rerank mismatch')
                    for candidate in ev['candidate_evidence']:
                        for path in candidate['evidence_paths']:
                            require(all(tuple(t) in observed for t in path['triples']),'Nonobserved evidence edge')
                    if method=='R4' and llm:
                        rp=work/'llm/test'/f'{i}_{d}.json';conf=confidence(read(rp)) if rp.exists() else {}
                        order=rerank(ev,cfg,conf,llm['strength'])['order']
                        require(dataset.rank('test',i,d,scores,ids,order)==ranks['R5'][qi],'LLM rank mismatch')
                inspected+=1
            for name,c in summary['primary_comparisons'].items():
                new,base=name.split('_vs_')
                require(abs(c['delta_MRR']-np.mean(1/ranks[new]-1/ranks[base]))<1e-12,'Primary contrast mismatch')
                pvalues.append(c['fact_sign_flip']['p'])
            seen=set()
            for line in (work/'per_query_ranks.jsonl').read_text(encoding='utf-8').splitlines():
                row=json.loads(line);pair=(row['fact_index'],row['direction']);require(pair not in seen,'Duplicate query output');seen.add(pair)
            require(seen=={(i,d) for i in range(expected//2) for d in (0,1)},'Missing query output')
            checks.append(dict(dataset=key,family=family,queries=expected,sampled_predictions_verified=inspected,passed=True))
    multiplicity=read(run/'multiplicity.json');require(len(multiplicity['comparisons'])==len(pvalues),'Missing primary comparison')
    require(np.allclose([r['p_holm'] for r in multiplicity['comparisons']],holm(pvalues),atol=1e-15),'Incorrect multiplicity correction')
    result=dict(passed=True,groups=checks,metric_rows=metric_count,primary_comparisons=len(pvalues),code_unchanged=True,
                historical_dependencies_unchanged=True,boundary='replication only; no new blind data')
    save(run/'final_verification.json',result);save(ROOT/'results/final_verification.json',result)
    print(json.dumps(dict(passed=True,groups=len(checks),metric_rows=metric_count),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--training-only',action='store_true');args=parser.parse_args()
    frozen=verify_training()
    if not args.training_only:verify_results(frozen)
    else:print('TRAINING_ISOLATION_VERIFIED')
