"""Resumable full experiment; no training, test filtering or tuning in inference."""
import fcntl
import gc
import os
import time
import traceback
import subprocess
import numpy as np
import torch
from tqdm import tqdm
from common_io import ROOT, SEEDS, MODES, read, dump, sha, identity, now
from regime_inspector import records, observation, vocabulary
from protocol_data import final_labels, validation
from backbone import bidirectional
from evidence import EvidenceGraph
from experiment import score_graph, score_rows, lexicon, public_inference, evaluate_graph
from ranking import grouped_validation
from offline_eval import historical, truth_index, baseline_rank, metrics

ROOT.joinpath('results').mkdir(exist_ok=True)
lock=(ROOT/'results/run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
started=time.monotonic()
state=dict(pid=os.getpid(),state='running',completed_graphs=[],started_at=now(),api_calls=0,prompt_tokens=0,completion_tokens=0)
totals={}
def progress(stage,**kw):
    if stage=='reranking':
        key=(kw.get('graph'),kw.get('split'),kw.get('seed'))
        totals[key]={k:kw.pop(k,0) for k in ['api_calls','prompt_tokens','completion_tokens','cache_hits']}
        kw.update({k:sum(item[k] for item in totals.values()) for k in ['api_calls','prompt_tokens','completion_tokens','cache_hits']})
    state.update(stage=stage,updated_at=now(),elapsed_seconds=time.monotonic()-started,**kw)
    state['estimated_cost_usd_upper']=(state.get('prompt_tokens',0)*.3+state.get('completion_tokens',0)*1.2)/1e6
    state['pricing_basis']='DeepSeek flash peak cache-miss upper estimate; 2026-11-16 official pricing'
    dump(ROOT/'status.json',state)

def main():
    torch.set_num_threads(4)
    config=read(ROOT/'config/experiment.json'); freeze=read(ROOT/'freeze_manifest.json')
    assert freeze['model_actually_loaded'] and read(ROOT/'results/smoke_verification.json')['passed']
    assert read(ROOT/'results/large_preflight.json')['passed']
    assert read(ROOT/'results/tests.json')['passed']
    for p,expected in freeze['protected_sources'].items(): assert sha(p)==expected,p
    code={str(p.relative_to(ROOT)):sha(p) for folder in ['src','scripts','config'] for p in (ROOT/folder).rglob('*') if p.is_file()}
    runid=identity(dict(code=code,freeze=freeze,config=config))[:20]
    run=ROOT/'results/runs'/runid;run.mkdir(parents=True,exist_ok=True)
    dump(run/'provenance.json',dict(code=code,freeze=freeze,config=config))
    progress('starting',run=str(run),run_id=runid)
    recs=records();recs.sort(key=lambda x:(x['key']!='NELLInductive_v1',x['key']=='ConceptNet100k_external',x['key']))
    graphbar=tqdm(recs,desc='Full experiment graphs',unit='graph')
    for rec in graphbar:
        key=rec['key'];out=run/key;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():
            state['completed_graphs'].append(key);continue
        progress('graph_preparation',graph=key)
        z=observation(rec);names,rels,_=vocabulary(rec,z)
        facts,alltrue=final_labels(rec)
        q=bidirectional(facts,z['R']//2);pairs=q[:,:2].copy()
        # Persist public payload and isolated evaluator inputs separately.
        np.save(out/'public_queries.npy',pairs)
        base,m5=historical(key,q)
        score_dir=ROOT/'cache/scores'/runid/key
        for j,seed in enumerate(SEEDS):
            score_graph(z,pairs,seed,score_dir/str(seed),progress,verify=True)
            truth=truth_index(alltrue,z['R']//2)
            for i,score in score_rows(score_dir/str(seed)):
                h,r,t=map(int,q[i]);assert baseline_rank(score,h,r,t,truth)==base[j,i],f'{key}/{seed}/{i} mismatch'
        dump(out/'baseline_alignment.json',dict(exact=True,queries=len(q),seeds=SEEDS,bitwise_operator_replay=True,
            stage7_mrr=metrics(base),same_query_M5=metrics(m5)))
        graph=EvidenceGraph(z,config);lex=lexicon(names,rels,graph)
        query_dirs=out/'test_queries'
        for seed in SEEDS:
            public_inference(graph,lex,key,'test',seed,pairs,score_dir/str(seed),query_dirs/str(seed),config,progress)
        del graph,lex;gc.collect()
        # Fixed method: no choice is made from test or validation outcomes.
        progress('validation_preparation',graph=key)
        vz,vfacts,vtruth,vnames,protocol=validation(rec,z,ROOT/'cache/validation_topology'/key)
        vq=bidirectional(vfacts,vz['R']//2)
        selection,confirm=grouped_validation(vfacts,vz['R']//2,config['validation_seed'])
        dump(out/'validation_split.json',dict(selection_val=selection,confirm_val=confirm,
            grouping='connected facts sharing forward or inverse public queries',protocol=protocol,
            tuned_parameters=False,high_margin_gate=False))
        vrels=rels if vz is z else vz['_relation_names']
        vg=EvidenceGraph(vz,config);vl=lexicon(vnames,vrels,vg)
        valdir=ROOT/'cache/scores'/runid/(key+'_valid')
        for seed in SEEDS:
            score_graph(vz,vq[:,:2],seed,valdir/str(seed),progress)
            public_inference(vg,vl,key,'valid',seed,vq[:,:2],valdir/str(seed),out/'valid_queries'/str(seed),config,progress)
        from evaluate_validation import evaluate
        dump(out/'validation_metrics.json',evaluate(vq,vtruth,vz['R']//2,valdir,out/'valid_queries',selection,confirm))
        # Identity across independent graphs is matched by public vocabulary token,
        # never by coincident internal integer indices.
        val_heads={vnames[int(h)] for h in vq[:,0] if vnames[int(h)] is not None}
        seen={i for i,token in enumerate(names) if token is not None and token in val_heads}
        outputs,summary=evaluate_graph(key,facts,alltrue,z,score_dir,query_dirs,seen)
        from unique_diagnostics import summarize
        summary['unique_public_query_diagnostics']=summarize(query_dirs)
        tmp=out/'rankings.tmp'
        with tmp.open('wb') as f:np.savez_compressed(f,queries=q,**outputs)
        os.replace(tmp,out/'rankings.npz');dump(out/'summary.json',summary)
        dump(out/'complete.json',dict(queries=len(q),ranking_sha256=sha(out/'rankings.npz'),completed_at=now()))
        state['completed_graphs'].append(key);progress('graph_complete',graph=key)
        del vz,vg,vl,z,outputs,alltrue;gc.collect();torch.cuda.empty_cache()
    for p,expected in freeze['protected_sources'].items():assert sha(p)==expected,p
    report=['# B2+D* 局部证据 LLM 重排正式结果','',
        '11 图、三个冻结 B2 种子；所有测试查询均保留。区间绑定同一事实的双向查询和三个种子，配对 Bootstrap 2,000 次。',
        '数据均有历史测试暴露，本实验不是首次盲测。原始全候选分数独立保存在 cache/scores；未改写原分数。','',
        '| 图谱 | 方法 | MRR | Hits@1 | Hits@3 | Hits@10 |','|---|---|---:|---:|---:|---:|']
    for rec in recs:
        summary=read(run/rec['key']/'summary.json')
        for name,item in summary['methods'].items():
            m=item['metrics'];report.append('| '+rec['key']+' | '+name+' | '+' | '.join(f'{m[k]:.6f}' for k in ['MRR','Hits1','Hits3','Hits10'])+' |')
    report+=['','逐图 summary.json 包含与骨干/M5 的配对区间、Seen/Unseen 分表及全部诊断。',
        '不以单图或点估计上升宣称总体优于 M5；请根据配对区间及图间一致性判断。']
    (ROOT/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    dump(ROOT/'results/final_summary.json',{rec['key']:read(run/rec['key']/'summary.json') for rec in recs})
    # Commit only this experiment's code, configuration and small verification/report
    # artifacts. Large scores, caches, prompts and rankings stay in ignored outputs.
    gitdir=(ROOT.parents[1]/'.git').read_text().strip().split('gitdir: ',1)[1]
    if gitdir.startswith('D:/'):gitdir='/mnt/d/'+gitdir[3:]
    git=['git','--git-dir='+gitdir,'--work-tree='+str(ROOT.parents[1])]
    subprocess.run(git+['add','--','experiments/b2_dstar_llm_rerank'],check=True)
    committed=subprocess.run(git+['commit','-m','Record completed frozen B2+Dstar LLM reranking experiment'],capture_output=True,text=True)
    state['final_commit_succeeded']=committed.returncode==0
    if committed.returncode:state['commit_error']=committed.stderr[-1000:]
    progress('completed',state='completed',report=str(ROOT/'report.md'))

try:main()
except BaseException as exc:
    progress('failed',state='failed',error=type(exc).__name__+': '+str(exc))
    traceback.print_exc()
    raise
