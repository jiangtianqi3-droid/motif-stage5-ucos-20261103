"""Resumable A-E diagnostics. Full run automatically verifies and writes report.md."""
import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.graph_adapter import ROOT,dump,sha,digest,load_observed,load_signature,load_query_inputs,CandidateProvider,extract_frozen_gates
from src.community_detection import detect_communities
from src.edge_weighter import CommunityEdgeWeighter
from src.path_extractor import retrieve
from src.evidence_exporter import export_json,validate_evidence
from src.diagnostics import load_offline_truth,evaluate_record,aggregate,paired_interval

def stamp():
    from datetime import datetime,timezone,timedelta
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def code_identity():
    files=[*sorted((ROOT/'src').glob('*.py')),*sorted((ROOT/'scripts').glob('*.py')),*sorted((ROOT/'config').glob('*')),ROOT/'run.sh']
    return {str(p.relative_to(ROOT)):sha(p) for p in files}

def progress(**kw):
    if kw.get('state') in ('starting','running') and shutil.disk_usage(ROOT).free<8*1024**3:
        raise RuntimeError('Less than 8 GiB free: stop safely, keep all existing results, resume after adding space')
    value=dict(at=stamp(),pid=os.getpid(),**kw)
    print(json.dumps(value,ensure_ascii=False),flush=True);dump(ROOT/'results/status.json',value)
    with (ROOT/'logs/events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,ensure_ascii=False)+'\n')

def process_dataset(key,cfg,run,limit,smoke):
    began=time.perf_counter()
    graph=load_observed(key);sig=load_signature(graph,key)
    query_forward,query_inverse=load_query_inputs(key)
    query_inverse[:,1]=(query_inverse[:,1]+graph.R//2)%graph.R
    provider=CandidateProvider(key,graph.n)
    assert len(query_forward)==len(query_inverse)==len(provider.scores)
    public_queries=np.c_[query_forward[:,0],query_inverse[:,0],query_forward[:,1]].astype(np.int64)
    assert hashlib.sha256(np.ascontiguousarray(public_queries).tobytes()).hexdigest()==provider.query_sha256,'Candidate query order mismatch'
    indices=np.arange(len(query_forward))
    if limit is not None and len(indices)>limit:
        indices=np.sort(np.random.default_rng(cfg['seed']).choice(indices,limit,replace=False))
    work=run/key;work.mkdir(parents=True,exist_ok=True)
    dump(work/'query_selection.json',dict(indices=indices.tolist(),seed=cfg['seed'],selection='uniform facts without inspecting answers',both_directions=True))
    common=detect_communities(graph,seed=cfg['seed']);labels=common['labels']
    dump(work/'community.json',dict(labels=labels.tolist(),members=common['members'],statistics=common['statistics']))
    factory=CommunityEdgeWeighter(graph,sig,cfg['eta_out'],cfg['eta_rel'],cfg['eta_in'],cfg['eta_topology'],cfg['support'])
    degree=factory.compute('degree_aware')
    weighted=detect_communities(graph,degree.weights,seed=cfg['seed'])
    dump(work/'community_degree.json',dict(labels=weighted['labels'].tolist(),statistics=weighted['statistics']))
    relations=np.r_[query_forward[indices,1],query_inverse[indices,1]]
    progress(state='running',stage='gated_model',dataset=key,run=str(run),relations=len(set(map(int,relations))))
    import torch
    torch.cuda.reset_peak_memory_stats()
    gate_start=time.perf_counter()
    gates=extract_frozen_gates(graph,sig,relations,ROOT/'cache/gates',
        lambda done,total,q:progress(state='running',stage='gated_model',dataset=key,done=done,total=total,relation=q,run=str(run)))
    gate_seconds=time.perf_counter()-gate_start
    # Ensure both real topology submodes distinguish relation types.
    rho_stats={}
    for q in sorted(set(map(int,relations))):
        a=factory.relation_importance(q);b=factory.relation_importance(q,'gated_model',gates[q])
        rho_stats[str(q)]=dict(stat_span=float(np.ptp(a)),gate_span=float(np.ptp(b)))
    assert any(v['stat_span']>1e-8 for v in rho_stats.values()),'Statistical topology is constant'
    assert any(v['gate_span']>1e-8 for v in rho_stats.values()),'Actual gate mapping is constant'
    pre_seconds=time.perf_counter()-began
    dump(work/'preprocessing.json',dict(nodes=graph.n,edges=len(graph.edge_type),relations=graph.R,hyperedges=len(sig['edge_type']),
        seconds=pre_seconds,gate_seconds=gate_seconds,rho=rho_stats,graph_key=graph.key,candidates_sha256=provider.source_sha256,
        peak_cuda_bytes=int(torch.cuda.max_memory_allocated()),community_statistics=common['statistics']))
    # Iterate by relation so only one set of entity transition matrices is resident.
    jobs={}
    for i in indices:
        for direction,queries in enumerate([query_forward,query_inverse]):
            h,q=map(int,queries[i]);jobs.setdefault(q,[]).append((int(i),direction,h))
    completed=0;total=len(indices)*2*len(cfg['methods'])*len(cfg['candidate_k'])*len(cfg['path_budgets'])
    retrieval_started=time.perf_counter();uncached_seconds=[]
    for q in sorted(jobs):
        modes=dict(naive_khop=('uniform',None),global_community=('uniform',None),ppr_uniform=('uniform',None),
            ppr_degree=('degree_aware',None),ppr_topology_signature=('topology_aware','topo_signature'),
            ppr_topology_gated=('topology_aware','gated_model'))
        for method in cfg['methods']:
            mode,submode=modes[method]
            weights=factory.compute(mode,q,submode or 'topo_signature',gates[q] if submode=='gated_model' else None)
            for i,direction,h in jobs[q]:
                candidates=provider.get(i,direction,max(cfg['candidate_k']))
                path_cache={};tail_cache={}
                # Largest budget first; reuse exact joins without changing their results.
                for k in sorted(cfg['candidate_k'],reverse=True):
                    for m in sorted(cfg['path_budgets'],reverse=True):
                        path=work/'evidence'/method/f'k{k}_m{m}'/f'{i}_{direction}.json'
                        if path.exists():
                            record=json.loads(path.read_text())
                            validate_evidence(record,graph)
                            assert record['graph_key']==graph.key and record['query_info']['head']==h and record['query_info']['relation']==q
                            assert record['provenance']['run_fingerprint']==run.name
                        else:
                            record=retrieve(graph,weights,labels,h,q,candidates[:k],method,sig,
                                cfg['max_nodes'],cfg['max_edges'],m,cfg['hops'],cfg['alpha'],path_cache,tail_cache)
                            record['query_info'].update(query_id=f'{key}:{i}:{direction}',fact_index=i,direction=direction)
                            record['provenance']=dict(candidate_source='B0 unfiltered reciprocal raw rank',
                                candidates_sha256=provider.source_sha256,run_fingerprint=run.name,ground_truth_used=False,
                                topology_submode=submode,gate_checkpoint='T_gate_only_0p25_s1024_step500.pth' if submode=='gated_model' else None)
                            export_json(path,record,graph)
                            if k==max(cfg['candidate_k']) and m==max(cfg['path_budgets']):uncached_seconds.append(record['seconds'])
                        completed+=1
                if completed%24==0:
                    progress(state='running',stage='retrieval',dataset=key,done=completed,total=total,run=str(run))
    # Only now load evaluation targets and hidden raw files, after every evidence record exists.
    truth,audit=load_offline_truth(key,graph)
    dump(work/'leakage_audit.json',dict(passed=True,observed_graph_key=graph.key,**audit,
        graph_weights_and_paths_share_only_this_graph=True,truth_opened_after_retrieval=True))
    hubs=set(map(int,np.argsort(-(graph.in_degree+graph.out_degree),kind='stable')[:max(1,int(np.ceil(.05*graph.n)))]))
    rows_by_group={};summaries=[]
    for method in cfg['methods']:
        for k in cfg['candidate_k']:
            for m in cfg['path_budgets']:
                rows=[]
                for i in indices:
                    for direction in [0,1]:
                        p=work/'evidence'/method/f'k{k}_m{m}'/f'{i}_{direction}.json'
                        record=json.loads(p.read_text())
                        answer=int(truth[i,2 if direction==0 else 0])
                        rows.append(evaluate_record(record,answer,labels,hubs))
                rows_by_group[method,k,m]=rows
                dump(work/'diagnostics'/f'{method}_k{k}_m{m}.json',rows)
                summaries.append(dict(dataset=key,method=method,k=k,m=m,**aggregate(rows)))
    comparisons=[]
    for k in cfg['candidate_k']:
        for m in cfg['path_budgets']:
            for base in ['global_community','ppr_uniform','ppr_degree']:
                for end in ['ppr_topology_signature','ppr_topology_gated']:
                    for metric in ['hit','hub_fraction','cross_hit','expansion_success']:
                        comparisons.append(dict(dataset=key,k=k,m=m,base=base,target=end,metric=metric,
                            **paired_interval(rows_by_group[base,k,m],rows_by_group[end,k,m],metric,cfg['seed'])))
    dump(work/'metrics.json',summaries);dump(work/'paired_intervals.json',comparisons)
    from src.report_writer import make_figures
    make_figures(work,graph,labels,indices,cfg)
    result=dict(dataset=key,state='complete',finished_at=stamp(),query_facts=len(indices),
        retrieval_seconds=time.perf_counter()-retrieval_started,preprocessing_seconds=pre_seconds,
        cold_retrieval_mean_seconds=float(np.mean(uncached_seconds)) if uncached_seconds else None,
        peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        peak_cuda_bytes=int(torch.cuda.max_memory_allocated()),groups=len(summaries))
    dump(work/'completion.json',result)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--limit',type=int,default=None)
    parser.add_argument('--datasets',nargs='+')
    args=parser.parse_args()
    cfg=yaml.safe_load((ROOT/'config/default.yaml').read_text())
    if args.datasets: cfg['datasets']=args.datasets
    limit=1 if args.smoke else (args.limit or cfg['query_limit'])
    if limit is not None and limit<1: raise ValueError('limit must be positive')
    code=code_identity();payload=dict(config=cfg,limit=limit,smoke=args.smoke,code=code,freeze_manifest_sha256=sha(ROOT/'freeze_manifest.json'))
    fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:20]
    run=ROOT/'results/runs'/fingerprint;run.mkdir(parents=True,exist_ok=True)
    lock=ROOT/'logs/run.lock'
    if lock.exists():
        pid=int(lock.read_text())
        try: os.kill(pid,0)
        except ProcessLookupError: lock.unlink()
        else: raise RuntimeError(f'Another experiment is active: PID {pid}')
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        progress(state='starting',stage='environment',run=str(run))
        import torch,scipy,networkx
        dump(run/'protocol.json',payload)
        import matplotlib,pytest
        dump(run/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
            scipy=scipy.__version__,networkx=networkx.__version__,torch=torch.__version__,cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0),seed=cfg['seed'],matplotlib=matplotlib.__version__,pytest=pytest.__version__))
        versions=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
        (run/'pip-freeze.txt').write_text(versions,encoding='utf-8')
        summaries=[];start=time.perf_counter()
        for key in cfg['datasets']:
            done=run/key/'completion.json'
            if done.exists():summaries.append(json.loads(done.read_text()));continue
            summaries.append(process_dataset(key,cfg,run,limit,args.smoke))
        from verify_isolation import verify
        isolation=verify(full=not args.smoke)
        from src.report_writer import write_report
        write_report(run,cfg,limit,args.smoke,isolation)
        result=dict(state='smoke_complete' if args.smoke else 'complete',stage='finished',run=str(run),
            seconds=time.perf_counter()-start,datasets=summaries,report=str(ROOT/'report.md'))
        dump(run/'completion.json',result);progress(**result)
    except BaseException as exc:
        failure=dict(state='failed',stage='exception',run=str(run),error=str(exc),traceback=traceback.format_exc())
        dump(run/'failure.json',failure);progress(**failure)
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__': main()
