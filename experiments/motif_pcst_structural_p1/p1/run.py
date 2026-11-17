"""Structural-only P1 worker. The experiment uses one local process and resumable query files."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from p1.assets import Assets, Evaluator, prepare
from p1.config import ROOT, RESULTS, load_config
from p1.memory import export
from p1.retrieval import ARMS, ABLATIONS, Retriever
from p1.scoring import evidence, filtered_after, permutations, raw_order
from p1.source import source_calibration
from src.io_utils import PRIMARY, STAGE7, dump, identity, read, sha, write_csv, atomic
from src.recovery import verify_runtime

TZ = timezone(timedelta(hours=8))
MAIN = ('evidence_only','fusion','gated','fixed_015')


def stage_plan(cfg, mode):
    target = cfg['target']
    phases = [('quick',key,target['quick_facts'],0) for key in target['primary']]
    if mode == 'quick':
        return phases
    phases += [('expanded',key,target['expanded_facts'],0) for key in target['primary']]
    phases += [('transfer',key,target['transfer_facts'],0) for key in target['transfer']]
    phases += [('sensitivity',key,target['sensitivity_facts'],0) for key in target['primary']]
    return phases


def indices_for(size,key,count,seed):
    # Both directions stay together. No answers, rank or correctness enter this draw.
    number = int(identity([seed,key])[:16],16)
    return np.random.default_rng(number).permutation(size)[:count].tolist()


def schemes(cfg, phase):
    r = cfg['retrieval']
    b = cfg['evidence_budget']
    k, facts = r['candidate_topk'],b['primary_max_facts']
    if phase!='sensitivity':
        return [(k,facts,ARMS + (ABLATIONS if phase=='quick' else ()),phase=='quick')]
    return [(k,bb,ARMS,False) for bb in b['sensitivity_max_facts']] + [
        (kk,facts,ARMS,False) for kk in cfg['candidate_sensitivity']['topk'] if kk!=k]


def score_job(asset, retriever, raw, index, direction, k, budget, arms, match, source, cfg, memory_slots):
    h,r = map(int,asset.queries[index,direction])
    top30 = raw_order(raw)[:30]
    candidates = top30[:k]
    query_id = f'{asset.key}:{index}:{direction}'
    selected = retriever.query(h,r,candidates,budget,arms,match)
    output = {}
    memories = None
    for arm,e in selected.items():
        ev = evidence(asset.graph,r,candidates,e,source['table'])
        orders = permutations(raw,candidates,ev,source['lambda_selected'],cfg['source']['gate_margin'])
        orders['fixed_015'] = permutations(raw,candidates,ev,.15,cfg['source']['gate_margin'])['fusion']
        output[arm] = dict(ids=e['ids'],metrics=e['metrics'],values=ev,
                           covered_candidates=sorted(e['paths']),orders=orders,paths=e['paths'])
        if arm=='R5' and memory_slots:
            memories = export(asset.graph,query_id,h,r,candidates,raw,e,ev,memory_slots,
                              asset.meta['parameter_sha256_before'],asset.entry['observed_sha256'])
    return dict(query_id=query_id,index=index,direction=direction,head=h,relation=r,
                candidates=candidates,top30=top30,arms=output,memory=memories)


def run_graph(asset, phase, folder, cfg, source, indices, on_progress):
    score,gates = prepare(asset,indices)
    retriever = Retriever(asset.graph,gates,source['table'],cfg['retrieval'])
    stage_root = folder/'runs'/phase/asset.key
    dump(stage_root/'samples.json',dict(dataset=asset.key,phase=phase,indices=indices,
         score_kind='reciprocal_raw_rank',checkpoint=asset.meta['parameter_sha256_before'],
         observations=asset.entry['observed_sha256']))
    jobs = []
    timing = perf_counter()
    progress = 0
    for k,budget,arms,match in schemes(cfg,phase):
        for i in indices:
            for d in (0,1):
                path = stage_root/f'k{k}_b{budget}'/f'{i}-{d}.json'
                if path.is_file():
                    job = read(path)
                else:
                    memory_slots = cfg['memory']['slots'] if phase!='sensitivity' else 0
                    job = score_job(asset,retriever,score[i][d],i,d,k,budget,arms,match,source,cfg,memory_slots)
                    dump(path,job)
                jobs.append((k,budget,job))
                progress+=1
                if progress%20==0:
                    on_progress(progress)
    # All structural jobs already exist on disk. Only now open the evaluation truth.
    evaluator = Evaluator(asset)
    rows, metrics = [], []
    diagnostics = []
    mem_path = stage_root/'structural_memory.jsonl'
    temp_mem = mem_path.with_suffix('.partial')
    with temp_mem.open('w',encoding='utf-8') as mem:
        for k,budget,job in jobs:
            if job['memory']:
                mem.write(json.dumps(job['memory'],ensure_ascii=False,allow_nan=False)+'\n')
            i,d = job['index'],job['direction']
            raw = score[i][d]
            target,excluded,before = evaluator.query(i,d,job['head'],job['relation'],raw)
            candidates = job['candidates']
            top30 = job['top30']
            common = dict(dataset=asset.key,phase=phase,query_id=job['query_id'],fact_index=i,direction=d,
                          k=k,budget=budget,original_rank=before,recall12=target in top30[:12],recall30=target in top30,
                          coverage_tier='top12' if target in top30[:12] else 'top13-30' if target in top30 else 'outside30',
                          query_relation=job['relation'],head_degree=int(asset.graph.out_degree[job['head']]))
            def add(method,fusion,rank,reciprocal=None,hits=None):
                rows.append(dict(common,method=method,fusion=fusion,rank=rank,
                    positive=rank<before if hits is None else hits[3],negative=rank>before if hits is None else hits[4],
                    reciprocal=1/rank if reciprocal is None else reciprocal,
                    hits1=float(rank<=1) if hits is None else hits[0],
                    hits3=float(rank<=3) if hits is None else hits[1],hits10=float(rank<=10) if hits is None else hits[2]))
            add('frozen_B2+Dstar','baseline',before)
            add('B0','baseline',int(evaluator.b0[i,d]))
            if evaluator.m5 is not None:
                m5 = evaluator.m5[:,i,d]
                # Average the metrics of three independently frozen seeds, not the ranks before MRR.
                add('M5_three_seed','baseline',float(m5.mean()),float((1/m5).mean()),
                    [float((m5<=n).mean()) for n in (1,3,10)]+[float((m5<before).mean()),float((m5>before).mean())])
            for arm,out in job['arms'].items():
                ms = out['metrics']
                alt = out['values']
                target_score = alt.get(target,alt.get(str(target)))
                negs = [v['score'] for c,v in alt.items() if int(c)!=target]
                delta = None if target_score is None or not negs else target_score['score']-mean(negs)
                metrics.append(dict(common,method=arm,**ms,target_path=target in out['covered_candidates'],
                                target_evidence_margin=delta))
                for fusion in MAIN:
                    order = out['orders'][fusion]
                    rank = filtered_after(raw,candidates,order,target,excluded,evaluator.rank)
                    add(arm,fusion,rank)
                diagnostics.append(dict(query_id=job['query_id'],method=arm,ids=out['ids'],target=target,
                     target_path=target in out['covered_candidates'],support=ms['motif_support'],
                     path=out['paths'].get(target,out['paths'].get(str(target),[])),baseline=before,rank=rows[-3]['rank'],
                     gated=rows[-2]['rank'],fixed=rows[-1]['rank'],
                     score=target_score['score'] if target_score is not None else None,
                     coverage=ms['covered'],nodes=ms['selected_nodes'],budget=budget,k=k))
    temp_mem.replace(mem_path)
    write_csv(stage_root/'retrieval.csv',metrics)
    write_csv(stage_root/'predictions.csv',rows)
    dump(stage_root/'diagnostics.json',diagnostics)
    dump(stage_root/'timing.json',dict(queries=progress,seconds=perf_counter()-timing))
    return rows,metrics,diagnostics


def boot_delta(pairs,seed,nboot):
    facts = defaultdict(list)
    for row,val in pairs:
        facts[row['fact_index']].append(val)
    x = np.array([mean(v) for v in facts.values()])
    if not x.size:
        return None
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0,len(x),(nboot,len(x)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def summaries(rows, metrics, cfg):
    actual = defaultdict(list)
    shots = defaultdict(list)
    for r in rows:
        actual[r['dataset'],r['k'],r['budget'],r['method'],r['fusion']].append(r)
    for r in metrics:
        shots[r['dataset'],r['k'],r['budget'],r['method']].append(r)
    predict, retract = [], []
    for (g,k,b,method,fusion),items in actual.items():
        paired = [(r,r['reciprocal']-1/r['original_rank']) for r in items]
        mrr = mean(r['reciprocal'] for r in items)
        rec = dict(dataset=g,k=k,budget=b,method=method,fusion=fusion,queries=len(items),mrr=mrr,
                   delta_mrr=mean(d for _,d in paired),ci=boot_delta(paired,cfg['random_seed'],cfg['target']['bootstrap']),
                   mean_rank_improvement=mean(r['original_rank']-r['rank'] for r in items),
                   tiers=dict(Counter(r['coverage_tier'] for r in items)))
        rec.update({key:mean(r[key] for r in items) for key in ('hits1','hits3','hits10','recall12','recall30')})
        rec['positive'] = sum(r['positive'] for r in items)
        rec['negative'] = sum(r['negative'] for r in items)
        predict.append(rec)
    for (g,k,b,method),items in shots.items():
        item = dict(dataset=g,k=k,budget=b,method=method,queries=len(items),
             reachable=sum(r['reachable'] for r in items),covered=sum(r['covered'] for r in items),
             path_length_distribution=dict(sum((Counter({int(k):v for k,v in r['path_length_distribution'].items()})
                                                  for r in items),Counter())))
        item['coverage_reachable']=item['covered']/item['reachable'] if item['reachable'] else None
        for key in ('candidate_coverage_all','selected_observed_edges','selected_nodes','path_count','motif_support',
                    'path_reliability','hub_ratio','evidence_redundancy','budget_utilization','retrieval_latency_ms',
                    'protected_facts','target_path','target_evidence_margin'):
            valid = [r[key] for r in items if r[key] is not None]
            item[key] = mean(valid) if valid else None
        retract.append(item)
    return predict,retract


def main(mode):
    from p1.report import make_report
    cfg = load_config()
    versions = {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'p1').glob('*.py'))}
    source_files = (STAGE7/'inputs.json', STAGE7/'model_manifest.json', STAGE7/'entity_frozen.json')
    sources = {p.name:sha(p) for p in source_files}
    fingerprint = identity(dict(cfg=cfg,code=versions,assets=sources,
                                poc_commit='9b92c77'))
    folder = RESULTS/fingerprint
    folder.mkdir(parents=True,exist_ok=True)
    done = []
    errors = []
    peaks = []
    current = 'asset_audit'
    def state(status,phase,**kw):
        v = dict(state=status,stage='P1',current_phase=phase,run_id=fingerprint,
                 pid=os.getpid(),last_update=datetime.now(TZ).isoformat(),completed_experiments=done,
                 gpu_peak_memory_mb=max(peaks,default=0)/1024**2,llm_api_requests=0,errors=errors,**kw)
        dump(RESULTS/'status.json',v)
        return v
    state('running',current)
    try:
        if shutil.disk_usage(ROOT).free < 8*1024**3:
            raise RuntimeError('free space under 8 GiB')
        verified = verify_runtime()
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        dump(folder/'audit.json',dict(commit=commit,config=cfg,fingerprint=fingerprint,stage7_files=verified,
              checkpoint=read(STAGE7/'model_manifest.json')['B2_s1024'],poc='9b92c77',mode=mode))
        current='unit_tests'
        state('running',current)
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],check=True,cwd=ROOT)
        current='source_path_validation'
        state('running',current)
        source = source_calibration(cfg,folder)
        done.append('source_path_validation')
        for phase,key,count,_ in stage_plan(cfg,mode):
            current = f'{phase}/{key}'
            aggregate_path = folder/'runs'/phase/key/'summary.json'
            if aggregate_path.is_file():
                done.append(current)
                continue
            state('running',current)
            started = perf_counter()
            try:
                asset = Assets(key,folder)
                indices = indices_for(len(asset.queries),key,count,cfg['random_seed'])
                total = len(indices)*2*len(schemes(cfg,phase))
                state('running',current,completed_queries=0,total_queries=total)
                rows,metrics,diag = run_graph(asset,phase,folder,cfg,source,indices,
                    lambda n:state('running',current,completed_queries=n,total_queries=total))
                p,r = summaries(rows,metrics,cfg)
                asset_meta = read(asset.folder/'prepared.json')
                peaks.append(asset_meta['peak_allocated'])
                dump(aggregate_path,dict(dataset=key,phase=phase,prediction=p,retrieval=r,
                     graph=dict(nodes=asset.graph.n,edges=len(asset.graph.edge_type),relations=asset.graph.R,
                                degree_p90=float(np.quantile(asset.graph.out_degree,.9))),
                     diagnostics=len(diag),wall_seconds=perf_counter()-started,total_queries=total,
                     asset=asset_meta))
            except RuntimeError as exc:
                if 'out of memory' not in str(exc).lower():
                    raise
                errors.append(dict(phase=current,error=str(exc)))
                continue
            done.append(current)
            make_report(folder,state('running',current,completed_queries=total,total_queries=total))
        status = 'complete' if mode=='full' else 'quick_complete'
        if errors:
            status += '_with_errors'
        make_report(folder,state(status,'report'))
    except BaseException as exc:
        errors.append(dict(phase=current,error=f'{type(exc).__name__}: {exc}'))
        make_report(folder,state('failed',current))
        raise


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode',choices=('quick','full'),default='full')
    mode = p.parse_args().mode
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(mode)
