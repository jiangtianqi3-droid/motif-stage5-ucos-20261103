import argparse
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median

import numpy as np

from src.api.budget_guard import Ledger, LIMITS, BudgetExhausted
from src.api.client import Client
from src.evaluation import OfflineEvaluation
from src.evidence_serializer import candidate_map, messages, label
from src.graph_loader import fact, open_graph
from src.io_utils import KEYS, PRIMARY, RESULTS, ROOT, SEED, atomic, dump, identity, read, sha, write_csv
from src.local_retrieval import retrieve
from src.reporting import aggregate, recovery_report, report

DATA = RESULTS / 'api'
LEDGER = DATA / 'ledger.sqlite'
ARMS = ('A1','A2','A3','A4','A5')
TZ = timezone(timedelta(hours=8))


def phase_plan(n, key):
    indices = random.Random(f'{SEED}:{key}').sample(range(n), min(n, 50))
    if len(indices) < 14:
        raise ValueError('not enough unique test facts for disjoint smoke and pilot')
    return dict(retrieval=indices, smoke=indices[:2], pilot=indices[2:14])


def stage_state(state, stage, **info):
    requests, inp, out = Ledger(LEDGER).usage() if LEDGER.is_file() else (0,0,0)
    info = dict(state=state, current_stage=stage, pid=os.getpid(), last_update=datetime.now(TZ).isoformat(),
                api_requests_used=requests, api_requests_limit=LIMITS[0], api_input_budget_used=inp,
                api_output_budget_used=out, **info)
    dump(RESULTS/'status.json', info)
    return info


def ranking_context():
    return {key: open_graph(key, PRIMARY) for key in KEYS}


def retrieval_snapshot(context):
    source = RESULTS / 'recovery/recovery.json'
    parts = ('local_retrieval.py','evidence_serializer.py','graph_loader.py','pcst_adapter.py')
    manifest = dict(model=PRIMARY, plans={key:phase_plan(len(context[key][3]),key) for key in KEYS},
                    seed=SEED, ranks=[r['cache_sha256'] for r in read(source)['records'] if r['model']==PRIMARY],
                    code={name:sha(ROOT/'src'/name) for name in parts})
    fingerprint = identity(manifest)
    paths = RESULTS / 'retrieval' / fingerprint
    dump(paths/'sample_manifest.json', dict(fingerprint=fingerprint, **manifest))
    jobs = {}
    metrics = []
    groups = defaultdict(list)
    for key in KEYS:
        graph, weighter, scores, queries = context[key]
        jobs[key] = {}
        for index in manifest['plans'][key]['retrieval']:
            for direction, (head, relation) in enumerate(queries[index]):
                query_id = f'{key}:{index}:{direction}'
                row = scores[index,direction]
                top30 = np.lexsort((np.arange(len(row)), -row))[:30].tolist()
                mapping = candidate_map(query_id, top30[:12])
                seed = int(identity([SEED,query_id])[:16], 16)
                dest = paths / key / f'{index}-{direction}.json'
                if dest.is_file():
                    job = read(dest)
                    if (job['candidate_ids'] != top30[:12] or job['aliases'] != mapping or
                            job['head'] != head or job['relation'] != relation):
                        raise ValueError('retrieval cache mismatch')
                else:
                    output = retrieve(graph, weighter, int(head), int(relation), mapping, seed)
                    for data in output.values():
                        data['supported_ids'] = sorted({label(fact(graph,i))
                                                       for path in data['paths'].values() for i in path})
                    job = dict(dataset=key,query_id=query_id,fact_index=index,direction=direction,
                               head=int(head),relation=int(relation),aliases=mapping,
                               candidate_ids=top30[:12],top30=top30,retrieval=output)
                    dump(dest,job)
                jobs[key][index,direction] = job
                for arm, data in job['retrieval'].items():
                    metrics.append(dict(dataset=key,query_id=query_id,arm=arm,**data))
                    groups[key,arm].append(data)
    stats = []
    for (key,arm), records in groups.items():
        reached = sum(r['reachable'] for r in records)
        covered = sum(r['covered'] for r in records)
        stats.append(dict(dataset=key,arm=arm,queries=len(records), coverage=covered/max(1,reached),
                          candidate_coverage=covered/(12*len(records)), reachable=reached, covered=covered,
                          path_lengths=dict(Counter(n for r in records for n in r['path_lengths'])),
                          **{k:mean(r[k] for r in records) for k in ('edges','nodes','local_edges','local_nodes',
                             'motif_support','duplication','hub_ratio','input_bound','milliseconds','no_evidence')}))
    write_csv(RESULTS/'retrieval_metrics.csv', metrics)
    dump(RESULTS/'retrieval_summary.json', stats)
    dump(RESULTS/'sample_manifest.json', dict(fingerprint=fingerprint,**manifest))
    return fingerprint, manifest['plans'], jobs


def cases(stage, rows, api_results, jobs, context):
    chosen = [(label, next((row for row in rows if row['arm']=='A4' and row['fusion']=='direct' and predicate(row)), None))
              for label,predicate in [('正向翻转',lambda r:r['positive']),('负向翻转',lambda r:r['negative']),
                                      ('弃权',lambda r:r['status']=='abstain'),
                                      ('普通和模体 PCST 证据不同',lambda r:
                                       jobs[r['dataset']][r['fact_index'],int(r['direction']=='head')]['retrieval']['A3']['ids']
                                       != jobs[r['dataset']][r['fact_index'],int(r['direction']=='head')]['retrieval']['A4']['ids'])]]
    lines = ['### 真实案例']
    for description, row in chosen:
        if row is None:
            lines.append(f'- {description}在当前样本中未出现。')
            continue
        key = row['dataset']
        direction = int(row['direction']=='head')
        job = jobs[key][row['fact_index'],direction]
        graph = context[key][0]
        lines += [f"- {description}对应 {row['query_id']}，查询为 ({job['head']}, {job['relation']}, ?)，"
                  f"真实目标为 {row['target']}，过滤名次从 {row['original_rank']} 变为 {row['new_rank']}。",
                  f"  原始 Top-12 为 {job['candidate_ids']}，别名映射为 {job['aliases']}，写回为 {row['written_candidate_ids']}。"]
        for arm in ('A3','A4'):
            ids = job['retrieval'][arm]['ids']
            edges = [f'{label(f)}=({f.src},{f.relation},{f.dst})' for f in map(lambda i:fact(graph,i),ids)]
            reply = api_results.get((row['query_id'],arm),{}).get('response')
            lines.append(f'  {arm} 观察事实为 {" ".join(edges) or "空"}。API 返回为 `{reply}`。')
    atomic(RESULTS/f'{stage}_cases.md','\n'.join(lines)+'\n')


def api_stage(stage, fingerprint, plans, jobs, context, client, limit=None):
    to_run = [(key, jobs[key][index, d]) for key in KEYS for index in plans[key][stage] for d in (0,1)]
    total = len(to_run)
    responses = {}
    done = 0
    error = None
    try:
        for key, job in to_run:
            graph = context[key][0]
            for arm in ARMS:
                if limit is not None and len(responses) >= limit:
                    return
                retrieval = job['retrieval'].get(arm, {})
                evidence = [fact(graph,i) for i in retrieval.get('ids',[])]
                prompt = messages(job['head'],job['relation'],job['aliases'],evidence)
                evidence_hash = identity([[f.id,f.src,f.relation,f.dst] for f in evidence])
                result = client.rank(job['query_id'],job['aliases'],prompt,
                                     {label(f) for f in evidence},evidence_hash)
                responses[job['query_id'],arm] = result
                dump(DATA/fingerprint/stage/f"{job['query_id'].replace(':','-')}-{arm}.json", result)
            done += 1
            stage_state('running',stage,completed_queries=done,total_queries=total,
                        completed_outputs=len(responses),model=client.provider.model,run_id=fingerprint)
        failures = [r['status'] for r in responses.values() if r['status'] not in ('valid','abstain')]
        if failures:
            raise RuntimeError(f'{stage} has unresolved responses: {dict(Counter(failures))}')
    except Exception as exc:
        error = exc
        raise
    finally:
        # The offline label reader is constructed only after retrieval and API submission.
        if responses:
            evaluators = {key:OfflineEvaluation(key) for key in KEYS}
            rows = []
            for key, job in to_run:
                scores = context[key][2][job['fact_index'],job['direction']]
                if not any((job['query_id'], arm) in responses for arm in ARMS):
                    continue
                rows.append(evaluators[key].row(job,scores,'A0',dict(status='baseline'),'baseline'))
                for arm in ARMS:
                    result = responses.get((job['query_id'],arm))
                    if result is not None:
                        for fusion in ('direct','conservative'):
                            rows.append(evaluators[key].row(job,scores,arm,result,fusion))
                rows.append(evaluators[key].row(job,scores,'A4-structure-only',dict(status='baseline'),'baseline'))
            write_csv(RESULTS/f'{stage}_ranks.csv',rows)
            summary = aggregate(rows)
            dump(RESULTS/f'{stage}_summary.json',summary)
            write_csv(RESULTS/f'{stage}_api_metrics.csv',summary)
            if stage == 'pilot' or not (RESULTS/'pilot_summary.json').exists():
                write_csv(RESULTS/'api_metrics.csv', [dict(stage=stage,**r) for r in summary])
            cases(stage,rows,responses,jobs,context)
            if error is None and limit is None:
                timing = [r['seconds'] for r in responses.values() if r['seconds'] > 0]
                _, probe_job = to_run[0]
                before = client.ledger.usage()
                prompt = messages(probe_job['head'],probe_job['relation'],probe_job['aliases'],[])
                probe = client.rank(probe_job['query_id'],probe_job['aliases'],prompt,set(),identity([]))
                if not probe['cache_hit'] or before != client.ledger.usage():
                    raise ValueError('cache replay was not free')
                dump(RESULTS/f'{stage}_verified.json',dict(queries=done,outputs=len(responses),
                     median_request_seconds=median(timing) if timing else None,
                     mean_request_seconds=mean(timing) if timing else None,
                     all_responses_valid=True, cache_replay_free=True, run_id=fingerprint,
                     model=client.provider.model))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('retrieval','probe','smoke','pilot','all'),default='all')
    selected = parser.parse_args().stage
    stage = 'preflight'
    run_id = None
    try:
        stage_state('running',stage)
        if shutil.disk_usage(ROOT).free < 8*1024**3:
            raise RuntimeError('free disk space below 8 GiB')
        if not (RESULTS/'recovery/recovery.json').is_file() or len(read(RESULTS/'recovery/recovery.json')['records']) != 4:
            stage = 'asset_recovery'
            subprocess.run([sys.executable,'-m','src.recovery'], cwd=ROOT, check=True)
        recovery_report()
        stage = 'tests'
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,check=True)
        stage = 'retrieval'
        stage_state('running',stage)
        context = ranking_context()
        run_id, plans, jobs = retrieval_snapshot(context)
        info = stage_state('retrieval_complete',stage,completed_queries=sum(map(len,jobs.values())),run_id=run_id)
        report(info)
        if selected == 'retrieval':
            return
        stage = 'api_preflight'
        stage_state('running',stage,run_id=run_id)
        client = Client.open(ROOT/'.env.local',LEDGER)
        client.preflight()
        stage = 'smoke'
        prior = RESULTS/'smoke_verified.json'
        if selected == 'pilot' and (not prior.is_file() or read(prior)['run_id']!=run_id or
                                     read(prior)['model']!=client.provider.model):
            raise RuntimeError('the matching API smoke has not passed')
        if selected != 'pilot':
            stage_state('running',stage,run_id=run_id,completed_queries=0,total_queries=8)
            api_stage(stage,run_id,plans,jobs,context,client,limit=1 if selected=='probe' else None)
        if selected in ('probe','smoke'):
            info = stage_state('probe_complete' if selected=='probe' else 'smoke_complete',stage,run_id=run_id)
        else:
            stage = 'pilot'
            stage_state('running',stage,run_id=run_id,completed_queries=0,total_queries=48)
            api_stage(stage,run_id,plans,jobs,context,client)
            info = stage_state('complete',stage,run_id=run_id,completed_queries=48,total_queries=48)
        report(info)
    except Exception as exc:
        status = 'budget_exhausted' if isinstance(exc,BudgetExhausted) else 'failed'
        info = stage_state(status,stage,run_id=run_id,error=f'{type(exc).__name__}: {exc}')
        report(info)
        raise


if __name__ == '__main__':
    import fcntl
    RESULTS.mkdir(parents=True,exist_ok=True)
    with (RESULTS/'run.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
