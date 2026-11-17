"""Resumable structural-only P2 task runner."""
import argparse
import csv
import fcntl
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from p2.assets import RESULTS, ROOT, GRAPHS, PAIR_GRAPHS, Target, config, preflight, verify
from p2.audit import run as run_audit
from p2.report import report
from p2.scoring import calibrate, transfer_graph
from p2.source import source_path_validation
from src.io_utils import dump, identity, read, sha, write_csv

TZ=timezone(timedelta(hours=8))
COPY_FILES=('propagation_trace.csv','propagation_summary.json','counterfactual_probe.csv','cases.json',
            'path_validation.csv','source_fusion_grid.csv','source_lock.json','path_summary.json',
            'source_mask_partition.json','scorer_ablation.csv','transfer_metrics.csv','per_query_rank_changes.csv')
SOURCE_FILES=('path_validation.csv','path_summary.json','source_mask_partition.json',
              'source_fusion_grid.csv','source_lock.json')


def source_identity(cfg):
    code={n:sha(ROOT/'p2'/n) for n in ('source.py','features.py','scoring.py','assets.py')}
    return identity(dict(config=cfg,code=code,p1=preflight(cfg)[1].name))


def atomic_copy(source,target):
    tmp=target.with_suffix(target.suffix+'.partial')
    shutil.copyfile(source,tmp)
    tmp.replace(target)


def reuse_source(folder,cfg):
    prep=RESULTS/'source_preflight'
    proof=prep/'reuse_manifest.json'
    if not proof.is_file():
        return False
    data=read(proof)
    approved=data.get('identity')==source_identity(cfg)
    if not approved:
        return False
    for name in SOURCE_FILES:
        verify(sha(prep/name)==data['files'][name],f'source preflight output changed: {name}')
        if not (folder/name).is_file():
            atomic_copy(prep/name,folder/name)
    return True


def mark_source(cfg):
    prep=RESULTS/'source_preflight'
    lock=read(prep/'source_lock.json')
    paths=read(prep/'path_summary.json')
    verify(lock['source_path_table_sha']==identity(paths['table']),'source lock mismatch')
    dump(prep/'reuse_manifest.json',dict(identity=source_identity(cfg),files={n:sha(prep/n) for n in SOURCE_FILES}))


def publish(folder):
    for n in COPY_FILES:
        if (folder/n).is_file():
            atomic_copy(folder/n,RESULTS/n)


def merge_rows(folder):
    path=folder/'per_query_rank_changes.csv'
    tmp=path.with_suffix('.partial')
    wrote=False
    with tmp.open('w',encoding='utf-8',newline='') as stream:
        for file in sorted((folder/'transfer').glob('*/*/per_query_rank_changes.csv')):
            with file.open(encoding='utf-8',newline='') as src:
                reader=csv.DictReader(src)
                if not wrote:
                    writer=csv.DictWriter(stream,reader.fieldnames)
                    writer.writeheader()
                    wrote=True
                writer.writerows(reader)
    tmp.replace(path)


def main(mode):
    cfg=config()
    fingerprint,p1root,prior_audit=preflight(cfg)
    folder=RESULTS/fingerprint
    folder.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    completed=[]
    phase='tests'
    peak=0
    def state(status,phase,done=0,total=0,**extras):
        value=dict(stage='P2',state=status,current_phase=phase,completed_queries=done,total_queries=total,
                   completed_experiments=completed,run_id=fingerprint,pid=os.getpid(),commit=commit,
                   last_update=datetime.now(TZ).isoformat(),llm_api_requests=0,gpu_peak_memory_mb=peak/1024**2,
                   frozen_checkpoint=prior_audit['checkpoint']['checkpoint_sha256'],
                   run_directory=str(folder),**extras)
        dump(RESULTS/'status.json',value)
        return value
    state('running',phase)
    try:
        if shutil.disk_usage(ROOT).free<8*1024**3:
            raise RuntimeError('free disk below 8 GiB')
        with (folder/'unit_tests.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],
                           cwd=ROOT,stderr=subprocess.STDOUT,stdout=log,check=True)
        phase='propagation_audit'
        state('running',phase)
        audit=run_audit(folder,p1root,cfg,mode,lambda name,n,t:state('running',name,n,t))
        completed.append(phase)
        publish(folder)
        report(folder,state('running',phase))
        if mode=='smoke':
            report(folder,state('smoke_complete','report'))
            return
        phase='source_path_validation'
        state('running',phase)
        reuse_source(folder,cfg)
        path=source_path_validation(folder,cfg)
        peak=max(peak,path['peak_allocated'])
        completed.append(phase)
        phase='source_calibration'
        state('running',phase)
        lock=calibrate(folder,p1root,path,cfg)
        verify(lock['source_path_table_sha']==identity(path['table']),'source-only lock table changed')
        completed.append(phase)
        publish(folder)
        report(folder,state('running',phase))
        results=[]
        for key in GRAPHS:
            view=Target(key,p1root,folder)
            work=[12,8,30] if key in PAIR_GRAPHS else [12]
            for k in work:
                phase=f'transfer/{key}/k{k}'
                state('running',phase,0,len(view.indices)*2 if k==12 else 100)
                stats=transfer_graph(view,folder,lock,path,cfg,k)
                results.append(stats)
                completed.append(phase)
                report(folder,state('running',phase,stats['queries'],stats['queries']))
        phase='report'
        metrics=[m for r in results for m in r['metrics']]
        write_csv(folder/'transfer_metrics.csv',metrics)
        write_csv(folder/'scorer_ablation.csv',[r for r in metrics if r['scorer'].startswith(('P2','agg','motif','relative'))])
        merge_rows(folder)
        publish(folder)
        positive=[r['dataset'] for r in metrics if r['scorer']=='P2_primary' and r['ci_b2'][0]>0]
        report(folder,state('complete',phase,transfer_graphs=4,
                            eleven_graphs='eligible_followup' if len(positive)>=2 else 'not_supported_by_four_graphs',
                            mlp='deferred_pending_fixed_scorer_evidence'))
    except BaseException as e:
        publish(folder)
        report(folder,state('failed',phase,error=f'{type(e).__name__}: {e}'))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('smoke','full','lock-preflight'),default='full')
    args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS/'run.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='lock-preflight':
            mark_source(config())
        else:
            main(args.mode)
