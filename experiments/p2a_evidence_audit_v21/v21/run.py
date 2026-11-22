"""Automatic V2.1 audit, smoke and formal frozen-backbone evaluation."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
from time import perf_counter
import argparse
import csv
import fcntl
import os
import resource
import shutil
import subprocess
import sys

import torch

from v21.common import ROOT,OUTPUT,config,read,dump,sha,identity,verify,write_csv
from v21.preflight import setup
from v21.graph import FactIndex
from v21.audit import collect
from v21.backbone import FrozenBackbone
from v21.evaluate import evaluate_queries
from v21.statistics import analyze,figures,make_report

TZ=timezone(timedelta(hours=8))


def publish(directory):
    for name in ('propagation','aggregation','sufficiency','necessity','figures'):
        source=directory/name
        if not source.is_dir():
            continue
        dest=OUTPUT/name
        dest.mkdir(parents=True,exist_ok=True)
        for path in source.iterdir():
            if path.is_file():
                tmp=(dest/path.name).with_suffix(path.suffix+'.partial')
                shutil.copyfile(path,tmp)
                tmp.replace(dest/path.name)
    for name in ('statistics.json','smoke_verified.json','timing.json'):
        path=directory/name
        if path.is_file():
            tmp=(OUTPUT/name).with_suffix('.partial')
            shutil.copyfile(path,tmp)
            tmp.replace(OUTPUT/name)


def main(smoke_only=False):
    cfg=config()
    directory,view,selection,demands,provenance=setup(cfg)
    start=perf_counter()
    done=[]
    phase='unit_tests'
    backbone=None
    info=None
    index=FactIndex(view.graph)
    table=read(view.dir.parents[2]/'source_calibration.json')['table']
    def state(status,phase,completed=0,total=None,**extra):
        value=dict(stage='P2-A Stage 1 V2.1',state=status,current_phase=phase,
             run_id=directory.name,run_directory=str(directory),pid=os.getpid(),
             completed_queries=completed,total_queries=cfg['experiment']['query_count'] if total is None else total,
             last_update=datetime.now(TZ).isoformat(),llm_api_requests=0,
             completed_experiments=done,git_commit=provenance['git_commit'],
             runtime_seconds=perf_counter()-start,**extra)
        dump(OUTPUT/'status.json',value)
        return value
    state('running',phase)
    try:
        if shutil.disk_usage(ROOT).free<8*1024**3:
            raise RuntimeError('disk space below 8 GiB')
        with (directory/'unit_tests.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],stdout=log,
                            stderr=subprocess.STDOUT,cwd=ROOT,check=True)
        done.append('unit_tests')
        phase='frozen_model_precompute'
        state('running',phase)
        torch.cuda.set_per_process_memory_fraction(min(1.,cfg['hardware']['max_gpu_memory_gb']*1024**3/
                                                      torch.cuda.get_device_properties(0).total_memory))
        backbone=FrozenBackbone(view.asset,demands,directory,cfg)
        done.append('strict_checkpoint_load_and_full_graph_freeze')
        smoke=selection[:cfg['experiment']['smoke_queries']]
        smoke_proof=directory/'smoke_verified.json'
        if not smoke_proof.is_file():
            began=perf_counter()
            phase='smoke_propagation'
            state('running',phase,0,len(smoke))
            selections,trace,agg=collect(view,smoke,index,cfg,directory,table)
            phase='smoke_fidelity'
            state('running',phase,0,len(smoke))
            records,timing=evaluate_queries(backbone,view,index,smoke,selections,directory,cfg,
                  lambda n,t:state('running',phase,n,t))
            info=backbone.close()
            elapsed=perf_counter()-began
            rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            estimate=(cfg['experiment']['query_count']-len(smoke))*elapsed/len(smoke)
            proof=dict(queries=len(smoke),all_checks_passed=True,seconds=elapsed,
                    estimated_remaining_seconds=estimate,forward_calls=info['forward_calls'],
                    gpu_peak_allocated=info['peak_allocated'],gpu_peak_reserved=info['peak_reserved'],
                    cpu_peak_rss_bytes=rss_bytes,full_model_parameter_hash=info['parameter_sha_after'],
                    relation_counterfactual_forward_calls=info['counterfactual_relation_encoder_calls'],
                    full_dstar_fixed=info['full_statistics_fixed'],entity_space=view.graph.n,
                    input_hash=backbone.static_fingerprint,selection_sha256=identity(smoke))
            dump(smoke_proof,proof)
            if smoke_only:
                stats=analyze(directory,records,cfg)
                figures(directory,stats,trace,agg,cfg)
                phase='smoke_complete'
                done.append(phase)
                value=state('smoke_complete',phase,len(smoke),len(smoke),gpu_peak_memory_mb=info['peak_allocated']/1024**2)
                make_report(directory,stats,records,trace,agg,cfg,value,proof)
                publish(directory)
                print(f'Smoke passed: {elapsed:.1f}s, remaining estimated {estimate:.0f}s',flush=True)
                return
        else:
            proof=read(smoke_proof)
            verify(proof['all_checks_passed'] and proof['input_hash']==backbone.static_fingerprint
                   and proof['selection_sha256']==identity(smoke),'smoke cache identity mismatch')
            if smoke_only:
                info=backbone.close()
                state('smoke_complete','smoke_cache_verified',len(smoke),len(smoke),
                      gpu_peak_memory_mb=info['peak_allocated']/1024**2)
                publish(directory)
                return
        done.append('10_query_smoke')
        phase='formal_propagation_aggregation'
        state('running',phase)
        selections,trace,agg=collect(view,selection,index,cfg,directory,table)
        done.append(phase)
        publish(directory)
        phase='formal_fidelity'
        state('running',phase)
        records,timing=evaluate_queries(backbone,view,index,selection,selections,directory,cfg,
             lambda n,t:state('running',phase,n,t))
        info=backbone.close()
        done.append('sufficiency_and_matched_necessity')
        dump(directory/'timing.json',dict(timing,seconds_total_current_process=perf_counter()-start,
             cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        phase='statistics_and_report'
        state('running',phase,len(records))
        stats=analyze(directory,records,cfg)
        figures(directory,stats,trace,agg,cfg)
        done.append('10000_bootstrap_and_seven_figures')
        value=state('complete','report',len(records),gpu_peak_memory_mb=info['peak_allocated']/1024**2,
                    gpu_peak_reserved_mb=info['peak_reserved']/1024**2)
        decision=make_report(directory,stats,records,trace,agg,cfg,value,proof)
        publish(directory)
        dump(directory/'complete.json',dict(state=value,decision=decision,files={str(p.relative_to(directory)):sha(p)
             for p in sorted(directory.rglob('*')) if p.is_file() and p.suffix in ('.csv','.json','.npz')
             and p.name!='complete.json'}))
        state('complete','report',len(records),decision=decision,gpu_peak_memory_mb=info['peak_allocated']/1024**2)
        print(f'Completed {len(records)} queries. Decision {decision}',flush=True)
    except BaseException as exc:
        if backbone is not None:
            try:
                info=backbone.close()
            except Exception:
                pass
        state('failed',phase,error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(args.smoke_only)
