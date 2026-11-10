"""Resumable route A runner. Each graph/model has a fresh bounded-memory process."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.util import *

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--skip-llm',action='store_true')
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    parser.add_argument('--families',nargs='+',choices=['B0','T-4D'],default=['B0','T-4D'])
    parser.add_argument('--worker',nargs=2,metavar=('DATASET','FAMILY'));parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if args.worker:
        assert source_identity()==read(args.run/'protocol.json')['source'],'Source changed after experiment fingerprint was frozen'
        from src.pipeline import worker
        worker(*args.worker,args.run,args.smoke,args.skip_llm);return
    payload=dict(schema=1,source=source_identity(),smoke=args.smoke,skip_llm=args.skip_llm,datasets=args.datasets,families=args.families,
                 seed=1024,validation_max_facts=256,bootstrap=2000,filter_protocol='answer isolation; all positives filtered offline only',
                 llm_model=os.environ.get('LLM_MODEL_NAME','deepseek-flash'),llm_calibration_max_queries=64)
    fingerprint=identity(payload)[:20]
    run=ROOT/'results/runs'/fingerprint;run.mkdir(parents=True,exist_ok=True)
    lock=ROOT/'logs'/f'{fingerprint}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    if lock.exists():
        try: os.kill(int(lock.read_text()),0)
        except ProcessLookupError: lock.unlink()
        else: raise RuntimeError('This exact experiment is already active')
    lock.write_text(str(os.getpid()))
    try:
        dump(run/'protocol.json',payload)
        snap=run/'historical_hashes.json'
        if not snap.exists(): dump(snap,frozen_snapshot())
        verify_snapshot(read(snap))
        if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Less than 8 GiB free; existing assets preserved')
        status(run,state='starting',stage='preflight',fingerprint=fingerprint)
        if not args.skip_llm:
            from src.smoke import api_smoke
            api_smoke(run)
        for key in args.datasets:
            for family in args.families:
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',key,family,'--run',str(run)]
                if args.smoke:command.append('--smoke')
                if args.skip_llm:command.append('--skip-llm')
                subprocess.run(command,check=True)
        assert source_identity()==payload['source'],'Source changed during experiment'
        verification=verify_snapshot(read(snap));dump(run/'historical_verification.json',verification)
        from src.report import report
        report(run,args.datasets,args.families,args.smoke)
        dump(run/'completion.json',dict(state='smoke_complete' if args.smoke else 'complete',at=now(),historical_verification=verification))
        status(run,state='smoke_complete' if args.smoke else 'complete',stage='finished',report=str(run/'report_route_a_rerank.md'))
    except BaseException as exc:
        status(run,state='failed',stage='exception',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
