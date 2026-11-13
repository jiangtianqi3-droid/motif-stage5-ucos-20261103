"""Immutable, resumable repair replication. All output stays in the new directory."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.util import *

FAMILIES=['B0','T-4D-s1024','T-4D-s2026','T-4D-s3407']

def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    p.add_argument('--families',nargs='+',choices=FAMILIES,default=FAMILIES)
    p.add_argument('--skip-llm',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--worker',nargs=2);p.add_argument('--run',type=Path);args=p.parse_args()
    if args.worker:
        protocol=read(args.run/'protocol.json')
        if source_identity()!=protocol['source']:raise RuntimeError('Code changed after freeze')
        from src.pipeline import worker
        worker(*args.worker,args.run,args.smoke,args.skip_llm);return
    frozen=STAGE5/'frozen.json'
    if not frozen.exists():raise RuntimeError('Strict three-seed training must finish before evaluation')
    if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Insufficient free space for independent replication')
    # Both code and dependency identities participate in the run key.
    dependencies=frozen_snapshot()
    runtime=HISTORY/'topology-ucos-stage5-20261103/runtime'
    dependencies.update({str(p):sha(p) for p in runtime.rglob('*.py')})
    dependencies.update({str(p):sha(p) for p in STAGE5.glob('*.py')})
    dependencies[str(STAGE5/'data/strict_manifest.json')]=sha(STAGE5/'data/strict_manifest.json')
    for rec in read(frozen)['formal']['T-4D'].values():dependencies[str(STAGE5/rec['checkpoint'])]=sha(STAGE5/rec['checkpoint'])
    protocol=dict(schema=3,source=source_identity(),dependency_fingerprint=identity(dependencies),training_frozen_sha256=sha(frozen),
        datasets=args.datasets,families=args.families,smoke=args.smoke,skip_llm=args.skip_llm,
        seed=20261112,validation_max_facts_per_role=512,structure_grid_strengths=[0.,.15,.3],llm_grid_strengths=[0.,.05],
        validation_partition='fact components connected by shared public queries; no query or fact crossover',
        selection='unique-query macro MRR paired one-SE simplicity rule',llm_min_unique_queries=32,llm_max_unique_queries=64,
        primary_comparisons=['R4_vs_R1','R4_vs_R3','R5_vs_R4'],primary_metric='MRR',guardrail='Hits10',
        multiplicity='Holm across all reported primary graph/family/comparison cells',
        boundary='historically exposed repair replication; no first-blind-test or broad generalization claim')
    run=ROOT/'results/runs'/identity(protocol)[:20];run.mkdir(parents=True,exist_ok=True)
    path=run/'protocol.json'
    if path.exists() and read(path)!=protocol:raise RuntimeError('Protocol mismatch')
    dump(path,protocol);dump(run/'dependency_hashes.json',dependencies)
    dump(ROOT/'results/active_run.json',dict(run=str(run),protocol_sha256=sha(path)))
    lock=ROOT/'logs'/f'{run.name}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    if lock.exists():
        try:os.kill(int(lock.read_text()),0)
        except ProcessLookupError:lock.unlink()
        else:raise RuntimeError('This experiment is already running')
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    try:
        for key in args.datasets:
            for family in args.families:
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',key,family,'--run',str(run)]
                if args.skip_llm:command.append('--skip-llm')
                if args.smoke:command.append('--smoke')
                subprocess.run(command,check=True)
        verify_snapshot(dependencies)
        if source_identity()!=protocol['source']:raise RuntimeError('Source changed during evaluation')
        from src.repair_report import report
        report(run,args.datasets,args.families,args.smoke)
        dump(run/'completion.json',dict(state='smoke_complete' if args.smoke else 'complete',at=now(),dependencies_unchanged=True))
        status(run,state='smoke_complete' if args.smoke else 'complete',stage='finished',report=str(run/'report.md'))
    except BaseException as exc:
        status(run,state='failed',stage='exception',error=str(exc));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
