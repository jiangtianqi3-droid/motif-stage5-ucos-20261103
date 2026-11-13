"""Run validation, training, evaluation and verification sequentially without polling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone,timedelta

ROOT=Path(__file__).resolve().parent
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def snapshot():
    paths=list((ROOT/'code/src').glob('*.py'))+list((ROOT/'code/scripts').glob('*.py'))+list((ROOT/'code/tests').glob('*.py'))+list((ROOT/'training').glob('*.py'))
    paths+=[ROOT/'training/protocol.json',ROOT/'workflow.py',ROOT/'verify_run.py',ROOT/'run_all.sh',ROOT/'run_training.sh',ROOT/'code/run.sh']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');parser.add_argument('--skip-llm',action='store_true');args=parser.parse_args()
    code=snapshot();plan=dict(code_hashes=code,skip_llm=args.skip_llm,seeds=[1024,2026,3407],
        steps=['tests','cached_integration','train','training_verification','gpu_smoke','full_evaluation','final_verification'],
        boundary='historically exposed repair replication, new blind data still required')
    fingerprint=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:16]
    write(ROOT/'results'/f'workflow_plan_{fingerprint}.json',plan)
    state=dict(workflow=fingerprint,pid=os.getpid(),started=now(),completed_steps=[],check_only=args.check_only)
    def stage(name,command,cwd=None):
        if snapshot()!=code:raise RuntimeError('Source changed during workflow; stop instead of mixing versions')
        log=ROOT/'logs'/f'{fingerprint}_{name}.log'
        state.update(state='running',stage=name,updated=now(),log=str(log));write(ROOT/'workflow_status.json',state)
        with log.open('a',encoding='utf-8') as output:
            output.write(f'\n--- {now()} start {name} ---\n');output.flush()
            subprocess.run(command,cwd=cwd or ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
        state['completed_steps'].append(name);state['updated']=now();write(ROOT/'workflow_status.json',state)
    try:
        if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('At least 8 GiB free space required on output drive')
        stage('tests',[sys.executable,'-B','-m','pytest','tests/test_repair.py','tests/test_route_a.py','-q','-s','-p','no:cacheprovider',
                       '--rootdir=.','--confcutdir=.','--basetemp',str(ROOT/'cache/pytest-workflow')],ROOT/'code')
        stage('cached_integration',[sys.executable,'-B',str(ROOT/'code/scripts/check_cached_pipeline.py')])
        if args.check_only:
            state.update(state='validated',stage='code_checks_complete',updated=now());write(ROOT/'workflow_status.json',state);return
        stage('train',['bash',str(ROOT/'run_training.sh')])
        stage('training_verification',[sys.executable,'-B',str(ROOT/'verify_run.py'),'--training-only'])
        stage('gpu_smoke',['bash',str(ROOT/'code/run.sh'),'scripts/run_repair.py','--smoke','--skip-llm',
                           '--datasets','NELLInductive_v1','--families','B0','T-4D-s1024'])
        command=['bash',str(ROOT/'code/run.sh'),'scripts/run_repair.py']
        if args.skip_llm:command.append('--skip-llm')
        stage('full_evaluation',command)
        stage('final_verification',[sys.executable,'-B',str(ROOT/'verify_run.py')])
        if snapshot()!=code:raise RuntimeError('Code changed before completion')
        state.update(state='complete',stage='verified',updated=now(),report=str(ROOT/'report.md'),
                     limitations=['small existing test sets','historically exposed targets; new independent blind data still required'])
        write(ROOT/'workflow_status.json',state);write(ROOT/'repair_completion.json',state)
    except BaseException as exc:
        state.update(state='failed',updated=now(),error=f'{type(exc).__name__}: {exc}')
        write(ROOT/'workflow_status.json',state)
        raise

if __name__=='__main__':main()
