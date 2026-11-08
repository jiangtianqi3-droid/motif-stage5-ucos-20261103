"""One-run local supervisor. Never changes or terminates a research process.

Kept outside scripts/*.py deliberately: adding supervision must not change the
running experiment fingerprint or make its existing query caches unreachable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone

ROOT=Path(__file__).resolve().parents[2]
ZONE=timezone(timedelta(hours=8))

def now(): return datetime.now(ZONE).isoformat()

def read(path,default=None):
    try: return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError: return default

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_matches(protocol,root=ROOT):
    mismatches=[name for name,value in protocol['code'].items() if not (root/name).exists() or sha(root/name)!=value]
    if sha(root/'freeze_manifest.json')!=protocol['freeze_manifest_sha256']:
        mismatches.append('freeze_manifest.json')
    return mismatches

def process_info(pid,root=ROOT):
    proc=Path('/proc')/str(pid)
    try:
        args=(proc/'cmdline').read_bytes().split(b'\0')
        stat=(proc/'stat').read_text().split(') ',1)[1].split()
        if stat[0]=='Z': return None
        if b'scripts/run_community_experiments.py' not in args or (proc/'cwd').resolve()!=root.resolve():
            return None
        return dict(pid=pid,start_ticks=int(stat[19]),cpu_ticks=int(stat[11])+int(stat[12]))
    except (FileNotFoundError,PermissionError,ProcessLookupError): return None

def classify(alive,status,complete,free_bytes,quiet_seconds,restarts,max_restarts,changed):
    if complete: return 'complete'
    if alive:
        return 'quiet' if quiet_seconds>=1800 else 'healthy'
    if changed: return 'needs_attention_source_changed'
    if free_bytes<8*1024**3: return 'needs_attention_disk'
    text=(str(status.get('error',''))+'\n'+str(status.get('traceback',''))).lower()
    fatal=['assertionerror','valueerror','typeerror','keyerror','modulenotfounderror',
           'zerodivisionerror','hidden target','hidden inverse','protected historical assets',
           'less than 8 gib','another experiment is active','fingerprint','signature mismatch']
    if status.get('state')=='failed' and any(token in text for token in fatal):
        return 'needs_attention_deterministic_failure'
    transient=['outofmemory','out of memory','cuda error','input/output error','errno 5',
               'temporarily unavailable','errno 11','interrupted system call','errno 4']
    if status.get('state')=='failed' and not any(token in text for token in transient):
        return 'needs_attention_unknown_failure'
    return 'restart' if restarts<max_restarts else 'needs_attention_restart_limit'

def notify(title,body,no_notify=False):
    if no_notify: return dict(notification='disabled')
    script=ROOT/'scripts/monitor/notify.ps1'
    windows='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    try:
        path=subprocess.check_output(['wslpath','-w',str(script)],text=True).strip()
        result=subprocess.run([windows,'-NoProfile','-NonInteractive','-WindowStyle','Hidden',
            '-File',path,'-Title',title,'-Body',body],capture_output=True,text=True,timeout=25)
        return dict(notification='queued' if result.returncode==0 else 'failed',
                    notification_detail=(result.stdout+result.stderr).strip()[-1500:])
    except Exception as exc: return dict(notification='failed',notification_detail=str(exc))

def run_monitor(args):
    run=ROOT/'results/runs'/args.run
    protocol=read(run/'protocol.json')
    assert protocol is not None and protocol['smoke'] is False and protocol['limit'] is None
    assert not source_matches(protocol),'Running experiment source changed; refuse a new supervisor'
    out=ROOT/'results/monitor'/args.run;out.mkdir(parents=True,exist_ok=True)
    lock=out/'watch.lock'
    if lock.exists():
        owner=int(lock.read_text())
        try: os.kill(owner,0)
        except ProcessLookupError: lock.unlink()
        else: raise RuntimeError(f'Supervisor already active: {owner}')
    with lock.open('x') as f:f.write(str(os.getpid()))
    config=dict(run=args.run,interval_seconds=args.interval,max_restarts=args.max_restarts,
        experiment_pid=args.pid,watcher_pid=os.getpid(),started_at=now(),
        monitor_sha256=sha(Path(__file__)),restart_policy='Only unexpected death or explicitly transient errors; never terminate live process',
        source_identity_preserved=True)
    dump(out/'config.json',config)
    pid=args.pid;initial=process_info(pid);expected_start=initial['start_ticks'] if initial else None
    restarts=0;previous=None;last_alert=None
    if not args.once:
        notify('MOTIF 实验看护已启用',f'每{args.interval//60}分钟检查进度与进程；临时退出最多断点续跑2次。',args.no_notify)
    try:
        while True:
            status=read(ROOT/'results/status.json',{})
            if status.get('run') and Path(status['run']).resolve()!=run.resolve():
                action='needs_attention_other_run'
                info=None;quiet=0;changed=[]
            else:
                info=process_info(pid)
                if info and expected_start is not None and info['start_ticks']!=expected_start: info=None
                # Adopt a manually resumed copy of this same run, never duplicate it.
                if info is None and status.get('pid') and status['pid']!=pid:
                    candidate=process_info(int(status['pid']))
                    if candidate:
                        pid=int(status['pid']);info=candidate;expected_start=candidate['start_ticks']
                if info and expected_start is None: expected_start=info['start_ticks']
                stamp=status.get('at')
                quiet=max(0,time.time()-datetime.fromisoformat(stamp).timestamp()) if stamp else float(args.interval)
                complete=read(run/'completion.json',{}).get('state')=='complete'
                changed=source_matches(protocol) if info is None else []
                action=classify(info is not None,status,complete,shutil.disk_usage(ROOT).free,
                                quiet,restarts,args.max_restarts,changed)
            value=dict(at=now(),watcher_pid=os.getpid(),experiment_pid=pid,run=args.run,action=action,
                process=info,quiet_seconds=quiet,restarts=restarts,interval_seconds=args.interval,
                next_check_at=(datetime.now(ZONE)+timedelta(seconds=args.interval)).isoformat(),
                progress={k:status.get(k) for k in ['at','state','stage','dataset','done','total','error']},
                source_mismatches=changed,free_bytes=shutil.disk_usage(ROOT).free)
            if action=='quiet':
                # A long phase may be legitimate. Report it, never kill or duplicate it.
                value['cpu_advanced']=bool(info and previous and previous.get('process') and
                    info['cpu_ticks']>previous['process']['cpu_ticks'])
            if action=='restart' and not args.once:
                log=ROOT/'logs'/f'recovery-{datetime.now(ZONE).strftime("%Y%m%dT%H%M%S")}.log'
                with log.open('a') as f:
                    child=subprocess.Popen(['bash',str(ROOT/'run.sh'),'scripts/run_community_experiments.py'],
                        cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                pid=child.pid;restarts+=1
                value.update(action='restarted',new_pid=pid,restarts=restarts,recovery_log=str(log))
                (ROOT/'logs/full.pid').write_text(str(pid)+'\n')
                (ROOT/'logs/full.logpath').write_text(str(log)+'\n')
                # Popen has execed bash; the same PID will exec the experiment.
                expected_start=None
                value.update(notify('MOTIF 实验已断点续跑',f'检测到临时退出，第{restarts}次恢复；已完成查询缓存继续复用。',args.no_notify))
            elif action=='complete':
                value.update(notify('MOTIF 实验已完成','四图诊断和正式报告已生成，本次看护自动结束。',args.no_notify))
            elif (action.startswith('needs_attention') or action=='quiet') and action!=last_alert:
                value.update(notify('MOTIF 实验需要检查',f'{action}；日志与断点已保留，未终止任何活跃实验。',args.no_notify))
                last_alert=action
            elif action=='healthy': last_alert=None
            dump(out/'status.json',value)
            with (out/'checks.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(value,ensure_ascii=False)+'\n')
            print(json.dumps(value,ensure_ascii=False),flush=True)
            previous=value
            if args.once or action=='complete' or action.startswith('needs_attention'): break
            end=time.monotonic()+args.interval
            while time.monotonic()<end:
                if (out/'STOP').exists():
                    dump(out/'stopped.json',dict(at=now(),reason='User stop marker; experiment was left untouched'))
                    return
                time.sleep(min(30,max(0,end-time.monotonic())))
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--pid',required=True,type=int)
    parser.add_argument('--interval',type=int,default=3600)
    parser.add_argument('--max-restarts',type=int,default=2)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--no-notify',action='store_true')
    args=parser.parse_args()
    assert args.interval>=60 and 0<=args.max_restarts<=3
    run_monitor(args)
