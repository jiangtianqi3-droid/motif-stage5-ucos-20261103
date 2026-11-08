"""Inventory historical assets without importing their code or modifying them."""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY = Path('D:/motif-phase4-work')
REPOS = [Path('D:/ultra-phase1-work/ULTRA'), Path('D:/motif-phase2-work/MOTIF'),
         Path('D:/motif-phase3-work/MOTIF'), HISTORY/'github-archives/motif-stage5-ucos-20261103',
         HISTORY/'motif-stage6-entity-degree-p0p1', HISTORY/'motif-stage6-entity-degree-p0p1-v2',
         HISTORY/'tsdc-stage6-20261103']

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])

def main():
    for name in ['config','src','tests','scripts','cache','results','figures','logs','preserved_changes']:
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    dest = ROOT/'freeze_manifest.json'
    if dest.exists():
        raise RuntimeError('Freeze manifest already exists; use verify, never overwrite')
    repositories = []
    for i, repo in enumerate(REPOS):
        status = git(repo, 'status','--porcelain=v1').decode()
        rec = dict(path=str(repo), branch=git(repo,'branch','--show-current').decode().strip(),
                   commit=git(repo,'rev-parse','HEAD').decode().strip(), clean=not status, status=status)
        if status:
            save = ROOT/'preserved_changes'/str(i)
            save.mkdir()
            (save/'working.patch').write_bytes(git(repo,'diff','--binary'))
            (save/'staged.patch').write_bytes(git(repo,'diff','--cached','--binary'))
            for rel in git(repo,'ls-files','--others','--exclude-standard','-z').decode().split('\0'):
                if rel and '.ai-handoff' not in Path(rel).parts:
                    target = save/'untracked'/rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(repo/rel, target)
            rec['preserved_at'] = str(save)
        repositories.append(rec)
    skip = {'.git','.ai-handoff','__pycache__','triton_cache','torch_extensions','github-archives'}
    assets = []
    roots = [HISTORY, Path('D:/motif-phase3-work'), Path('D:/motif-phase2-work'), Path('D:/ultra-phase1-work')]
    for base in roots:
        for directory, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in skip and d != ROOT.parents[1].name)
            for name in sorted(files):
                p = Path(directory)/name
                if p.suffix.lower() not in {'.py','.md','.json','.yaml','.yml','.sh','.pth','.pt','.npz','.npy','.csv','.txt'}:
                    continue
                # Preserve all research data/results/checkpoints; derived compiler caches excluded.
                assets.append(dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)))
    base = REPOS[-1]
    payload = dict(created_at=datetime.now(timezone.utc).isoformat(), baseline_commit=git(base,'rev-parse','HEAD').decode().strip(),
        baseline_branch=git(base,'branch','--show-current').decode().strip(), baseline_clean=not git(base,'status','--porcelain'),
        tag='baseline-4d-gate-frozen-20261108', branch='experiment/query-community-v1',
        worktree=str(ROOT.parents[1]), repositories=repositories, assets=assets,
        running_experiments=False, process_audit='Windows CIM + WSL ps inspected before freeze; no research processes running',
        excluded=['compiler caches','.git','.ai-handoff','github archive duplicate'],
        environment=dict(python=sys.version, platform=platform.platform()),
        protocol='Existing frozen observed graphs only; candidate rankings are unfiltered B0 reciprocal ranks, no truth masking at retrieval; targets used only by offline evaluator; both directions; seed 1024')
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(assets=len(assets),bytes=sum(x['bytes'] for x in assets),baseline=payload['baseline_commit']),indent=2))

if __name__ == '__main__':
    main()
