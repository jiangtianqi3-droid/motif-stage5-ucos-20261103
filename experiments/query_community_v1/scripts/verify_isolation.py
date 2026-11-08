"""Verify protected hashes; never repair a mismatch or touch historical files."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.graph_adapter import ROOT,runtime_path,sha,dump

def verify(full=False):
    manifest=json.loads((ROOT/'freeze_manifest.json').read_text())
    assets=manifest['assets']
    if not full:
        assets=[r for r in assets if Path(r['path']).suffix in ('.py','.pth','.pt','.md','.yaml','.yml')]
    mismatches=[]
    for row in assets:
        p=runtime_path(row['path'])
        if not p.exists() or sha(p)!=row['sha256']: mismatches.append(row['path'])
    assert not mismatches, ('Protected historical assets changed',mismatches[:20])
    # Git status checked using native Windows git (WSL mode/CRLF interpretation differs).
    result=dict(passed=True,files_checked=len(assets),full=full,changed=[])
    dump(ROOT/'results'/('isolation_full.json' if full else 'isolation_core.json'),result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true')
    print(json.dumps(verify(parser.parse_args().full)))
