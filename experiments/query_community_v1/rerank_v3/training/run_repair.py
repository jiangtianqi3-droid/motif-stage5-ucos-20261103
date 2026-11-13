"""Fixed historical architecture, three seeds; source labels isolated."""
import json
import os
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
from common import HISTORY,dump,sha
from strict_data import prepare

def main():
    prepare(ROOT,HISTORY)
    import torch
    torch.set_num_threads(4)
    from train import run
    cfg=dict(alpha=.25,freedom='gate_only',method='topology',semantic=False,shuffle=False)
    rows={}
    for seed in [1024,2026,3407]:
        rows[str(seed)]=run('T4D_strict',cfg,seed)
        dump(ROOT/'training_progress.json',dict(completed_seeds=list(rows),state='running'))
    dump(ROOT/'frozen.json',dict(formal={'T-4D':rows},protocol_sha256=sha(ROOT/'protocol.json'),
         strict_manifest_sha256=sha(ROOT/'data/strict_manifest.json'),source_test_used=False,
         architecture_selection='historical fixed config, no new architecture search',
         boundary='repair replication on historically exposed targets; not a blind test'))
    dump(ROOT/'training_progress.json',dict(completed_seeds=list(rows),state='complete'))

if __name__=='__main__':main()
