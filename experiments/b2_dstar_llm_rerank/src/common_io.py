"""Experiment-owned files only; never mutate historical assets."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
HISTORY = Path(os.environ.get('MOTIF_HISTORY', '/mnt/d/motif-phase4-work'))
S7 = HISTORY / 'motif-stage7-asymmetric-20261106'
OLD11 = ROOT.parent / 'universal_11g_v1'
M5RUN = HISTORY / 'conceptnet-zeroshot-boost/experiments/universal_11g_v1/results/runs/5443b773ab1db4353493'
SEEDS = [1024, 2026, 3407]
MODES = ['LLM', 'Paths+LLM', 'Paths+Motifs+LLM', 'Community+Paths+Motifs+LLM']

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def dump(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, p)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def save_array(p, a):
    import numpy as np
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    with tmp.open('wb') as f: np.save(f, a)
    os.replace(tmp, p)
