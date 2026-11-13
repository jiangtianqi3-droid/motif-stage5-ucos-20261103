"""Isolated runtime, bounded memory and atomic disk checkpoints."""
import gc
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
HISTORY = Path('/mnt/d/motif-phase4-work' if os.name != 'nt' else 'D:/motif-phase4-work')
ROOT = Path(os.environ.get('ROUTE_A_OUTPUT', str(HISTORY / 'route-a-repair-20261112')))
OLD = HISTORY / 'query-community-v1-20261108/experiments/query_community_v1'
STAGE5 = ROOT / 'training'
DATA = HISTORY / 'motif-stage6-entity-degree-p0p1-v2/data'
DATASETS = ['FB15k237Inductive_v1', 'WN18RRInductive_v1', 'NELLInductive_v1', 'FBIngram_25']
METHODS = ['R1', 'R2', 'R3', 'R4']
RSS_LIMIT = 1_800_000_000
sys.dont_write_bytecode = True

def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def peak_memory():
    import psutil
    info=psutil.Process().memory_info()
    if os.name=='nt': return getattr(info,'peak_wset',info.rss)
    import resource
    return max(info.rss,int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))

def memory(enforce=True):
    import psutil
    rss = psutil.Process().memory_info().rss
    peak=peak_memory()
    if enforce and peak > RSS_LIMIT: raise MemoryError(f'Peak RSS {peak} exceeds {RSS_LIMIT}; checkpoint retained')
    return rss

def cleanup():
    gc.collect()
    if 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def status(run, **kw):
    value = dict(at=now(), pid=os.getpid(), rss_bytes=memory(enforce=kw.get('state')!='failed'),peak_rss_bytes=peak_memory(), run=str(run), **kw)
    dump(run / 'status.json', value)
    dump(ROOT / 'results/rerank_status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)

def historical_package():
    if 'qc_v1' not in sys.modules:
        spec = importlib.util.spec_from_file_location('qc_v1', OLD / 'src/__init__.py', submodule_search_locations=[str(OLD / 'src')])
        module = importlib.util.module_from_spec(spec); sys.modules['qc_v1'] = module
        spec.loader.exec_module(module)
    from qc_v1 import graph_adapter
    # Historical adapter's optional signature-audit writes are redirected here.
    graph_adapter.ROOT = ROOT
    return graph_adapter

def model_environment():
    for path in [HISTORY / 'topology-ucos-stage5-20261103/runtime', STAGE5]:
        if str(path) not in sys.path: sys.path.insert(0, str(path))

def source_identity():
    return {str(p.relative_to(SOURCE_ROOT)): sha(p) for p in sorted([*SOURCE_ROOT.glob('src/*.py'), *SOURCE_ROOT.glob('scripts/*.py'), SOURCE_ROOT / 'run.sh'])}

def frozen_snapshot():
    paths = [*OLD.glob('src/*.py'), *OLD.glob('scripts/*.py'), *OLD.glob('config/*'), OLD / 'run.sh', OLD / 'report.md', OLD / 'freeze_manifest.json']
    run = OLD / 'results/runs/d48ae432a2533cdaaec0'
    paths += [*run.glob('*.json'), *run.glob('*.md'), *run.glob('*/*.json')]
    frozen = read(STAGE5 / 'frozen.json')['formal']['T-4D']['1024']
    paths += [STAGE5 / 'frozen.json', STAGE5 / frozen['checkpoint'], HISTORY.parent / 'motif-phase3-work/MOTIF/ckpts/motif_3g.pth']
    paths += [*STAGE5.glob('*.py'),*STAGE5.glob('runtime/motif/*.py'),*STAGE5.glob('runtime/motif/rspmm/*.py')]
    semantic=HISTORY/'topology-semantic-stage4-20261102/semantic'
    paths += [semantic/'relation_mappings.json',semantic/'target_mapping_provenance.json']
    provenance=read(semantic/'target_mapping_provenance.json')
    adapter=historical_package()
    for key in DATASETS:
        paths += [DATA/f'{key}.npz',HISTORY/'tsdc-stage6-20261103/stage6/observations'/f'{key}.npz']
        paths += list(adapter.runtime_path(provenance[key]['raw']).glob('*.txt'))
    paths += list((HISTORY/'topology-semantic-stage4-20261102/cache/downstream').glob('*.npz'))
    return {str(p): sha(p) for p in paths if p.is_file()}

def verify_snapshot(snapshot):
    changed = [p for p, h in snapshot.items() if not Path(p).is_file() or sha(p) != h]
    if changed: raise RuntimeError(f'Historical assets changed: {changed}')
    return dict(passed=True, files=len(snapshot), changed=[])
