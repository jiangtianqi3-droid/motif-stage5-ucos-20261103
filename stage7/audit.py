import csv
from pathlib import Path
from stage7.util import ROOT, read, dump, sha, log


def run(args):
    s6 = ROOT.parent/'stage6'
    manifest = read(s6/'ARTIFACT_MANIFEST.json')
    snapshot = {}
    for p in s6.rglob('*'):
        if p.is_file() and not set(p.relative_to(s6).parts) & {'cache','triton_cache','torch_extensions','__pycache__'}:
            snapshot[str(p.relative_to(ROOT.parent))] = sha(p)
    # Snapshot all Stage5 tracked files, not .git or machine-generated caches.
    import subprocess
    files = subprocess.check_output(['git','-C',str(ROOT.parent),'ls-files','-z']).decode().split('\0')
    for name in files:
        if name:
            snapshot[name] = sha(ROOT.parent/name)
    dump(ROOT/'preserved_assets.json', snapshot, exclusive=True)
    names = ['proxy_formal_correlation','selected_topology_prediction','source_density_metrics',
             'source_density_sensitivity','selection_stability','ablation','factorial_interaction']
    tables = {}
    for name in names:
        p = s6/'results'/f'{name}.csv'
        rows = list(csv.DictReader(p.open(encoding='utf-8-sig')))
        tables[name] = dict(sha256=sha(p), rows=len(rows), columns=list(rows[0]) if rows else [])
    tables['verification'] = read(s6/'results/verification.json')
    dump(ROOT/'results/stage6_audit.json', dict(inputs=tables,
         stage6_frozen_sha256=sha(s6/'frozen.json'), preserved_files=len(snapshot),
         mechanisms={
           'fixed_cap':'min(256,floor(.15*E)) gives ConceptNet 256/99997=.256%; NELLv1 124/833=14.886%',
           'constraints':'Random spanning forest, relation coverage and sequential endpoint residual >=2 shrink sparse graphs further.',
           'distribution':'Round-robin 2x2 bins imposes equal bucket mass rather than natural relation/degree distribution. Forest protection and deletion constraints favor cyclic/high-degree relations.',
           'query':'Formal query process and relation vocabulary differ independently of sampler; structural audit exposure must be tracked.',
           'topology':'Source real-density gains change sign; recalculated signatures after masking are a distinct intervention.',
           'numeric':'Entity first linear differs with identical inputs under changed batch shapes; ranks near ties amplify FP32 errors.'}))
    log('audit_complete', preserved_files=len(snapshot), tables=tables.keys().__iter__().__str__())
