"""Read-only audit. No downstream dataset processor or ranking reader is imported."""
import argparse
import subprocess
from stage6.util import ROOT, dump, read, runtime_path, sha


def audit(stage5, history):
    ROOT.joinpath('results').mkdir(exist_ok=True)
    deps = read(stage5 / 'EXTERNAL_DEPENDENCIES.json')
    entries = dict(deps['inputs'])
    entries[deps['official_checkpoint']['original_path']] = deps['official_checkpoint']['sha256']
    rows = []
    for path, expected in entries.items():
        p = runtime_path(path)
        actual = sha(p) if p.is_file() else None
        rows.append(dict(path=str(p), expected=expected, actual=actual,
                         status='matched' if actual == expected else 'missing' if actual is None else 'mismatch'))
    frozen = read(history / 'frozen.json')
    baselines = {}
    for tag, r in frozen['records'].items():
        if tag.split('_')[0] in ('B1', 'B2'):
            fields = ['seed', 'config', 'checkpoint', 'checkpoint_sha256', 'best_step']
            record = {k: r[k] for k in fields}
            p = history / r['checkpoint']
            actual = sha(p) if p.is_file() else None
            record.update(path=str(p), actual_sha256=actual, recovered=actual == r['checkpoint_sha256'],
                          source_record_sha256=sha(history / 'frozen.json'),
                          implementation=str(history / 'stable_gate.py'), implementation_sha256=sha(history / 'stable_gate.py'))
            baselines[tag] = record
    dump(ROOT / 'results/dependency_audit.json', dict(external=rows, baselines=baselines))
    choices = read(stage5 / 'analysis/source_choices.json')['formal']['T-4D']
    model_manifest = dict(official_sha256=deps['official_checkpoint']['sha256'],
        T4={seed: {key: rec[key] for key in ['seed', 'config', 'checkpoint', 'checkpoint_sha256', 'best_step']}
            for seed, rec in choices.items()},
        historical={tag: {key: rec[key] for key in ['seed', 'config', 'checkpoint', 'checkpoint_sha256', 'best_step']}
            for tag, rec in baselines.items()})
    model_path = ROOT / 'model_manifest.json'
    if model_path.exists():
        assert read(model_path) == model_manifest
    else:
        dump(model_path, model_manifest, exclusive=True)
    protected = {str(p.relative_to(stage5)): sha(p) for p in stage5.rglob('*')
                 if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
    dest = ROOT / 'protected_stage5.json'
    if dest.exists():
        normalized = lambda d: {k.replace('\\', '/'): v for k, v in d.items()}
        assert normalized(read(dest)) == normalized(protected), 'stage5 changed since initial snapshot'
    else:
        dump(dest, protected, exclusive=True)
    # Protect historical code and checkpoint assets actually used, without writing there.
    extra = {str(history / name): sha(history / name) for name in
             ['frozen.json', 'stable_gate.py', 'gated_model.py', 'common.py']}
    extra.update({r['path']: r['actual_sha256'] for r in baselines.values() if r['recovered']})
    historical_path = ROOT / 'protected_history.json'
    if historical_path.exists():
        def canonical(path):
            s = str(runtime_path(path)).replace('\\', '/')
            return s[5].upper() + ':' + s[6:] if s.startswith('/mnt/') else s
        assert {canonical(k): v for k, v in read(historical_path).items()} == {canonical(k): v for k, v in extra.items()}
    else:
        dump(historical_path, extra)
    print('EXTERNAL', {s: sum(r['status'] == s for r in rows) for s in ['matched', 'missing', 'mismatch']})
    print('BASELINES', {k: v['recovered'] for k, v in baselines.items()})
    assert all(r['status'] == 'matched' for r in rows)
    return baselines


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage5', type=runtime_path, required=True)
    p.add_argument('--history', type=runtime_path, required=True)
    a = p.parse_args()
    audit(a.stage5, a.history)
