import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
KEYS = ('WN18RRInductive_v1', 'FBIngram_25')
METHODS = ('B0', 'B2_s1024+Dstar')
PRIMARY = METHODS[1]
HISTORY = Path('/mnt/d/motif-phase4-work')
STAGE7 = HISTORY / 'motif-stage7-asymmetric-20261106'
SNAPSHOT = ROOT.parent / 'query_community_v1/artifacts/inputs'
SEED = 20261117


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(value, encoding='utf-8')
    tmp.replace(path)


def dump(path, value):
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def snapshot(key, filename):
    rec = next(x for x in read(SNAPSHOT / 'manifest.json')['datasets'] if x['dataset'] == key)
    details = next(x for x in rec['files'] if x['file'] == f'artifacts/inputs/{key}/{filename}')
    path = SNAPSHOT / key / filename
    if sha(path) != details['sha256']:
        raise ValueError(f'snapshot checksum mismatch: {key}/{filename}')
    return path
