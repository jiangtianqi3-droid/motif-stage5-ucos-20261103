import hashlib
import json
import os
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(*arrays, metadata=None):
    h = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, str(a.dtype))).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def dump(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if exclusive:
        with path.open('x', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    else:
        tmp = path.with_suffix('.tmp')
        tmp.write_text(text, encoding='utf-8')
        tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def log(event, **kw):
    print(json.dumps(dict(event=event, **kw), ensure_ascii=False), flush=True)


def runtime_path(path):
    text = str(path).replace('\\', '/')
    if os.name != 'nt' and len(text) > 2 and text[1] == ':':
        text = '/mnt/' + text[0].lower() + text[2:]
    if os.name == 'nt' and text.startswith('/mnt/'):
        text = text[5] + ':' + text[6:]
    return Path(text)
