import json
import os
from pathlib import Path
from stage6.util import sha, identity, read, runtime_path

ROOT = Path(__file__).resolve().parent


def log(event, **kw):
    from datetime import datetime,timezone
    value=dict(event=event,utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kw)
    text=json.dumps(value,ensure_ascii=False)
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/events.jsonl').open('a',encoding='utf8') as f:
        f.write(text+'\n')
    print(text,flush=True)


def dump(path, value, exclusive=False):
    path = Path(path).resolve()
    if ROOT not in path.parents:
        raise ValueError('Stage 7 may write only inside stage7/')
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if exclusive:
        with path.open('x', encoding='utf8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    else:
        tmp = path.with_suffix('.tmp')
        tmp.write_text(text, encoding='utf8')
        tmp.replace(path)
