"""Fetch public labels only; never derive names from held-out facts."""
import urllib.request
from common_io import ROOT, dump, sha

sources = {
    'FB15k-237': 'https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/FB15k-237/entity2text.txt',
    'WN18RR': 'https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/WN18RR/entity2text.txt'
}
result = {}
for name, url in sources.items():
    path = ROOT / 'cache/lexical' / (name+'.tsv')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not path.exists():
            with urllib.request.urlopen(url, timeout=45) as f: data = f.read()
            if b'\t' not in data: raise ValueError('Not a mapping')
            path.write_bytes(data)
        result[name] = dict(url=url, sha256=sha(path), bytes=path.stat().st_size)
    except (OSError, ValueError) as exc:
        result[name] = dict(url=url, error=type(exc).__name__)
dump(ROOT/'results/lexical_sources.json', result)
print(result)
