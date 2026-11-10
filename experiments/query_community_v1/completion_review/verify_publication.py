"""Verify and index the compact Git publication without reading private caches."""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN_ID='f20ab88ff89d82d8506c'
ARCHIVE=ROOT/'rerank_v2/artifacts'/RUN_ID
EXCLUDED={'cache','logs','results','__pycache__','.pytest_cache','.secrets','.git'}
PATTERNS=[rb'sk-[a-zA-Z0-9]{20,}',rb'github_pat_[a-zA-Z0-9_]{20,}',rb'gh[pousr]_[a-zA-Z0-9]{20,}',rb'AKIA[A-Z0-9]{16}']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

manifest_path=ARCHIVE/'publication_manifest.json'
manifest=read(manifest_path)
protocol=read(ARCHIVE/'protocol.json')
for relative,digest in protocol['source'].items():
    assert sha(ROOT/'rerank_v2'/relative)==digest,relative
assert sha(ROOT/'rerank_v2/report_route_a_rerank.md')==sha(ARCHIVE/'report_route_a_rerank.md')
assert read(ARCHIVE/'final_delivery_verification.json')['passed']
files=[ROOT/'ROUTE_A_LATEST.md']
for module in ('rerank_v2','progress_window','completion_review'):
    for path in (ROOT/module).rglob('*'):
        if not path.is_file() or path==manifest_path:continue
        if any(part in EXCLUDED for part in path.relative_to(ROOT/module).parts):continue
        files.append(path)
records=[]
for path in sorted(files):
    content=path.read_bytes()
    assert not any(re.search(pattern,content) for pattern in PATTERNS),f'Credential-shaped content found in {path.relative_to(ROOT)}'
    assert path.suffix.lower() not in ('.docx','.env','.pth','.npy'),path
    records.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
manifest.update(file_count=len(records),total_bytes=sum(r['bytes'] for r in records),files=records,
                verification=dict(frozen_source_files=len(protocol['source']),report_copies_match=True,credential_scan_passed=True))
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,files=len(records),MiB=round(manifest['total_bytes']/1024**2,2),frozen_source_files=len(protocol['source'])),ensure_ascii=False))
