"""Restore byte-exact files into the phase directory, or verify without writing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'cache/archive_dependencies'))
import zstandard as zstd

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def safe_target(root,name):
    if '\\' in name or ':' in name:raise ValueError('Unsafe archive name')
    p=(root/name).resolve()
    if p==root.resolve() or not p.is_relative_to(root.resolve()):raise ValueError('Unsafe archive path')
    return p

def restore(archive_dir,destination,verify_only=False,selected=None):
    archive_dir=Path(archive_dir);destination=Path(destination)
    manifest=json.loads((archive_dir/'archive_manifest.json').read_text(encoding='utf-8'))
    count=0;source_bytes=0
    for part in manifest['parts']:
        if selected and Path(part['file']).name not in selected:continue
        archive=archive_dir/part['file'];index=archive_dir/part['index']
        assert sha(archive)==part['sha256'] and sha(index)==part['index_sha256'],'Archive or index corrupt'
        records=json.loads(zstd.ZstdDecompressor().decompress(index.read_bytes()))
        expected={r['path']:r for r in records['members']};seen=set()
        with archive.open('rb') as raw,zstd.ZstdDecompressor().stream_reader(raw) as stream,tarfile.open(fileobj=stream,mode='r|') as tar:
            for entry in tar:
                if not entry.isfile():raise ValueError('Archive contains unsafe non-file member')
                if entry.name not in expected or entry.name in seen:raise ValueError('Unexpected/duplicate member')
                info=expected[entry.name];target=safe_target(destination,entry.name)
                write=not verify_only and not target.exists()
                if not verify_only and target.exists():
                    assert sha(target)==info['sha256'],'Existing file differs; refusing to overwrite '+str(target)
                if write:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    temporary=target.with_name(target.name+'.restore.tmp')
                    handle=temporary.open('xb')
                else:handle=None
                h=hashlib.sha256();n=0
                try:
                    with tar.extractfile(entry) as source:
                        for block in iter(lambda:source.read(4*1024*1024),b''):
                            h.update(block);n+=len(block)
                            if handle:handle.write(block)
                finally:
                    if handle:handle.close()
                assert n==info['bytes']==entry.size and h.hexdigest()==info['sha256'],entry.name
                if write:temporary.replace(target)
                seen.add(entry.name);count+=1;source_bytes+=n
        assert len(seen)==part['files']
        print(json.dumps(dict(part=part['file'],files_checked=count)),flush=True)
    if not selected:assert count==manifest['source_files'] and source_bytes==manifest['source_bytes']
    return dict(verified=True,files=count,source_bytes=source_bytes,verify_only=verify_only)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=ROOT/'artifacts/data_archive_v1')
    p.add_argument('--destination',type=Path,default=ROOT);p.add_argument('--verify-only',action='store_true')
    p.add_argument('--parts',nargs='+')
    a=p.parse_args();print(json.dumps(restore(a.archive,a.destination,a.verify_only,a.parts)))
