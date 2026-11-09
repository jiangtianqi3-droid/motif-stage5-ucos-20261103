"""Lossless, verified, resumable small Git blobs for complete evidence files."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor,as_completed

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'cache/archive_dependencies'))
import zstandard as zstd

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_name(path.name+'.tmp')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(path)

def zread(path):return json.loads(zstd.ZstdDecompressor().decompress(Path(path).read_bytes()))

def zwrite(path,value):
    b=json.dumps(value,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    Path(path).write_bytes(zstd.ZstdCompressor(level=9).compress(b))

def safe_target(root,name):
    # Treat both slash conventions and drive syntax as potentially hostile.
    if '\\' in name or ':' in name:raise ValueError('Unsafe archive name')
    target=(root/name).resolve()
    if not target.is_relative_to(root.resolve()) or target==root.resolve():raise ValueError('Unsafe archive target')
    return target

class HashReader:
    def __init__(self,stream):self.stream=stream;self.h=hashlib.sha256();self.size=0
    def read(self,n=-1):
        data=self.stream.read(n);self.h.update(data);self.size+=len(data);return data

def verify_part(path,members):
    expected={r['path']:r for r in members};seen=set()
    with Path(path).open('rb') as raw,zstd.ZstdDecompressor().stream_reader(raw) as stream,tarfile.open(fileobj=stream,mode='r|') as tar:
        for entry in tar:
            if not entry.isfile():raise ValueError('Archive contains non-file member')
            # Verification never writes files: syntax checks suffice and avoid
            # millions of unnecessary NTFS realpath calls. Restore still checks
            # resolved destinations before writing to defend against symlinks.
            member_name=PurePosixPath(entry.name)
            if member_name.is_absolute() or '..' in member_name.parts or '\\' in entry.name or ':' in entry.name:
                raise ValueError('Unsafe archive name')
            if entry.name not in expected or entry.name in seen:raise ValueError('Unexpected/duplicate member')
            record=expected[entry.name];assert entry.size==record['bytes']
            reader=HashReader(tar.extractfile(entry))
            while reader.read(4*1024*1024):pass
            assert reader.h.hexdigest()==record['sha256'],entry.name
            assert reader.size==record['bytes'];seen.add(entry.name)
    assert seen==set(expected),'Missing members'
    return len(seen)

def write_part(root,output,name,files,max_compressed=40*1024**2):
    archive=output/'parts'/(name+'.tar.zst');index=output/'index'/(name+'.json.zst')
    if archive.exists() and index.exists():
        old=zread(index)
        assert [m['path'] for m in old['members']]==[p.relative_to(root).as_posix() for p in files]
        assert sha(archive)==old['archive_sha256']
        verify_part(archive,old['members'])
        return [dict(file=str(archive.relative_to(output)).replace('\\','/'),index=str(index.relative_to(output)).replace('\\','/'),
            sha256=old['archive_sha256'],index_sha256=sha(index),files=len(files),source_bytes=sum(m['bytes'] for m in old['members']),bytes=archive.stat().st_size)]
    temp=archive.with_name(archive.name+'.tmp');members=[]
    with temp.open('wb') as raw,zstd.ZstdCompressor(level=6,threads=2).stream_writer(raw,closefd=False) as stream,tarfile.open(fileobj=stream,mode='w|',format=tarfile.PAX_FORMAT) as tar:
        for p in files:
            before=p.stat();rel=p.relative_to(root).as_posix();entry=tarfile.TarInfo(rel)
            entry.size=before.st_size;entry.mode=0o644;entry.mtime=0
            with p.open('rb') as source:
                reader=HashReader(source);tar.addfile(entry,reader)
            after=p.stat()
            assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,'Source changed during archive'
            assert reader.size==before.st_size
            members.append(dict(path=rel,bytes=reader.size,sha256=reader.h.hexdigest()))
    if temp.stat().st_size>max_compressed:
        assert len(files)>1,'Single input exceeds safe archive size'
        temp.unlink()  # Only a newly created temporary file under this archive directory.
        half=len(files)//2
        return write_part(root,output,name+'a',files[:half],max_compressed)+write_part(root,output,name+'b',files[half:],max_compressed)
    verify_part(temp,members)
    digest=sha(temp);temp.replace(archive)
    zwrite(index,dict(archive_sha256=digest,members=members))
    return [dict(file=archive.relative_to(output).as_posix(),index=index.relative_to(output).as_posix(),
        sha256=digest,index_sha256=sha(index),files=len(files),source_bytes=sum(m['bytes'] for m in members),bytes=archive.stat().st_size)]

def evidence_sort(path,root):
    parts=path.relative_to(root).parts
    if len(parts)>=7 and parts[0:2]==('results','runs') and 'evidence' in parts:
        try:
            fact,direction=map(int,path.stem.split('_'))
            at=parts.index('evidence')
            return (parts[2],parts[3],fact,direction,parts[at+1],parts[at+2])
        except ValueError:pass
    return ('z',path.as_posix(),0,0,'','')

def build(root=ROOT,output=None,chunk_bytes=384*1024**2,workers=4):
    output=Path(output or root/'artifacts/data_archive_v1')
    (output/'parts').mkdir(parents=True,exist_ok=True);(output/'index').mkdir(parents=True,exist_ok=True)
    selection=[]
    # All successful, smoke and failed-run artifacts; gates are scientific outputs,
    # while compiler caches and installed dependency packages are reproducible tools.
    roots=['results/runs','results/monitor','figures','logs','cache/gates','cache/signatures']
    for rel in roots:
        source=root/rel
        if source.exists():
            for directory,dirs,names in os.walk(source):
                dirs[:]=sorted(d for d in dirs if d not in ('__pycache__','.pytest_cache'))
                for name in sorted(names):
                    p=Path(directory)/name
                    if rel=='logs' and name.startswith('archive'):continue
                    if name.endswith('.tmp') or name.endswith('.tmp.npy') or name.endswith('.tmp.npz'):continue
                    selection.append(p)
    selection.append(root/'results/status.json')
    files=sorted(set(selection),key=lambda p:evidence_sort(p,root))
    total_bytes=sum(p.stat().st_size for p in files)
    expected=321648
    formal=[p for p in files if 'd48ae432a2533cdaaec0' in p.parts and 'evidence' in p.parts and p.suffix=='.json']
    assert len(formal)==expected,(len(formal),expected)
    progress=root/'logs/archive_build_status.json'
    dump(progress,dict(state='packing',files=len(files),formal_evidence=len(formal),source_bytes=total_bytes,done_files=0))
    print(json.dumps(dict(state='inventory_complete',files=len(files),source_GB=round(total_bytes/1e9,3))),flush=True)
    chunks=[];chunk=[];size=0;count=0;began=time.monotonic()
    for p in files:
        size_next=p.stat().st_size
        if chunk and (size+size_next>chunk_bytes or len(chunk)>=2000):
            chunks.append(chunk)
            chunk=[];size=0
        chunk.append(p);size+=size_next
    if chunk:chunks.append(chunk)
    parts=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(write_part,root,output,f'part-{i+1:05d}',items):len(items) for i,items in enumerate(chunks)}
        for future in as_completed(futures):
            parts+=future.result();count+=futures[future]
            status=dict(state='packing_and_verifying',files=len(files),done_files=count,parts=len(parts),workers=workers,
                seconds=round(time.monotonic()-began,1),archive_bytes=sum(x['bytes'] for x in parts))
            dump(progress,status)
            if len(parts)%5==0:print(json.dumps(status),flush=True)
    parts=sorted(parts,key=lambda p:p['file'])
    # Rebuild the membership total from persisted verified sidecars, not loop counters.
    members=sum(p['files'] for p in parts)
    assert members==len(files)
    source_bytes=sum(p['source_bytes'] for p in parts)
    assert source_bytes==total_bytes
    manifest=dict(format='motif-byte-exact-tar-zstd-v1',formal_run='d48ae432a2533cdaaec0',
        formal_evidence_records=expected,source_files=members,source_bytes=source_bytes,
        compressed_bytes=sum(p['bytes'] for p in parts),parts=parts,
        verification='Every archived file was decompressed and its SHA256 checked against the bytes read from the source',
        relative_to='experiments/query_community_v1',included=roots+['results/status.json'],
        excluded=['compiler caches','installed dependency packages','Python bytecode','previous-stage assets already referenced by freeze_manifest.json'],
        original_sources_modified=False,zstandard_version=zstd.__version__)
    dump(output/'archive_manifest.json',manifest)
    dump(progress,dict(state='complete',files=members,parts=len(parts),source_bytes=source_bytes,
        compressed_bytes=manifest['compressed_bytes'],seconds=round(time.monotonic()-began,1)))
    print(json.dumps(dict(state='complete',files=members,parts=len(parts),source_GB=round(source_bytes/1e9,3),
        archive_GB=round(manifest['compressed_bytes']/1e9,3),seconds=round(time.monotonic()-began,1))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();assert 1<=args.workers<=8;build(args.root,workers=args.workers)
