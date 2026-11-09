import importlib.util
import json
from pathlib import Path
import pytest

def module(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

archive=module('archive_data');restore=module('restore_data')

def test_byte_exact_roundtrip_and_no_overwrite(tmp_path):
    root=tmp_path/'source';root.mkdir();files=[]
    for name,data in [('a.json',b'{\n  "a": 1\n}\n'),('nested/data.bin',bytes(range(256))*8)]:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);files.append(p)
    out=tmp_path/'archive';(out/'parts').mkdir(parents=True);(out/'index').mkdir()
    parts=archive.write_part(root,out,'part-00001',files)
    archive.dump(out/'archive_manifest.json',dict(parts=parts,source_files=2,source_bytes=sum(p.stat().st_size for p in files)))
    dest=tmp_path/'restored';result=restore.restore(out,dest)
    assert result['files']==2
    for p in files:assert (dest/p.relative_to(root)).read_bytes()==p.read_bytes()
    assert restore.restore(out,dest)['verified']
    (dest/'a.json').write_bytes(b'local user changes')
    with pytest.raises(AssertionError):restore.restore(out,dest)
    assert (dest/'a.json').read_bytes()==b'local user changes'

def test_corrupt_part_is_rejected(tmp_path):
    root=tmp_path/'source';root.mkdir();p=root/'file.json';p.write_bytes(b'{}')
    out=tmp_path/'archive';(out/'parts').mkdir(parents=True);(out/'index').mkdir()
    parts=archive.write_part(root,out,'part-00001',[p])
    archive.dump(out/'archive_manifest.json',dict(parts=parts,source_files=1,source_bytes=2))
    (out/parts[0]['file']).write_bytes(b'corrupt')
    with pytest.raises(AssertionError):restore.restore(out,tmp_path/'restored')

@pytest.mark.parametrize('name',['../escape.json','/absolute.json','C:/escape.json','..\\escape.json'])
def test_unsafe_paths_rejected(tmp_path,name):
    with pytest.raises(ValueError):restore.safe_target(tmp_path,name)
