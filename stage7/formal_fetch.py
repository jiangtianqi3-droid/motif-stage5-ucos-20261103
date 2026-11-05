"""Called only after global selection freeze, before formal parsing."""
from datetime import datetime,timezone
from stage7.util import ROOT,read,dump,sha,log
from stage7.prepare import fetch


def main():
    frozen=read(ROOT/'frozen.json');protocol=read(ROOT/'protocol.json')
    assert frozen['protocol_sha256']==sha(ROOT/'protocol.json')
    records={}
    meta=read(ROOT/'observations/manifest.json')
    for _,key in protocol['targets']:
        if key not in meta or key in frozen.get('unavailable_targets',{}):continue
        rec=meta[key]
        if 'archive' in rec:continue
        for which in ['valid','test']:
            if which not in rec:continue
            path=ROOT/'formal_inputs'/(key+'_'+which+'.txt')
            fetch(rec[which],path)
            records[path.name]=dict(sha256=sha(path),url=rec[which])
            log('formal_input_downloaded_after_freeze',key=key,split=which,sha256=sha(path))
    dump(ROOT/'formal_inputs/download_manifest.json',dict(records=records,
        downloaded_utc=datetime.now(timezone.utc).isoformat(),frozen_sha256=sha(ROOT/'frozen.json')))


if __name__=='__main__':main()
