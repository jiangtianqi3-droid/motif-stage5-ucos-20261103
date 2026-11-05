"""Search historical research artifacts, excluding public runtime catalogs and raw labels."""
import os
import re
from pathlib import Path
from stage7.util import ROOT,dump,sha,log,runtime_path


def run(roots=None):
    roots=[runtime_path(p) for p in roots or ['D:/ultra-phase1-work','D:/motif-phase2-work','D:/motif-phase3-work','D:/motif-phase4-work']]
    exclude={'.git','.ai-handoff','stage7','runtime','MOTIF','ULTRA','ultra','data','raw','downloads',
             'cache','triton_cache','torch_extensions','__pycache__','output','models','ckpts','checkpoints','evaluation','github-archives'}
    patterns={
        'HM_1k_new':re.compile(r'Hamaguchi-BM_both-1000|HM[_ -]1k',re.I),
        'WikiTopicsMT4_health_new':re.compile(r'wikidata_healthv2|WikiTopics.?MT4.{0,40}health',re.I),
        'AristoV4_external_new':re.compile(r'AristoV4|Aristo-v4',re.I),
        'UMLS_external_new':re.compile(r'\bUMLS\b',re.I)}
    files=[];matches=[]
    for root in roots:
        for folder,dirs,names in os.walk(root):
            dirs[:]=[d for d in dirs if d not in exclude]
            for name in names:
                path=Path(folder)/name
                if path.suffix.lower() not in {'.csv','.json','.md','.py','.yaml','.yml','.sh'} or path.stat().st_size>3*1024*1024:
                    continue
                try:text=path.read_text(encoding='utf-8-sig')
                except UnicodeError:continue
                files.append(dict(path=str(path),sha256=sha(path)))
                for key,pattern in patterns.items():
                    for i,line in enumerate(text.splitlines(),1):
                        if pattern.search(line):matches.append(dict(candidate=key,path=str(path),line=i,evidence=line[:500]))
    assert files,'No historical artifacts found; configure --roots explicitly.'
    dump(ROOT/'results/historical_exposure_search.json',dict(roots=list(map(str,roots)),excluded_dirs=sorted(exclude),
        scanned_files=len(files),files=files,matches=matches,
        interpretation='Mention in a public runtime catalog is not a prior evaluation. Hits in research outputs require manual interpretation; lineage overlap is separately documented.'))
    log('historical_exposure_search',files=len(files),matches=len(matches))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--roots',nargs='+')
    run(parser.parse_args().roots)
