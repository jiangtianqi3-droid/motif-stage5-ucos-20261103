"""Export the completed route-A run without copying runtime caches or credentials."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

RUN_ID='f20ab88ff89d82d8506c'
MODULES=('rerank_v2','progress_window','completion_review')
EXCLUDED={'cache','logs','results','__pycache__','.pytest_cache','.secrets','.git'}
ALLOWED={'.py','.md','.sh','.ps1','.cs','.exe','.vbs'}
GROUP_FILES=('completion.json','summary.json','ranks.npz','per_query_ranks.jsonl',
             'model_audit.json','operator_audit.json','retrieval_completion.json',
             'selected_structure.json','selected_llm.json','selection.json',
             'split_leakage_audit.json','verbalizer_audit.json','llm_test_stats.json')

def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-mirror',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--result-root',type=Path,required=True)
    args=parser.parse_args()
    source=args.source_mirror/'experiments/query_community_v1'
    dest=args.repo/'experiments/query_community_v1'
    run=args.result_root/'results/runs'/RUN_ID
    protocol=read(run/'protocol.json')
    assert read(run/'completion.json')['state']=='complete'
    assert read(run/'final_delivery_verification.json')['passed']
    for relative,expected in protocol['source'].items():
        assert sha(source/'rerank_v2'/relative)==expected,relative
    for module in MODULES:
        if (dest/module).exists(): raise RuntimeError(f'Refuse to overwrite an existing module: {dest/module}')
    records=[]
    def copy(src,target):
        if target.exists():raise RuntimeError(f'Refuse to overwrite: {target}')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,target)
        digest=sha(src);assert sha(target)==digest
        records.append(dict(path=target.relative_to(dest).as_posix(),bytes=target.stat().st_size,sha256=digest))
    for module in MODULES:
        for path in sorted((source/module).rglob('*')):
            if not path.is_file():continue
            rel=path.relative_to(source/module)
            if any(part in EXCLUDED for part in rel.parts):continue
            if path.name!='.gitignore' and path.suffix.lower() not in ALLOWED:continue
            copy(path,dest/module/rel)
    overview=source/'ROUTE_A_LATEST.md'
    if overview.exists():copy(overview,dest/overview.name)
    archive=dest/'rerank_v2/artifacts'/RUN_ID
    for path in sorted(run.iterdir()):
        if path.is_file() and path.suffix in ('.json','.csv','.md'):
            copy(path,archive/path.name)
    copy(run/'report_route_a_rerank.md',dest/'rerank_v2/report_route_a_rerank.md')
    for dataset in protocol['datasets']:
        for family in protocol['families']:
            work=run/dataset/family
            for filename in GROUP_FILES:
                path=work/filename
                if path.exists():copy(path,archive/dataset/family/filename)
            # Exactly the first four lexicographic examples used by the report.
            for case in sorted((work/'cases').glob('*.json'))[:4]:
                copy(case,archive/dataset/family/'cases'/case.name)
    validations={
        'smoke_audit.json':args.result_root/'results/runs/488c9569690cb3a60133/smoke_audit.json',
        'llm_contract_recheck.json':args.result_root/'results/runs/488c9569690cb3a60133/contract_recheck/audit.json',
        'final_operator_memory_smoke.json':args.result_root/'results/runs/041e6fa726965fff741f/completion.json',
        'progress_window_verification.json':args.result_root/'monitor/verification.json',
    }
    for name,path in validations.items():
        if path.exists():copy(path,archive/'validation'/name)
    manifest=dict(run_id=RUN_ID,source_code_matches_frozen_run=True,
                  scope='code, reports, full numerical ranks and diagnostics, bounded representative evidence',
                  excluded=['all-entity logits cache','complete per-query evidence trees','LLM response cache',
                            'credentials and supplied prompt document','model checkpoints','runtime logs'],
                  file_count=len(records),total_bytes=sum(r['bytes'] for r in records),
                  files=sorted(records,key=lambda r:r['path']))
    path=archive/'publication_manifest.json'
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(manifest=str(path),files=len(records),MiB=round(manifest['total_bytes']/1024**2,2)),ensure_ascii=False))

if __name__=='__main__':main()
