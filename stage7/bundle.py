"""Seal a portable result/code bundle; heavyweight regenerable caches excluded."""
import zipfile
from datetime import datetime,timezone
from stage7.util import ROOT,read,dump,sha,log


def main():
    v=read(ROOT/'results/verification.json')
    assert not v['errors'] and (ROOT/'report.md').exists()
    model=read(ROOT.parent/'stage6/model_manifest.json')
    dump(ROOT/'EXTERNAL_DEPENDENCIES.json',dict(
        stage5='Unmodified parent checkout of motif-stage5-ucos-20261103, upstream173017b7c3a4d88edf73d2ce46b083a5fcb69354',
        stage6='Unmodified sibling package stage6/; observed/model/graph adapters and observation bundles reused read-only',
        stage6_artifact_manifest_sha256=sha(ROOT.parent/'stage6/ARTIFACT_MANIFEST.json'),
        stage6_frozen_sha256=sha(ROOT.parent/'stage6/frozen.json'),model_manifest=model,
        environment=read(ROOT/'results/environment.json'),
        analysis_environment=read(ROOT/'results/analysis_environment.json'),
        figure_environment=read(ROOT/'results/figure_environment.json'),
        source_validation='Existing topology-stage2 source validation truth and Stage4 frozen512 query subsets',
        metadata='Paths can be overridden in stage7.run; download bytes and hashes retained inside the bundle.'))
    excluded={'cache','triton_cache','torch_extensions','__pycache__'}
    files=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or set(p.relative_to(ROOT).parts)&excluded:continue
        if p.name in ['ARTIFACT_MANIFEST.json','stage7_artifacts.zip'] or p.suffix in ['.pyc','.tmp','.part']:continue
        files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    files.sort(key=lambda r:r['path'])
    dump(ROOT/'ARTIFACT_MANIFEST.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        files=files,count=len(files),bytes=sum(r['bytes'] for r in files),excluded=sorted(excluded),
        frozen_sha256=sha(ROOT/'frozen.json'),report_sha256=sha(ROOT/'report.md'),verification=v['status']))
    archive=ROOT/'stage7_artifacts.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for r in files:z.write(ROOT/r['path'],'stage7/'+r['path'])
        z.write(ROOT/'ARTIFACT_MANIFEST.json','stage7/ARTIFACT_MANIFEST.json')
    import json
    print(json.dumps(dict(event='bundle_complete',files=len(files),bytes=archive.stat().st_size,archive_sha256=sha(archive))),flush=True)


if __name__=='__main__':main()
