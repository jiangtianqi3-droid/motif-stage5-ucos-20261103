import io
import unittest
import numpy as np
from datetime import datetime,timezone
from stage7.util import ROOT,read,dump,sha,log


def run(args):
    errors=[]
    preserved=read(ROOT/'preserved_assets.json')
    for name,h in preserved.items():
        if sha(ROOT.parent/name)!=h:errors.append('Changed preserved asset '+name)
    model_manifest=read(ROOT.parent/'stage6/model_manifest.json')
    assert sha(args.official)==model_manifest['official_sha256']
    model_files=1
    for rec in model_manifest['T4'].values():
        assert sha(args.stage5/rec['checkpoint'])==rec['checkpoint_sha256'];model_files+=1
    for rec in model_manifest['historical'].values():
        assert sha(args.history/rec['checkpoint'])==rec['checkpoint_sha256'];model_files+=1
    observation_files=0
    for folder in [args.observations,ROOT/'observations']:
        for rec in read(folder/'manifest.json').values():
            if rec.get('status')=='unavailable':continue
            assert sha(folder/rec['file'])==rec['sha256'];observation_files+=1
    p=read(ROOT/'protocol.json');frozen=read(ROOT/'frozen.json')
    registered_targets=len(p['targets'])
    unavailable=frozen.get('unavailable_targets',{})
    p['targets']=[row for row in p['targets'] if row[1] not in unavailable]
    assert sha(ROOT/'protocol.json')==frozen['protocol_sha256']
    diagnostic_seals={
        'source_mechanism_summary_sha256':'results/mechanism_summary.json',
        'numeric_checks_sha256':'results/gpu_checks.json',
        'historical_exposure_search_sha256':'results/historical_exposure_search.json',
        'pretraining_observation_overlap_sha256':'results/pretraining_observation_overlap.json',
        'compatibility_preflight_sha256':'results/new_graph_preflight.json',
        'proxy_multiplicity_verification_sha256':'results/proxy_multiplicity_verification.json'}
    for field,path in diagnostic_seals.items():
        assert sha(ROOT/path)==p[field],('Changed preregistration evidence',path)
    for file,h in frozen['code_hashes'].items():
        if sha(ROOT/file)!=h:errors.append('Frozen code mismatch '+file)
    downloads=read(ROOT/'formal_inputs/download_manifest.json')
    assert downloads['frozen_sha256']==sha(ROOT/'frozen.json')
    assert downloads['downloaded_utc']>=frozen['created_utc']
    for name,rec in downloads['records'].items():
        assert '/' not in name and '\\' not in name
        assert sha(ROOT/'formal_inputs'/name)==rec['sha256'],('Changed formal download',name)
    alpha0_audit=read(ROOT/'results/alpha0_relation_kernel_audit.json')
    assert alpha0_audit['frozen_sha256']==sha(ROOT/'frozen.json')
    assert alpha0_audit['no_weight_update'] and alpha0_audit['no_frozen_outputs_changed']
    count=0;rankfiles=0;no_gradient=True
    incomplete_formal={}
    for group,key in p['targets']:
        for ps in p['proxy_seeds']:
            path=ROOT/'splits'/f'{key}__p{ps}.json';meta=read(path)
            if sha(path.with_suffix('.npz'))!=meta['split_sha256']:errors.append('Split hash mismatch '+str(path))
            with np.load(path.with_suffix('.npz')) as z:
                facts=z['facts'];masks=[z['hidden'+str(i)] for i in range(3)]
                assert set(masks[0])<=set(masks[1])<=set(masks[2])
                select=set(np.concatenate([z[e+'_selection'] for e in 'ABCD']))
                confirm=set(np.concatenate([z[e+'_confirmation'] for e in 'ABCD']))
                assert not select&confirm and select|confirm<=set(masks[0])
                assert len({len(z[e+'_selection']) for e in 'ABCD'})==1
                assert len({len(z[e+'_confirmation']) for e in 'ABCD'})==1
            for seed in p['training_seeds']:
                label=f'{key}__p{ps}__s{seed}'
                path=ROOT/'adaptation'/(label+'.json');rec=read(path)
                assert sha(path)==frozen['selections'][label]['sha256']
                plan=ROOT/'adaptation/plans'/(label+'.json')
                assert sha(plan)==rec['selection_plan_sha256']
                decisions=read(plan)['candidates']
                for name,d in rec['decisions'].items():
                    assert d['candidate_alpha']==decisions[name]['candidate_alpha']
                    if d['fallback']:assert d['final_alpha']==0.
                assert not rec['formal_labels_accessed']
                no_gradient &=rec['no_weight_update']
                count+=1
        marker=read(ROOT/'formal_inputs'/(key+'_opened.json'))
        assert marker['opened_utc']>=frozen['created_utc']
        for family,seeds in [('T-4D',p['training_seeds']),('B0',[1024]),('B1',p['training_seeds']),('B2',p['training_seeds'])]:
            for seed in seeds:
                path=ROOT/'evaluation'/f'{key}__{family}_s{seed}.json'
                if not path.exists():
                    incomplete_formal.setdefault(key,[]).append(f'{family}_s{seed}')
                    continue
                rec=read(path)
                assert rec['frozen_sha256']==sha(ROOT/'frozen.json') and rec['no_weight_update']
    for folder in ['adaptation/ranks','mechanism/ranks','evaluation','source_validation']:
        for path in (ROOT/folder).glob('*.json'):
            rec=read(path)
            assert sha(path.with_suffix('.npz'))==rec['ranks_sha256']
            with np.load(path.with_suffix('.npz')) as z:
                r=z['ranks'];assert r.shape[0]==len(z['alphas']) and r.shape[1]==len(z['queries']) and r.shape[2]==2
                assert r.dtype==np.int64 and (r>=1).all()
            rankfiles+=1
    buffer=io.StringIO()
    result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    (ROOT/'results/tests.log').write_text(buffer.getvalue(),encoding='utf8')
    if not result.wasSuccessful():errors.append('Unit tests failed')
    verification=dict(status=('passed_with_incomplete_targets' if unavailable or incomplete_formal else 'passed') if not errors else 'failed',errors=errors,preserved_files=len(preserved),
        registered_targets=registered_targets,completed_targets=len(p['targets'])-len(incomplete_formal),unavailable_targets=list(unavailable),incomplete_formal_targets=incomplete_formal,
        stage5_stage6_unchanged=not any('preserved' in e for e in errors),target_selection_count=count,
        checkpoint_files_verified=model_files,observation_bundles_verified=observation_files,
        preregistration_diagnostic_seals_verified=len(diagnostic_seals),
        postfreeze_formal_downloads_verified=len(downloads['records']),
        alpha0_relation_path_diagnosis_completed=True,
        expected_target_selection_count=len(p['targets'])*9,rank_artifacts_verified=rankfiles,no_gradient_updates=no_gradient,
        nested_masks=True,global_selection_confirmation_disjoint=True,equal_environment_n=True,
        formal_opened_after_global_selection_freeze=True,unit_tests=result.testsRun,
        gpu_mechanisms=read(ROOT/'results/gpu_checks.json')['status'],
        frozen_sha256=sha(ROOT/'frozen.json'),verified_utc=datetime.now(timezone.utc).isoformat())
    dump(ROOT/'results/verification.json',verification)
    log('verification_complete',**verification)
    assert not errors,errors
