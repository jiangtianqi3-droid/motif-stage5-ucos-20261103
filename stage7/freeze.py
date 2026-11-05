from datetime import datetime,timezone
from stage7.util import ROOT,read,dump,sha,log


FILES=['proxy_environments.py','proxy_density.py','proxy_split.py','robust_selector.py','confirmation.py',
       'numeric_stability.py','evaluator.py','isolation.py','experiment.py','formal.py','formal_fetch.py','prepare.py','run.py','util.py']


def run(args):
    p=read(ROOT/'protocol.json')
    records={}
    unavailable={}
    for group,key in p['targets']:
        complete=all((ROOT/'adaptation'/f'{key}__p{ps}__s{seed}.json').exists() for seed in p['training_seeds'] for ps in p['proxy_seeds'])
        if not complete and (ROOT/'failures'/(key+'.json')).exists():
            unavailable[key]=dict(record=read(ROOT/'failures'/(key+'.json')),sha256=sha(ROOT/'failures'/(key+'.json')))
            continue
        for seed in p['training_seeds']:
            for ps in p['proxy_seeds']:
                label=f'{key}__p{ps}__s{seed}'
                path=ROOT/'adaptation'/(label+'.json')
                r=read(path)
                assert r['no_weight_update'] and not r['formal_labels_accessed']
                assert sha(ROOT/'adaptation/plans'/(label+'.json'))==r['selection_plan_sha256']
                assert sha(ROOT/'splits'/f'{key}__p{ps}.npz')==r['split_sha256']
                records[label]=dict(sha256=sha(path),decisions=r['decisions'],checkpoint_sha256=r['checkpoint_sha256'])
    dump(ROOT/'frozen.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),
        code_hashes={f:sha(ROOT/f) for f in FILES},selections=records,selection_count=len(records),unavailable_targets=unavailable,
        model_manifest_sha256=sha(ROOT.parent/'stage6/model_manifest.json'),
        preserved_manifest_sha256=sha(ROOT/'preserved_assets.json'),
        old_observation_manifest_sha256=sha(args.observations/'manifest.json'),
        new_observation_manifest_sha256=sha(ROOT/'observations/manifest.json'),
        no_formal_new_queries_loaded=True),exclusive=True)
    log('all_selections_frozen',count=len(records),sha256=sha(ROOT/'frozen.json'))
