"""Observation-only post-registration audit; does not change any split or choice."""
import numpy as np
from scipy.spatial.distance import jensenshannon
from stage6.observed_loader import load_observed
from stage7.util import ROOT, read, sha, dump
from stage7.analysis import write


def run(observations=None):
    protocol=read(ROOT/'protocol.json')
    rows=[]; checks=[]
    for _,key in protocol['targets']:
        folder=ROOT/'observations' if key.endswith('_new') else observations or ROOT.parent/'stage6/observations'
        graph=load_observed(folder,key);facts=graph.forward
        degree=np.bincount(facts[:,[0,2]].ravel(),minlength=graph.n)
        frequency=np.bincount(facts[:,1],minlength=graph.R//2)
        for seed in protocol['proxy_seeds']:
            path=ROOT/'splits'/f'{key}__p{seed}.npz'
            if not path.exists():continue
            with np.load(path) as z:
                assert np.array_equal(facts,z['facts'])
                masks=[z['hidden'+str(i)].copy() for i in range(3)]
            meta=read(path.with_suffix('.json'))
            parent=np.arange(graph.n)
            def root(v):
                while parent[v]!=v:
                    parent[v]=parent[parent[v]];v=parent[v]
                return v
            protected=[]
            for i in np.random.default_rng(seed).permutation(len(facts)):
                a,b=root(facts[i,0]),root(facts[i,2])
                if a!=b:parent[a]=b;protected.append(int(i))
            assert len(protected)==meta['mask_metadata'][0]['forest_facts']
            assert not set(protected)&set(masks[-1])
            not_forest=np.setdiff1d(np.arange(len(facts)),protected)
            groups={'all_observed':np.arange(len(facts)),'forest_protected':np.asarray(protected),
                    'non_forest':not_forest,**{'hidden_'+str(rate):ids for rate,ids in zip([.05,.10,.15],masks)}}
            for name,ids in groups.items():
                q=facts[ids];counts=np.bincount(q[:,1],minlength=graph.R//2)
                rows.append(dict(key=key,proxy_seed=seed,subset=name,n=len(q),fraction=len(q)/len(facts),
                    observed_relations=int((frequency>0).sum()),represented_relations=int((counts>0).sum()),
                    relation_coverage=float((counts>0).sum()/max((frequency>0).sum(),1)),
                    js_relation_to_observed=float(jensenshannon(counts,frequency)) if len(q) else None,
                    mean_log_endpoint_degree=float(np.log1p(degree[q[:,[0,2]]]).mean()) if len(q) else None,
                    mean_log_relation_frequency=float(np.log1p(frequency[q[:,1]]).mean()) if len(q) else None,
                    facts_with_endpoint_degree_at_most_2=int((degree[q[:,[0,2]]].min(1)<=2).sum()) if len(q) else 0,
                    endpoint_degree_at_most_2_fraction=float((degree[q[:,[0,2]]].min(1)<=2).mean()) if len(q) else None,
                    singleton_relation_facts=int((frequency[q[:,1]]==1).sum()),
                    definition='Unique forward facts; observed covariates only; no formal queries or outcomes'))
            checks.append(dict(key=key,proxy_seed=seed,split_sha256=sha(path),
                forest_count_verified=True,no_hidden_forest_fact=True,
                exclusions_during_scan_to_largest_mask=meta['mask_metadata'][-1]['excluded']))
    write('observation_constraint_bias.csv',rows)
    dump(ROOT/'results/observation_constraint_bias_verification.json',dict(
        status='passed',checks=checks,selection_rules_changed=False,formal_queries_accessed=False))
    print('Observation constraint audit:',len(rows),'rows;',len(checks),'verified splits',flush=True)


if __name__=='__main__':run()
