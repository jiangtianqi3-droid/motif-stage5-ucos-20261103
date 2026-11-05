from datetime import datetime,timezone
from stage7.util import ROOT,read,dump,sha,log


def run(args):
    assert read(ROOT/'results/mechanism_summary.json')['status']=='passed'
    assert read(ROOT/'results/gpu_checks.json')['status']=='passed'
    mult=read(ROOT/'results/proxy_multiplicity_verification.json')
    assert mult['status']=='passed' and mult['masking_code_sha256']==sha(ROOT/'proxy_density.py')
    old=read(ROOT.parent/'stage6/protocol.json')
    new=read(ROOT/'observations/manifest.json')
    compatibility=read(ROOT/'results/new_graph_preflight.json')
    targets=old['targets']+[['new_prospective',k] for k,r in new.items() if r.get('status')!='unavailable' and compatibility[k]['status']=='compatible_topology']
    value=dict(name='Distribution-Robust Test-Time Topology Adaptation',version=1,
        created_utc=datetime.now(timezone.utc).isoformat(),mechanism_passed=True,
        setting='No target valid/test labels or target gradients; only observation self-supervision. Historical group exposed; prospective new results with provenance dependencies declared.',
        training_seeds=[1024,2026,3407],proxy_seeds=[6101,6102,6103],primary_proxy_seed=6101,
        targets=targets,alpha_grid=[0.,.1,.25,.5,1.],rates=[.05,.10,.15],
        physical_mask='Unique forward facts; uniform random order; sequential min endpoint residual2, preserved random spanning forest and at least1 fact per observed relation; no relaxation. Nested prefixes, no fixed fact cap.',
        duplicate_policy='Sampling units deduplicated; every masked copy and inverse removed; all unmasked multiplicities retained. Stage6 original sampler baseline retains its original deduplication behavior.',
        environments=dict(A='Uniform within common hidden pool (natural frequency)',B='Round-robin relations (long-tail coverage subject to available facts)',
             C='Round-robin 3x3 observed head/tail degree bins',D='Round-robin terciles of relation hypergraph incidence support (forward+inverse)'),
        matching='Shared physical graph and common5% hidden fact pool for A-D at all densities; equal query counts, globally disjoint selection/confirmation; overlaps across environments and density documented.',
        query_budget='min(192,max(64,ceil(2*sqrt(unique_forward_facts)))) is an inference query budget, never caps removed facts. Per environment min(60%/40% budget, available common pool split).',
        primary_method='robust_lcb',robust_objective='argmax_alpha min over12 environment x density selection LCB; ties smaller alpha; then candidate only confirmation minLCB>0 else0',
        methods=['reference','fixed025','stage6_mean','stage6_lcb','single_mean','single_lcb','equal_mean_unconfirmed','equal_mean','sample_weighted','longtail_weighted','worst_mean','robust_lcb','four_env_lcb_10','three_density_A_lcb','four_env_mean_10','random'],
        weights=dict(equal='1/12',sample='selection query counts',longtail='B weight2, others1, equal across densities'),
        single_environment='A at10% density; Stage6 baselines use original capped15% sampler and independent selection/confirmation',
        factorial_controls='four_env_lcb_10: four environments at10% only; three_density_A_lcb: environmentA at3 densities only; four_env_mean_10: equal4-environment mean at10% then meanLCB confirmation. Isolates distribution from masking density.',
        confirmation='Per independent method one candidate fixed before confirmation; no retries. single_mean/stage6_mean/equal_mean_unconfirmed deliberate no-confirmation controls; average methods use weighted sum of environment LCBs; worst/robust require every LCB>0.',
        confirmation_min_facts=16,selection_min_facts=24,lcb_kappa=1.64,
        reference='Same Stage5 T-4D checkpoint and relation weights at alpha0; no calibration; not called old B1',
        random='Uniform fixed alpha grid per graph/training/proxy seed using SHA256 plus seed7197; no test labels',
        numeric=dict(path='Query-wise FP32 entity Linear GEMMs, canonical relation pairs IDs(2k,2k+1)',tf32=False,batch_size=8,oom_halving=True,
                     ties='Unchanged pessimistic >=; no score rounding or tie tolerance; no ensemble',all_methods_same=True),
        numeric_choice='Chosen by exact batch invariance on source/historical mechanism probes, never MRR',
        inference='Actual forward per alpha, no interpolation. Source-trained weights frozen. Exact reuse within identical model/graph/config/query identities only.',
        paired_unit='One forward fact, average head/tail RR first; train-seed averages do not pool scores/ranks',
        bootstrap=5000,bootstrap_seed=7197,
        multiplicity='Primary robust_lcb vs stage6_lcb prespecified. Other methods/correlations exploratory; unadjusted paired95% intervals plus Holm correction on main graphwise comparisons. Overlapping environments/densities not IID.',
        formal_grid='All5 alpha for diagnostic correlations (never target-driven selection); B0 official, real historical B1/B2 checkpoints under the same numeric path.',
        historical_replay='Post-freeze replay old Stage6 frozen topology_lcb alphas on the new common numeric path for historical11 only. Numeric regression, not a new selector or a baseline on new graphs.',
        exposure='HM1k and WikiTopicsMT4 health prospective label results but source-family dependence; Aristo/UMLS external sources have transductive within-corpus splits. Not three fully independent standard inductive benchmarks.',
        compatibility=compatibility,
        failure='Save actual OOM/failure and continue independent graphs. No hidden smaller query count or different masks. No test-driven hyperparameters.',
        stage6_frozen_sha256=sha(ROOT.parent/'stage6/frozen.json'),model_manifest_sha256=sha(ROOT.parent/'stage6/model_manifest.json'),
        new_observation_manifest_sha256=sha(ROOT/'observations/manifest.json'),
        source_mechanism_summary_sha256=sha(ROOT/'results/mechanism_summary.json'),numeric_checks_sha256=sha(ROOT/'results/gpu_checks.json'))
    value['historical_exposure_search_sha256']=sha(ROOT/'results/historical_exposure_search.json')
    value['pretraining_observation_overlap_sha256']=sha(ROOT/'results/pretraining_observation_overlap.json')
    value['compatibility_preflight_sha256']=sha(ROOT/'results/new_graph_preflight.json')
    value['proxy_multiplicity_verification_sha256']=sha(ROOT/'results/proxy_multiplicity_verification.json')
    dump(ROOT/'protocol.json',value,exclusive=True)
    log('protocol_frozen',targets=len(targets),sha256=sha(ROOT/'protocol.json'))
