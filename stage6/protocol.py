from stage6.util import ROOT, dump, read, sha


PROTOCOL = dict(
    name='TSDC', version=1,
    setting='historically exposed targets; no target valid/test labels; no target gradient updates; observation-only self-supervised parameter selection',
    checkpoint_family='stage5 T-4D gate_only; one independent calibration per training seed',
    training_seeds=[1024, 2026, 3407], proxy_seeds=[6101, 6102, 6103], primary_proxy_seed=6101,
    alpha_grid=[0., .1, .25, .5, 1.], gamma_grid=[-.03, -.01, 0., .01, .03, .1],
    global_gamma_grid=[0., .001, .003, .01, .03, .1, .3],
    source_rates=[.05, .10, .15], target_rate=.15, max_proxy_facts=256,
    min_endpoint_remaining_degree=2, protect_spanning_forest=True, preserve_relation_coverage=True,
    sampling='unique forward facts; frequency/endpoint-degree 2x2 strata; seeded round-robin; no constraint relaxation',
    select_fraction=.6, min_selection_facts=24, min_confirmation_facts=16, lcb_kappa=1.64,
    reference=dict(checkpoint='same T-4D checkpoint and all its relation-layer weights', alpha=0., gamma=0., mode='conditional'),
    reference_warning='not called old B1; T-4D gate_only backbone verified bitwise equal to official, but inference reduction path can differ',
    ranking='stage5 all candidates, two directions, pessimistic >= ties, legal observation positives only in proxy filter',
    paired_unit='one original fact, mean reciprocal rank over head and tail', bootstrap=5000, bootstrap_seed=6713,
    primary_method='joint_lcb', comparisons=['reference', 'fixed', 'cv', 'joint_mean', 'joint_lcb', 'random',
        'topology_lcb', 'entity_lcb', 'global_lcb', 'joint_global_lcb'],
    random_rule='uniform joint conditional grid, RNG from sha256(graph key, training seed, proxy seed, 6629)',
    cv_rule='alpha=.15+.10*CV/(1+CV), CV of current graph raw exclusivity support, gamma=0',
    formal_policy='freeze ALL 11 graphs x 3 training seeds x 3 proxy seeds and control decisions before opening any formal query or group mask',
    confirm_policy='one selection winner per predeclared independent ablation; no tuning or candidate retry using confirmation',
    diagnostics='source-only alpha stability, masking/inverse leakage, directions, real forward difference, B0 repeat; no target-driven changes',
    multiplicity='unadjusted paired intervals are exploratory; primary joint_lcb separately declared; no family-wise significance claim',
    targets=[['locked', 'NELLInductive_v2'], ['locked', 'NELLInductive_v3'], ['locked', 'NELLInductive_v4'], ['locked', 'NLIngram_50'],
             ['historical', 'NELLInductive_v1'], ['historical', 'NLIngram_25'], ['historical', 'FB15k237Inductive_v1'],
             ['historical', 'WN18RRInductive_v1'], ['historical', 'FBIngram_25'], ['historical', 'WKIngram_25'],
             ['external', 'ConceptNet100k_external']],
)


def register():
    p = ROOT / 'protocol.json'
    if p.exists():
        assert read(p) == PROTOCOL
    else:
        dump(p, PROTOCOL, exclusive=True)
    return PROTOCOL


def configs(protocol, cv_alpha):
    cond = [[a, g, 'conditional'] for a in protocol['alpha_grid'] for g in protocol['gamma_grid']]
    glob = [[a, g, 'global'] for a in protocol['alpha_grid'] for g in protocol['global_gamma_grid'] if g != 0]
    return cond + glob + [[cv_alpha, 0., 'conditional']]
