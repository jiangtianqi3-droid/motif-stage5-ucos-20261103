"""Create isolated runtime; read historical inputs without changing them."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'topology-semantic-stage4-20261102'
if (ROOT/'frozen.json').exists():raise RuntimeError('Do not reinitialize a frozen experiment')
if list((ROOT/'results').glob('training_*_s*.json')):raise RuntimeError('Do not overwrite an experiment after source training starts')
for name in ['analysis','results','data','semantic','cache','curves','checkpoints','evaluation','logs','triton_cache']:
    (ROOT/name).mkdir(exist_ok=True)
for name in ['plans.py','weighted.py','incident_api.py','topology.py','gated_model.py']:
    shutil.copy2(OLD/name,ROOT/name)
shutil.copytree(OLD/'runtime',ROOT/'runtime',dirs_exist_ok=True)
shutil.copytree(OLD/'triton_cache',ROOT/'triton_cache',dirs_exist_ok=True)
run=(OLD/'run.sh').read_text().replace(OLD.name,ROOT.name)
(ROOT/'run.sh').write_bytes(run.replace('\r\n','\n').encode())
maps=json.loads((OLD/'semantic/relation_mappings.json').read_text(encoding='utf-8'))
maps.update(json.loads((OLD/'semantic/external_relation_mappings.json').read_text(encoding='utf-8')))
(ROOT/'semantic/relation_mappings.json').write_text(json.dumps(maps,ensure_ascii=False,indent=2),encoding='utf-8')
protocol=dict(stage='5-corrected',seeds=[1024,2026,3407],source_validation_queries=512,
    validation_steps=[0,100,250,500],steps=500,batch_size=2,num_negative=64,
    gate_lr=.001,relation_lr=.000002,weight_decay=.01,gradient_clip=1.,
    alpha_topo_grid=[.15,.2,.25,'adaptive'],freedoms=['gate_only','joint_low_lr'],
    alpha_adaptive='0.15+0.10*CV/(1+CV), CV=std(raw_s_pmi)/max(abs(mean(raw_s_pmi)),1e-6); observed graph only',
    uncertainty=dict(alpha_max=.20,gamma=2.,tau0=.5,formula='0.2*sigmoid(-2*max(z_freq,z_pmi)-0.5)*opacity'),
    semantic_features=['z_type(0.5*(mean_nonself_cos+max_nonself_cos))','z_type(mean_adjacent_slot_cos)'],
    query_alignment='exclude slots with relation==query; empty context zero after standardization; only transparent context slots',
    opacity='normalized lexical name contains letters and is not an ID pattern; all occupied slots AND query must be transparent',
    semantic_parameters='exactly two shared scalars per layer; no bias/projection/semantic-query state interaction',
    hub_lambda_grid=[0,.001,.003,.01,.03,.1,.3],hub_degree='observed inverse-augmented entity graph incoming degree; same definition for both directions',
    source_schedule='same stage4 RNG; 25-step blocks across three sources and five stage2 masked folds',
    selection='screen all8 topology settings seed1024 using source macroMRR; retain best setting per freedom and replicate both at2026,3407; choose main topology freedom by seed1024 source macroMRR; U and shuffle use that fixed architecture; each seed chooses own step; include step0; no target selection',
    shuffle_seed=8111,bootstrap_replicates=5000,bootstrap_seed=20261105,
    bootstrap_unit='paired original triple with both directions; average seed-wise reciprocal ranks then resample',
    targets=json.loads((OLD/'frozen.json').read_text())['target_specs'],
    boundary='historically exposed locked/historical/external replication; P0 zero target updates and no target validation selection; no claim of globally unseen tests',
    corrections=['sigmoid suppression is not an exact protection guarantee','two scalar kernel removes coordinate lookup channel but does not prove information-theoretic impossibility of ID inference','shuffle can still outperform by chance; empirical outcome never constrained','variable edge alpha means final incident group mean need not equal1; no final renormalization that would reactivate masked edges'],
    report_reference='report_2.md absent; local completed stage4 report.md matches cited table values')
(ROOT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
paths=[p for p in OLD.glob('*.py')]+[OLD/'report.md',OLD/'protocol.json',OLD/'frozen.json',OLD/'semantic/relation_mappings.json',OLD/'semantic/external_relation_mappings.json']
paths+=list((OLD/'analysis').glob('query_masks_*.npz'))
hashes={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(ROOT/'analysis/readonly_inputs_sha256.json').write_text(json.dumps(hashes,indent=2))
print('INITIALIZED',ROOT,'protected references',len(hashes))
