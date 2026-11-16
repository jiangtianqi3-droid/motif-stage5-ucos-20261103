"""Audit assets before any scoring or API call; this does not claim a loaded model."""
import importlib.metadata
import subprocess
import numpy as np
from common_io import ROOT, S7, OLD11, M5RUN, HISTORY, SEEDS, read, dump, sha
from regime_inspector import records, observation, vocabulary
from evidence import EvidenceGraph
from prompting import Lexicon

config = read(ROOT/'config/experiment.json')
manifest = read(S7/'model_manifest.json')
checkpoints = {}
for seed in SEEDS:
    item = manifest['B2_s'+str(seed)]
    p = HISTORY/'topology-semantic-stage4-20261102'/item['checkpoint']
    actual = sha(p)
    assert actual == item['checkpoint_sha256']
    assert item['config']['id'] == 'B2' and item['config']['topology'] == 'pmi'
    checkpoints[str(seed)] = dict(path=str(p), sha256=actual, config=item['config'])
protected = {}
for folder in [S7, S7/'runtime', OLD11/'src', ROOT.parent/'conceptnet_zeroshot_v1/src']:
    for p in folder.rglob('*.py'): protected[str(p)] = sha(p)
for name in ['model_manifest.json','inputs.json','frozen.json']:
    protected[str(S7/name)] = sha(S7/name)
graphs = []
for rec in records():
    z = observation(rec); names, rels, source = vocabulary(rec,z)
    # Lightweight graph surface suffices for coverage; signatures are observation-only at runtime.
    lex = Lexicon(names, rels, None, [ROOT/'cache/lexical/FB15k-237.tsv',ROOT/'cache/lexical/WN18RR.tsv'])
    old = M5RUN/'per_graph'/rec['key']
    complete = read(old/'complete.json')
    assert sha(old/'rankings.npz') == complete['ranking_sha256']
    with np.load(old/'rankings.npz') as data:
        assert data['B2/M2'].shape == data['B2/M5'].shape
        count = len(data['queries'])
    graphs.append(dict(key=rec['key'], n=z['n'], forward_relations=z['R']//2,
        observation_sha256=rec['sha256'], query_count=count, lexical=lex.audit(),
        m5_rankings=str(old/'rankings.npz'), m5_sha256=complete['ranking_sha256']))
    print(rec['key'], count, 'queries; lexical', round(lex.audit()['real_text_coverage'],3), flush=True)
versions = {}
for name in ['torch','numpy','scipy','networkx','tiktoken','pytest','tqdm']:
    versions[name] = importlib.metadata.version(name)
gitdir = (ROOT.parents[1]/'.git').read_text().strip().split('gitdir: ',1)[1]
if gitdir.startswith('D:/'): gitdir = '/mnt/d/'+gitdir[3:]
git = lambda *args: subprocess.check_output(['git','--git-dir='+gitdir,*args],text=True).strip()
result = dict(family='B2+D*', config=config, checkpoints=checkpoints, dependencies=versions,
    main_commit=git('rev-parse','HEAD'), query_community_commit=git('rev-parse','origin/experiment/query-community-v1'),
    protected_sources=protected, target_graphs=graphs, model_actually_loaded=False,
    qualification='Checkpoint SHA and source audit only. Actual load/state hash proof is produced by score smoke.',
    historical_protocol='GraIL valid_ind+test_ind; independent training graph validation; all targets previously exposed')
dump(ROOT/'freeze_manifest.json',result)
