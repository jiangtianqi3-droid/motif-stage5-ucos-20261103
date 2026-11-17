"""Hash and adapt P1 frozen assets without writing to the P1 tree."""
from pathlib import Path
import subprocess

import numpy as np
import yaml

from p1.assets import Assets, Evaluator, verify, load
from p1.config import load_config as p1_config
from src.io_utils import HISTORY, STAGE7, PRIMARY, atomic, identity, read, sha

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results'
P1 = ROOT.parent/'motif_pcst_structural_p1'
P1_SHA = '180f77feccfea1f7dfc26a86cb2162073444df55'
CHECKPOINT = '1e85a7e85b57afa14b5b79813bab85c0fd075323b16ac05a5948f200c2aed790'
GRAPHS = dict(WN18RRInductive_v1='expanded', FBIngram_25='expanded',
              FB15k237Inductive_v1='transfer', NELLInductive_v1='transfer')
PAIR_GRAPHS = tuple(GRAPHS)[:2]


def config():
    cfg = yaml.safe_load((ROOT/'configs/experiment.yaml').read_text(encoding='utf-8'))
    strict = dict(stage='P2',structural_only=True,enable_api=False,enable_llm=False,train_backbone=False)
    verify(all(type(cfg[k]) is type(v) and cfg[k]==v for k,v in strict.items()),'P2 must be structural only')
    verify(cfg['candidate_topk']==12 and cfg['evidence_budget']==32,'P1 cache protocol mismatch')
    return cfg


def p1_root():
    state = read(P1/'results/status.json')
    verify(state['state']=='complete','P1 has not completed')
    return P1/'results'/state['run_id']


def preflight(cfg):
    root = p1_root()
    audit = read(root/'audit.json')
    verify(audit['commit']==P1_SHA and audit['checkpoint']['checkpoint_sha256']==CHECKPOINT,'P1 audit identity')
    for path, digest in audit['stage7_files'].items():
        verify(sha(STAGE7/path)==digest,f'Stage-7 source changed: {path}')
    cp = HISTORY/'topology-semantic-stage4-20261102'/audit['checkpoint']['checkpoint']
    verify(sha(cp)==CHECKPOINT,'actual checkpoint hash differs')
    cfg_dstar = read(STAGE7/'entity_frozen.json')['Dstar']['config']
    verify(cfg_dstar==dict(kind='D-Sender',gamma=.75,alpha=.20,anchor=True),'Dstar config changed')
    for path in (P1/'p1').glob('*.py'):
        past = subprocess.check_output(['git','show',f'{P1_SHA}:experiments/{P1.name}/p1/{path.name}'],cwd=ROOT)
        verify(past.replace(b'\r\n',b'\n')==path.read_bytes().replace(b'\r\n',b'\n'),f'P1 code changed: {path.name}')
    run_hash = identity(dict(code={p.name:sha(p) for p in sorted((ROOT/'p2').glob('*.py'))},config=cfg,
                             p1=root.name,p1_report=sha(P1/'report_structural_p1.md'),checkpoint=CHECKPOINT))
    lines = ['# P1 资产审计','',f'P1 提交 `{P1_SHA}`，运行 `{root.name}`，正式报告 SHA256 为 `{sha(P1/"report_structural_p1.md")}`。',
             f'检查点已逐字节校验为 `{CHECKPOINT}`。D* 为均值锚定的发送端校准，γ=0.75、α=0.20。',
             'P1 的 R2、R3、R4 使用同一候选骨干。R3 新增 B2 奖励只进入 PCST 扩展。R4 调整边成本。'
             '候选评分重新从选中事实提取路径，并在最多四个关系模式中取一个最高值，候选评分函数不接收 B2 奖励。',
             '代码见 P1 retrieval.py 的 Retriever.query、Retriever.describe，scoring.py 的 evidence、permutations、filtered_after。'
             '先用独立重算检查缓存来源，再用干预区分路径截断、单条最佳聚合和秩层级写回。',
             '沿用 P1 GraIL 合并 valid_ind/test_ind 和 FBIngram 测试划分，历史目标标签只在离线排行核验读取。'
             '历史 M5 三种子按同一查询行核验，适配协议与 P2 分列。']
    atomic(RESULTS/'p2_p1_audit.md','\n\n'.join(lines)+'\n')
    return run_hash,root,audit


class Target:
    def __init__(self,key,root,working):
        self.key = key
        self.asset = Assets(key,working)
        self.graph = self.asset.graph
        self.dir = root/'runs'/GRAPHS[key]/key
        samples=read(self.dir/'samples.json')
        self.indices = samples['indices']
        provenance=read(root/'assets'/key/'prepared.json')
        identity_pair=(self.asset.meta['parameter_sha256_before'],self.asset.entry['observed_sha256'])
        verify((samples['checkpoint'],samples['observations'])==identity_pair
               and (provenance['checkpoint'],provenance['observation'])==identity_pair,f'{key} sample/score identity')
        path = self.asset.open_previous()
        if path:
            self.raw = np.load(path,mmap_mode='r',allow_pickle=False)
        else:
            path = root/'assets'/key/'scores.npz'
            proof = read(path.with_name('scores_hash.json'))
            verify(proof['sha256']==sha(path) and (proof['checkpoint'],proof['graph'])==identity_pair,'local recovery cache')
            a=load(path)
            self.raw=dict(zip(map(int,a['indices']),a['scores']))
        self.gate_dir = root/'assets'/key/'gates'
        self.gates = {}
        self.provenance = identity_pair

    def gate(self,r):
        if r not in self.gates:
            a=load(self.gate_dir/f'{r}.npz')
            verify(a['layers'].shape==(6,self.graph.R) and np.isfinite(a['layers']).all(),'gate vocabulary')
            self.gates[r]=a['layers'].mean(0)
        return self.gates[r]

    def job(self,i,d,k=12):
        path = self.dir/f'k{k}_b32'/f'{i}-{d}.json'
        if k!=12:
            path=self.dir.parent.parent/'sensitivity'/self.key/f'k{k}_b32'/f'{i}-{d}.json'
        job=read(path)
        raw=self.raw[i][d]
        from p1.scoring import raw_order
        expected=raw_order(raw)[:30]
        verify(job['query_id']==f'{self.key}:{i}:{d}' and job['candidates']==expected[:k] and job['top30']==expected,
               'cached job/score mismatch')
        verify(list(self.asset.queries[i,d])==[job['head'],job['relation']],'candidate query identity')
        return job,raw
