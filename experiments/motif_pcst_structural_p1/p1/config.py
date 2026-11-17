from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
FLAGS = dict(stage='P1', structural_only=True, enable_llm=False, enable_api=False,
             enable_krlm=False, allow_backbone_training=False)


def load_config():
    cfg = yaml.safe_load((ROOT/'config/p1.yaml').read_text(encoding='utf-8'))
    if cfg['experiment'] != FLAGS or any(type(cfg['experiment'][k]) is not type(v) for k,v in FLAGS.items()):
        raise ValueError('P1 requires all structural-only flags')
    r = cfg['retrieval']
    if r['max_hops'] != 3 or not all(r[k] is True for k in ('use_query_ppr','preserve_direction')):
        raise ValueError('P1 uses directed three-hop PPR')
    return cfg
