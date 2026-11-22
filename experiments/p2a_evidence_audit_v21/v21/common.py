from pathlib import Path
import yaml

from src.io_utils import read, dump, sha, identity, atomic, write_csv
from p1.assets import verify

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'outputs'


def config():
    cfg = yaml.safe_load((ROOT/'configs/audit_config.yaml').read_text(encoding='utf-8'))
    exp = cfg['experiment']
    verify(exp['name']=='p2a_evidence_audit_v21' and exp['dataset']=='FBIngram_25','unsupported Stage 1 protocol')
    verify(all(exp[k] is False for k in ('enable_api','enable_llm','train_backbone')),'forbidden Stage 1 flag')
    verify(all(v is True for v in cfg['evaluation'].values()),'all V2.1 freezing constraints are required')
    verify(cfg['audit']['max_path_hops']==3 and cfg['audit']['candidate_k']==12,'historical path/candidate protocol')
    verify(cfg['hardware']['device']=='cuda','use original CUDA operator')
    return cfg
