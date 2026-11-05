"""Adapters use the reviewed stage5 and historical gate interfaces."""
import types
import torch
import common
import model as t4
import gated_model
from stage6.util import ROOT, read, sha


def load_model(stage5, history, official, family, seed):
    manifest = read(ROOT / 'model_manifest.json')
    assert sha(official) == manifest['official_sha256']
    common.CKPT = official
    m = common.base_model(seed).cuda().eval()
    if family == 'T-4D':
        r = manifest['T4'][str(seed)]
        path = stage5 / r['checkpoint']
        assert sha(path) == r['checkpoint_sha256']
        m = t4.install(m, dict(r['config']))
        m.load_state_dict(torch.load(path, map_location='cpu')['model'], strict=True)
    elif family in ('B1', 'B2'):
        r = manifest['historical'][family + '_s' + str(seed)]
        path = history / r['checkpoint']
        assert sha(path) == r['checkpoint_sha256']
        # Historical stable_gate.py method=topology is exactly this Fdim=1 forward,
        # with type normalization and NO residual alpha. Keep its native path at 1.
        m = gated_model.install(m, method='C0' if family == 'B1' else 'T3', freedom='joint', normalization='type')
        m.load_state_dict(torch.load(path, map_location='cpu')['model'], strict=True)
    else:
        assert family == 'B0'
        path = official
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    m.tsdc_native_forward = m.relation_model.forward
    m.tsdc_family = family
    m.tsdc_checkpoint_sha256 = sha(path)
    return m


def set_alpha(m, alpha):
    if m.tsdc_family == 'T-4D':
        m.relation_model.stage5_config['alpha'] = alpha
    elif m.tsdc_family == 'B2' and alpha != 1.:
        # Separate diagnostic intervention on the TRUE single-channel B2 gate.
        m.relation_model.stage5_config = dict(alpha=alpha, semantic=False)
        m.relation_model.forward = types.MethodType(t4.forward, m.relation_model)
    else:
        m.relation_model.forward = m.tsdc_native_forward


def state_hash(m):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(m.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
