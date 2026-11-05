"""Supplemental fixed-score control registered before formal queries are opened."""
from stage6.util import ROOT, read, dump, sha


CONTROL = dict(name='matched signed global degree control', version=1,
    gamma_grid=[-.03, -.01, 0., .01, .03, .1],
    scope='fixed parameter diagnostic only; no additional target parameter selection',
    reason='disentangle query conditioning from the sign/range of the global degree coefficient',
    registration='added after observation-proxy diagnostics, before any stage6 formal query evaluation',
    primary_policy='primary protocol, selected configurations and LCB decisions are untouched',
    infer='reuse the same exact neural scores; only the permitted candidate-score correction differs')


def register():
    path = ROOT / 'matched_global_protocol.json'
    if path.exists():
        assert read(path) == CONTROL
    else:
        dump(path, CONTROL, exclusive=True)
    return sha(path)


def append(configs, alphas):
    result = list(configs)
    for alpha in alphas:
        for gamma in CONTROL['gamma_grid']:
            c = [alpha, gamma, 'global']
            if gamma != 0 and c not in result:
                result.append(c)
    return result
