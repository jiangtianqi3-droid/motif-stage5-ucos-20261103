"""Capability boundary: this module reads only whitelisted observation bundles.

No loader from common.downstream, no queries / alltrue / evaluation paths.
The bundle schema is closed; extra fields fail validation.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from stage6.util import identity, read, sha


@dataclass(frozen=True)
class ObservedGraph:
    edge_index: np.ndarray
    edge_type: np.ndarray
    n: int
    R: int
    provenance: dict

    def __post_init__(self):
        e, r = self.edge_index, self.edge_type
        assert e.shape == (2, len(r)) and self.R % 2 == 0
        assert self.n > 1 and self.R > 0
        assert not len(r) or (e.min() >= 0 and e.max() < self.n and r.min() >= 0 and r.max() < self.R)
        forward = sorted(map(tuple, np.c_[e[:, r < self.R // 2].T, r[r < self.R // 2]].tolist()))
        backward = sorted(map(tuple, np.c_[e[::-1, r >= self.R // 2].T, r[r >= self.R // 2] - self.R // 2].tolist()))
        assert forward == backward, 'incomplete inverse pairs (including multiplicities)'
        e.setflags(write=False)
        r.setflags(write=False)

    @property
    def key(self):
        return identity(self.edge_index, self.edge_type, metadata=[self.n, self.R])

    @property
    def forward(self):
        keep = self.edge_type < self.R // 2
        return np.unique(np.c_[self.edge_index[0, keep], self.edge_type[keep], self.edge_index[1, keep]], axis=0)

    def inverse(self, relation):
        return (np.asarray(relation) + self.R // 2) % self.R


def load_observed(folder, key):
    folder = Path(folder)
    manifest = read(folder / 'manifest.json')
    rec = manifest[key]
    assert Path(rec['file']).name == rec['file']
    path = folder / rec['file']
    assert sha(path) == rec['sha256']
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'edge_index', 'edge_type', 'n', 'R'}, z.files
        return ObservedGraph(z['edge_index'].astype(np.int64), z['edge_type'].astype(np.int64),
                             int(z['n']), int(z['R']), dict(rec))


def from_forward(facts, n, R, provenance=None):
    f = np.asarray(facts, dtype=np.int64).reshape(-1, 3)
    e = np.concatenate([f[:, [0, 2]].T, f[:, [2, 0]].T], axis=1)
    r = np.r_[f[:, 1], f[:, 1] + R // 2]
    return ObservedGraph(e, r, n, R, provenance or {})
