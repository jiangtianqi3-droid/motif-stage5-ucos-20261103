"""Sparse observation-only structural context. No label or truth-index input."""
import numpy as np
import scipy.sparse as sp

def normalize(x):
    if sp.issparse(x):
        length = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        return sp.diags(1 / np.maximum(length, 1e-12)) @ x
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

def safe_z(x):
    x = np.asarray(x, dtype=np.float64)
    return np.clip((x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6), -5, 5)

class Context:
    def __init__(self, z, lexical=None):
        self.n, self.R = z['n'], z['R']
        self.u, self.v = z['edge_index']
        self.r = z['edge_type']
        u, v, r = self.u, self.v, self.r
        self.degree = np.bincount(u, minlength=self.n)
        counts = sp.csr_matrix((np.ones(len(u)), (u, r)), shape=(self.n, self.R))
        frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        signature = counts.copy()
        signature.data = np.log1p(signature.data)
        signature = normalize(signature.multiply(np.log(self.n / (1 + frequency))))
        adjacency = sp.csr_matrix((np.ones(len(u)), (u, v)), shape=(self.n, self.n))
        self.profile = normalize(signature + .3 * normalize(adjacency @ signature)).tocsr()
        self.tail_counts = sp.csr_matrix((np.ones(len(u)), (r, v)), shape=(self.R, self.n))
        self.center = normalize(self.tail_counts @ self.profile).tocsr()
        self.local = {}
        for a, b, c in zip(u, v, r):
            self.local.setdefault((int(a), int(c)), []).append(int(b))
        local_degree = np.asarray(counts[u, r]).ravel()
        self.edge_weight = ((1 + self.degree[u]) ** -.75 * (1 + local_degree) ** -.50).astype(np.float32)
        self.rho = np.zeros((self.R, self.R), np.float32)
        he, ht, pmi = z['hyper_edge'], z['hyper_type'], z['raw'][:, 3]
        for kind in range(7):
            ix = np.flatnonzero(ht == kind)
            arity = 2 if kind < 3 else 3
            for a in range(arity):
                for b in range(arity):
                    if a != b:
                        np.maximum.at(self.rho, (he[a, ix], he[b, ix]), pmi[ix])
        self.rho /= np.maximum(self.rho.max(axis=1, keepdims=True), 1e-12)
        np.fill_diagonal(self.rho, 1.)
        self.lexical = None if lexical is None else normalize(np.asarray(lexical, dtype=np.float32))
        if self.lexical is not None:
            self.lex_context = normalize(self.lexical + .3 * normalize(adjacency @ self.lexical))
            self.lex_rel = z.get('_lex_rel', np.zeros((self.R, self.lexical.shape[1]), np.float32))
            self.lex_center = normalize(self.tail_counts @ self.lexical)
            self.translation = np.zeros((self.R, self.lexical.shape[1]), np.float32)
            for at in range(0, len(u), 4096):
                sl = slice(at, at + 4096)
                np.add.at(self.translation, r[sl], self.lexical[v[sl]] - self.lexical[u[sl]])
            self.translation = normalize(self.translation)

    def score(self, queries):
        # Query rows contain heads and relations only; withheld tails are not accepted.
        assert queries.ndim == 2 and queries.shape[1] == 2
        reach = np.zeros((len(queries), self.n), np.float32)
        topo, lex = np.zeros_like(reach), np.zeros_like(reach)
        for q in np.unique(queries[:, 1]):
            ix = np.flatnonzero(queries[:, 1] == q)
            transition = sp.csr_matrix((self.edge_weight * self.rho[q, self.r], (self.u, self.v)), shape=(self.n, self.n))
            x = sp.csr_matrix((np.ones(len(ix)), (np.arange(len(ix)), queries[ix, 0])), shape=(len(ix), self.n))
            total = sp.csr_matrix(x.shape)
            for factor in (2., 1., .5):
                x = x @ transition
                total += factor * x
            reach[ix] = np.log1p(total.toarray())
            global_score = (self.center[q] @ self.profile.T).toarray().ravel()
            bias = safe_z(np.log1p(self.tail_counts[q].toarray()).ravel())
            topo[ix] = .6 * global_score + .2 * bias
            for i in ix:
                h = queries[i, 0]
                neighbors = self.local.get((int(h), int(q)), [])
                if neighbors:
                    p = normalize(np.asarray(self.profile[neighbors].sum(axis=0)))
                    topo[i] += np.asarray(self.profile @ p.T).ravel()
                if self.lexical is not None:
                    translation = normalize(self.lex_context[h] + .5 * self.lex_rel[q] + .5 * self.translation[q])
                    lex[i] = self.lex_context @ translation + .5 * (self.lexical @ self.lex_center[q]) + .15 * bias
                    if neighbors:
                        lex[i] += self.lexical @ normalize(self.lexical[neighbors].sum(axis=0))
        zr, zt, zl = safe_z(reach), safe_z(topo), safe_z(lex)
        pure_consensus = (reach > 0) * zr * np.maximum(0., zt)
        full_consensus = (reach > 0) * zr * np.maximum(0., zt + zl)
        return np.stack([zr, zt, zl, safe_z(pure_consensus), safe_z(full_consensus)], axis=1).astype(np.float32)
