"""No calibration/training. Canonical relation batches and exact pessimistic ranks."""
import time
import types
import gc
import numpy as np
import torch
from motif import tasks
from stage6.models import set_alpha


def metrics(r):
    r = np.asarray(r, dtype=float)
    return dict(mrr=float((1/r).mean()), hits_at_1=float((r<=1).mean()),
                hits_at_3=float((r<=3).mean()), hits_at_10=float((r<=10).mean()))


@torch.no_grad()
def evaluate(model, data, truth, alphas, batch_size=8, collect_scores=False):
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    triples = torch.cat([data.target_edge_index, data.target_edge_type[None]]).T
    ranks = np.zeros((len(alphas), len(triples), 2), dtype=np.int64)
    scores_out = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    records = []
    for ia, alpha in enumerate(alphas):
        set_alpha(model, alpha)
        native = model.relation_model.forward
        # Canonical sorted pairs independent of query order and entity batch size.
        needed=set(triples[:,2].cpu().tolist())
        needed |= {(r+data.num_relations//2)%data.num_relations for r in list(needed)}
        rs=sorted({v for r in needed for v in (2*(r//2),2*(r//2)+1)})
        memory = {}
        for i in range(0, len(rs), 2):
            keys = rs[i:i+2]
            padded = keys if len(keys) == 2 else keys * 2
            out = native(data.relation_hypergraph, torch.tensor(padded, device=triples.device))
            for j, r in enumerate(keys):
                memory[r] = out[j].clone()
        def cached(self, hyper, query):
            return torch.stack([memory[r] for r in query.cpu().tolist()])
        model.relation_model.forward = types.MethodType(cached, model.relation_model)
        scores = []
        began = time.perf_counter()
        try:
            for at in range(0, len(triples), batch_size):
                b = triples[at:at+batch_size]
                tail, head = tasks.all_negative(data, b)
                st, sh = model(data, tail), model(data, head)
                assert torch.isfinite(st).all() and torch.isfinite(sh).all()
                tm, hm = tasks.strict_negative_mask(truth, b)
                ranks[ia, at:at+len(b), 0] = tasks.compute_ranking(st, b[:, 1], tm).cpu().numpy()
                ranks[ia, at:at+len(b), 1] = tasks.compute_ranking(sh, b[:, 0], hm).cpu().numpy()
                if collect_scores:
                    scores.append(torch.stack([st, sh], 1).cpu().numpy())
        finally:
            model.relation_model.forward = native
        torch.cuda.synchronize()
        records.append(dict(alpha=alpha, entity_seconds=time.perf_counter()-began, cached_relations=len(rs)))
        if collect_scores:
            scores_out.append(np.concatenate(scores))
        del memory
    return ranks, dict(seconds=time.perf_counter()-started, peak_allocated=torch.cuda.max_memory_allocated(),
                       batch_size=batch_size, facts=len(triples), alpha_records=records), scores_out


def safe_evaluate(model, data, truth, alphas, batch_size=8, collect_scores=False):
    while True:
        try:
            return evaluate(model, data, truth, alphas, batch_size, collect_scores)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size //= 2
            gc.collect()
            torch.cuda.empty_cache()
