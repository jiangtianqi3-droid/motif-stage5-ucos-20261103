"""Exact all-candidate double-direction evaluator; one real forward per alpha.

Only gamma calibrations reuse the SAME scores, because they act after propagation.
Relation representation cache lifetime is one model/graph/alpha invocation.
"""
import gc
import time
import types
import numpy as np
import torch
from motif import tasks
from stage6.degree_calibrator import DegreeCalibrator
from stage6.models import set_alpha


def metrics(ranks):
    r = np.asarray(ranks, dtype=float)
    return dict(mrr=float((1 / r).mean()), hits_at_1=float((r <= 1).mean()),
                hits_at_3=float((r <= 3).mean()), hits_at_10=float((r <= 10).mean()))


@torch.no_grad()
def evaluate(model, data, truth, graph, configs, batch_size=8, collect=False):
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    triples = torch.cat([data.target_edge_index, data.target_edge_type[None]]).T
    ranks = np.zeros((len(configs), len(triples), 2), dtype=np.int64)
    cal = DegreeCalibrator(graph, triples.device)
    began = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    alpha_records = []
    evidence = []
    for alpha in dict.fromkeys(c[0] for c in configs):
        set_alpha(model, alpha)
        original = model.relation_model.forward
        memory = {}
        def cached_forward(self, hyper, query):
            missing = sorted(set(query.cpu().tolist()) - set(memory))
            for at in range(0, len(missing), 2):
                ix = missing[at:at + 2]
                out = original(hyper, torch.tensor(ix, device=query.device))
                assert not out.requires_grad and torch.isfinite(out).all()
                for j, k in enumerate(ix):
                    memory[k] = out[j].clone()
            return torch.stack([memory[k] for k in query.cpu().tolist()])
        model.relation_model.forward = types.MethodType(cached_forward, model.relation_model)
        started = time.perf_counter()
        try:
            for at in range(0, len(triples), batch_size):
                batch = triples[at:at + batch_size]
                tail, head = tasks.all_negative(data, batch)
                t, h = model(data, tail), model(data, head)
                assert torch.isfinite(t).all() and torch.isfinite(h).all()
                tm, hm = tasks.strict_negative_mask(truth, batch)
                for j, (a, gamma, mode) in enumerate(configs):
                    if a != alpha:
                        continue
                    tc = cal.apply(t, batch[:, 2], gamma, False, mode)
                    hc = cal.apply(h, batch[:, 2], gamma, True, mode)
                    ranks[j, at:at + len(batch), 0] = tasks.compute_ranking(tc, batch[:, 1], tm).cpu().numpy()
                    ranks[j, at:at + len(batch), 1] = tasks.compute_ranking(hc, batch[:, 0], hm).cpu().numpy()
                if collect:
                    # Compact true-vs-filtered-error evidence, using only eval-phase data.
                    for direction, scores, mask, target in [(0, t, tm, batch[:, 1]), (1, h, hm, batch[:, 0])]:
                        q = cal.transformed_relation(batch[:, 2], bool(direction))
                        gl = cal.global_log[None].expand(len(batch), -1)
                        cl = cal.conditional_log[q]
                        wrong = scores.masked_fill(~mask, -torch.inf).argmax(1)
                        true_score = scores.gather(1, target[:, None]).squeeze(1)
                        wrong_score = scores.gather(1, wrong[:, None]).squeeze(1)
                        for k in range(len(batch)):
                            evidence.append([alpha, at + k, direction, int(target[k]), int(wrong[k]),
                                float(gl[k, target[k]]), float(cl[k, target[k]]), float(gl[k, wrong[k]]),
                                float(cl[k, wrong[k]]), float(true_score[k]), float(wrong_score[k]),
                                int((scores[k] == true_score[k]).sum())])
        finally:
            model.relation_model.forward = original
        torch.cuda.synchronize()
        alpha_records.append(dict(alpha=alpha, seconds=time.perf_counter() - started, cached_relations=len(memory)))
        del memory
    torch.cuda.synchronize()
    return ranks, dict(seconds=time.perf_counter() - began, peak_allocated=torch.cuda.max_memory_allocated(),
                       batch_size=batch_size, queries=len(triples), alpha_records=alpha_records), np.asarray(evidence)


def safe_evaluate(model, data, truth, graph, configs, batch_size=8, collect=False):
    while True:
        try:
            return evaluate(model, data, truth, graph, configs, batch_size, collect)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            gc.collect()
            torch.cuda.empty_cache()
