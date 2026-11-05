"""Candidate scoring only. Global degree counts inverse-augmented incoming edges."""
import numpy as np
import torch


class DegreeCalibrator:
    def __init__(self, graph, device='cuda'):
        self.n, self.R = graph.n, graph.R
        global_degree = np.bincount(graph.edge_index[1], minlength=graph.n)
        conditional = np.bincount(graph.edge_type * graph.n + graph.edge_index[1],
                                  minlength=graph.R * graph.n).reshape(graph.R, graph.n)
        assert np.array_equal(conditional.sum(0), global_degree)
        self.global_log = torch.tensor(np.log1p(global_degree), dtype=torch.float32, device=device)
        self.conditional_log = torch.tensor(np.log1p(conditional), dtype=torch.float32, device=device)

    def transformed_relation(self, relation, head=False):
        return (relation + self.R // 2) % self.R if head else relation

    def penalty(self, relation, head=False, mode='conditional'):
        if mode == 'global':
            return self.global_log[None].expand(len(relation), -1)
        assert mode == 'conditional'
        q = self.transformed_relation(relation, head)
        return self.global_log[None] - self.conditional_log[q]

    def apply(self, scores, relation, gamma, head=False, mode='conditional'):
        if gamma == 0:
            return scores
        return scores - gamma * self.penalty(relation, head, mode)
