import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stage6.observed_loader import from_forward, load_observed
from stage6.degree_calibrator import DegreeCalibrator
from stage6.proxy_split import make_split
from stage6.select import choose
from stage6.util import ROOT, read


class CoreTests(unittest.TestCase):
    def setUp(self):
        facts = [(h, (h + t) % 3, t) for h in range(14) for t in range(14) if h != t]
        self.g = from_forward(np.asarray(facts + facts[:3]), 15, 6)

    def test_zero_is_identity(self):
        c = DegreeCalibrator(self.g, 'cpu')
        s = torch.randn(2, 15)
        self.assertIs(c.apply(s, torch.tensor([0, 1]), 0), s)

    def test_directions_and_consistent_counts(self):
        c = DegreeCalibrator(self.g, 'cpu')
        self.assertEqual(c.transformed_relation(torch.tensor([0]), True).item(), 3)
        f = self.g.edge_index
        degree = np.bincount(f[1], minlength=self.g.n)
        for q in range(6):
            cond = np.bincount(f[1, self.g.edge_type == q], minlength=self.g.n)
            expected = np.log((1 + degree) / (1 + cond))
            np.testing.assert_allclose(c.penalty(torch.tensor([q]))[0].numpy(), expected, atol=2e-7)

    def test_unsupported_is_finite(self):
        c = DegreeCalibrator(from_forward([[0, 0, 1]], 3, 4), 'cpu')
        p = c.penalty(torch.tensor([1]))
        self.assertTrue(torch.isfinite(p).all())
        self.assertAlmostEqual(p[0, 0].item(), np.log(2), places=6)
        self.assertEqual(p[0, 2].item(), 0)

    def test_no_repeat_penalty(self):
        c = DegreeCalibrator(from_forward([[0, 0, 1], [2, 0, 1]], 3, 2), 'cpu')
        self.assertEqual(c.penalty(torch.tensor([0]))[0, 1].item(), 0.)

    def test_joint_removal_and_dedup(self):
        rem, q, cut, info = make_split(self.g, 6101, .15)
        self.assertEqual(info['actual'], int(.15 * len(self.g.forward)))
        self.assertEqual(len(q), len(np.unique(q, axis=0)))
        for h, r, t in q:
            self.assertFalse(np.any((rem.edge_index[0] == h) & (rem.edge_index[1] == t) & (rem.edge_type == r)))
            self.assertFalse(np.any((rem.edge_index[0] == t) & (rem.edge_index[1] == h) & (rem.edge_type == r + 3)))
        self.assertEqual(rem.n, self.g.n)
        self.assertEqual(rem.R, self.g.R)

    def test_split_repeatability_and_cache_identity(self):
        a = make_split(self.g, 6101)
        b = make_split(self.g, 6101)
        c = make_split(self.g, 6102)
        self.assertEqual(a[0].key, b[0].key)
        self.assertNotEqual(a[0].key, c[0].key)

    def test_insufficient_eligible_facts_do_not_relax(self):
        g = from_forward([[0, 0, 1], [1, 0, 2]], 3, 2)
        rem, q, cut, info = make_split(g, 6101, .9)
        self.assertEqual(info['actual'], 0)
        self.assertEqual(q.shape, (0, 3))
        self.assertEqual(len(rem.forward), len(g.forward))

    def test_lcb_fails_and_small_sample_fallback(self):
        cfg = [[0., 0., 'conditional'], [.1, 0., 'conditional']]
        r = np.full((2, 60, 2), 2)
        r[1, :36] = 1
        r[1, 36:] = 4
        result = choose(r, cfg, 36)
        self.assertEqual(result['candidate_index'], 1)
        self.assertEqual(result['final_index'], 0)
        self.assertTrue(result['fallback'])
        self.assertTrue(choose(r[:, :10], cfg, 6)['fallback'])

    def test_confirmation_never_reselects(self):
        cfg = [[0., 0., 'conditional'], [.1, 0., 'conditional'], [.25, 0., 'conditional']]
        r = np.full((3, 60, 2), 3)
        r[1, :36] = 1
        r[1, 36:] = 10
        r[2, 36:] = 1
        result = choose(r, cfg, 36)
        self.assertEqual(result['candidate_index'], 1)
        self.assertEqual(result['final_index'], 0)

    def test_unselected_confirmation_is_inaccessible(self):
        cfg = [[0., 0., 'conditional'], [.1, 0., 'conditional'], [.25, 0., 'conditional']]
        r = np.full((3, 60, 2), np.nan)
        r[:, :36] = 3
        r[1, :36] = 1
        r[0, 36:] = 3
        r[1, 36:] = 1
        result = choose(r, cfg, 36)
        self.assertEqual(result['final_index'], 1)
        self.assertTrue(np.isnan(r[2, 36:]).all())

    def test_pair_unit(self):
        from stage6.select import rr
        np.testing.assert_allclose(rr(np.array([[1, 2], [2, 4]])), [.75, .375])

    def test_actual_nbfnet_direction_conversion(self):
        from motif.base_nbfnet import BaseNBFNet
        h = torch.tensor([[0, 0, 0], [0, 1, 2]])
        t = torch.tensor([[0, 1, 2], [1, 1, 1]])
        r = torch.tensor([[1, 1, 1], [1, 1, 1]])
        nh, nt, nr = BaseNBFNet.negative_sample_to_tail(None, h, t, r, 3)
        self.assertTrue(torch.equal(nr, torch.tensor([[1, 1, 1], [4, 4, 4]])))
        self.assertTrue(torch.equal(nh, torch.tensor([[0, 0, 0], [1, 1, 1]])))

    def test_filtered_ranking_independent_bruteforce(self):
        from motif import tasks
        from torch_geometric.data import Data
        truth = Data(edge_index=torch.tensor([[0, 0, 2], [1, 2, 1]]), edge_type=torch.tensor([0, 0, 0]), num_nodes=4)
        batch = torch.tensor([[0, 1, 0]])
        tm, hm = tasks.strict_negative_mask(truth, batch)
        s = torch.tensor([[.5, .4, .7, .6]])
        for scores, target, mask, expected_mask in [(s, 1, tm, [True, False, False, True]),
                                                   (s, 0, hm, [False, True, False, True])]:
            self.assertEqual(mask[0].tolist(), expected_mask)
            expected = 1 + sum(float(scores[0, i]) >= float(scores[0, target]) for i, ok in enumerate(expected_mask) if ok)
            actual = tasks.compute_ranking(scores, torch.tensor([target]), mask).item()
            self.assertEqual(actual, expected)

    def test_observation_isolation_all_real_graphs(self):
        # Fail on any read outside the sanitized observations capability while
        # loading/sampling all 14 real graphs. Also run the complete pure selector.
        folder = ROOT / 'observations'
        if not folder.exists():
            self.skipTest('observations not prepared')
        original = builtins.open
        path_open = Path.open
        def guard(file, *a, **kw):
            if isinstance(file, (str, Path)):
                self.assertTrue(Path(file).resolve().is_relative_to(folder.resolve()), str(file))
            return original(file, *a, **kw)
        def guard_path(obj, *a, **kw):
            self.assertTrue(obj.resolve().is_relative_to(folder.resolve()), str(obj))
            return path_open(obj, *a, **kw)
        keys = list(read(folder / 'manifest.json'))
        with patch('builtins.open', guard), patch.object(Path, 'open', guard_path):
            for key in keys:
                g = load_observed(folder, key)
                rem, q, cut, info = make_split(g, 6101)
                cfg = [[0., 0., 'conditional'], [.1, 0., 'conditional']]
                ranks = np.full((2, len(q), 2), 2)
                choose(ranks, cfg, cut)


if __name__ == '__main__':
    unittest.main()
