import unittest
import numpy as np
from stage7.analysis import bootstrap,correlation,random_alpha,holm


class Statistics(unittest.TestCase):
    def test_bootstrap_pairs_and_seed(self):
        d=np.array([.2,-.1,.05,.1,-.02])
        a,b=bootstrap(d),bootstrap(d)
        self.assertEqual(a,b);self.assertAlmostEqual(a['delta'],d.mean())
        self.assertLess(a['lo'],a['hi']);self.assertEqual(a['n'],5)

    def test_bootstrap_identity(self):
        a=bootstrap(np.zeros(50))
        self.assertEqual((a['delta'],a['lo'],a['hi'],a['p']),(0,0,0,1))

    def test_random_control_reproducible(self):
        self.assertEqual(random_alpha('x',1024,6101),random_alpha('x',1024,6101))
        self.assertIn(random_alpha('x',1024,6101),[0,.1,.25,.5,1])

    def test_holm_family(self):
        rows=[dict(p=.01),dict(p=.03),dict(p=.8)]
        holm(rows)
        self.assertEqual([r['holm_p'] for r in rows],[.03,.06,.8])

    def test_degenerate_correlation_is_not_evidence(self):
        self.assertIsNone(correlation([0,0,0],[1,2,3])['pearson'])
        self.assertAlmostEqual(correlation([1,2,3],[2,4,6])['spearman'],1.)

    def test_small_spearman_uses_exact_nonzero_probability(self):
        result=correlation([1,2,3],[2,4,6])
        self.assertAlmostEqual(result['p_spearman'],2/6)
        self.assertIn('exact',result['spearman_p_method'])


if __name__=='__main__':unittest.main()
