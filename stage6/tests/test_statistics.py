import unittest
import numpy as np
from stage6.statistics import paired_bootstrap


class StatisticsTests(unittest.TestCase):
    def test_constant_difference(self):
        r, draws = paired_bootstrap(np.full(20, .125))
        self.assertEqual(r['delta'], .125)
        self.assertEqual(r['lo'], .125)
        self.assertEqual(r['hi'], .125)

    def test_reproducible_paired_mean(self):
        d = np.array([.5, -.25, 0, .1])
        a, x = paired_bootstrap(d)
        b, y = paired_bootstrap(d)
        self.assertTrue(np.array_equal(x, y))
        self.assertAlmostEqual(a['delta'], .0875)
