import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pcst_adapter import Fact, transform, restore
from ranking_fusion import validate_response, writeback


class CoreTests(unittest.TestCase):
    def test_parallel_and_inverse_mapping(self):
        facts = [Fact('a', 'h', 'r', 't'), Fact('b', 'h', 's', 't'), Fact('c', 't', 'r_inv', 'h')]
        p, edges, costs, emap, vmap = transform(['h', 't'], facts, [1, 1], [0, 3, 0], [1, 1, 1])
        self.assertTrue(all(x >= 0 for x in p + costs))
        nodes, chosen = restore(['h', 't'], facts, [0, 1, 2], list(emap), emap, vmap)
        self.assertEqual(chosen, facts)
        self.assertEqual(nodes, {'h', 't'})
        self.assertEqual(len(edges), 4)

    def test_reject_negative_and_duplicate(self):
        f = [Fact('x', 'a', 'r', 'b')]
        with self.assertRaises(ValueError): transform(['a', 'b'], f, [1, 1], [-1], [1])
        with self.assertRaises(ValueError): transform(['a', 'b'], f * 2, [1, 1], [1, 1], [1, 1])

    def test_ranking_and_validation(self):
        self.assertEqual(writeback(['a','b','c','d'], ['a','b'], ['b','a']), ['b','a','c','d'])
        self.assertFalse(validate_response({'ranked_candidate_ids':['a','a'], 'used_evidence_ids':[], 'abstain':False}, ['a','b'], []))
        self.assertFalse(validate_response({'ranked_candidate_ids':['a','b'], 'used_evidence_ids':['bad'], 'abstain':False}, ['a','b'], []))
        self.assertTrue(validate_response({'ranked_candidate_ids':['b','a'], 'used_evidence_ids':['E1'], 'abstain':False}, ['a','b'], ['E1']))
        with self.assertRaises(ValueError): writeback(['a','b'], ['a','b'], ['a','a'])

    def test_real_solver_when_available(self):
        try:
            import pcst_fast  # noqa: F401
        except ImportError:
            self.skipTest('pcst_fast is not installed in this interpreter')
        from pcst_adapter import solve
        f = Fact('f', 'h', 'r', 't')
        nodes, facts = solve(['h', 't'], [f], [1, 1], [2], [1])
        self.assertEqual(facts, [f])
        self.assertEqual(nodes, {'h', 't'})


if __name__ == '__main__': unittest.main()
