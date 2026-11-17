import ast
import unittest

import numpy as np

from p2.assets import ROOT,config
from p2.audit import category,edge_diff,inspect_selected
from p2.features import all_paths,aggregate
from p2.scoring import fuse
from p2.source import sample_controls
from p1.paths import masked_graph
from p1.scoring import filtered_after,permutations
from src.graph_loader import ObservedGraph
from src.evaluation import OfflineEvaluation


def graph():
    triples=[(0,0,1),(1,1,2),(0,1,3),(3,0,2),(0,0,4)]
    triples += [(t,r+2,h) for h,r,t in triples]
    return ObservedGraph(np.array([(h,t) for h,r,t in triples]).T,
                         np.array([r for h,r,t in triples]),5,4)


class Tests(unittest.TestCase):
    def setUp(self):
        self.g=graph()
        self.job=dict(head=0,relation=0,candidates=[2])
        self.table={'qf':[9,10],'fq':[1,9]}
        self.prior=np.array([.9,.2,0.,0.])
        self.get=lambda ids:inspect_selected(self.g,self.job,ids,lambda _:0.,self.prior,self.table)

    def test_edge_diff_and_extending_real_path(self):
        self.assertEqual(edge_diff([0,1],[0,2,3]),([2,3],[1],.25))
        self.assertEqual(all_paths(self.g,0,[2],[0,1,2,3])[2],[(0,1),(2,3)])

    def test_outer_edges_do_not_enter_candidates(self):
        a,b=self.get([0,1]),self.get([0,1,4])
        np.testing.assert_equal(a[3][2],b[3][2])
        self.assertEqual(category(1,False,False,False),'T1')

    def test_remove_best_path_and_check_remaining_path(self):
        a,b=self.get([0,1,2,3]),self.get([1,2,3])
        self.assertGreater(a[1][2]['score'],b[1][2]['score'])
        self.assertEqual(b[0][2],[(2,3)])

    def test_lower_path_changes_features_without_changing_max(self):
        a,b=self.get([0,1]),self.get([0,1,2,3])
        self.assertEqual(a[1][2]['score'],b[1][2]['score'])
        self.assertNotEqual(a[3][2][1],b[3][2][1])
        self.assertEqual(category(2,True,False,False),'T2')

    def test_no_path_or_duplicate_edge_counts(self):
        self.assertNotIn(2,all_paths(self.g,0,[2],[0,0,2,4]))
        self.assertEqual(all_paths(self.g,0,[2],[0,0,1,1])[2],[(0,1)])

    def test_arm_isolation_and_trace_category(self):
        a,b=self.get([0,1]),self.get([2,3])
        self.assertNotEqual(a[0],b[0])
        self.assertEqual(self.get([0,1])[0],a[0])
        self.assertEqual(category(1,True,True,False),'T3')
        self.assertEqual(category(1,True,True,True),'T4')

    def test_motif_feature_is_independent_of_p1_score(self):
        a=self.get([0,1])
        b=inspect_selected(self.g,self.job,[0,1],lambda _:0.,self.prior/2,self.table)
        self.assertEqual(a[1],b[1])
        self.assertNotEqual(a[3][2][7],b[3][2][7])

    def test_filtered_rank_zero_fusion_and_outside_rank(self):
        raw=1/(60+np.array([1,2,3,3,5],dtype=float))
        values={0:dict(score=-.5,valid=True),1:dict(score=.8,valid=True)}
        self.assertEqual(permutations(raw,[0,1],values,0,.23)['fusion'],[0,1])
        rank=OfflineEvaluation.filtered_rank
        self.assertEqual(filtered_after(raw,[0,1],[1,0],2,{0,2},rank),rank(raw,2,{0,2}))
        self.assertEqual(rank(raw,2,{2}),4)

    def test_aggregate_temperature_and_neutral_missing(self):
        self.assertEqual(aggregate([],'S2-0.08'),0.)
        self.assertEqual(aggregate([.2,.4,.5],'S1-2'),.45)
        self.assertLess(aggregate([.2,.4,.5],'S2-0.08'),.5)

    def test_source_mask_removes_forward_and_reverse(self):
        g=masked_graph(self.g,[(0,0,1)])
        g.assert_hidden_absent([(0,0,1)])
        self.assertEqual(len(g.edge_type),len(self.g.edge_type)-2)

    def test_fusion_activates_only_with_nonzero_weight(self):
        raw=1/(60+np.arange(1,6,dtype=float))
        cs=[0,1,2]
        values={0:dict(score=-.4,valid=True),1:dict(score=.5,valid=True),
                2:dict(score=0.,valid=False)}
        for mode in ('F2','F3','F4'):
            self.assertEqual(fuse(raw,cs,values,mode,0.,.2),(cs,False,False,False))
        order,on,shift,_=fuse(raw,cs,values,'F2',.35,.2)
        self.assertEqual(order,[1,0,2])
        self.assertTrue(on and shift)

    def test_controls_exclude_observed_positives_and_answer(self):
        raw=1/(60+np.arange(1,6,dtype=float))
        controls,*_=sample_controls(self.g,0,0,2,{1},raw,2,np.random.default_rng(5))
        chosen=set(x for group in controls.values() for x in group)
        self.assertFalse(chosen & {0,1,2})

    def test_no_llm_clients(self):
        self.assertFalse(config()['enable_llm'])
        self.assertFalse(config()['enable_api'])
        self.assertFalse(config()['train_backbone'])
        forbidden={'api','requests','urllib','openai','transformers','dotenv','krlm'}
        for file in (ROOT/'p2').glob('*.py'):
            for node in ast.walk(ast.parse(file.read_text(encoding='utf-8'))):
                if isinstance(node,ast.Import):
                    names=[x.name for x in node.names]
                elif isinstance(node,ast.ImportFrom):
                    names=[node.module]
                else:
                    continue
                self.assertFalse({segment for name in names if name for segment in name.split('.')} & forbidden)
