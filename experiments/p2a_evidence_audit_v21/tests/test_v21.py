import ast
import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path

import numpy as np

from v21.graph import FactIndex,matched_random,strata
from v21.audit import category,aggregate,AGGREGATORS
from v21.common import ROOT,config
from src.graph_loader import ObservedGraph
from p2.features import all_paths


def toy():
    facts=[(0,0,1),(0,1,1),(1,0,2),(2,1,3),(4,0,5),(0,0,1)]
    expanded=facts+[(v,r+2,u) for u,r,v in facts]
    return ObservedGraph(np.array([(u,v) for u,r,v in expanded]).T,np.array([r for u,r,v in expanded]),6,4)


class FactTests(unittest.TestCase):
    def setUp(self):
        self.g=toy()
        self.index=FactIndex(self.g)

    def test_unique_fact_reverse_duplicate_and_parallel(self):
        self.assertEqual(len(self.index.facts),5)
        ids=self.index.canonical([0,5,6,11])
        self.assertEqual(len(ids),1)
        self.assertEqual(len(self.index.closure(ids)),4)
        self.assertNotEqual(self.index.directed_to_id[0],self.index.directed_to_id[1])

    def test_deletion_removes_all_inverse_records(self):
        f=self.index.canonical([0,1])
        left=self.index.complement(f)
        self.index.assert_paired(left)
        self.assertFalse(set(left)&set(self.index.closure(f)))
        self.assertEqual(len(left)+len(self.index.closure(f)),len(self.g.edge_type))

    def test_local_matched_random_excludes_and_matches_fact_budget(self):
        available=tuple(range(len(self.index.facts)))
        bins,_,_=strata(self.index,available,0)
        selected=(0,1)
        a,detail=matched_random(selected,available,bins,42)
        b,other=matched_random(selected,available,bins,42)
        self.assertEqual(a,b)
        self.assertEqual(detail,other)
        self.assertFalse(set(a)&set(selected))
        self.assertEqual(len(a),len(selected))
        self.assertTrue(all(not d['overlap_selected'] for d in detail))

    def test_no_global_random_fallback(self):
        selected=(0,1)
        bins,_,_=strata(self.index,selected,0)
        a,detail=matched_random(selected,selected,bins,4)
        self.assertEqual(a,selected)
        self.assertTrue(all(d['relaxation']=='local_overlap_last_resort' for d in detail))

    def test_original_selected_direction_is_preserved_for_paths(self):
        paths=all_paths(self.g,0,[2],[0,2])
        self.assertEqual(paths[2],[(0,2)])
        self.assertNotIn(0,all_paths(self.g,2,[0],[0,2]))

    def test_t0_t4_v21_definition(self):
        self.assertEqual(category(False,False,False,False,False),'T0')
        self.assertEqual(category(True,False,False,False,False),'T1')
        self.assertEqual(category(True,True,False,False,False),'T2')
        self.assertEqual(category(True,True,True,False,False),'T3')
        self.assertEqual(category(True,True,True,True,False),'T4')

    def test_stable_lse_uniform_fallback(self):
        for agg in AGGREGATORS:
            self.assertEqual(aggregate([],agg,.08),0.)
            self.assertTrue(np.isfinite(aggregate([1e5,1e5+1],agg,.08)))
        self.assertAlmostEqual(aggregate([.4,.4],'NormalizedLogSumExp',.08),.4)
        self.assertGreater(aggregate([.4,.4],'LogSumExp',.08),.4)
        self.assertEqual(aggregate([.4,.4],'Max',.08),.4)

    def test_audit_inverse_counter_is_json_serializable(self):
        from v21.audit import collect
        from p1.retrieval import Retriever
        from src.evaluation import OfflineEvaluation
        raw=1/(60+np.arange(1,7,dtype=np.float32))
        cfg=config()
        gates={r:dict(layers=np.full((6,4),.4)) for r in range(4)}
        retriever=Retriever(self.g,gates,{},dict(max_local_nodes=300,max_local_edges=800))
        selected=retriever.query(0,0,[2,3],32)
        job=dict(query_id='toy:0:0',head=0,relation=0,candidates=[2,3],top30=[2,3],
                 arms={k:dict(ids=v['ids'],paths=v['paths'],values={
                      c:dict(score=0.,valid=False) for c in (2,3)}) for k,v in selected.items()})
        from p1.scoring import evidence
        for k,v in selected.items():
            job['arms'][k]['values']=evidence(self.g,0,[2,3],v,{})
        job=json.loads(json.dumps(job))
        class View:
            key='toy'
            graph=self.g
            asset=None
            def gate(self,r):return np.full(4,.4)
            def job(self,i,d):return job,raw
        class Offline:
            rank=staticmethod(OfflineEvaluation.filtered_rank)
            def __init__(self,a):pass
            def query(self,*args):return 2,{2},3
        with tempfile.TemporaryDirectory() as tmp, patch('v21.audit.Evaluator',Offline):
            collect(View(),[(0,0)],self.index,cfg,Path(tmp),{})
            rows=json.loads((Path(tmp)/'propagation/scoring_inputs.json').read_text())
            self.assertEqual(len(rows),6)
            self.assertTrue(all(type(r['inverse_records']) is int for r in rows))

    def test_no_api_train_or_changed_evaluation_flag(self):
        c=config()
        self.assertFalse(c['experiment']['enable_api'])
        self.assertFalse(c['experiment']['enable_llm'])
        self.assertFalse(c['experiment']['train_backbone'])
        forbidden={'requests','httpx','urllib','openai','transformers','dotenv','api'}
        for p in (ROOT/'v21').glob('*.py'):
            for node in ast.walk(ast.parse(p.read_text())):
                mods=[x.name for x in node.names] if isinstance(node,ast.Import) else (
                    [node.module or ''] if isinstance(node,ast.ImportFrom) else [])
                self.assertFalse({part for m in mods for part in m.split('.')} & forbidden)

    def test_plot_runtime_is_available(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.assertTrue(callable(plt.subplots))

    def test_full_dstar_is_frozen_after_subset(self):
        import torch
        from calibration import factors
        edge=torch.tensor(self.g.edge_index)
        cfg=dict(kind='D-Sender',gamma=.75,alpha=.2,anchor=True)
        a,b,z,_=factors(edge,6,cfg)
        before=[t.clone() for t in (a,b,z)]
        selected=torch.tensor(self.index.closure(self.index.canonical([0])))
        subset=edge[:,selected]
        self.assertEqual(len(a),6)
        self.assertEqual(len(b),6)
        self.assertFalse(torch.equal(factors(subset,6,cfg)[0],a))
        self.assertTrue(all(torch.equal(t,u) for t,u in zip((a,b,z),before)))


if __name__=='__main__':
    unittest.main()
