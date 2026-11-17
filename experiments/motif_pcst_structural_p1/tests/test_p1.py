import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from p1.assets import M5, Assets
from p1.config import FLAGS, ROOT, load_config
from p1.memory import export
from p1.paths import counts_from_rows, masked_graph, path_choices, reliability, signature
from p1.retrieval import ARMS, Retriever, capped, true_fact, verify_connected
from p1.run import boot_delta, indices_for, schemes
from p1.scoring import evidence, filtered_after, permutations, raw_order
from src.evaluation import OfflineEvaluation
from src.graph_loader import ObservedGraph
from src.pcst_adapter import Fact, solve, restore, transform


def toy():
    triples = [(0,0,1),(0,1,1),(1,1,2),(1,0,3),(3,1,2),(0,0,1)]
    augmented = triples+[(t,r+2,h) for h,r,t in triples]
    return ObservedGraph(np.array([(h,t) for h,r,t in augmented]).T,
                         np.array([r for h,r,t in augmented]),5,4)


class StructuralTests(unittest.TestCase):
    def setUp(self):
        self.g = toy()
        self.cfg = load_config()
        self.gates = {r:dict(layers=np.tile([.2,.65,.2,.4],(6,1)))
                      for r in range(self.g.R)}
        self.table = {'f': [3,5], 'qf': [9,10], 'f/0': [2,3]}

    def test_01_candidate_vocabulary_and_stable_ties(self):
        self.assertEqual(raw_order(np.array([.04,.05,.05,.02,.01])),[1,2,0,3,4])
        self.g.assert_inverse_pairs()
        self.assertEqual(self.g.n,5)

    def test_02_mask_removes_duplicate_fact_and_inverse(self):
        m = masked_graph(self.g,[(0,0,1)])
        self.assertEqual(len(m.edge_type),len(self.g.edge_type)-4)
        m.assert_hidden_absent([(0,0,1)])

    def test_03_restore_preserves_direction_and_parallel_identity(self):
        a,b,c = Fact(7,1,0,2),Fact(8,1,1,2),Fact(9,2,2,1)
        p,e,cost,nmap,vmap = transform([1,2],[a,b,c],[1,1],[.7,.1,.1],[.2,.2,.2])
        self.assertTrue(all(x>=0 for x in p+cost))
        _, got = restore([1,2],[a,b,c],[0,1,2],list(nmap),nmap,vmap)
        self.assertEqual(got,[a,b,c])
        self.assertEqual(len(e),4)

    def test_04_actual_solver_restores_virtual_fact(self):
        f = Fact(5,1,2,0)
        nodes,facts = solve([0,1],[f],[1,1],[2],[1],root=0)
        self.assertEqual(facts,[f])
        self.assertEqual(nodes,{0,1})

    def test_05_budget_counts_shared_ids_once(self):
        self.assertEqual(capped([(2,6),(6,9),(10,11)],3),[2,6,9])
        self.assertNotEqual(true_fact(self.g,0),true_fact(self.g,1))

    def test_06_07_all_and_reachable_coverage_use_distinct_denominators(self):
        r = Retriever(self.g,self.gates,self.table,self.cfg['retrieval'])
        choices = r.query(0,0,[2,4],3)
        self.assertEqual(tuple(choices),ARMS)
        out = choices['R2']['metrics']
        self.assertEqual(out['candidate_coverage_all'],.5)
        self.assertEqual(out['candidate_coverage_reachable'],1.)
        self.assertIsNone(r.query(0,0,[4],3)['R2']['metrics']['candidate_coverage_reachable'])

    def test_08_global_gate_cannot_fabricate_an_instance(self):
        r = Retriever(self.g,self.gates,self.table,self.cfg['retrieval'])
        choice = r.query(0,0,[2,4],4)['R3']
        self.assertNotIn(4,choice['paths'])
        self.assertEqual(evidence(self.g,0,[2,4],choice,self.table)[4]['score'],0.)

    def test_09_pcst_three_hop_direction_and_connectivity(self):
        r = Retriever(self.g,self.gates,self.table,self.cfg['retrieval'])
        for arm,out in r.query(0,0,[2,3,4],3).items():
            self.assertLessEqual(len(out['ids']),3)
            if arm!='R0':
                verify_connected(self.g,out['ids'],0)
            for dest,paths in out['paths'].items():
                for path in paths:
                    u=0
                    for i in path:
                        self.assertEqual(self.g.edge_index[0,i],u)
                        u=self.g.edge_index[1,i]
                    self.assertEqual(u,dest)

    def test_10_hub_cost_is_an_explicit_extension_change(self):
        r = Retriever(self.g,self.gates,self.table,self.cfg['retrieval'])
        out = r.query(0,0,[2],5)
        for i,c in zip(out['R4']['ids'],out['R4']['costs']):
            sender=self.g.edge_index[1,i]
            self.assertAlmostEqual(c,.34+(.26*r.hub[sender] if sender not in (0,2) else 0.))
        self.assertEqual(out['R2']['protected'],out['R3']['protected'])
        self.assertEqual(out['R3']['protected'],out['R4']['protected'])

    def test_11_masked_path_signatures_use_masked_graph(self):
        m=masked_graph(self.g,[(0,0,1)])
        p=path_choices(m,0,[2],list(range(len(m.edge_type))),lambda _:0.)[2][0]
        self.assertNotIn('q',signature(m,0,p,False))
        self.assertTrue(0<=reliability(self.table,m,0,p)<=1)
        rows=[dict(has_path=True,observed_positive=True,signature='a',coarse='b'),
              dict(has_path=True,observed_positive=False,signature='a',coarse='b')]
        self.assertEqual(counts_from_rows(rows),{'a':[1,2],'b':[1,2]})

    def test_12_13_fusion_and_filter_full_rank(self):
        raw=np.array([.09,.07,.05,.04,.02])
        values={0:dict(score=0.,valid=False),1:dict(score=.3,valid=True)}
        out=permutations(raw,[0,1],values,0.,.23)
        self.assertEqual(out['evidence_only'],[1,0])
        self.assertEqual(out['fusion'],[0,1])
        fun=OfflineEvaluation.filtered_rank
        self.assertEqual(filtered_after(raw,[0,1],[1,0],1,{1},fun),1)
        self.assertEqual(filtered_after(raw,[0,1],[1,0],4,{4},fun),5)
        self.assertEqual(fun(np.array([.3,.3,.2]),1,{1}),2)

    def test_14_query_sample_and_pair_bootstrap(self):
        self.assertEqual(indices_for(300,'a',100,11),indices_for(300,'a',200,11)[:100])
        self.assertEqual(len({i for i in indices_for(300,'a',100,11)}),100)
        self.assertEqual(boot_delta([({'fact_index':1},.2),({'fact_index':1},-.2)],11,10),[0.,0.])

    def test_15_budget_schemes_change_one_factor(self):
        work=schemes(self.cfg,'sensitivity')
        self.assertEqual([(k,b) for k,b,_,_ in work],[(12,16),(12,32),(12,48),(8,32),(30,32)])

    def test_16_cache_rejects_other_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)
            cache=folder/'results/recovery'
            cache.mkdir(parents=True)
            file=cache/'candidate.npy'
            file.write_bytes(b'fixed test vector')
            from p1.assets import PRIMARY, sha
            (cache/'recovery.json').write_text(json.dumps({'records':[dict(dataset='toy',model=PRIMARY,
                 cache_sha256=sha(file),observation_sha256='wrong',path=str(file))]}))
            asset=Assets.__new__(Assets)
            asset.key='toy'
            asset.meta={'feature_sha256':sha(file)}
            asset.entry={'observed_sha256':'right'}
            with patch('p1.assets.POC',folder), self.assertRaisesRegex(ValueError,'identity'):
                asset.open_previous()

    def test_17_memory_has_direction_mask_and_no_test_labels(self):
        r=Retriever(self.g,self.gates,self.table,self.cfg['retrieval'])
        row=r.query(0,0,[2,3],4)['R5']
        scores=evidence(self.g,0,[2,3],row,self.table)
        out=export(self.g,'q',0,0,[2,3],np.ones(5),row,scores,8,'cp','obs')
        self.assertEqual(len(out['memory_mask']),8)
        self.assertEqual(out['memory_mask'],[i>=0 for i in out['memory_entity_ids']])
        self.assertTrue(set(out).isdisjoint({'target','answer','label','truth'}))
        self.assertEqual(out['entity_embedding_ref']['checkpoint_parameter_sha256'],'cp')

    def test_18_p1_entrypoints_import_no_api_or_language_clients(self):
        self.assertEqual(self.cfg['experiment'],FLAGS)
        deny=('api','urllib','requests','krlm','transformers','openai','httpx','dotenv')
        for path in (ROOT/'p1').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names=[x.name for x in node.names] if isinstance(node,ast.Import) else (
                       [node.module or ''] if isinstance(node,ast.ImportFrom) else [])
                self.assertFalse(set(part for name in names for part in name.lower().split('.')) & set(deny))
