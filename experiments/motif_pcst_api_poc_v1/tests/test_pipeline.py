import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.api.budget_guard import Ledger, BudgetExhausted
from src.api.client import Client
from src.api.provider_adapter import ProviderError, UncertainDelivery
from src.api.response_validator import parse
from src.evaluation import OfflineEvaluation
from src.evidence_serializer import candidate_map, estimate, messages, MAX_INPUT
from src.graph_loader import fact
from query_community_v1.src.edge_weighter import CommunityEdgeWeighter
from query_community_v1.src.graph_adapter import ObservedGraph
from src.local_retrieval import ARMS, budget, directed_paths, retrieve
from src.pipeline import phase_plan
from src.reporting import aggregate


class GraphTests(unittest.TestCase):
    def setUp(self):
        direct = [(0,1),(1,2),(0,1),(1,1)]
        u,v = zip(*direct)
        self.graph = ObservedGraph(np.array([u+v,v+u]), np.array([0,0,1,1,2,2,3,3]), 5,4)
        sig = dict(edge_index=np.array([[0,0],[1,2],[-1,-1]]),raw=np.ones((2,4)),z=np.zeros((2,4)))
        self.weighter = CommunityEdgeWeighter(self.graph,sig)

    def test_only_observed_directional_facts_and_inverse(self):
        self.graph.assert_inverse_pairs()
        self.graph.assert_hidden_absent([(0,0,2)])
        with self.assertRaises(AssertionError):
            self.graph.assert_hidden_absent([(2,2,1)])
        self.assertEqual(fact(self.graph,4).relation,self.graph.inverse(fact(self.graph,0).relation))
        self.assertEqual(directed_paths(self.graph,0,[2,4],lambda _:0),{2:(0,1)})

    def test_bounded_coverage_and_matched_candidates(self):
        mapping = candidate_map('q',[2,4])
        self.assertEqual(mapping, candidate_map('q',[2,4]))
        result = retrieve(self.graph,self.weighter,0,0,mapping,42)
        self.assertEqual(tuple(result), ARMS)
        for arm,row in result.items():
            self.assertLessEqual(row['input_bound'],MAX_INPUT)
            self.assertTrue(set(row['ids']) <= set(range(8)))
            self.assertEqual(row['reachable'],1)
            self.assertNotIn(4,row['paths'])
            self.assertLessEqual(row['edges'],60)
        self.assertEqual(result['A4']['paths'][2],(0,1))
        self.assertEqual(result['A4']['edges'],result['A5']['edges'])

    def test_budget_counts_distinct_fact_ids_and_makes_paths_atomic(self):
        mapping = candidate_map('q',[2,4])
        self.assertEqual(budget(self.graph,0,0,mapping,[(0,1),(0,2)]), [0,1,2])
        self.assertLess(estimate(messages(0,0,mapping,[])),MAX_INPUT)

    def test_disjoint_fixed_fact_sets(self):
        plan = phase_plan(100,'key')
        self.assertEqual(plan, phase_plan(100,'key'))
        self.assertEqual((len(plan['retrieval']),len(plan['smoke']),len(plan['pilot'])),(50,2,12))
        self.assertFalse(set(plan['smoke']) & set(plan['pilot']))

    def test_prompt_only_exposes_public_query_schema(self):
        prompt = messages(0, 0, {'C01': 2, 'C02': 4}, [fact(self.graph, 0)])
        body = prompt[1]['content']
        self.assertIn('(N0, R0, ?)', body)
        self.assertIn('N0 --R0--> N1', body)
        self.assertNotIn('N0 --R0--> N2', body)


class FakeProvider:
    base, model = 'https://example.test','unit-test'
    generation = {'max_tokens':512,'temperature':0}
    max_retries = 2

    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = 0

    def complete(self, prompt, cap):
        self.calls += 1
        result = next(self.answers)
        if isinstance(result,Exception):
            raise result
        return result


def reply(candidates, used=(), abstain=False):
    return dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(
        dict(ranked_candidate_ids=candidates,used_evidence_ids=list(used),abstain=abstain,
             confidence='low'))))],usage=dict(prompt_tokens=100,completion_tokens=25))


class ApiTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ledger = Ledger(Path(tmp.name)/'ledger.sqlite')
        self.candidates = {'C01':0,'C02':1}
        self.prompt = messages(2,0,self.candidates,[])

    def rank(self, provider, query='q'):
        return Client(provider,self.ledger,lambda _:None).rank(query,self.candidates,self.prompt,set(),'ev')

    def test_retry_reopen_cache_and_abstentions(self):
        provider = FakeProvider([ProviderError(429),reply(['C02','C01'],abstain=True)])
        value = self.rank(provider)
        self.assertEqual(value['status'],'abstain')
        self.assertEqual(self.ledger.usage(), (2,100,25))
        other = FakeProvider([])
        self.assertTrue(self.rank(other)['cache_hit'])
        self.assertEqual(other.calls,0)
        self.assertEqual(self.ledger.usage(), (2,100,25))

    def test_invalid_response_falls_back_and_resumes(self):
        provider = FakeProvider([reply(['C01','C01'])]*3)
        self.assertEqual(self.rank(provider)['status'],'invalid')
        self.assertIsNone(self.rank(FakeProvider([]))['response'])
        self.assertEqual(self.ledger.usage()[0],3)
        self.assertIsNone(parse(reply(['C01','C02'],['Ehidden']),self.candidates,set()))

    def test_uncertain_delivery_stays_reserved(self):
        result = self.rank(FakeProvider([UncertainDelivery('timeout')]))
        self.assertEqual(result['status'],'uncertain')
        provider = FakeProvider([])
        self.assertEqual(self.rank(provider)['status'],'uncertain')
        self.assertEqual(provider.calls,0)
        self.assertEqual(self.ledger.usage()[2],512)

    def test_budget_guard_reservation_stops_before_send(self):
        restricted = Ledger(self.ledger.path,(1,1000,600))
        restricted.reserve('other',650,512)
        provider = FakeProvider([])
        with self.assertRaises(BudgetExhausted):
            Client(provider,restricted).rank('q',self.candidates,self.prompt,set(),'ev')
        self.assertEqual(provider.calls,0)

    def test_truncated_json_never_reorders(self):
        value = reply(['C01','C02'])
        value['choices'][0]['finish_reason'] = 'length'
        self.assertIsNone(parse(value,self.candidates,set()))

    def test_filter_removes_other_truths_but_preserves_pessimistic_ties(self):
        scores = np.array([.6,.4,.4,.2])
        self.assertEqual(OfflineEvaluation.filtered_rank(scores,1,{0,1}),2)

    def test_fake_response_is_not_a_metric_and_flips_are_counted(self):
        rows = [dict(dataset='synthetic',arm='A4',fusion='direct',original_rank=2,new_rank=b,
                     positive=b<2,negative=b>2,status=status,top12=True,top30=True,
                     input_tokens=0,output_tokens=0,seconds=0)
                for b,status in ((1,'valid'),(3,'valid'),(2,'abstain'),(2,'invalid'))]
        a = aggregate(rows)[0]
        self.assertEqual([a[k] for k in ('positive','negative','unchanged','abstain','invalid')], [1,1,2,1,1])
