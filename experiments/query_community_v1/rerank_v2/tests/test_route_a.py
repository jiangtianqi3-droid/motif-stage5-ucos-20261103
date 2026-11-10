import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.edge_weighter import UnifiedEdgeWeighter
from src.structural_reranker import Config, rerank, trigger
from src.llm_reranker import validate, confidence
from src.offline import Dataset, bootstrap
from qc_v1.graph_adapter import ObservedGraph

def graph():
    edge=np.array([[0,0,0,1,2,2,3],[1,2,3,2,3,4,4]])
    r=np.array([0,0,1,1,0,0,1])
    return ObservedGraph(np.concatenate([edge,edge[::-1]],axis=1),np.r_[r,r+2],5,4)

def test_unified_stochastic_and_mean_anchor():
    g=graph();factory=UnifiedEdgeWeighter(g)
    factory.relation_importance=lambda q,*args:np.array([.2,.8,.2,.8])
    w=factory.compute()
    np.testing.assert_allclose(w.weights.mean(),1.)
    np.testing.assert_allclose(np.asarray(w.transition.sum(1)).ravel(),1.)
    np.testing.assert_allclose(np.asarray(w.reverse_transition.sum(1)).ravel(),1.)
    # Independent formula confirms non-row-constant factors survive normalization.
    u,v=g.edge_index;r=g.edge_type
    raw=factory.bounded_map(factory.relation_importance(0))[r]*(1+g.out_degree[u])**(-.25)*(1+g.edge_rel_out)**(-.5)*(1+g.out_degree[v])**(-.25)
    sums=np.bincount(u,weights=raw,minlength=g.n)
    np.testing.assert_allclose(w.edge_probability,raw/sums[u])
    topo=factory.compute('topology_aware')
    assert not np.allclose(w.transition.toarray(),topo.transition.toarray())

def test_row_constant_cancellation_is_still_a_mathematical_invariant():
    g=graph();factory=UnifiedEdgeWeighter(g);factory.relation_importance=lambda q,*args:np.array([.2,.8,.2,.8])
    first=factory.compute('topology_aware').transition.toarray();factory.eta=1.7
    second=factory.compute('topology_aware').transition.toarray()
    np.testing.assert_allclose(first,second)

def record(n=30):
    return dict(candidate_evidence=[dict(entity=i,base_logit=1-i*.01,base_rank=i+1,status='ok',max_path_weight=1.,ppr_affinity_score=1.,
               evidence_paths=[dict(transition_path_probability=1.)]) for i in range(n)])

def test_gate_protection_and_zero_residual_identity():
    r=record();r['candidate_evidence'][12]['max_path_weight']=100
    cfg=Config(strength=10.,beta_w=1.,beta_p=0.)
    result=rerank(r,cfg)
    assert result['order'][0]==12
    assert all(result['gate'][i] for i in result['order'][:10])
    assert [i for i in result['order'] if not result['gate'][i]]==[i for i in range(30) if not result['gate'][i]]
    assert rerank(r,Config())['order']==list(range(30))
    r['candidate_evidence'][0]['max_path_weight']=1000
    r['candidate_evidence'][0]['base_logit']=10
    assert rerank(r,cfg)['order'][0]==0

def test_no_evidence_is_zero_and_does_not_trigger():
    r=record()
    for c in r['candidate_evidence']: c.update(status='no_evidence',evidence_paths=[],max_path_weight=0)
    cfg=Config(strength=10)
    result=rerank(r,cfg)
    assert result['support']==[0]*30
    assert not trigger(r,result,cfg)

def test_unknown_llm_id_rejected_and_abstention_exact_identity():
    decision,error=validate(json.dumps(dict(abstain=False,top_candidates=[dict(candidate_id=999,confidence=.9)],reasoning='test')),{1,2})
    assert error and decision['abstain']
    r=record();cfg=Config(strength=.3);before=rerank(r,cfg)
    after=rerank(r,cfg,confidence(dict(decision=decision)),.1)
    assert before==after

def test_pool_permutation_preserves_outside_rank_and_pessimistic_ties():
    ds=object.__new__(Dataset)
    ds.truth={'test':np.array([[0,0,4]])};ds.public={'test':np.array([[[0,0],[4,1]]])};ds.filters={'test':{(0,0):{4}}}
    scores=np.array([5.,4.,3.,2.,1.]);ids=np.array([0,1,2])
    assert ds.rank('test',0,0,scores,ids,[2,0,1])==5
    ds.truth['test'][0,2]=1;ds.filters['test'][0,0]={1}
    ties=np.array([5.,4.,4.,2.,1.])
    assert ds.rank('test',0,0,ties,ids)==3
    assert ds.rank('test',0,0,ties,ids,[1,0,2])==1

def test_bootstrap_clusters_opposite_direction_changes_by_fact():
    result=bootstrap([[1,2],[1,2]],[[2,1],[2,1]])
    # Within every fact, the two directions cancel; resampling directions apart
    # would create a spurious nonzero interval for MRR and Hits@1.
    for metric in ['MRR','Hits1']:
        assert result[metric]['lo']==result[metric]['hi']==result[metric]['delta']==0
