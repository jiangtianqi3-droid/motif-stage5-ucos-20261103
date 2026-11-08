import sys
from pathlib import Path
import numpy as np
import pytest
import networkx as nx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.graph_adapter import ObservedGraph
from src.edge_weighter import CommunityEdgeWeighter
from src.community_detection import detect_communities
from src.local_ppr import local_ppr,intersections,local_relation_distribution
from src.path_extractor import retrieve,top_paths
from src.evidence_exporter import validate_evidence

def graph(facts,n=None,R=4):
    facts=list(facts)
    f=np.asarray(facts,dtype=np.int64).reshape(-1,3)
    return ObservedGraph(np.c_[f[:,[0,2]].T,f[:,[2,0]].T],np.r_[f[:,1],f[:,1]+R//2],
                         n if n is not None else int(f[:,[0,2]].max()+1),R)

def signature(R=4):
    # All seven motif kinds, independent exclusive/RW support for relations 0,1.
    he=np.tile(np.array([[0,1],[0,1],[-1,-1]]),(1,7))
    ht=np.repeat(np.arange(7),2)
    he[2,ht>=3]=he[0,ht>=3]
    raw=np.tile([[2.,.5,1.,.1],[2.,.8,3.,.9]],(7,1))
    z=np.tile([[-1.,-1.,-1.,-1.],[1.,1.,1.,1.]],(7,1))
    return dict(edge_index=he,edge_type=ht,raw=raw,z=z)

def candidate(t): return dict(entity=t,base_rank=1,base_score=.1)

def test_01_community_modularity():
    f=[(a,0,b) for start in [0,5] for a in range(start,start+5) for b in range(a+1,start+5)]+[(4,1,5)]
    g=graph(f);result=detect_communities(g)
    assert len(result['members'])==2 and result['statistics']['modularity']>.4
    assert result['labels'][0]!=result['labels'][9]
    net=nx.Graph();net.add_nodes_from(range(g.n))
    for u,v in g.edge_index.T: net.add_edge(int(u),int(v),weight=2.)
    assert result['statistics']['modularity']==pytest.approx(nx.community.modularity(net,[set(m) for m in result['members']]))

def test_02_inverse_directed_recovery():
    g=graph([(0,0,1),(1,1,2)]);g.assert_inverse_pairs()
    w=CommunityEdgeWeighter(g).compute()
    p=top_paths(g,w,2,0)
    assert any([int(g.edge_type[e]) for e in ids]==[3,2] for ids in p)

def test_03_parallel_relations():
    g=graph([(0,0,1),(0,1,1),(1,0,2)])
    p=top_paths(g,CommunityEdgeWeighter(g).compute(),0,2)
    assert {tuple(g.edge_type[list(ids)]) for ids in p}>={(0,0),(1,0)}

def test_04_hub_suppression():
    g=graph([(0,0,i) for i in range(2,12)]+[(1,0,12)],n=13)
    factory=CommunityEdgeWeighter(g,signature())
    for mode in ['degree_aware','topology_aware']:
        w=factory.compute(mode,q=0)
        assert w.weights[0]<w.weights[10]

def test_05_cross_community_three_hop_bridge():
    f=[(0,1,1),(1,1,2),(2,1,3)]
    f += [(0,0,i) for i in range(4,45)]+[(i,0,j) for i in range(4,15) for j in range(i+1,15)]
    g=graph(f,n=45);sig=signature();factory=CommunityEdgeWeighter(g,sig)
    # Query relation 1 cooccurs with the high purity chain relation.
    w=factory.compute('topology_aware',q=1)
    labels=np.zeros(g.n,dtype=int);labels[2:4]=1
    r=retrieve(g,w,labels,0,1,[candidate(3)],'ppr_topology_4d',sig,max_nodes=10,max_edges=12)
    c=r['candidate_evidence'][0]
    assert c['status']=='ok' and c['is_cross_community']
    assert c['evidence_paths'][0]['triples']==[[0,1,1],[1,1,2],[2,1,3]]
    assert c['evidence_paths'][0]['contains_bridge'];validate_evidence(r,g)
    same=retrieve(g,w,labels,0,1,[candidate(3)],'global_community',sig)
    assert same['candidate_evidence'][0]['status']=='no_evidence'

def test_06_isolate_no_evidence():
    g=graph([(0,0,1)],n=3);w=CommunityEdgeWeighter(g).compute()
    r=retrieve(g,w,np.arange(3),0,0,[candidate(2)],'ppr_uniform')
    assert r['candidate_evidence'][0]['status']=='no_evidence'
    assert local_ppr(w.transition,2)=={2:1.}

def test_07_dual_seed_intersection():
    g=graph([(0,0,1),(1,0,2),(2,0,3)])
    w=CommunityEdgeWeighter(g).compute()
    meeting=dict(intersections(local_ppr(w.transition,0),local_ppr(w.reverse_transition,3)))
    assert meeting[1]>0 and meeting[2]>0

def test_08_nontrivial_topology_both_modes():
    g=graph([(0,0,1),(0,1,2)])
    factory=CommunityEdgeWeighter(g,signature())
    rho=factory.relation_importance(1)
    assert rho[1]>.8 and rho[1]!=rho[0]
    gates=np.tile([.25,2.],7)
    rho=factory.relation_importance(0,'gated_model',gates)
    assert rho[1]>rho[0]
    assert np.ptp(factory.compute('topology_aware',0,'gated_model',gates).weights)>0

def test_09_seed_reproducibility():
    g=graph([(0,0,1),(1,0,2),(2,0,0),(2,1,3),(3,0,4)])
    a=detect_communities(g);b=detect_communities(g)
    assert np.array_equal(a['labels'],b['labels'])
    w=CommunityEdgeWeighter(g).compute()
    assert top_paths(g,w,0,4)==top_paths(g,w,0,4)

def test_10_zero_leakage_target_and_inverse():
    g=graph([(0,0,1),(1,0,2)])
    hidden=np.array([[0,1,2],[0,0,2]])
    g.assert_hidden_absent(hidden)
    bad=graph([(0,0,1),(1,0,2),(0,1,2)])
    with pytest.raises(AssertionError): bad.assert_hidden_absent(hidden)
    inverse_only=ObservedGraph(np.array([[2],[0]]),np.array([3]),3,4)
    with pytest.raises(AssertionError): inverse_only.assert_hidden_absent(hidden)
    for mode in ['uniform','degree_aware','topology_aware']:
        w=CommunityEdgeWeighter(g,signature()).compute(mode,0)
        r=retrieve(g,w,np.zeros(3,dtype=int),0,0,[candidate(2)],'ppr_topology_4d',signature())
        used={tuple(t) for c in r['candidate_evidence'] for p in c['evidence_paths'] for t in p['triples']}
        assert not used & set(map(tuple,hidden))
        validate_evidence(r,g)

@pytest.mark.parametrize('mode',['uniform','degree_aware','topology_aware'])
def test_probability_and_finite(mode):
    g=graph([(0,0,1),(0,1,1),(1,0,2)],n=4)
    w=CommunityEdgeWeighter(g,signature()).compute(mode,0)
    assert np.allclose(np.asarray(w.transition.sum(1)).ravel()[:3],1)
    assert np.isclose(sum(local_ppr(w.transition,0).values()),1)
    if mode=='degree_aware': assert w.weights.mean()==pytest.approx(1.)

def test_empty_graph_safe():
    g=graph([],n=3);d=detect_communities(g)
    w=CommunityEdgeWeighter(g).compute()
    r=retrieve(g,w,d['labels'],0,0,[candidate(2)],'ppr_uniform')
    assert r['subgraph']['edge_ids']==[];validate_evidence(r,g)

def test_sender_factor_cancels_from_row_transition():
    # Mathematical invariant: sender-only damping cannot alter row-stochastic PPR.
    g=graph([(0,0,1),(0,1,2),(2,0,3)])
    factory=CommunityEdgeWeighter(g,signature(),eta=0)
    a=factory.compute('topology_aware',0)
    factory.eta=.9;b=factory.compute('topology_aware',0)
    assert np.allclose(a.transition.toarray(),b.transition.toarray())
    assert not np.allclose(a.weights,b.weights)

def test_weighted_community_supported():
    g=graph([(0,0,1),(1,1,2),(2,0,3)])
    d=detect_communities(g,np.array([1.,.01,1.,1.,.01,1.]))
    assert np.isfinite(d['statistics']['modularity'])

def test_no_gate_fallback():
    g=graph([(0,0,1)])
    with pytest.raises(ValueError): CommunityEdgeWeighter(g,signature()).compute('topology_aware',0,'gated_model')

def test_pi_is_head_only():
    g=graph([(0,0,1),(1,1,2)],n=4);s=signature();w=CommunityEdgeWeighter(g,s).compute()
    a=retrieve(g,w,np.zeros(4,dtype=int),0,0,[candidate(2)],'ppr_uniform',s)
    b=retrieve(g,w,np.zeros(4,dtype=int),0,0,[candidate(3)],'ppr_uniform',s)
    assert a['query_info']['pi_h']==b['query_info']['pi_h']
    assert sum(a['query_info']['pi_h'])==pytest.approx(1.)

def test_budgets_are_query_wide():
    g=graph([(0,0,i) for i in range(1,20)])
    w=CommunityEdgeWeighter(g).compute()
    r=retrieve(g,w,np.zeros(20,dtype=int),0,0,[candidate(t) for t in range(1,20)],'ppr_uniform',max_nodes=5,max_edges=5)
    validate_evidence(r,g)
    assert len(r['subgraph']['nodes'])<=5

def test_real_signature_has_counts_not_raw_placeholders():
    from src.graph_adapter import load_observed,load_signature,HISTORY
    if not HISTORY.exists(): pytest.skip('Local frozen assets unavailable')
    g=load_observed('FB15k237Inductive_v1');s=load_signature(g,'FB15k237Inductive_v1')
    assert np.all(s['raw'][:,0]>0)
    assert np.ptp(CommunityEdgeWeighter(g,s).relation_importance(26))>1e-6

@pytest.mark.parametrize('method',['naive_khop','global_community','ppr_uniform','ppr_degree','ppr_topology_signature','ppr_topology_gated'])
def test_all_methods_export_json(method):
    g=graph([(0,0,1),(0,1,2),(1,1,3),(2,0,3)])
    s=signature();w=CommunityEdgeWeighter(g,s).compute('degree_aware' if method=='ppr_degree' else 'uniform')
    r=retrieve(g,w,np.zeros(g.n,dtype=int),0,0,[candidate(3)],method,s)
    assert validate_evidence(r,g)

def test_paths_match_independent_multigraph_oracle():
    g=graph([(0,0,1),(0,1,1),(1,0,2),(1,1,3),(2,0,3),(3,0,0)])
    w=CommunityEdgeWeighter(g).compute('degree_aware')
    net=nx.MultiDiGraph()
    for e in range(len(g.edge_type)):
        net.add_edge(int(g.edge_index[0,e]),int(g.edge_index[1,e]),key=e)
    oracle=[tuple(edge[2] for edge in p) for p in nx.all_simple_edge_paths(net,0,3,cutoff=3)]
    oracle=sorted(oracle,key=lambda p:(-np.prod(w.weights[list(p)]),p))[:10]
    assert top_paths(g,w,0,3,limit=10)==oracle

def test_adaptive_join_both_directions_match_oracle():
    g=graph([(0,0,i) for i in range(1,12)]+[(i,1,j) for i in range(1,5) for j in range(5,12)]+[(11,0,12)],n=13)
    w=CommunityEdgeWeighter(g).compute('degree_aware')
    net=nx.MultiDiGraph()
    for e in range(len(g.edge_type)):net.add_edge(int(g.edge_index[0,e]),int(g.edge_index[1,e]),key=e)
    for h,t in [(0,12),(12,0),(2,11),(11,2)]:
        oracle=[tuple(e[2] for e in p) for p in nx.all_simple_edge_paths(net,h,t,cutoff=3)]
        oracle=sorted(oracle,key=lambda p:(-np.prod(w.weights[list(p)]),p))[:10]
        assert top_paths(g,w,h,t)==oracle
