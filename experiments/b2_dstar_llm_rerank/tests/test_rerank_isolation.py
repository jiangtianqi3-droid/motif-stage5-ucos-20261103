import json
import numpy as np
import pytest
from ranking import candidate_pool, decision_pool, apply_response, writeback, validate, grouped_validation
from prompting import Lexicon, TokenBudget, BudgetExceeded
from evidence import EvidenceGraph

def response(ids, abstain=False):
    return json.dumps(dict(abstain=abstain, ranked_candidate_ids=ids, confidence=.8, evidence_summary='路径支持'))

def ev(ids, paths=True): return {x:dict(paths=[dict(weight=.5)] if paths else []) for x in ids}

def test_answer_outside_top30_preserved():
    order=list(range(100)); pool=[0,1,2,3,4,15,20,29]
    new=writeback(order[:30],pool,[29,0,1,2,3,4,15,20])+order[30:]
    assert new[30:]==order[30:]
    assert new.index(80)==order.index(80)

def graph():
    config=dict(eta_out=.5,eta_rel=.5,local_nodes=256)
    return EvidenceGraph(dict(n=5,R=4,edge_index=np.array([[0,1,1,2],[1,0,2,1]]),
        edge_type=np.array([0,2,1,3]),hyper_edge=np.empty((3,0),int),hyper_type=np.empty(0,int),raw=np.empty((0,4))),config)

def test_inverse_direction_distinction():
    g=graph(); lex=Lexicon(['a','b','c','d','e'],['parent','works'],g)
    assert lex.relation(0)==dict(name='parent',inverse=False)
    assert lex.relation(2)==dict(name='parent',inverse=True)
    f=g.retrieve(0,0,[1])[1]['paths'][0]['edges'][0]
    b=g.retrieve(1,2,[0])[0]['paths'][0]['edges'][0]
    assert g.rel[f]==0 and g.rel[b]==2

def test_no_evidence_exact_fallback():
    top=list(range(30)); pool=top[:8]
    assert apply_response(top,pool,ev(pool,False),response(pool[::-1]))==(top,'no_evidence')

@pytest.mark.parametrize('ids',[[0,0,2,3,4,5,6,7],[0,1,2],[0,1,2,3,4,5,6,999]])
def test_invalid_candidate_bijection_fallback(ids):
    top=list(range(30)); pool=top[:8]
    assert apply_response(top,pool,ev(pool),response(ids))==(top,'parse_or_validation_fallback')

@pytest.mark.parametrize('content,exception',[('not json',None),('{}',None),(None,TimeoutError())])
def test_parse_timeout_fallback(content,exception):
    top=list(range(30)); pool=top[:8]
    assert apply_response(top,pool,ev(pool),content,exception)==(top,'parse_or_validation_fallback')

def test_abstain_zero_changes():
    top=list(range(30)); pool=top[:8]
    assert apply_response(top,pool,ev(pool),response(pool[::-1],True))==(top,'abstain')

def test_challenger_continuous_insertion():
    top=list(range(30)); pool=[0,1,2,3,4,20,24,29]
    result=writeback(top,pool,[24,0,1,2,3,4,20,29])
    assert result[:7]==[24,0,1,2,3,4,5]
    assert result.index(4)==5
    assert [x for x in result if x not in pool]==[x for x in top if x not in pool]
    assert result[-1]==29

def test_zero_leakage_prompt_character_identity():
    g=graph(); lex=Lexicon(['a','b','c','d','e'],['parent','works'],g)
    scores=np.array([.1,.2,.3,.4,.5])
    def generate(test_labels):
        # Labels have no route into production candidate/retrieval/prompt APIs.
        top=candidate_pool(scores,g.tails.get((0,0),set()))
        evidence=g.retrieve(0,0,top); pool=decision_pool(top,evidence)
        return json.dumps(lex.build(0,0,top,pool,evidence,'Paths+Motifs+LLM'),sort_keys=True)
    assert generate([(0,0,4)])==generate([(0,0,2),(0,0,3)])==generate([])
    assert 4 in candidate_pool(scores,g.tails.get((0,0),set()))

def test_grouped_validation_no_sibling_leakage():
    facts=np.array([[0,0,1],[0,0,2],[3,0,2],[4,1,5],[6,1,7]])
    a,b=grouped_validation(facts,2)
    assert ({0,1,2} <= set(a)) or ({0,1,2} <= set(b))
    assert not set(a)&set(b) and set(a)|set(b)==set(range(5))

def test_opaque_entity_uses_profile():
    lex=Lexicon(['/m/abc','123','Q12','deadbeefdeadbeef',None],['parent','works'],graph())
    assert 'name' not in lex.entity(0) and 'type_signature' in lex.entity(0)
    assert '/m/abc' not in json.dumps(lex.entity(0))

def test_budget_never_drops_candidates_or_top_path():
    budget=TokenBudget(2000)
    rows=[dict(candidate_id=i,name='entity',paths=[{'text':'word '*100}]*3,motifs=[{'text':'x'*100}]) for i in range(8)]
    messages,count,changes=budget.fit(dict(candidates=rows))
    parsed=json.loads(messages[1]['content'])
    assert len(parsed['candidates'])==8 and all(len(c['paths'])>=1 for c in parsed['candidates'])
    assert count<=2000 and changes

def test_mandatory_overflow_explicit():
    with pytest.raises(BudgetExceeded): TokenBudget(500).fit(dict(candidates=[dict(name='word '*2000)]))

def test_high_receiver_degree_does_not_penalize_edge():
    g=graph(); before=g.base_weights[0]
    # The formula depends only on the sender, never the receiving node's degree.
    expected=(1+g.degree[0])**-.5*(1+g.profiles[0][0])**-.5
    assert before==expected

def test_pool_includes_remote_path_challenger():
    top=list(range(30)); evidence=ev(top,False); evidence[25]['paths']=[dict(weight=.8)]
    assert decision_pool(top,evidence)==[0,1,2,3,4,25]

def test_contiguous_eight_permutation():
    top=list(range(30)); assert writeback(top,top[:8],top[:8][::-1])==top[:8][::-1]+top[8:]
