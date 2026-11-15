"""Small independent oracles for direction, normalization, masking and ranking."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from structural_calibrator import edge_weights,plus_forward,sparsify,Reachability
from evaluator import safe_z,ranks,truth_index,bootstrap
from types import SimpleNamespace

def main():
    edge=torch.tensor([[0,0,1,2],[1,2,0,0]])
    rel=torch.tensor([0,0,1,1]);degree,raw,mean=edge_weights(edge,rel,3,2)
    expected=torch.tensor([(1+1)**-.75*(1+1)**-.5]*2+[(1+2)**-.75*(1+2)**-.5]*2)
    torch.testing.assert_close(raw,expected)
    torch.testing.assert_close((raw/mean).mean(),torch.tensor(1.))
    x=torch.arange(6,dtype=torch.float32).reshape(1,3,2)
    relation=torch.tensor([[[2.,3.],[4.,5.]]])
    layer=SimpleNamespace(relation_projection=lambda a:a,relation=relation,boost_raw=raw,boost_mean=mean,boost_degree=degree,boost_alpha=.9,boost_layer=0,boost_heads=torch.tensor([0]),update=lambda a,b:a)
    got=plus_forward(layer,x,None,torch.zeros_like(x),edge,rel,(3,3))
    want=torch.zeros_like(x)
    for e in range(4):
        dst,src=edge[:,e];w=1. if src==0 and degree[src]<=2 else raw[e]
        want[:,dst]+=x[:,src]*relation[:,rel[e]]*(.1+.9*w/mean)
    torch.testing.assert_close(got,want)
    assert torch.isfinite(safe_z(torch.ones(2,3))).all()
    score=torch.tensor([[2.,2.,1.]])
    query=np.array([[0,0,0]])
    assert ranks(score,query,{})[0]==2
    assert ranks(score,query,{(0,0):np.array([0,1])})[0]==1
    assert score[0,1]==2,'Filtered ranking must not mutate scores'
    rng=np.random.default_rng(2)
    r=rng.integers(1,100,20)
    assert np.allclose(bootstrap(r,r)['paired_delta_ci95'],0)
    print('PASS: actual sender direction; joint fanout; mean anchor; first-hop exemption; finite zero variance; pessimistic filtered ties; immutable scores; paired bootstrap')

if __name__=='__main__':main()
