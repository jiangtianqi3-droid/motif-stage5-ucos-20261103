import json
import numpy as np
import torch
from common import ROOT,base_model,load_cached,dump
from plans import build_plan,to_device
from mixed import aggregate,weights
from incident_api import reference_incident
from cosine import attach,features
from model import install,CosGate
def main():
    torch.manual_seed(3)
    he=np.array([[0,1,2,0],[1,2,0,2],[-1,0,1,-1]],dtype=np.int32);ht=np.array([0,3,4,2])
    p=to_device(build_plan(he,ht,3),'cuda');dtype=torch.float64
    args=[torch.randn(2,4,3,device='cuda',dtype=dtype,requires_grad=True),torch.randn(7,3,device='cuda',dtype=dtype,requires_grad=True),torch.randn(7,3,device='cuda',dtype=dtype,requires_grad=True),torch.rand(2,4,device='cuda',dtype=dtype,requires_grad=True)+.3,torch.rand(2,4,device='cuda',dtype=dtype,requires_grad=True)+.3,torch.rand(2,4,device='cuda',dtype=dtype)*.2]
    a=aggregate(*args,.2,p);b=reference_incident(*args[:3],weights(args[3],args[4],.2,args[5],p),p)
    up=torch.randn_like(a);ga=torch.autograd.grad((a*up).sum(),args[:5],retain_graph=True);gb=torch.autograd.grad((b*up).sum(),args[:5],retain_graph=True)
    error=float((a-b).abs().max());gradient_errors=[float((x-y).abs().max()) for x,y in zip(ga,gb)]
    assert error<1e-10 and max(gradient_errors)<1e-10
    assert torch.autograd.gradcheck(lambda *v:aggregate(*v,args[5],.2,p),tuple(args[:5]),fast_mode=True)
    d,t=load_cached('wn18rr_train','validation');q=torch.cat([d.target_edge_index,d.target_edge_type[None]]).T[:2,None,:];m=base_model().cuda().eval()
    with torch.no_grad():ref=m(d,q)
    cfg=dict(method='topology',alpha=.2,freedom='gate_only',semantic=False)
    m=install(m,cfg).eval()
    with torch.no_grad():out=m(d,q)
    identity_error=float((ref-out).abs().max());assert identity_error<1e-3,identity_error
    attach(d.relation_hypergraph,'wn18rr_train');h=d.relation_hypergraph;c0=features(h,torch.tensor([0,1],device='cuda'))
    um=install(base_model().cuda(),dict(cfg,semantic=True)).eval()
    with torch.no_grad():uo=um(d,q)
    semantic_identity_error=float((ref-uo).abs().max());assert semantic_identity_error<1e-3
    v=np.load(ROOT/'semantic/wn18rr_train.npz')['vectors'];ortho,_=np.linalg.qr(np.random.default_rng(3).normal(size=(384,384)));gram=(v@ortho)@(v@ortho).T
    rotational_error=float(abs(gram-h.cos_gram.cpu().numpy()).max());assert rotational_error<1e-5
    wk=np.load(ROOT/'semantic/WKIngram_25.npz');assert not wk['transparent'].any()
    assert sum(p.numel() for p in CosGate().parameters())==2
    # Zero uncertainty weights force the semantic residual to zero exactly.
    w0=weights(args[3],args[4],.2,torch.zeros_like(args[5]),p)
    w1=weights(args[3],args[4]*3,.2,torch.zeros_like(args[5]),p);assert torch.equal(w0,w1)
    dump(ROOT/'results/operator_tests.json',dict(status='passed',forward_error=error,gradient_errors=gradient_errors,gradcheck=True,model_identity_max_score_error=identity_error,semantic_model_identity_max_score_error=semantic_identity_error,rotation_invariance_error=rotational_error,semantic_parameters_per_layer=2,wk_all_masked=True,zero_semantic_beta_exact=True))
    print('ALL OPERATOR TESTS PASSED',flush=True)
if __name__=='__main__':main()
