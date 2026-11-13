"""General [B,I] incident-weight API, in addition to the compressed gated API.

Backward returns the literal dot product dL/dw_i. Production gating fuses the
same derivative through its group normalization to save six I-sized tensors.
"""
import torch
import triton
from weighted import forward_parts,backward_edges,backward_sources,reduce

class DirectIncidentWeightedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,rels,pos,incident_weight,plan):
        N,B,D=h.shape; P=pos.shape[0]; T=rels.shape[0]; I=plan['I']
        assert incident_weight.shape==(B,I)
        p=dict(plan); p['edge_ids']=torch.arange(I,device=h.device,dtype=torch.int32); p['E']=I
        hp=(h[:,None]+pos[None]).reshape(N*P,B,D); hp[:P]=1
        mean=h.new_ones(N*T,B); nc=p['starts'].numel(); part=h.new_empty(nc,B,D)
        forward_parts[(nc,B)](hp,rels,incident_weight,mean,p['hp_index'],p['edge_ids'],p['starts'],p['ends'],p['chunk_group'],part,
            I,I,B,D,T,128,triton.next_power_of_2(D),False,num_warps=4)
        seg=reduce(part,p['chunk_ptr'],N*T,B,D).reshape(N,T,B,D)
        ctx.save_for_backward(h,rels,pos,incident_weight,mean); ctx.plan=p
        return (seg*rels[None]).sum(1)
    @staticmethod
    def backward(ctx,grad):
        h,rels,pos,w,mean=ctx.saved_tensors; p=ctx.plan; grad=grad.contiguous()
        N,B,D=h.shape; P=pos.shape[0]; T=rels.shape[0]; I=p['I']; hp=(h[:,None]+pos[None]).reshape(N*P,B,D); hp[:P]=1
        center=h.new_zeros(N*T,B); dw=torch.zeros_like(w); nc=p['starts'].numel(); part=h.new_empty(nc,B,D)
        backward_edges[(nc,B)](hp,rels,w,mean,center,grad,p['hp_index'],p['edge_ids'],p['starts'],p['ends'],p['chunk_group'],part,dw,
            I,I,B,D,T,128,triton.next_power_of_2(D),False,True,num_warps=4)
        dr=reduce(part,p['chunk_ptr'],N*T,B,D).reshape(N,T,B,D).sum(0)
        ng=p['grad_starts'].numel(); gp=h.new_empty(ng,B,D)
        backward_sources[(ng,B)](hp,rels,w,mean,grad,p['hp_index'],p['groups'],p['edge_ids'],p['grad_pair'],p['grad_starts'],p['grad_ends'],gp,
            I,I,B,D,T,128,triton.next_power_of_2(D),False,num_warps=4)
        dhp=reduce(gp,p['grad_ptr'],N*P,B,D).reshape(N,P,B,D)
        return dhp.sum(1),dr,dhp.sum(0),dw,None

def aggregate_incident(features,types,pos,w,plan):
    B,N,D=features.shape
    return DirectIncidentWeightedSum.apply(features.transpose(0,1).contiguous(),types[:,None].expand(-1,B,-1).contiguous(),
        pos[:,None].expand(-1,B,-1).contiguous(),w.contiguous(),plan).transpose(0,1)

def reference_incident(h,types,pos,w,plan):
    B,N,D=h.shape; P=len(pos)
    hp=h[:,:,None,:]+pos[None,None]
    hp=torch.where(torch.arange(N,device=h.device)[None,:,None,None]>0,hp,torch.ones_like(hp)).reshape(B,N*P,D)
    rows=plan['hp_index'].long(); group=plan['groups'].long()
    msg=hp[:,rows[0]]*hp[:,rows[1]]*types[group%7][None]*w[:,:,None]
    return h.new_zeros(B,N,D).index_add(1,group//7,msg)

def check():
    import json,numpy as np
    from plans import build_plan,to_device
    from setup import ROOT
    torch.manual_seed(19)
    he=np.array([[0,1,2,0],[1,2,0,2],[-1,0,1,-1]],dtype=np.int32); ht=np.array([0,3,4,2])
    p=to_device(build_plan(he,ht,3),'cuda'); args=[torch.randn(2,4,3,device='cuda',dtype=torch.float64,requires_grad=True),
        torch.randn(7,3,device='cuda',dtype=torch.float64,requires_grad=True),torch.randn(7,3,device='cuda',dtype=torch.float64,requires_grad=True),
        torch.rand(2,p['I'],device='cuda',dtype=torch.float64,requires_grad=True)]
    a=aggregate_incident(*args,p); b=reference_incident(*args,p); up=torch.randn_like(a)
    ga=torch.autograd.grad((a*up).sum(),args,retain_graph=True); gb=torch.autograd.grad((b*up).sum(),args,retain_graph=True)
    err=float((a-b).abs().max()); errs=[float((x-y).abs().max()) for x,y in zip(ga,gb)]
    assert err<1e-10 and max(errs)<1e-10
    assert torch.autograd.gradcheck(lambda *v:aggregate_incident(*v,p),tuple(args),fast_mode=True,atol=2e-5,rtol=2e-4)
    result=dict(status='passed',shape=[2,p['I']],output_error=err,gradient_errors=errs,gradcheck=True,
        derivative='grad_incident_weight = dot(unweighted source-product * type_embedding, grad_output_at_destination)')
    (ROOT/'results/direct_incident_tests.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))
if __name__=='__main__': check()
