"""Exact incident mixing with recomputation: no per-layer I-sized saved weights."""
import torch
from incident_api import aggregate_incident
def weights(at,asem,alpha,beta,plan):
    ids=plan['edge_ids'].long();groups=plan['groups'].long();G=len(plan['counts']);B=len(at)
    def normalize(a):
        ai=a[:,ids];s=a.new_zeros(B,G).scatter_add(1,groups[None].expand(B,-1),ai)
        means=s/plan['counts'].clamp_min(1)[None]
        return ai/means[:,groups].clamp_min(1e-12)
    return 1+alpha*(normalize(at)-1)+beta[:,ids]*(normalize(asem)-1)
class RecomputedMixed(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,typ,pos,at,asem,beta,alpha,plan):
        ctx.save_for_backward(h,typ,pos,at,asem,beta);ctx.alpha=alpha;ctx.plan=plan
        w=weights(at,asem,alpha,beta,plan)
        return aggregate_incident(h,typ,pos,w,plan)
    @staticmethod
    def backward(ctx,grad):
        saved=ctx.saved_tensors
        with torch.enable_grad():
            args=[v.detach().requires_grad_(ctx.needs_input_grad[i]) for i,v in enumerate(saved)]
            h,typ,pos,at,asem,beta=args
            y=aggregate_incident(h,typ,pos,weights(at,asem,ctx.alpha,beta,ctx.plan),ctx.plan)
            active=[v for v in args if v.requires_grad]
            gg=torch.autograd.grad(y,active,grad,allow_unused=True)
        it=iter(gg)
        return tuple(next(it) if v.requires_grad else None for v in args)+(None,None)
def aggregate(h,typ,pos,at,asem,beta,alpha,plan):return RecomputedMixed.apply(h,typ,pos,at,asem,beta,alpha,plan)
