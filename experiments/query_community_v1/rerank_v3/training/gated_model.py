import types
import torch
from torch import nn
from torch.nn import functional as F
from weighted import aggregate

class TopologyGate(nn.Module):
    def __init__(self,D=64,Fdim=4,interactive=True):
        super().__init__(); self.Fdim=Fdim; self.interactive=interactive
        self.w1=nn.Parameter(torch.zeros(7,Fdim)); self.bias=nn.Parameter(torch.zeros(7))
        if interactive:
            self.Wq=nn.Linear(D,16,bias=False); nn.init.zeros_(self.Wq.weight)
            self.Ws=nn.Linear(Fdim,16,bias=False); nn.init.xavier_normal_(self.Ws.weight)
    def forward(self,hq,z,bounds):
        z=z.to(dtype=hq.dtype)
        if self.Fdim==1: z=z[:,3:4]
        # Algebraic contraction avoids materializing E x 16 projected features.
        basic=torch.cat([z[bounds[k]:bounds[k+1]]@self.w1[k]+self.bias[k] for k in range(7)])
        if self.interactive:
            beta=self.Wq(hq)@self.Ws.weight
            g=beta@z.T+basic[None]
        else: g=basic[None].expand(len(hq),-1)
        return torch.exp(2*torch.tanh(g/2))

def forward(self,hyper,query):
    q=query+1; B=len(q); N=int(hyper.num_nodes)+1; D=self.dims[0]
    state=torch.zeros(B,N,D,device=query.device,dtype=self.layers[0].linear.weight.dtype)
    state=state.scatter(1,q[:,None,None].expand(B,1,D),torch.ones(B,1,D,device=query.device,dtype=state.dtype))
    for layer in self.layers:
        old=state; hq=state[torch.arange(B,device=q.device),q]
        if hasattr(layer,'topology_gate'): a=layer.topology_gate(hq,hyper.topology_z,hyper.type_bounds)
        else: a=state.new_ones(B,hyper.plan['E'])
        out=aggregate(state,layer.rel_embedding.weight,layer.pos_embedding.weight,a,hyper.plan,
                      self.incident_normalization,not hasattr(layer,'topology_gate'))
        output=layer.linear(torch.cat([out,state],dim=-1))
        output=layer.dropout(output)
        if layer.norm_type=='layer_norm': output=layer.norm(output)
        state=F.relu(output)
        if self.short_cut and state.shape==old.shape: state=state+old
    return self.mlp(state)[:,1:]

def install(model,method='T2',freedom='gate',normalization='type'):
    rel=model.relation_model
    if method!='C0':
        for layer in rel.layers:
            layer.topology_gate=TopologyGate(layer.in_channels,1 if method=='T3' else 4,method!='T1').to(layer.linear.weight.device)
    rel.incident_normalization=normalization
    rel.forward=types.MethodType(forward,rel)
    for p in model.parameters(): p.requires_grad_(False)
    if freedom=='joint' or method=='C0':
        for name,p in rel.named_parameters():
            # The unused static initialization table is frozen in official MOTIF.
            if name!='position.weight': p.requires_grad_(True)
    else:
        for layer in rel.layers:
            for p in layer.topology_gate.parameters(): p.requires_grad_(True)
    return model
