import types
import torch
from torch import nn
from torch.nn import functional as F
from gated_model import TopologyGate
from weighted import aggregate as normalized_aggregate
from mixed import aggregate as mixed_aggregate
from cosine import features
class CosGate(nn.Module):
    def __init__(self):super().__init__();self.theta=nn.Parameter(torch.zeros(2))
    def forward(self,cq,ci):return torch.exp(2*torch.tanh((self.theta[0]*cq+self.theta[1]*ci)/2))
def forward(self,hyper,query):
    q=query+1;B=len(q);D=self.dims[0];N=int(hyper.num_nodes)+1
    state=self.layers[0].linear.weight.new_zeros(B,N,D).scatter(1,q[:,None,None].expand(B,1,D),1.)
    alpha=hyper.alpha_adaptive if self.stage5_config['alpha']=='adaptive' else self.stage5_config['alpha']
    semantic=self.stage5_config.get('semantic',False)
    if semantic:
        cq,ci,active=features(hyper,query);beta=hyper.sem_alpha[None].expand(B,-1)*active
        semantic=bool(beta.any())
    for layer in self.layers:
        old=state;hq=state[torch.arange(B,device=q.device),q];a=state.new_ones(B,hyper.plan['E'])
        original=normalized_aggregate(state,layer.rel_embedding.weight,layer.pos_embedding.weight,a,hyper.plan,'type',True)
        if hasattr(layer,'topology_gate'):
            a=layer.topology_gate(hq,hyper.topology_z,hyper.type_bounds)
            if semantic:
                ase=layer.cos_gate(cq,ci)
                out=mixed_aggregate(state,layer.rel_embedding.weight,layer.pos_embedding.weight,a,ase,beta,alpha,hyper.plan)
            else:
                weighted=normalized_aggregate(state,layer.rel_embedding.weight,layer.pos_embedding.weight,a,hyper.plan,'type',False)
                out=original+alpha*(weighted-original)
        else:out=original
        state=F.relu(layer.norm(layer.dropout(layer.linear(torch.cat([out,state],-1))))) if layer.norm_type=='layer_norm' else F.relu(layer.dropout(layer.linear(torch.cat([out,state],-1))))
        if self.short_cut and state.shape==old.shape:state=state+old
    return self.mlp(state)[:,1:]
def install(model,cfg):
    rel=model.relation_model;rel.stage5_config=cfg
    if cfg['method']!='plain':
        for layer in rel.layers:
            layer.topology_gate=TopologyGate(layer.in_channels,4,True).to(layer.linear.weight.device)
            if cfg.get('semantic'):layer.cos_gate=CosGate().to(layer.linear.weight.device)
    rel.forward=types.MethodType(forward,rel)
    for p in model.parameters():p.requires_grad_(False)
    for n,p in rel.named_parameters():
        if ('gate' in n) or (cfg['freedom']=='joint_low_lr' and n!='position.weight'):p.requires_grad_(True)
    return model
