"""Frozen coordinate-invariant features. No trainable vector projection."""
import numpy as np
import torch
from common import ROOT
@torch.no_grad()
def attach(h,key,shuffle=False):
    a=np.load(ROOT/'semantic'/(key+'.npz'));v=a['vectors'].copy();trans=a['transparent'];R=len(v)//2
    if shuffle:
        ix=np.random.default_rng(8111).permutation(R);v=v[np.r_[ix,ix+R]]
    v=torch.from_numpy(v).to(h.edge_index.device);he=h.edge_index
    h.cos_gram=(v@v.T).clamp(-1,1);h.cos_trans=torch.from_numpy(trans).to(he.device)
    h.cos_edge_trans=torch.ones(he.shape[1],device=he.device,dtype=torch.bool)
    for slot in range(3):h.cos_edge_trans&=(he[slot]<0)|h.cos_trans[he[slot].clamp_min(0)]
    cin=torch.zeros(he.shape[1],device=he.device);count=torch.zeros_like(cin)
    for i,j in [(0,1),(1,2)]:
        mask=(he[i]>=0)&(he[j]>=0)
        cin[mask]+=h.cos_gram[he[i,mask],he[j,mask]];count[mask]+=1
    cin/=count.clamp_min(1);h.cos_internal=standardize(cin[None],h.type_bounds,h.cos_edge_trans[None])[0]
    h.sem_alpha=.2*torch.sigmoid(-2*torch.maximum(h.topology_z[:,0],h.topology_z[:,3])-.5)*h.cos_edge_trans
@torch.no_grad()
def standardize(x,bounds,mask):
    y=torch.zeros_like(x)
    for k in range(7):
        at,to=bounds[k:k+2]
        if to==at:continue
        m=mask[:,at:to].expand(len(x),-1);v=x[:,at:to];n=m.sum(1,keepdim=True)
        mu=(v*m).sum(1,keepdim=True)/n.clamp_min(1)
        sd=(((v-mu).square()*m).sum(1,keepdim=True)/n.clamp_min(1)).sqrt()
        y[:,at:to]=torch.where(m&(sd>1e-6),(v-mu)/sd.clamp_min(1e-6),torch.zeros_like(v))
    return y
@torch.no_grad()
def features(h,query):
    he=h.edge_index;B=len(query);E=he.shape[1]
    total=torch.zeros(B,E,device=he.device);maximum=torch.full_like(total,-2);count=torch.zeros_like(total)
    for slot in range(3):
        r=he[slot];mask=(r[None]>=0)&(r[None]!=query[:,None])&h.cos_trans[r.clamp_min(0)][None]
        c=h.cos_gram[query[:,None],r.clamp_min(0)[None]]
        total+=c*mask;count+=mask;maximum=torch.maximum(maximum,torch.where(mask,c,torch.full_like(c,-2)))
    active=(count>0)&h.cos_edge_trans[None]&h.cos_trans[query,None]
    cq=.5*(total/count.clamp_min(1)+maximum)
    cq=standardize(cq,h.type_bounds,active)
    ci=h.cos_internal[None].expand(B,-1)
    return cq,ci,active
