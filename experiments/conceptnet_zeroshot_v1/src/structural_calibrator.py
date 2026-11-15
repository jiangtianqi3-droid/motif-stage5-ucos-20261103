import types, math
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F
from weighted import aggregate

def sparsify(z,p=.30,beta=.15):
    """Preserve original standardized PMI; recompute incidence plan after pruning."""
    out=dict(z)
    if z['n']/(z['R']/2)<=500:return out
    keep=[]
    for k in range(7):
        ids=np.flatnonzero(z['hyper_type']==k)
        if k>=3:
            ids=ids[np.argsort(-z['raw'][ids,3],kind='stable')[:math.ceil(len(ids)*p)]] if beta else ids[:0]
        keep.extend(sorted(ids.tolist()))
    keep=np.array(keep,dtype=np.int64)
    for key in ['hyper_type','raw','z']:out[key]=z[key][keep]
    out['hyper_edge']=z['hyper_edge'][:,keep]
    return out

def relation_forward(self,hyper,query):
    q=query+1;B=len(q);N=int(hyper.num_nodes)+1;D=self.dims[0]
    state=torch.zeros(B,N,D,device=q.device)
    state.scatter_(1,q[:,None,None].expand(B,1,D),1.)
    for layer in self.layers:
        old=state;hq=state[torch.arange(B,device=q.device),q]
        a=layer.topology_gate(hq,hyper.topology_z,hyper.type_bounds)
        # Damping type embeddings AFTER gate normalization avoids cancellation.
        scale=state.new_tensor([1,1,1,self.tri_beta,self.tri_beta,self.tri_beta,self.tri_beta])
        out=aggregate(state,layer.rel_embedding.weight*scale[:,None],layer.pos_embedding.weight,a,hyper.plan,self.incident_normalization,False)
        output=layer.dropout(layer.linear(torch.cat([out,state],-1)))
        if layer.norm_type=='layer_norm':output=layer.norm(output)
        state=F.relu(output)
        if self.short_cut:state=state+old
    return self.mlp(state)[:,1:]

def install_sparse(m,beta=.15):
    m.relation_model.tri_beta=beta
    m.relation_model.forward=types.MethodType(relation_forward,m.relation_model)

def edge_weights(edge,rel,n,R):
    # Triton receives at stored row 0, sends from row 1. Match historical operator.
    receiver,sender=edge
    degree=torch.bincount(sender,minlength=n).float()
    local=torch.bincount(sender*R+rel,minlength=n*R).reshape(n,R)
    raw=(1+degree[sender]).pow(-.75)*(1+local[sender,rel]).float().pow(-.5)
    return degree,raw,raw.mean()

def plus_bellman(self,data,h_index,r_index,separate_grad=False):
    for layer in self.layers:layer.boost_heads=h_index
    return self.boost_original_bellman(data,h_index,r_index,separate_grad)

def plus_forward(self,input,query,boundary,edge_index,edge_type,size,edge_weight=None):
    relation=self.relation_projection(self.relation)
    out=torch.zeros_like(input)
    receiver,sender=edge_index
    for at in range(0,len(edge_type),2048):
        sl=slice(at,at+2048);src=sender[sl]
        raw=self.boost_raw[sl][None].expand(len(input),-1)
        if self.boost_layer==0:
            protect=(src[None]==self.boost_heads[:,None]) & (self.boost_degree[self.boost_heads,None]<=2)
            raw=torch.where(protect,torch.ones_like(raw),raw)
        w=1-self.boost_alpha+self.boost_alpha*raw/self.boost_mean
        msg=input[:,src]*relation[:,edge_type[sl]]*w[:,:,None]
        out.index_add_(1,receiver[sl],msg)
    return self.update(out+boundary,input)

def install_plus(m,d,alpha=.90):
    degree,raw,mean=edge_weights(d.edge_index,d.edge_type,d.num_nodes,d.num_relations)
    e=m.entity_model;e.boost_original_bellman=e.bellmanford;e.bellmanford=types.MethodType(plus_bellman,e)
    for i,l in enumerate(e.layers):
        l.boost_raw=raw;l.boost_mean=mean;l.boost_degree=degree;l.boost_alpha=alpha;l.boost_layer=i
        l.forward=types.MethodType(plus_forward,l)

class Reachability:
    def __init__(self,z):
        self.n=z['n'];self.R=z['R'];u,v=z['edge_index'];r=z['edge_type']
        self.degree=np.bincount(u,minlength=self.n)
        local=np.bincount(u*self.R+r,minlength=self.n*self.R).reshape(self.n,self.R)
        self.weight=((1+self.degree[u])**-.75*(1+local[u,r])**-.5).astype(np.float32)
        self.u,self.v,self.r=u,v,r
        self.rho=np.zeros((self.R,self.R),np.float32)
        he=z['hyper_edge'];ht=z['hyper_type'];s=z['raw'][:,3]
        for k in range(7):
            ids=np.flatnonzero(ht==k);arity=2 if k<3 else 3
            for a in range(arity):
                for b in range(arity):
                    if a!=b:np.maximum.at(self.rho,(he[a,ids],he[b,ids]),s[ids])
        self.rho/=np.maximum(self.rho.max(1,keepdims=True),1e-12)
        np.fill_diagonal(self.rho,1.)
        self.current=None;self.matrix=None
        self.adjacency=sp.csr_matrix((np.ones(len(u),dtype=bool),(u,v)),shape=(self.n,self.n))

    def reachable_targets(self,queries):
        x=sp.csr_matrix((np.ones(len(queries),dtype=bool),(np.arange(len(queries)),queries[:,0])),shape=(len(queries),self.n))
        found=np.zeros(len(queries),bool)
        for _ in range(3):
            x=x@self.adjacency
            found|=np.asarray(x[np.arange(len(queries)),queries[:,2]]).ravel()
        return found

    def score(self,queries):
        out=np.zeros((len(queries),self.n),np.float32)
        for q in np.unique(queries[:,1]):
            ix=np.flatnonzero(queries[:,1]==q)
            if self.current!=q:
                self.matrix=sp.csr_matrix((self.weight*self.rho[q,self.r],(self.u,self.v)),shape=(self.n,self.n))
                self.current=q
            x=sp.csr_matrix((np.ones(len(ix),np.float32),(np.arange(len(ix)),queries[ix,0])),shape=(len(ix),self.n))
            total=sp.csr_matrix(x.shape,dtype=np.float32)
            for lam in [1.,.5,.25]:
                x=x@self.matrix;total=total+lam*x
            out[ix]=np.log1p(total.toarray())
        return out
