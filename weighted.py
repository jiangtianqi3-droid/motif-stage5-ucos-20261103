"""Weighted segmented aggregation with complete h/type/position/normalizer gradients.

Incidence weights are represented exactly as a_e / mean_group. This avoids
retaining I x B weights at every layer. All reductions stay bounded by chunks.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def norm_parts(A,Ids,Starts,Ends,Out,E:tl.constexpr,B:tl.constexpr,C:tl.constexpr):
    ch=tl.program_id(0); b=tl.program_id(1)
    i=tl.load(Starts+ch)+tl.arange(0,C); end=tl.load(Ends+ch)
    edge=tl.load(Ids+i,i<end,0)
    v=tl.load(A+b*E+edge,i<end,0.)
    tl.store(Out+ch*B+b,tl.sum(v,0))

@triton.jit
def reduce_parts(Parts,Ptr,Out,B:tl.constexpr,D:tl.constexpr,F:tl.constexpr,R:tl.constexpr):
    group=tl.program_id(0); b=tl.program_id(1)
    start=tl.load(Ptr+group); end=tl.load(Ptr+group+1)
    d=tl.arange(0,F)
    acc=tl.full((F,),0,Parts.dtype.element_ty)
    for at in range(start,end,R):
        ix=at+tl.arange(0,R)
        value=tl.load(Parts+(ix[:,None]*B+b)*D+d[None,:],(ix[:,None]<end)&(d[None,:]<D),0.)
        acc+=tl.sum(value,0)
    tl.store(Out+(group*B+b)*D+d,acc,d<D)

@triton.jit
def forward_parts(HP,Rel,A,Mean,Rows,Ids,Starts,Ends,Groups,Out,
                  I:tl.constexpr,E:tl.constexpr,B:tl.constexpr,D:tl.constexpr,T:tl.constexpr,
                  C:tl.constexpr,F:tl.constexpr,UNIT:tl.constexpr):
    ch=tl.program_id(0); b=tl.program_id(1)
    group=tl.load(Groups+ch); typ=group%T
    i=tl.load(Starts+ch)+tl.arange(0,C); end=tl.load(Ends+ch)
    d=tl.arange(0,F)
    r0=tl.load(Rows+i,i<end,0); r1=tl.load(Rows+I+i,i<end,0)
    x=tl.load(HP+(r0[:,None]*B+b)*D+d[None,:],(i[:,None]<end)&(d[None,:]<D),0.)
    y=tl.load(HP+(r1[:,None]*B+b)*D+d[None,:],(i[:,None]<end)&(d[None,:]<D),0.)
    if UNIT:
        w=tl.full((C,),1,HP.dtype.element_ty)
    else:
        edge=tl.load(Ids+i,i<end,0)
        a=tl.load(A+b*E+edge,i<end,0.)
        mean=tl.load(Mean+group*B+b)
        w=a/mean
    value=x*y*w[:,None]
    total=tl.sum(value,0)
    tl.store(Out+(ch*B+b)*D+d,total,d<D)

@triton.jit
def backward_edges(HP,Rel,A,Mean,Center,Grad,Rows,Ids,Starts,Ends,Groups,DR,DA,
                   I:tl.constexpr,E:tl.constexpr,B:tl.constexpr,D:tl.constexpr,T:tl.constexpr,
                   C:tl.constexpr,F:tl.constexpr,UNIT:tl.constexpr,NEED_A:tl.constexpr):
    ch=tl.program_id(0); b=tl.program_id(1)
    group=tl.load(Groups+ch); dst=group//T; typ=group%T
    i=tl.load(Starts+ch)+tl.arange(0,C); end=tl.load(Ends+ch)
    d=tl.arange(0,F)
    r0=tl.load(Rows+i,i<end,0); r1=tl.load(Rows+I+i,i<end,0)
    x=tl.load(HP+(r0[:,None]*B+b)*D+d[None,:],(i[:,None]<end)&(d[None,:]<D),0.)
    y=tl.load(HP+(r1[:,None]*B+b)*D+d[None,:],(i[:,None]<end)&(d[None,:]<D),0.)
    upstream=tl.load(Grad+(dst*B+b)*D+d,d<D,0.)
    rel=tl.load(Rel+(typ*B+b)*D+d,d<D,0.)
    edge=tl.load(Ids+i,i<end,0)
    if UNIT:
        w=tl.full((C,),1,HP.dtype.element_ty)
    else:
        a=tl.load(A+b*E+edge,i<end,0.)
        mean=tl.load(Mean+group*B+b)
        w=a/mean
    product=x*y
    dr=tl.sum(product*w[:,None]*upstream[None,:],0)
    tl.store(DR+(ch*B+b)*D+d,dr,d<D)
    if NEED_A:
        gw=tl.sum(product*(upstream*rel)[None,:],1)
        center=tl.load(Center+group*B+b)
        grad_a=(gw-center)/mean
        tl.atomic_add(DA+b*E+edge,grad_a,i<end)

@triton.jit
def backward_sources(HP,Rel,A,Mean,Grad,Rows,Groups,Ids,Pairs,Starts,Ends,Out,
                     I:tl.constexpr,E:tl.constexpr,B:tl.constexpr,D:tl.constexpr,T:tl.constexpr,
                     C:tl.constexpr,F:tl.constexpr,UNIT:tl.constexpr):
    ch=tl.program_id(0); b=tl.program_id(1)
    ix=tl.load(Starts+ch)+tl.arange(0,C); end=tl.load(Ends+ch)
    pair=tl.load(Pairs+ix,ix<end,0); inc=pair%I; own=pair//I
    other=1-own
    row=tl.load(Rows+other*I+inc,ix<end,0)
    group=tl.load(Groups+inc,ix<end,0); dst=group//T; typ=group%T
    d=tl.arange(0,F)
    factor=tl.load(HP+(row[:,None]*B+b)*D+d[None,:],(ix[:,None]<end)&(d[None,:]<D),0.)
    up=tl.load(Grad+(dst[:,None]*B+b)*D+d[None,:],(ix[:,None]<end)&(d[None,:]<D),0.)
    rel=tl.load(Rel+(typ[:,None]*B+b)*D+d[None,:],(ix[:,None]<end)&(d[None,:]<D),0.)
    if UNIT:
        w=tl.full((C,),1,HP.dtype.element_ty)
    else:
        edge=tl.load(Ids+inc,ix<end,0)
        a=tl.load(A+b*E+edge,ix<end,0.)
        mean=tl.load(Mean+group*B+b,ix<end,1.)
        w=a/mean
    value=tl.sum(factor*up*rel*w[:,None],0)
    tl.store(Out+(ch*B+b)*D+d,value,d<D)

def reduce(part,ptr,groups,B,D):
    out=torch.empty((groups,B,D),device=part.device,dtype=part.dtype)
    reduce_parts[(groups,B)](part,ptr,out,B,D,triton.next_power_of_2(D),32,num_warps=4)
    return out

class HyperRelConvWeightedSumAggrSegmented(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,rels,pos,a,plan,normalization='type',unit=False):
        N,B,D=h.shape; P=pos.shape[0]; T=rels.shape[0]; E=plan['E']; I=plan['I']; G=N*T
        hp=(h[:,None]+pos[None]).reshape(N*P,B,D); hp[:P]=1
        nc=plan['starts'].numel()
        if unit: mean=h.new_ones(G,B)
        else:
            parts=h.new_empty(nc,B)
            if nc: norm_parts[(nc,B)](a,plan['edge_ids'],plan['starts'],plan['ends'],parts,E,B,128,num_warps=4)
            sums=reduce(parts[:,:,None],plan['chunk_ptr'],G,B,1)[:,:,0]
            counts=plan['counts'].to(h.dtype)
            if normalization=='global':
                ss=sums.reshape(N,T,B).sum(1); cc=counts.reshape(N,T).sum(1)
                mm=ss/cc.clamp_min(1)[:,None]
                mean=torch.where(cc[:,None]>0,mm,torch.ones_like(mm))[:,None].expand(N,T,B).reshape(G,B).contiguous()
            else: mean=torch.where(counts[:,None]>0,sums/counts.clamp_min(1)[:,None],torch.ones_like(sums))
        parts=h.new_empty(nc,B,D)
        if nc: forward_parts[(nc,B)](hp,rels,a,mean,plan['hp_index'],plan['edge_ids'],plan['starts'],plan['ends'],plan['chunk_group'],parts,I,E,B,D,T,128,triton.next_power_of_2(D),unit,num_warps=4)
        seg=reduce(parts,plan['chunk_ptr'],G,B,D).reshape(N,T,B,D)
        output=(seg*rels[None]).sum(1)
        ctx.save_for_backward(h,rels,pos,a,mean,seg); ctx.plan=plan; ctx.norm=normalization; ctx.unit=unit
        return output
    @staticmethod
    def backward(ctx,grad):
        h,rels,pos,a,mean,seg=ctx.saved_tensors; plan=ctx.plan; grad=grad.contiguous()
        N,B,D=h.shape; T=rels.shape[0]; P=pos.shape[0]; E=plan['E']; I=plan['I']; G=N*T
        hp=(h[:,None]+pos[None]).reshape(N*P,B,D); hp[:P]=1
        counts=plan['counts'].to(h.dtype).reshape(N,T)
        value=(grad[:,None]*rels[None]*seg).sum(-1)
        if ctx.norm=='global':
            center=(value.sum(1)/counts.sum(1).clamp_min(1)[:,None])[:,None].expand(N,T,B).reshape(G,B).contiguous()
        else: center=(value/counts.clamp_min(1)[:,:,None]).reshape(G,B)
        da=torch.zeros_like(a) if ctx.needs_input_grad[3] else None
        nc=plan['starts'].numel(); part=h.new_empty(nc,B,D)
        if nc: backward_edges[(nc,B)](hp,rels,a,mean,center,grad,plan['hp_index'],plan['edge_ids'],plan['starts'],plan['ends'],plan['chunk_group'],part,da if da is not None else a,I,E,B,D,T,128,triton.next_power_of_2(D),ctx.unit,da is not None and not ctx.unit,num_warps=4)
        dr=reduce(part,plan['chunk_ptr'],G,B,D).reshape(N,T,B,D).sum(0)
        ng=plan['grad_starts'].numel(); gp=h.new_empty(ng,B,D)
        if ng: backward_sources[(ng,B)](hp,rels,a,mean,grad,plan['hp_index'],plan['groups'],plan['edge_ids'],plan['grad_pair'],plan['grad_starts'],plan['grad_ends'],gp,I,E,B,D,T,128,triton.next_power_of_2(D),ctx.unit,num_warps=4)
        dhp=reduce(gp,plan['grad_ptr'],N*P,B,D).reshape(N,P,B,D)
        dh=dhp.sum(1); dp=dhp.sum(0)
        return dh,dr,dp,da,None,None,None

def aggregate(features,type_embedding,position_embedding,a,plan,normalization='type',unit=False):
    B,N,D=features.shape
    h=features.transpose(0,1).contiguous()
    rels=type_embedding[:,None,:].expand(-1,B,-1).contiguous()
    pos=position_embedding[:,None,:].expand(-1,B,-1).contiguous()
    return HyperRelConvWeightedSumAggrSegmented.apply(h,rels,pos,a.contiguous(),plan,normalization,unit).transpose(0,1)

def reference(features,types,pos,a,he,ht,normalization='type'):
    # Explicit small-graph oracle, with padding source factors forced to one.
    B,N,D=features.shape; E=he.shape[1]
    shifted=he.T+1; gathered=features[:,shifted]
    position=torch.arange(1,he.shape[0]+1,device=features.device)
    factors=gathered+pos[position][None,None]
    factors=torch.where((shifted>0)[None,:,:,None],factors,torch.ones_like(factors))
    out=features.new_zeros(B,N,D)
    incidences=[]
    for slot in range(he.shape[0]):
        ids=torch.nonzero(shifted[:,slot]>0).flatten(); dest=shifted[ids,slot]
        product=features.new_ones(B,len(ids),D)
        for other in range(he.shape[0]):
            if other!=slot: product=product*factors[:,ids,other]
        incidences.append((ids,dest,product*types[ht[ids]][None]))
    counts=features.new_zeros(N*7 if normalization=='type' else N)
    sums=features.new_zeros(B,len(counts))
    for ids,dest,msg in incidences:
        group=dest*7+ht[ids] if normalization=='type' else dest
        counts=counts.index_add(0,group,features.new_ones(len(ids)))
        sums=sums.index_add(1,group,a[:,ids])
    mean=sums/counts.clamp_min(1)[None]
    for ids,dest,msg in incidences:
        group=dest*7+ht[ids] if normalization=='type' else dest
        w=a[:,ids]/mean[:,group]
        out=out.index_add(1,dest,msg*w[:,:,None])
    return out
