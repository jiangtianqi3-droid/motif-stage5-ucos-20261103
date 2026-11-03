"""CPU incidence layouts; no entity path enumeration. Int32 large index buffers."""
import numpy as np
import torch

def chunks(counts, width=128):
    ptr=np.r_[0,np.cumsum(counts,dtype=np.int64)]
    number=(counts+width-1)//width
    cp=np.r_[0,np.cumsum(number,dtype=np.int64)]
    group=np.repeat(np.arange(len(counts),dtype=np.int32),number)
    local=np.arange(len(group),dtype=np.int64)-cp[group]
    starts=ptr[group]+local*width
    ends=np.minimum(starts+width,ptr[group+1])
    return starts.astype(np.int32),ends.astype(np.int32),group,cp.astype(np.int32)

def build_plan(he, types, R, positions=7, normalization='type'):
    he=np.asarray(he,dtype=np.int32); types=np.asarray(types,dtype=np.int32)
    K,E=he.shape; N=R+1; T=7
    src=[]; dest=[]; eid=[]; hp=[]
    for slot in range(K):
        ix=np.flatnonzero(he[slot]>=0)
        eid.append(ix.astype(np.int32)); dest.append((he[slot,ix]+1)*T+types[ix])
        other=[]
        for j in range(K):
            if j!=slot: other.append(np.where(he[j,ix]>=0,(he[j,ix]+1)*positions+j+1,0))
        hp.append(np.stack(other).astype(np.int32))
    groups=np.concatenate(dest); order=np.argsort(groups,kind='stable')
    edge=np.concatenate(eid)[order]; rows=np.concatenate(hp,axis=1)[:,order]
    groups=groups[order].astype(np.int32); I=len(edge)
    counts=np.bincount(groups,minlength=N*T)
    starts,ends,cgroup,cptr=chunks(counts)
    # Forward is sorted by (destination,type). Reuse its pair IDs for reverse
    # source reductions; each pair is a distinct source slot occurrence.
    flat=rows.ravel(); valid=np.flatnonzero(flat>=positions)
    perm=valid[np.argsort(flat[valid],kind='stable')].astype(np.int32)
    gc=np.bincount(flat[valid],minlength=N*positions)
    gs,ge,gg,gp=chunks(gc)
    output=dict(edge_ids=edge,hp_index=rows,groups=groups,counts=counts.astype(np.int32),
        starts=starts,ends=ends,chunk_group=cgroup,chunk_ptr=cptr,grad_pair=perm,
        grad_starts=gs,grad_ends=ge,grad_group=gg,grad_ptr=gp,
        n=np.array(N),T=np.array(T),P=np.array(positions),I=np.array(I),E=np.array(E),K=np.array(K))
    return output

def to_device(plan, device):
    return {k:torch.from_numpy(np.ascontiguousarray(v)).to(device) if np.asarray(v).ndim else int(v)
            for k,v in plan.items()}
