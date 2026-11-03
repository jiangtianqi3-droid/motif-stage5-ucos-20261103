"""Four-channel sparse walk signatures. No entity-path enumeration here."""
import hashlib
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp

VERSION = 'slot4-rw-support-marginal-exclusivity-v2'
TYPES = ['HH','TT','HT','TFH','TFT','HFT','HFH']
SIDES = [(0,0),(1,1),(0,1),(1,0),(1,1),(0,1),(0,0)]

def graph_key(edge, rel, n, R):
    h=hashlib.sha256((VERSION+repr((n,R))).encode())
    for a in (edge,rel):
        a=np.ascontiguousarray(a,dtype=np.int64)
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def powered(C, exponent):
    out=C.copy(); out.data **= exponent
    return out

def gather(M, rows, cols):
    if not len(rows): return np.empty(0,dtype=np.float64)
    # SciPy 1.13 mutates index writeability flags; PyTorch-backed ndarray views
    # cannot always accept that mutation. Own the small index buffers explicitly.
    return np.asarray(M.tocsr()[np.array(rows,copy=True),np.array(cols,copy=True)]).ravel()

def contract(left, Fflat, right, n, R, B):
    """(a,u)(u,b*v) -> (a*b,v)(v,c). Only sparse multiplication."""
    mid=(left.T@Fflat).tocoo()
    unfolded=sp.coo_matrix((mid.data,(mid.row*B+mid.col//n,mid.col%n)),shape=(R*B,n)).tocsr()
    return (unfolded@right).tocsr()

def standardize(raw, kinds, eps=1e-6):
    z=np.zeros_like(raw); means=np.zeros((7,4)); stds=np.zeros((7,4))
    for k in range(7):
        ids=np.flatnonzero(kinds==k)
        if len(ids):
            means[k]=raw[ids].mean(0); stds[k]=raw[ids].std(0)
            z[ids]=(raw[ids]-means[k])/(stds[k]+eps)
    return z,means,stds

def extract(edge, rel, n, R, block=8, cache=None):
    edge=np.asarray(edge,dtype=np.int64); rel=np.asarray(rel,dtype=np.int64)
    if edge.shape!=(2,len(rel)): raise ValueError('edge/rel shape')
    if len(rel) and (edge.min()<0 or edge.max()>=n or rel.min()<0 or rel.max()>=R): raise ValueError('index range')
    key=graph_key(edge,rel,n,R)
    path=Path(cache)/(key+'.npz') if cache else None
    start=time.perf_counter()
    if path and path.exists():
        with np.load(path) as z: result={k:z[k].copy() for k in z.files}
        result['cache_hit']=True; result['load_seconds']=time.perf_counter()-start
        return result
    ones=np.ones(len(rel),dtype=np.float64)
    C=[sp.coo_matrix((ones,(edge[s],rel)),shape=(n,R)).tocsr() for s in (0,1)]
    U=[(c>0).astype(np.float64) for c in C]
    degree=np.bincount(edge[0],minlength=n).astype(np.float64)
    indegree=np.bincount(edge[1],minlength=n).astype(np.float64)
    if not np.array_equal(degree,indegree): raise ValueError('Expected inverse-augmented graph')
    safe=np.maximum(degree,1)
    # Incoming first-step mass after row-stochastic traversal. For head incidence,
    # traversal is reversed and must divide by original relation's tail-local degree.
    local_h=gather(C[0],edge[0],rel); local_t=gather(C[1],edge[1],rel)
    X=[sp.coo_matrix((1/local_t,(edge[0],rel)),shape=(n,R)).tocsr(),
       sp.coo_matrix((1/local_h,(edge[1],rel)),shape=(n,R)).tocsr()]
    all_he=[]; all_ht=[]; all_count=[]; all_q=[]; all_rw=[]; all_rel=[]
    for k,(L,S) in enumerate(SIDES[:3]):
        M=(C[L].T@C[S]).tocoo(); M.sum_duplicates()
        order=np.lexsort((M.col,M.row)); a=M.row[order]; c=M.col[order]
        q=C[L].T@sp.diags(safe**-.5)@C[S]
        rw=X[L].T@U[S]
        rr=powered(C[L],.75).T@powered(C[S],.75)
        all_he.append(np.vstack([a,c,np.full(len(a),-1)]).astype(np.int32))
        all_ht.append(np.full(len(a),k,dtype=np.int8)); all_count.append(M.data[order])
        all_q.append(gather(q,a,c)); all_rw.append(gather(rw,a,c)); all_rel.append(gather(rr,a,c))
    # Work in bounded central-relation blocks. No loops over entities or walks.
    Q=[sp.diags(safe**-.25)@c for c in C]
    RelC=[powered(c,.875) for c in C]
    type_parts={k:[] for k in range(3,7)}
    relation_order=np.argsort(rel,kind='stable')
    sorted_rel=rel[relation_order]
    for b0 in range(0,R,block):
        B=min(block,R-b0)
        lo,hi=np.searchsorted(sorted_rel,[b0,b0+B]); ids=relation_order[lo:hi]
        flatcols=(rel[ids]-b0)*n+edge[1,ids]
        shape=(n,B*n)
        F=sp.coo_matrix((ones[ids],(edge[0,ids],flatcols)),shape=shape).tocsr()
        FP=sp.coo_matrix((1/local_h[ids],(edge[0,ids],flatcols)),shape=shape).tocsr()
        FR=sp.coo_matrix(((local_h[ids]*local_t[ids])**(-.125),(edge[0,ids],flatcols)),shape=shape).tocsr()
        for k,(L,S) in enumerate(SIDES[3:],3):
            M=contract(C[L],F,C[S],n,R,B).tocoo(); M.sum_duplicates()
            row,col=M.row,M.col
            he=np.vstack([row//B,row%B+b0,col]).astype(np.int32)
            q=gather(contract(Q[L],F,Q[S],n,R,B),row,col)
            rw=gather(contract(X[L],FP,U[S],n,R,B),row,col)
            rr=gather(contract(RelC[L],FR,RelC[S],n,R,B),row,col)
            type_parts[k].append((he,M.data.copy(),q,rw,rr))
    for k in range(3,7):
        parts=type_parts.pop(k)
        he=np.concatenate([p[0] for p in parts],axis=1)
        arrays=[np.concatenate([p[j] for p in parts]) for j in range(1,5)]
        order=np.lexsort((he[2],he[1],he[0])); he=he[:,order]
        all_he.append(he); all_ht.append(np.full(he.shape[1],k,dtype=np.int8))
        for dest,a in zip((all_count,all_q,all_rw,all_rel),arrays): dest.append(a[order])
        del parts,arrays
    he=np.concatenate(all_he,axis=1); ht=np.concatenate(all_ht)
    count=np.concatenate(all_count); q=np.concatenate(all_q); rw=np.concatenate(all_rw)
    ra_rel=np.concatenate(all_rel)/count
    # Marginals are accumulated over the complete type, never within a block only.
    exclusivity=np.zeros(len(ht),dtype=np.float64)
    for k in range(7):
        ids=np.flatnonzero(ht==k); arity=2 if k<3 else 3
        denom=np.ones(len(ids),dtype=np.float64)
        for slot in range(arity):
            marginal=np.bincount(he[slot,ids],weights=q[ids],minlength=R)
            denom*=marginal[he[slot,ids]]
        exclusivity[ids]=q[ids]/denom**(1/arity)
    raw=np.column_stack([np.log1p(count),q/count,rw,exclusivity])
    if not np.isfinite(raw).all() or (raw<0).any(): raise ValueError('Nonfinite/negative signature')
    if len(raw) and (raw[:,[1,3]].max()>1+1e-10): raise ValueError('Bounded channel exceeds one')
    if len(raw) and raw[:,2].max()>n+1e-10: raise ValueError('RW support exceeds number of starts')
    z,mu,sigma=standardize(raw,ht)
    result=dict(edge_index=he,edge_type=ht,raw=raw,z=z,count=count,weighted_sum=q,
                ra_rel=ra_rel,means=mu,stds=sigma,degree=degree,
                seconds=np.array(time.perf_counter()-start),key=np.array(key),version=np.array(VERSION))
    if path:
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.tmp.npz'); np.savez_compressed(temp,**result); temp.replace(path)
    result['cache_hit']=False
    return result

def align(result, hyper_edge, hyper_type, R):
    """Verify exact canonical support and return features in external edge order."""
    he=np.asarray(hyper_edge); ht=np.asarray(hyper_type)
    ours=result['edge_index']; kinds=result['edge_type']
    if np.array_equal(he,ours) and np.array_equal(ht,kinds): return np.arange(len(ht))
    base=R+1
    def keys(e,t): return (((t.astype(np.int64)*base+e[0]+1)*base+e[1]+1)*base+e[2]+1)
    a=keys(ours,kinds); b=keys(he,ht)
    order=np.argsort(a); at=np.searchsorted(a[order],b)
    if len(a)!=len(b) or (at>=len(a)).any() or not np.array_equal(a[order[at]],b):
        raise ValueError('Official hyperedge support differs')
    return order[at]
