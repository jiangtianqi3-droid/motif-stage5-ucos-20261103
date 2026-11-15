import gc
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import normalize

ENCODER=Path('/mnt/c/Users/10250/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a')

@torch.inference_mode()
def encode(texts,relation_texts,cache,log):
    path=cache/'conceptnet_text_emb.pt'
    import hashlib,json
    identity=hashlib.sha256(json.dumps(texts+relation_texts).encode()).hexdigest()
    if path.exists():
        value=torch.load(path,map_location='cpu',weights_only=False)
        assert value['identity']==identity
        return value['entities'],value['relations']
    from transformers import AutoModel,AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(ENCODER),local_files_only=True)
    model=AutoModel.from_pretrained(str(ENCODER),local_files_only=True).cuda().half().eval()
    for p in model.parameters():p.requires_grad_(False)
    alltexts=texts+relation_texts;output=torch.empty(len(alltexts),384)
    for at in range(0,len(alltexts),512):
        batch=tokenizer(alltexts[at:at+512],padding=True,truncation=True,max_length=128,return_tensors='pt').to('cuda')
        vectors=normalize(model(**batch).last_hidden_state[:,0].float(),dim=-1)
        assert torch.isfinite(vectors).all()
        output[at:at+len(vectors)]=vectors.cpu()
        if at%8192==0:log('text_encoding',done=at,total=len(alltexts))
    value=dict(entities=output[:len(texts)],relations=output[len(texts):],identity=identity,encoder=str(ENCODER),pooling='CLS',precision='FP16 model, FP32 normalization')
    torch.save(value,path)
    del model;gc.collect();torch.cuda.empty_cache()
    return value['entities'],value['relations']

class Semantic:
    @torch.inference_mode()
    def __init__(self,z,lex,relations,rho):
        self.lex=lex.cuda();self.rel=relations.cuda();self.n=z['n'];self.R=z['R'];self.rho=rho
        u,v=z['edge_index'];r=z['edge_type'];self.u,self.v,self.r=u,v,r
        self.degree=np.bincount(u,minlength=self.n)
        ut=torch.tensor(u,device='cuda');vt=torch.tensor(v,device='cuda');rt=torch.tensor(r,device='cuda')
        self.emp=torch.zeros(self.R,lex.shape[1],device='cuda');self.center=torch.zeros_like(self.emp)
        ctx=torch.zeros_like(self.lex)
        for at in range(0,len(u),4096):
            a,b,c=ut[at:at+4096],vt[at:at+4096],rt[at:at+4096]
            self.emp.index_add_(0,c,self.lex[b]-self.lex[a]);self.center.index_add_(0,c,self.lex[b]);ctx.index_add_(0,a,self.lex[b])
        # Per-node positive degree scale cancels under inner normalization.
        self.ctx=normalize(self.lex+.3*normalize(ctx,dim=1),dim=1)
        self.emp=normalize(self.emp,dim=1);self.center=normalize(self.center,dim=1)
        counts=np.bincount(r*self.n+v,minlength=self.R*self.n).reshape(self.R,self.n)
        bias=np.log1p(counts).astype(np.float32)
        bias=(bias-bias.mean(1,keepdims=True))/(bias.std(1,keepdims=True)+1e-6)
        self.bias=torch.tensor(np.clip(bias,-5,5),device='cuda')
        self.neighbors={}
        for a,b,c in zip(u,v,r):self.neighbors.setdefault((int(a),int(c)),[]).append(int(b))

    @torch.inference_mode()
    def scores(self,queries):
        h=torch.tensor(queries[:,0],device='cuda');q=torch.tensor(queries[:,1],device='cuda')
        query=normalize(self.ctx[h]+.5*self.rel[q]+.5*self.emp[q],dim=1)
        trans=query@self.ctx.T
        p=torch.zeros(len(h),self.lex.shape[1],device='cuda')
        for i,(a,b,_) in enumerate(queries):
            ids=self.neighbors.get((int(a),int(b)),[])
            if ids:p[i]=normalize(self.lex[ids].sum(0),dim=0)
            # Exact local prototype as specified; cross-relation information enters Reach.
        proto=p@self.lex.T+.5*(self.center[q]@self.lex.T)+.15*self.bias[q]
        return trans,proto
