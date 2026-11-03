import gc,json,time,hashlib
import numpy as np
import torch
from torch.nn import functional as F
from motif import tasks
from common import ROOT,SOURCES,base_model,load_cached,dump,sha,log
from model import install
from cosine import attach
from evaluate import evaluate,metrics
def config(alpha=.2,freedom='gate_only',method='topology',semantic=False,shuffle=False):
    return dict(alpha=alpha,freedom=freedom,method=method,semantic=semantic,shuffle=shuffle)
def code_hashes():
    return {p.name:sha(p) for p in [ROOT/n for n in ['train.py','model.py','common.py','cosine.py','mixed.py','evaluate.py','weighted.py','incident_api.py','gated_model.py','protocol.json']]}
def cpu_state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def run(tag,cfg,seed):
    dest=ROOT/'results'/f'training_{tag}_s{seed}.json';hashes=code_hashes()
    if dest.exists():
        rr=json.loads(dest.read_text());assert rr['code_hashes']==hashes and rr['config']==cfg;return rr
    proto=json.loads((ROOT/'protocol.json').read_text());name=f'{tag}_s{seed}';m=install(base_model(seed).cuda(),cfg)
    gates=[p for n,p in m.named_parameters() if p.requires_grad and 'gate' in n];backbone=[p for n,p in m.named_parameters() if p.requires_grad and 'gate' not in n]
    groups=[]
    if gates:groups.append(dict(params=gates,lr=proto['gate_lr']))
    if backbone:groups.append(dict(params=backbone,lr=proto['relation_lr']))
    opt=torch.optim.AdamW(groups,weight_decay=proto['weight_decay']);curves=[];valid=[];best=-1.;best_step=0;start=1;data=truth=active=None;began=time.perf_counter()
    resume=ROOT/'checkpoints'/f'{name}_resume.pth'
    if resume.exists():
        old=torch.load(resume,map_location='cpu');assert old['code_hashes']==hashes and old['config']==cfg
        m.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);curves=old['curves'];valid=old['validation'];best=old['best'];best_step=old['best_step'];start=old['step']+1
    def validate(step):
        nonlocal data,truth,active,best,best_step
        data=truth=active=None;gc.collect();torch.cuda.empty_cache();rows=[]
        for source in SOURCES:
            d,t=load_cached(source,'validation')
            if cfg['semantic']:attach(d.relation_hypergraph,source,cfg['shuffle'])
            info,ranks,triples=evaluate(m,d,t,4);rr=ranks[0]
            np.savez_compressed(ROOT/'curves'/f'{name}_step{step}_{source}.npz',ranks=rr,triples=triples)
            rows.append(dict(source=source,**info,**metrics(rr)));del d,t;gc.collect();torch.cuda.empty_cache()
        score=float(np.mean([r['mrr'] for r in rows]));valid.append(dict(step=step,macro_mrr=score,sources=rows))
        if score>best:best=score;best_step=step
        payload=dict(model=cpu_state(m),config=cfg,seed=seed,step=step,code_hashes=hashes)
        torch.save(payload,ROOT/'checkpoints'/f'{name}_step{step}.pth')
        torch.save(dict(**payload,optimizer=opt.state_dict(),curves=curves,validation=valid,best=best,best_step=best_step),resume)
        dump(ROOT/'curves'/f'{name}_validation.json',valid);dump(ROOT/'curves'/f'{name}_loss.json',curves)
        log('source_validation',tag=name,step=step,macro_mrr=score,best_step=best_step)
    log('train_begin',tag=name,config=cfg,start=start)
    if not valid:validate(0)
    for step in range(start,501):
        block=(step-1)//25;source=SOURCES[block%3];fold=(block//3)%5
        if active!=(source,fold):
            data=truth=None;gc.collect();torch.cuda.empty_cache();data,truth=load_cached(source,f'fold{fold}');active=(source,fold)
            if cfg['semantic']:attach(data.relation_hypergraph,source,cfg['shuffle'])
        m.train();m.entity_model.eval();torch.manual_seed(seed+31*step);torch.cuda.manual_seed_all(seed+31*step)
        triples=torch.cat([data.target_edge_index,data.target_edge_type[None]]).T
        ix=np.random.default_rng(seed+step).integers(len(triples),size=proto['batch_size']);positive=triples[torch.from_numpy(ix).cuda()]
        batch=tasks.negative_sampling(truth,positive,proto['num_negative'],strict=True)
        opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();started=time.perf_counter();pred=m(data,batch);target=torch.zeros_like(pred);target[:,0]=1
        raw=F.binary_cross_entropy_with_logits(pred,target,reduction='none');ww=torch.ones_like(pred);ww[:,1:]=F.softmax(pred[:,1:].detach(),-1)
        loss=((raw*ww).sum(-1)/ww.sum(-1)).mean();assert torch.isfinite(loss);loss.backward()
        def norm(pp):return float(torch.sqrt(sum((p.grad.square().sum() for p in pp if p.grad is not None),torch.zeros((),device='cuda'))))
        gn,bn=norm(gates),norm(backbone);assert np.isfinite(gn+bn)
        torch.nn.utils.clip_grad_norm_(gates+backbone,proto['gradient_clip']);opt.step();torch.cuda.synchronize()
        row=dict(step=step,source=source,fold=fold,loss=float(loss),gate_gradient_norm=gn,backbone_gradient_norm=bn,seconds=time.perf_counter()-started,peak_allocated=torch.cuda.max_memory_allocated(),batch_sha256=hashlib.sha256(batch.cpu().numpy().tobytes()).hexdigest())
        curves.append(row)
        if step==1 or step%50==0:log('train_step',tag=name,**row);dump(ROOT/'results/training_status.json',dict(tag=name,**row))
        if step in proto['validation_steps']:validate(step)
    path=ROOT/'checkpoints'/f'{name}_step{best_step}.pth'
    result=dict(tag=name,family=tag,seed=seed,config=cfg,steps=500,best_step=best_step,best_source_validation_mrr=best,checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),validation=valid,seconds=time.perf_counter()-began,gate_parameters=sum(v.numel() for v in gates),semantic_parameters=sum(v.numel() for n,v in m.named_parameters() if 'cos_gate' in n),backbone_trainable_parameters=sum(v.numel() for v in backbone),code_hashes=hashes,data_stream_sha256=hashlib.sha256(''.join(r['batch_sha256'] for r in curves).encode()).hexdigest(),peak_allocated=max(r['peak_allocated'] for r in curves))
    dump(dest,result);log('train_complete',tag=name,best_step=best_step,best=best);del m,opt,data,truth;gc.collect();torch.cuda.empty_cache();return result
