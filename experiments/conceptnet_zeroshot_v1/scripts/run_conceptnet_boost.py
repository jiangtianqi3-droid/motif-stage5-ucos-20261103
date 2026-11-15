#!/usr/bin/env python
"""Frozen-weight, observation-only experiment; run using scripts/run.sh."""
import os,sys,json,time,gc,argparse,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from dataset_inspector import *
from evaluator import *
from structural_calibrator import *
from semantic_calibrator import encode,Semantic
from fusion_prober import PRIOR,calibrate

STATE={};START=time.time()
def log(stage,**kwargs):
    global STATE
    STATE.update(stage=stage,pid=os.getpid(),updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),elapsed_seconds=time.time()-START,**kwargs)
    atomic_json(ROOT/'results/status.json',STATE)
    print(json.dumps(dict(stage=stage,**kwargs),ensure_ascii=False),flush=True)

class GNN:
    def __init__(self,z,kind):
        from engine import model,RelationCache,state_hash
        from common import make_data
        self.kind=kind;self.z=sparsify(z) if kind=='M2' else z
        data=dict(self.z,queries=np.empty((0,3),np.int64),alltrue=np.empty((0,3),np.int64))
        self.data,_=make_data(data)
        self.model=model('B0' if kind=='M0' else 'B2',1024)
        if kind=='M1':
            from calibration import install
            install(self.model,self.data,dict(kind='D-Sender',gamma=.75,alpha=.2,anchor=True))
        elif kind=='M2':install_sparse(self.model);install_plus(self.model,self.data)
        self.relation_cache=RelationCache(self.model)
        self.before=state_hash(self.model)

    @torch.inference_mode()
    def scores(self,queries):
        # GNN batches are deliberately smaller than cheap semantic batches.
        scores=[];n=self.z['n']
        for at in range(0,len(queries),2):
            q=torch.tensor(queries[at:at+2],device='cuda')
            b=torch.empty(len(q),n,3,device='cuda',dtype=torch.long)
            b[:,:,0]=q[:,0,None];b[:,:,1]=torch.arange(n,device='cuda');b[:,:,2]=q[:,1,None]
            s=self.model(self.data,b)
            assert torch.isfinite(s).all()
            scores.append(s.cpu())
            resources()
        return torch.cat(scores).cuda()

    def close(self):
        from engine import state_hash
        assert self.before==state_hash(self.model),'Frozen weights changed'
        # Break the bound-method cache cycle before releasing GPU allocations.
        self.model.relation_model.forward=self.relation_cache.original
        del self.model,self.data,self.relation_cache
        gc.collect();torch.cuda.empty_cache()

def build_features(z,queries,name,cache,lex,rel):
    folder=cache/name;folder.mkdir(parents=True,exist_ok=True)
    paths=[folder/f'{at:06d}.npz' for at in range(0,len(queries),16)]
    if all(p.exists() for p in paths):return paths
    reach=Reachability(z);semantic=Semantic(z,lex,rel,reach.rho);gnn=GNN(z,'M2')
    for at,path in zip(range(0,len(queries),16),paths):
        if path.exists():continue
        q=queries[at:at+16]
        with torch.inference_mode():
            gnn_score=gnn.scores(q);rs=torch.tensor(reach.score(q),device='cuda')
            trans,proto=semantic.scores(q)
            zr,zt=safe_z(rs),safe_z(trans)
            consensus=(rs>0)*zr*zt.clamp_min(0)
            fs=torch.stack([safe_z(gnn_score),zr,zt,safe_z(proto),safe_z(consensus)],dim=1)
            assert torch.isfinite(fs).all()
            target_reachable=reach.reachable_targets(q)
            temp=path.with_suffix('.tmp.npz')
            np.savez(temp,features=fs.cpu().numpy(),raw_gnn=gnn_score.cpu().numpy(),queries=q,reachable=target_reachable);temp.replace(path)
            del fs,gnn_score,rs,trans,proto,zr,zt,consensus
        log('feature_stream',split=name,done=min(at+16,len(queries)),total=len(queries),**resources())
    gnn.close();del semantic,reach;gc.collect();torch.cuda.empty_cache()
    return paths

def baseline(z,queries,kind,truth,cache):
    path=cache/(kind+'_ranks.npz')
    if path.exists():return np.load(path)['ranks']
    gnn=GNN(z,kind);rows=[];topids=[];topvals=[]
    checkpoint=cache/(kind+'_partial.npz')
    if checkpoint.exists():
        with np.load(checkpoint) as f:rows=list(f['ranks']);topids=list(f['top_ids']);topvals=list(f['top_scores'])
    for at in range(len(rows),len(queries),16):
        q=queries[at:at+16];s=gnn.scores(q)
        rows.extend(ranks(s,q,truth));v,i=s.topk(50,dim=1)
        topids.extend(i.cpu().numpy());topvals.extend(v.cpu().numpy())
        tmp=checkpoint.with_suffix('.tmp.npz');np.savez(tmp,ranks=rows,top_ids=topids,top_scores=topvals);tmp.replace(checkpoint)
        log('baseline',model=kind,done=len(rows),total=len(queries),**resources())
    gnn.close();np.savez(path,ranks=rows,top_ids=topids,top_scores=topvals)
    return np.array(rows)

def evaluate_shards(paths,weights,truth,degree):
    collected={k:[] for k in weights};tops={k:[] for k in weights};values={k:[] for k in weights};reach=[];qs=[]
    for p in paths:
        with np.load(p) as f:
            features=torch.tensor(f['features'],device='cuda');raw_gnn=torch.tensor(f['raw_gnn'],device='cuda');q=f['queries'];reach.extend(f['reachable']);qs.extend(q)
        for k,w in weights.items():
            s=raw_gnn if k=='M2' else fuse(features,degree[q[:,0]],w)
            collected[k].extend(ranks(s,q,truth));v,i=s.topk(50,dim=1)
            tops[k].extend(i.cpu().numpy());values[k].extend(v.cpu().numpy())
        del features
    return {k:np.array(v) for k,v in collected.items()},tops,values,np.array(qs),np.array(reach)

def main():
    import fcntl
    lock=(ROOT/'results/run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(1024);np.random.seed(1024)
    STATE.update(mode='smoke' if args.smoke else 'full',completed_models=[])
    log('starting')
    # Source code, metadata, observations and weights all participate in run identity.
    historical=[DATA/'vocab.json',DATA/'train.txt',DATA/'valid.txt',DATA/'test.txt',DATA/'observed.npz',Path('/mnt/d/motif-phase3-work/MOTIF/ckpts/motif_3g.pth')]
    from s7util import HISTORY,ROOT as S7
    manifest=json.loads((S7/'model_manifest.json').read_text())
    historical += [HISTORY/manifest[k]['checkpoint'] for k in ['B1_s1024','B2_s1024']]
    historical += list(S7.glob('*.py'))+list((S7/'runtime').rglob('*.py'))
    hashes={str(p):digest(p) for p in historical}
    own={str(p.relative_to(ROOT)):digest(p) for folder in ['src','scripts'] for p in (ROOT/folder).glob('*') if p.is_file()}
    fingerprint=hashlib.sha256(json.dumps([hashes,own,args.smoke],sort_keys=True).encode()).hexdigest()[:16]
    run=ROOT/'results'/('smoke' if args.smoke else 'full')/fingerprint;run.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'cache'/fingerprint;cache.mkdir(parents=True,exist_ok=True)
    log('inspecting',run=str(run),fingerprint=fingerprint)
    atomic_json(run/'historical_hashes.json',hashes)
    vocab,texts,names=load_vocab();train=read_split('train',vocab)
    log('vocabulary_verified',entities=len(texts),relations=len(names),entity_examples=texts[:5],relation_examples=names[:5],numeric_concepts=sum(t.isdecimal() for t in texts))
    z=observed(train,len(texts),len(names));degree=np.bincount(z['edge_index'][0],minlength=len(texts))
    lex,rel=encode(texts,relation_texts(names),ROOT/'cache',log)
    protocol=dict(structure=dict(p_tri=.30,beta_tri=.15,alpha=.90,gamma_out=.75,gamma_rel=.50,lambda_hops=[1,.5,.25]),prior=PRIOR,structural_selection='fixed a priori; no target label selection',structural_grid_not_run=True,query_chunk=16,gnn_chunk=2,encoder='BAAI/bge-small-en-v1.5',source_weight_training_caveat='Inherited B2 source-training protocol is unchanged; this run makes no backbone updates.',test_exposure='Previously evaluated ConceptNet100k; not a first blind test',numeric_concepts='Original numeric concepts retained; never synthesize text from entity IDs')
    atomic_json(run/'protocol.json',protocol)
    # P0 self-supervision is completed before loading valid/test triples.
    rng=np.random.default_rng(20261113)
    eligible=np.flatnonzero((degree[train[:,0]]>1)&(degree[train[:,2]]>1))
    ids=rng.choice(eligible,size=16 if args.smoke else 500,replace=False)
    keep=np.ones(len(train),bool);keep[ids]=False
    probe=train[ids];masked=train[keep]
    assert not set(map(tuple,probe)) & set(map(tuple,masked))
    selfz=observed(masked,len(texts),len(names),cache=cache/'masked_topology',historical=False)
    selfq=bidirectional(probe)
    atomic_json(run/'selfsup_mask.json',dict(indices=ids.tolist(),removed_forward=len(ids),removed_inverse=len(ids),recomputed=['hypergraph','PMI','degree','fanout','relation translation','context','prototype','reachability']))
    paths=build_features(selfz,selfq,'selfsup',cache,lex,rel)
    self_degree=np.bincount(selfz['edge_index'][0],minlength=len(texts))
    selfweights=calibrate(paths,truth_index(train),self_degree,run/'P0-SelfSup.json',log,args.smoke)
    del selfz;gc.collect()
    valid=read_split('valid',vocab)
    if args.smoke:valid=valid[rng.choice(len(valid),8,replace=False)]
    valpaths=build_features(z,bidirectional(valid),'valid',cache,lex,rel)
    valweights=calibrate(valpaths,truth_index(np.concatenate([train,valid])),degree,run/'P1-ValCalib.json',log,args.smoke)
    atomic_json(run/'locked_weights.json',dict(P0_Prior=PRIOR,P0_SelfSup=selfweights,P1_ValCalib=valweights,locked_before_test_scoring=True))
    test=read_split('test',vocab)
    assert not set(map(tuple,test))&set(map(tuple,train))
    if args.smoke:test=test[rng.choice(len(test),8,replace=False)]
    q=bidirectional(test)
    # Full valid/test truths are used only in the final filtered evaluator.
    truth=truth_index(np.concatenate([train,read_split('valid',vocab),read_split('test',vocab)]))
    results={}
    for k in ['M0','M1']:
        results[k]=baseline(z,q,k,truth,cache)
        log('model_complete',model=k,latest_MRR=metrics(results[k])['MRR'],completed_models=list(results))
    paths=build_features(z,q,'test',cache,lex,rel)
    weights={'M2':[1,0,0,0,0,0],'M3':[1,.8,0,0,0,0],'M4':[0,0,1,1,0,0],'M5':PRIOR,'M6':selfweights,'M7':valweights}
    extra,tops,values,queries,reachable=evaluate_shards(paths,weights,truth,degree);results.update(extra)
    assert np.array_equal(q,queries)
    summary={};diagnostics={}
    for k,r in results.items():
        summary[k]=dict(**metrics(r),**bootstrap(r,results['M1']))
        np.savez(run/(k+'_ranks.npz'),ranks=r,queries=q)
        if k in tops:np.savez(run/(k+'_top50.npz'),ids=tops[k],scores=values[k])
        groups={'degree_le2':degree[q[:,0]]<=2,'degree_3_15':(degree[q[:,0]]>=3)&(degree[q[:,0]]<=15),'degree_gt15':degree[q[:,0]]>15,'reachable':reachable,'unreachable':~reachable}
        diagnostics[k]={g:dict(count=int(mask.sum()),**metrics(r[mask])) if mask.any() else dict(count=0) for g,mask in groups.items()}
    assert all(digest(p)==h for p,h in hashes.items()),'Historical asset changed'
    atomic_json(run/'summary.json',summary);atomic_json(run/'diagnostics.json',diagnostics)
    verification=dict(historical_hashes_unchanged=True,no_backbone_updates=True,finite_all_branches=True,**resources(),seconds=time.time()-START)
    atomic_json(run/'verification.json',verification)
    lines=['# ConceptNet100k 结构与语义协同实验','',('本文件为冒烟报告，不是全量效果结论。' if args.smoke else '全量评估已完成；负结果和全部方法均保留。'),'','MRR 0.18–0.22 为研究目标，不是保证。该数据已有历史测试暴露；P1 使用目标验证标签，不属于严格 P0。','', '|模型|MRR|Hits@1|Hits@3|Hits@10|','|---|---:|---:|---:|---:|']
    for k,v in summary.items():lines.append('|'+k+'|'+'|'.join(f'{v[a]:.6f}' for a in ['MRR','Hits1','Hits3','Hits10'])+'|')
    lines+=['','M0–M7 定义遵循任务；M3 固定 GNN:reach=1:0.8，M4 为两个语义分支等权。P0-Prior 固定先验，P0-SelfSup 在掩蔽500条观察事实后重算全部图特征，P1 在验证集坐标寻优。','', '## 实现与解释边界','结构参数预固定 p_tri=.30、beta_tri=.15、alpha=.90；未进行结构参数全网格扫描。','关系兼容度取每对关系共现超边的最大 s_pmi 并按行归一化；局部原型采用精确同关系邻居，跨关系邻居通过可达先验参与。','M0、M1、M2 使用原始GNN分数做全候选悲观并列排名。融合分支按查询安全截尾Z-score，可能产生新的并列。','文本语义模型可能包含预训练知识；无本轮测试事实输入不等于预训练语料无重合。','所有新文件仅存本实验目录，历史依赖只读。旧 B2 的源训练协议沿用，不能声称已修复历史训练负采样问题。','', '## 基线复现与目标达成']
    for k,expected in [('M0',.0189),('M1',.0354)]:lines.append(f'- {k}：MRR={summary[k]["MRR"]:.6f}，历史近似值={expected:.4f}，差值={summary[k]["MRR"]-expected:+.6f}。'+('冒烟抽样不可用于判断复现。' if args.smoke else '差异如较大，应结合原始逐事实排名和批次核查，不改参数追逐测试基线。'))
    lines.append('M5/M6/M7 是否进入目标区间：'+str({k:.18<=summary[k]['MRR']<=.22 for k in ['M5','M6','M7']}))
    lines+=['','## 置信区间','2000次按事实绑定双向的配对 bootstrap；主比较参考 M1，不涵盖调参和训练种子不确定性。','|模型|指标|95%区间|相对M1差值95%区间|','|---|---|---|---|']
    for k,v in summary.items():
        for i,metric in enumerate(v['order']):
            lo,hi=v['metric_ci95'][i];dl,dh=v['paired_delta_ci95'][i]
            lines.append(f'|{k}|{metric}|[{lo:.6f}, {hi:.6f}]|[{dl:+.6f}, {dh:+.6f}]|')
    lines+=['','## 度数与可达性诊断','可达组依据未加权观察图的1–3跳路径，与用于融合的关系加权可达分区分。','|模型|分组|查询数|MRR|Hits@1|Hits@3|Hits@10|','|---|---|---:|---:|---:|---:|---:|']
    for k,groups in diagnostics.items():
        for g,v in groups.items():lines.append(f'|{k}|{g}|{v["count"]}|'+('|'.join(f'{v[a]:.6f}' for a in ['MRR','Hits1','Hits3','Hits10']) if v['count'] else '—|—|—|—')+'|')
    lines+=['','## 锁定权重',f'P0-Prior: {PRIOR}',f'P0-SelfSup: {selfweights}',f'P1-ValCalib: {valweights}','',f'完整指标与四项区间：{run}/summary.json',f'度数及三跳可达分组：{run}/diagnostics.json',f'协议与权重：{run}/protocol.json、locked_weights.json',f'内存及完整性：{verification}']
    report=run/'report_conceptnet_boost.md';report.write_text('\n'.join(lines),encoding='utf-8')
    if not args.smoke:(ROOT/'report_conceptnet_boost.md').write_text(report.read_text(),encoding='utf-8')
    log('smoke_passed' if args.smoke else 'completed',completed_models=list(results),metrics={k:metrics(v) for k,v in results.items()},report=str(report),**verification)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        log('failed',error=repr(exc));traceback.print_exc();sys.exit(1)
