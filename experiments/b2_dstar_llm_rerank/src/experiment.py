"""Frozen scoring, pure public-query inference, then one offline evaluation."""
import gc
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
from common_io import ROOT, S7, SEEDS, MODES, read, dump, sha, identity, save_array
from ranking import candidate_pool, decision_pool, writeback, grouped_validation
from evidence import EvidenceGraph
from prompting import Lexicon, TokenBudget, BudgetExceeded
from llm_client import Client

def lexicon(names, rels, graph):
    return Lexicon(names,rels,graph,[ROOT/'cache/lexical/FB15k-237.tsv',ROOT/'cache/lexical/WN18RR.tsv'])

def bind_resources():
    # Monitoring only: do not replace model operations or impose an obsolete 4GB cap.
    import evaluator, torch, resource
    def resources():
        return dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    evaluator.resources=resources

def score_graph(z, public_pairs, seed, directory, progress, verify=False):
    """Only (h,r) pairs enter. Regroup original 16-fact forward/inverse batches."""
    import torch
    from backbone import Backbone
    from engine import state_hash
    bind_resources()
    directory.mkdir(parents=True,exist_ok=True)
    completed=directory/'complete.json'
    if completed.exists():
        info=read(completed)
        assert info['public_query_hash']==identity(public_pairs.tolist())
        assert all(sha(directory/k)==v for k,v in info['shards'].items())
        return info
    # Use the original 16-fact scoring path for all graphs. If memory is
    # insufficient, fail explicitly; never silently reduce batch size.
    model=Backbone(z,'B2',seed,calibrated=True,adaptive=False)
    before=state_hash(model.model)
    assert all(not p.requires_grad for p in model.model.parameters())
    refs={}; align=[]
    total=len(public_pairs)//2
    bar=tqdm(range(0,total,16),desc=f'B2+D* seed={seed}',unit='batch',mininterval=1)
    try:
        for start in bar:
            pairs=public_pairs[2*start:2*(start+16)].reshape(-1,2,2)
            facts=np.c_[pairs[:,0,0],pairs[:,0,1],pairs[:,1,0]]
            assert np.array_equal(pairs[:,1,1],pairs[:,0,1]+z['R']//2)
            if z['n']>30000:
                # Backbone.scores has a newer micro=2 branch. Replay the
                # historical operator at 16 explicitly, without changing it.
                from motif import tasks
                with torch.inference_mode():
                    triples=torch.as_tensor(facts[:,[0,2,1]],device='cuda')
                    s=[model.model(model.data,b).cpu().numpy() for b in tasks.all_negative(model.data,triples)]
                    scores=np.stack(s,axis=1).reshape(-1,z['n'])
            else: scores=model.scores(facts)
            if verify and start==0:
                # Independent historical all_negative batch construction.
                from motif import tasks
                with torch.inference_mode():
                    triples=torch.as_tensor(facts[:,[0,2,1]],device='cuda')
                    expected=np.stack([model.model(model.data,b).cpu().numpy() for b in tasks.all_negative(model.data,triples)],axis=1).reshape(-1,z['n'])
                assert np.array_equal(scores,expected),'Stage-7 raw score bit mismatch'
                align.append(dict(batch=start,raw_score_bitwise_equal=True,compared='original Stage-7 all_negative operator replay'))
            path=directory/(f'{start:07d}.npy')
            if path.exists(): assert np.array_equal(np.load(path),scores),'Resume changed raw scores'
            else: save_array(path,scores)
            refs[path.name]=sha(path)
            progress('scoring',done=min(start+16,total)*2,total=total*2,seed=seed)
        assert before==state_hash(model.model)
    finally: model.close()
    info=dict(family='B2+D*',seed=seed,parameter_hash=before,parameter_hash_unchanged=True,
        public_query_hash=identity(public_pairs.tolist()),shards=refs,raw_alignment=align,
        checkpoint_sha256=read(ROOT/'freeze_manifest.json')['checkpoints'][str(seed)]['sha256'])
    dump(completed,info)
    return info

def score_rows(directory):
    for path in sorted(directory.glob('*.npy')):
        data=np.load(path,mmap_mode='r')
        start=int(path.stem)*2
        for j, row in enumerate(data): yield start+j,row

def prepare_query(graph, lex, h, r, score, config):
    top=candidate_pool(score,graph.tails.get((h,r),set()),config['top_k'])
    evidence=graph.retrieve(h,r,top,community=False)
    pool=decision_pool(top,evidence)
    # Every ablation gets identical candidates, selected before mode-specific evidence.
    return top,pool,evidence

def public_inference(graph, lex, key, split, seed, pairs, score_dir, directory, config, progress, limit=None):
    directory.mkdir(parents=True,exist_ok=True)
    budget=TokenBudget(config['prompt_budget']); client=Client()
    manifest=read(ROOT/'freeze_manifest.json')
    dependency=identity(dict(config=config,observation=hashlib.sha256(graph.edge.tobytes()+graph.rel.tobytes()).hexdigest(),
        lex=lex.audit(),seed=seed,model=client.model,code={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').rglob('*.py')}))
    # Scan score stream in original batch order; public duplicates share a record.
    unique={}
    for i,(h,r) in enumerate(pairs): unique.setdefault((int(h),int(r)),i)
    selected=set(list(unique)[:limit]) if limit else set(unique)
    total=len(selected)*len(MODES); done=0; began=time.monotonic()
    counts=dict(api_calls=0,prompt_tokens=0,completion_tokens=0,cache_hits=0)
    bar=tqdm(total=total,desc=f'{key} / {seed} / {split}',unit='query-mode',mininterval=1)
    def execute(job):
        qid,messages,pool,count=job
        return client.call(qid,messages,pool,count,config['prompt_budget'])
    with ThreadPoolExecutor(max_workers=config['concurrency']) as executor:
        for index,score in score_rows(score_dir):
            h,r=map(int,pairs[index]); public=(h,r)
            if public not in selected or unique[public]!=index: continue
            token=f'{h}_{r}'; record_path=directory/(token+'.json')
            if record_path.exists():
                record=read(record_path)
                assert record['dependency']==dependency,'Stale inference cache; use a new run identity'
                for item in record['modes'].values():
                    counts['cache_hits']+=int(item.get('attempted',0)>0)
                    counts['prompt_tokens']+=item.get('usage',{}).get('prompt_tokens',0)
                    counts['completion_tokens']+=item.get('usage',{}).get('completion_tokens',0)
                done+=len(MODES);bar.update(len(MODES));continue
            top,pool,base=prepare_query(graph,lex,h,r,score,config)
            record=dict(dependency=dependency,query=[h,r],top30=top,pool=pool,modes={})
            gate=not any(base[t]['paths'] for t in pool)
            jobs=[]
            full=None
            for mode in MODES:
                if gate:
                    record['modes'][mode]=dict(order=top,status='no_evidence',attempted=0);continue
                ev=base
                if mode.startswith('Community'):
                    full=graph.retrieve(h,r,pool,community=True);ev=full
                obj=lex.build(h,r,top,pool,ev,mode)
                try: messages,ntokens,changes=budget.fit(obj)
                except BudgetExceeded:
                    record['modes'][mode]=dict(order=top,status='token_budget_fallback',attempted=0);continue
                qid=dict(dataset=key,split=split,h=h,r=r%graph.half,direction='inverse' if r>=graph.half else 'forward',
                    ablation_mode=mode,seed=seed,dependency=dependency)
                # Retain exact prompt privately for zero-leakage audits.
                prompt_path=directory/'prompts'/(token+'_'+str(MODES.index(mode))+'.json')
                dump(prompt_path,dict(query=qid,messages=messages,local_prompt_tokens=ntokens,cuts=changes))
                jobs.append((mode,executor.submit(execute,(qid,messages,pool,ntokens))))
            for mode,future in jobs:
                result=future.result()
                counts['cache_hits']+=int(result['cache_hit'])
                counts['api_calls']+=0 if result['cache_hit'] else result['attempted']
                usage=result['usage'];counts['prompt_tokens']+=usage.get('prompt_tokens',0);counts['completion_tokens']+=usage.get('completion_tokens',0)
                order=writeback(top,pool,result['decision']['ranked_candidate_ids']) if result['status']=='reranked' else top
                record['modes'][mode]=dict(result,order=order)
            dump(record_path,record)
            done+=len(MODES);bar.update(len(MODES));bar.set_postfix(calls=counts['api_calls'])
            elapsed=time.monotonic()-began
            progress('reranking',graph=key,split=split,seed=seed,done=done,total=total,
                queries_per_second=done/max(elapsed,.001),eta_seconds=(total-done)*elapsed/max(done,1),**counts)
            if done%64==0:gc.collect()
    bar.close()
    dump(directory/'summary.json',dict(unique_queries=len(selected),query_modes=total,seconds=time.monotonic()-began,**counts))
    return read(directory/'summary.json')

def evaluate_graph(key, facts, truth_facts, z, ranks_dir, query_dirs, validation_heads):
    from offline_eval import truth_index, baseline_rank, changed_rank, historical, metrics, paired_bootstrap
    from backbone import bidirectional
    q=bidirectional(facts,z['R']//2); truth=truth_index(truth_facts,z['R']//2)
    base,m5=historical(key,q); outputs={'B2+D*':base,'B2+D*+M5':m5}
    for mode in MODES: outputs['B2+D*+'+mode]=np.zeros_like(base)
    diagnostics={mode:dict(total=0,recall30=0,recall_pool=0,triggered=0,abstain=0,parse_fallback=0,positive=0,negative=0,promoted_11_30_to10=0,budget_fallback=0) for mode in MODES}
    for j,seed in enumerate(SEEDS):
        records={}
        for i,score in score_rows(ranks_dir/str(seed)):
            h,r,t=map(int,q[i]); public=(h,r)
            b=baseline_rank(score,h,r,t,truth)
            assert b==base[j,i],f'Historical rank mismatch {key}/{seed}/{i}'
            if public not in records: records[public]=read(query_dirs/str(seed)/(f'{h}_{r}.json'))
            rec=records[public]
            for mode in MODES:
                item=rec['modes'][mode]; rank=changed_rank(score,h,r,t,truth,rec['top30'],item['order'])
                outputs['B2+D*+'+mode][j,i]=rank
                d=diagnostics[mode];d['total']+=1;d['recall30']+=int(t in rec['top30']);d['recall_pool']+=int(t in rec['pool'])
                d['triggered']+=int(item.get('attempted',0)>0);d['abstain']+=int(item['status']=='abstain')
                d['parse_fallback']+=int(item['status']=='parse_or_validation_fallback');d['budget_fallback']+=int(item['status']=='token_budget_fallback')
                d['positive']+=int(rank<b);d['negative']+=int(rank>b);d['promoted_11_30_to10']+=int(11<=b<=30 and rank<=10)
            if len(records)>256: records.clear()
    seen=np.array([int(h) in validation_heads for h in q[:,0]])
    summary={name:dict(metrics=metrics(a),vs_baseline=paired_bootstrap(a,base),vs_M5=paired_bootstrap(a,m5),
        seen_query_heads=metrics(a[:,seen]),unseen_query_heads=metrics(a[:,~seen])) for name,a in outputs.items()}
    for d in diagnostics.values():
        d.update(Recall30=d['recall30']/d['total'],RecallDecisionPool=d['recall_pool']/d['total'],
            trigger_rate=d['triggered']/d['total'],abstain_rate=d['abstain']/max(d['triggered'],1),
            parse_or_validation_fallback_rate=d['parse_fallback']/max(d['triggered'],1))
    return outputs,dict(methods=summary,diagnostics=diagnostics,seen_queries=int(seen.sum()),unseen_queries=int((~seen).sum()))
