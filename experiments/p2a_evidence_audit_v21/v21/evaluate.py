"""Full-space frozen-forward sufficiency and canonical local matched deletion."""
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

import numpy as np

from v21.common import verify, identity, read, dump, sha
from v21.graph import matched_random, strata
from p1.assets import save_npz, Evaluator
from p1.scoring import raw_order
from src.io_utils import STAGE7


def score_summary(scores, target, excluded, offline, full=None):
    rank=offline.rank(scores,target,excluded)
    # Prediction consistency uses the full unfiltered entity space, independently of labels.
    prediction=np.lexsort((np.arange(len(scores)),-scores))[:10]
    return dict(rank=rank,rr=1/rank,hits1=rank<=1,hits10=rank<=10,top1=int(prediction[0]),
                top10=prediction.tolist(),top1_agreement=bool(prediction[0]==full['top1']) if full else True,
                top10_overlap=len(set(prediction)&set(full['top10']))/10 if full else 1.)


def baseline(backbone,view,selection,directory):
    path=directory/'full_scores.npz'
    proof=directory/'full_scores_proof.json'
    needed=sorted({i//view.asset.meta['batch'] for i,d in selection})
    static=backbone.static_fingerprint
    matrix={}
    if path.is_file():
        previous=read(proof)
        verify(previous['sha256']==sha(path) and previous['static']==static,'full score cache changed')
        with np.load(path,allow_pickle=False) as z:
            matrix={(int(i),int(d)):row for (i,d),row in zip(z['indices'],z['scores'])}
    missing=[(i,d) for i,d in selection if (i,d) not in matrix]
    if missing:
        requested=set(selection)
        for b in needed:
            first=b*view.asset.meta['batch']
            for d in (0,1):
                query=view.asset.queries[first:first+view.asset.meta['batch'],d].tolist()
                logits=backbone.score(query,backbone.full_ids)
                for offset,row in enumerate(logits):
                    if (first+offset,d) in requested:
                        matrix[first+offset,d]=row
        keys=sorted(matrix)
        save_npz(path,indices=np.array(keys),scores=np.stack([matrix[k] for k in keys]))
        dump(proof,dict(sha256=sha(path),static=static,historical_batch=view.asset.meta['batch']))
    offline=Evaluator(view.asset)
    result={}
    checks=[]
    for i,d in selection:
        h,r=map(int,view.asset.queries[i,d])
        old=view.raw[i][d]
        target,excluded,expected=offline.query(i,d,h,r,old)
        row=matrix[i,d]
        score=score_summary(row,target,excluded,offline)
        verify(score['rank']==expected,f'{view.key}:{i}:{d}: full graph historical filtered rank changed')
        order=np.sort(row)
        raw_rank=len(row)-np.searchsorted(order,row,side='left')
        reconstructed=np.float32(1.)/(60+raw_rank.astype(np.float32))
        differences=np.flatnonzero(reconstructed!=old)
        checks.append(dict(query_id=f'{view.key}:{i}:{d}',filtered_rank=score['rank'],
                           expected_filtered_rank=expected,proxy_differences=len(differences),
                           proxy_exact=not bool(len(differences))))
        if len(differences):
            dump(directory/'baseline_failure.json',dict(checks=checks,failing_query=f'{view.key}:{i}:{d}',
                   entities=differences.tolist(),new_raw_ranks=raw_rank[differences].tolist(),
                   old_proxy=old[differences].tolist(),new_proxy=reconstructed[differences].tolist()))
            raise ValueError(f'{view.key}:{i}:{d}: {len(differences)} full-entity proxies differ from archive')
        result[i,d]=score
    dump(directory/'baseline_verification.json',dict(queries=len(checks),all_exact=True,checks=checks,
         static=static,full_scores_sha256=sha(path)))
    return matrix,result,offline


def evaluate_queries(backbone,view,index,selection,selections,directory,cfg,on_progress):
    full_matrix,full_summary,offline=baseline(backbone,view,selection,directory)
    query_results=[]
    repeats=cfg['audit']['random_deletion_repeats']
    generated_cache={}
    counter=0
    calls=0
    started=perf_counter()
    cache_dir=directory/'query_checkpoints'
    for ordinal,(i,d) in enumerate(selection):
        query_started=perf_counter()
        query_id=f'{view.key}:{i}:{d}'
        job=selections[query_id]['job']
        local_ids=selections[query_id]['local_ids']
        local_facts=index.canonical(local_ids)
        partitions,freq,cuts=strata(index,local_facts,job['head'])
        full=full_summary[i,d]
        target,excluded,original=offline.query(i,d,job['head'],job['relation'],view.raw[i][d])
        input_record=dict(query_id=query_id,groups={k:job['arms'][k]['ids'] for k in cfg['retrieval_groups']},
                          local_ids=local_ids,full_static=backbone.static_fingerprint,
                          seed=cfg['experiment']['seed'],repeats=repeats)
        fingerprint=identity(input_record)
        dest=cache_dir/f'{i}-{d}.json'
        if dest.is_file():
            record=read(dest)
            verify(record['input_hash']==fingerprint,'checkpoint query input mismatch')
            query_results.append(record)
            on_progress(ordinal+1,len(selection))
            continue
        cache={}
        manifests=[]
        def compute(facts,mode):
            nonlocal calls
            # Exact canonical fact sets deduplicate identical randomized or retrieval tests.
            key=(mode,tuple(facts))
            if key not in cache:
                ids=index.closure(facts) if mode=='keep' else index.complement(facts)
                index.assert_paired(ids)
                result=backbone.score([[job['head'],job['relation']]],ids)[0]
                cache[key]=score_summary(result,target,excluded,offline,full)
                calls+=1
            return dict(cache[key])
        if ordinal<cfg['experiment']['smoke_queries']:
            score=backbone.score([[job['head'],job['relation']]],backbone.full_ids)[0]
            verify(np.array_equal(score,backbone.score([[job['head'],job['relation']]],backbone.full_ids)[0]),
                   'repeat entity forward not deterministic')
            # Original batch effects are explicitly checked rather than silently changing the reference.
            summary=score_summary(score,target,excluded,offline,full)
            verify(summary['rank']==full['rank'] and summary['top1']==full['top1'] and
                   summary['top10']==full['top10'],'single-query/full-historical-batch predictions changed')
            # IEEE float32 GEMM grouping may perturb logits, record its actual magnitude.
            singleton_error=float(np.max(np.abs(score-full_matrix[i,d])))
        else:
            singleton_error=None
        sufficient={}
        deletion={}
        random_results={}
        random_suff={}
        suff_ids={}
        group_facts={name:index.canonical(job['arms'][name]['ids']) for name in cfg['retrieval_groups']}
        for name,facts in dict(G_full=tuple(range(len(index.facts))),G_local=local_facts,**group_facts).items():
            score=full if name=='G_full' else compute(facts,'keep')
            score=dict(score,unique_facts=len(facts),directed_messages=len(index.closure(facts)),
                       delta_rr=score['rr']-full['rr'],loss_rr=full['rr']-score['rr'])
            sufficient[name]=score
            suff_ids[name]=list(facts)
        for name,facts in group_facts.items():
            delete=compute(facts,'remove')
            deletion[name]=dict(delete,unique_facts=len(facts),removed_messages=len(index.closure(facts)),
                                comp_rr=full['rr']-delete['rr'])
            random_results[name]=[]
            random_suff[name]=[]
            for rep in range(repeats):
                seed=int(identity([cfg['experiment']['seed'],query_id,name,rep])[:16],16)
                drawn,details=matched_random(facts,local_facts,partitions,seed)
                # The random control uses exactly the same draw for sufficiency and necessity.
                removed=compute(drawn,'remove')
                random_results[name].append(dict(removed,repeat=rep,comp_rr=full['rr']-removed['rr']))
                kept=compute(drawn,'keep')
                random_suff[name].append(dict(kept,repeat=rep,unique_facts=len(drawn),
                      directed_messages=len(index.closure(drawn)),loss_rr=full['rr']-kept['rr']))
                manifests.append(dict(query_id=query_id,group=name,repeat=rep,seed=seed,
                         unique_facts=len(drawn),message_records=len(index.closure(drawn)),
                         canonical_fact_ids=list(drawn),strata_details=details,
                         relaxations={tag:sum(x['relaxation']==tag for x in details)
                                      for tag in sorted({x['relaxation'] for x in details})},
                         local_only=True,overlap_selected=sum(x['overlap_selected'] for x in details)))
        record=dict(input_hash=fingerprint,query_id=query_id,fact_index=i,direction=d,full=full,
              sufficiency=sufficient,necessity=deletion,random_necessity=random_results,
              random_sufficiency=random_suff,random_details=manifests,unique_fact_sets=suff_ids,
              singleton_max_abs_error=singleton_error,full_dstar_buffer_hash=backbone.buffer_before,
              full_relation_sha=sha(backbone.directory/'full_relation_embeddings.npz'),
              query_wall_seconds=perf_counter()-query_started)
        # Checkpoint contains no model input labels; evaluation outputs retain only ranks and IDs.
        dump(dest,record)
        query_results.append(record)
        on_progress(ordinal+1,len(selection))
    return query_results,dict(wall_seconds=perf_counter()-started,new_forward_calls=calls,
                             completed_queries=len(query_results))
