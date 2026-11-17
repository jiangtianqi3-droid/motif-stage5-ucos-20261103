"""Source-only pair probes with three explicitly distinct control distributions."""
from collections import defaultdict
from copy import copy
from statistics import mean
from time import perf_counter

import numpy as np

from p2.assets import verify
from p2.features import all_paths, profile, aggregate
from p1.assets import Assets, open_model, predict, gates, load, save_npz
from p1.paths import masked_graph, signature, counts_from_rows
from p1.scoring import raw_order
from src.io_utils import dump, read, write_csv, sha


def canonical_pair(h,t):
    return min(h,t),max(h,t)


def visibility(g, h):
    frontier={h}
    identities=set()
    for _ in range(3):
        step={int(i) for u in frontier for i in g.edges_at(u)}
        identities|=step
        frontier={int(g.edge_index[1,i]) for i in step}
    return identities,{int(g.edge_index[1,i]) for i in identities}-{h}


def masked_partition(asset,cfg):
    g=asset.graph
    with np.load(asset.bundle,allow_pickle=False) as z:
        validation={canonical_pair(int(h),int(t)) for h,r,t in z['queries']}
    examples=defaultdict(list)
    directions=[(int(h),int(r),int(t)) for h,r,t in zip(*[g.edge_index[0],g.edge_type,g.edge_index[1]])
                if r<g.R//2 and h!=t]
    for h,r,t in sorted(set(directions)):
        key=canonical_pair(h,t)
        if key not in validation:
            examples[key].append((h,r,t))
    fit,probe=cfg['source_fit_facts'],cfg['source_probe_facts']
    rng=np.random.default_rng(cfg['seed']+29)
    pairs=sorted(examples)
    picked=[pairs[z] for z in rng.choice(len(pairs),fit+probe,replace=False)]
    fit_pair=set(picked[:fit])
    probe_pair=set(picked[fit:])
    verify(not fit_pair & probe_pair and not set(picked)&validation,'source held-out canonical pairs overlap')
    chosen=[examples[p][int(rng.integers(len(examples[p])))] for p in picked]
    removed=[fact for p in picked for fact in examples[p]]
    masked=masked_graph(g,removed)
    remaining={canonical_pair(int(h),int(t)) for h,t in masked.edge_index.T}
    verify(not set(picked)&remaining,'masked pairs still observable')
    return masked,chosen,removed


def sample_controls(graph,head,rel,answer,known,raw,n,rng):
    ids,shell=visibility(graph,head)
    banned=set(known)|{head,answer}
    random=np.array(sorted(set(range(graph.n))-banned))
    c1=rng.choice(random,min(n,len(random)),replace=False).tolist()
    shell_pool=sorted(set(shell)-banned-set(c1))
    c2=[shell_pool[i] for i in rng.choice(len(shell_pool),min(n,len(shell_pool)),replace=False)]
    order=raw_order(raw)
    ranked={v:i+1 for i,v in enumerate(order)}
    hard=[v for v in order[:80] if v not in banned and v not in c1 and v not in c2]
    paths=all_paths(graph,head,{answer,*c1,*c2,*hard},ids)
    lengths=lambda v:min(map(len,paths.get(v,[])),default=4)
    def bucket(v):
        return int(np.log2(max(1,graph.out_degree[v])))
    pos_length=lengths(answer)
    def distance(v):
        return (abs(int(v in paths)-int(answer in paths)),abs(lengths(v)-pos_length),
                abs(bucket(v)-bucket(answer)),abs(min(80,ranked[v])-min(80,ranked[answer])),ranked[v],v)
    c3=sorted(hard,key=distance)[:n]
    return {'C1':c1,'C2':c2,'C3':c3},paths,ranked,lengths,bucket


def frozen_masked_scores(asset,masked,triples,folder):
    import torch
    from engine import state_hash
    from topology import extract
    adapted=copy(asset)
    adapted.graph=masked
    adapted.queries=np.asarray([[(h,r),(t,masked.inverse(r))] for h,r,t in triples],dtype=np.int64)
    layout=folder/'masked_signature'
    layout.mkdir(parents=True,exist_ok=True)
    sig=extract(masked.edge_index,masked.edge_type,masked.n,masked.R,cache=layout)
    adapted.arr=dict(n=masked.n,R=masked.R,edge_index=masked.edge_index,edge_type=masked.edge_type,
                     hyper_edge=sig['edge_index'],hyper_type=sig['edge_type'],raw=sig['raw'],z=sig['z'])
    ids=list(range(len(triples)))
    output=folder/'masked_rrf.npz'
    gate_dir=folder/'masked_gates'
    demand=sorted(set(map(int,adapted.queries[:,:,1].ravel())))
    if output.is_file():
        proof=read(folder/'masked_rrf_proof.json')
        verify(proof['sha256']==sha(output) and proof['graph']==masked.key and
               proof['checkpoint']==asset.meta['parameter_sha256_before'],'masked backbone cache mismatch')
        matrix=load(output)['rr']
        peak=proof['peak_allocated']
    else:
        model,data=open_model(adapted)
        gates(adapted,model,data,ids,gate_dir)
        matrix=predict(adapted,model,data,ids)
        verify(state_hash(model)==asset.meta['parameter_sha256_before'],'masked run changed B2')
        peak=torch.cuda.max_memory_allocated()
        save_npz(output,rr=matrix)
        dump(folder/'masked_rrf_proof.json',dict(sha256=sha(output),graph=masked.key,peak_allocated=peak,
                 checkpoint=asset.meta['parameter_sha256_before']))
        del model,data
        torch.cuda.empty_cache()
    verify(matrix.shape==(len(ids),2,masked.n),'masked candidate vocabulary')
    return matrix,{q:load(gate_dir/f'{q}.npz')['layers'].mean(0) for q in demand},peak


def source_path_validation(folder,cfg):
    dest=folder/'path_summary.json'
    if dest.is_file():
        return read(dest)
    started=perf_counter()
    asset=Assets('wn18rr_train',folder)
    masked, chosen,removed=masked_partition(asset,cfg)
    dump(folder/'source_mask_partition.json',dict(masked_key=masked.key,original_key=asset.graph.key,
         fit=chosen[:cfg['source_fit_facts']],probe=chosen[cfg['source_fit_facts']:],fully_removed=removed,
         probe_and_fit_pair_disjoint=True,validation_pair_disjoint=True))
    matrix,gate,peak=frozen_masked_scores(asset,masked,chosen,folder)
    known=defaultdict(set)
    for h,r,t in zip(asset.graph.edge_index[0],asset.graph.edge_type,asset.graph.edge_index[1]):
        known[int(h),int(r)].add(int(t))
    rng=np.random.default_rng(cfg['seed']+31)
    fit_rows,probes=[],[]
    for i,(h,r,t) in enumerate(chosen):
        for d,(u,q,v) in enumerate(((h,r,t),(t,masked.inverse(r),h))):
            c,paths,order,lengths,bucket=sample_controls(masked,u,q,v,known[u,q],matrix[i,d],
                                                         cfg['source_control_count'],rng)
            if i<cfg['source_fit_facts']:
                for x in [v]+[x for values in c.values() for x in values]:
                    p=min(paths.get(x,[]),key=lambda p:(len(p),p),default=None)
                    fit_rows.append(dict(has_path=bool(p),observed_positive=x==v,
                                         signature=signature(masked,q,p) if p else '',
                                         coarse=signature(masked,q,p,False) if p else ''))
            else:
                for cls,ids in c.items():
                    record=dict(fact_index=i,direction=d,head=u,relation=q,positive=v,kind=cls,
                                controls=ids,positive_rank=order[v],degree_bin=bucket(v),
                                positive_paths=paths.get(v,[]),control_paths=[paths.get(x,[]) for x in ids],
                                matches=[lengths(x)==lengths(v) and bucket(x)==bucket(v) for x in ids],
                                rank_distance=[abs(order[x]-order[v]) for x in ids])
                    probes.append(record)
    table=counts_from_rows(fit_rows)
    variants=('S0','S1-2','S1-4','S2-0.08','motif_only','length_degree','S0_motif')
    details=[]
    for r in probes:
        for ix,c in enumerate(r['controls']):
            ppos,pneg=r['positive_paths'],r['control_paths'][ix]
            prof=[profile(masked,r['relation'],p,gate[r['relation']],table) for p in (ppos,pneg)]
            def compute(p,feats,v):
                if not p:
                    return 0.
                if v=='length_degree':
                    return max(1/(len(path)*(1+np.log1p(max([int(masked.in_degree[masked.edge_index[1,i]])
                              for i in path[:-1]],default=0)))) for path in p)
                if v=='motif_only':
                    return max(m-.5 for s,m,path in feats)
                if v=='S0_motif':
                    return max(s+.20*(m-.5) for s,m,path in feats)
                return aggregate([s for s,_,_ in feats],v)
            refsign=0
            for version in variants:
                a,b=[compute(p,f,version) for p,f in zip((ppos,pneg),prof)]
                sign=int(np.sign(a-b)) if abs(a-b)>1e-12 else 0
                if version=='S0':
                    refsign=sign
                details.append(dict(fact_index=r['fact_index'],direction=r['direction'],control_class=r['kind'],
                     mode=version,positive_entity=r['positive'],control_entity=c,positive_path=bool(ppos),
                     control_path=bool(pneg),matched=r['matches'][ix],rank_distance=r['rank_distance'][ix],
                     positive_score=a,control_score=b,gap=a-b,pairwise=1 if sign>0 else .5 if sign==0 else 0,
                     same_order_as_s0=refsign==sign))
    grouped=defaultdict(list)
    for item in details:
        grouped[item['control_class'],item['mode']].append(item)
    summary=[]
    for (kind,variant),rows in grouped.items():
        both=[r for r in rows if r['positive_path'] and r['control_path']]
        summary.append(dict(control=kind,variant=variant,pairs=len(rows),
            pairwise=mean(r['pairwise'] for r in rows),both_path_pairwise=mean(r['pairwise'] for r in both) if both else None,
            gap=mean(r['gap'] for r in rows),positive_path_rate=mean(r['positive_path'] for r in rows),
            control_path_rate=mean(r['control_path'] for r in rows),matched_coverage=mean(r['matched'] for r in rows),
            ranking_consistency_s0=mean(r['same_order_as_s0'] for r in rows),
            positive_score_quantiles=np.quantile([r['positive_score'] for r in rows],[.1,.5,.9]).tolist(),
            control_score_quantiles=np.quantile([r['control_score'] for r in rows],[.1,.5,.9]).tolist()))
    write_csv(folder/'path_validation.csv',details)
    out=dict(table=table,summary=summary,fit_facts=cfg['source_fit_facts'],probe_facts=cfg['source_probe_facts'],
             removed_direct_facts=len(removed),peak_allocated=peak,seconds=perf_counter()-started)
    dump(dest,out)
    return out
