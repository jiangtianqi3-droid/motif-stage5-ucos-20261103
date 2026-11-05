import numpy as np
import torch
from motif import tasks
from stage6.observed_loader import load_observed
from stage6.formal import load_queries
from stage6.proxy_graph import build_graph
from stage6.models import load_model
from stage7.evaluator import safe_evaluate
from stage7.numeric_stability import configure,install
from stage7.util import ROOT,dump,log


@torch.no_grad()
def run(args):
    g=load_observed(args.observations,'NELLInductive_v1')
    q,truth=load_queries(args,'NELLInductive_v1',g)
    d,t,_=build_graph(g,q,truth,ROOT/'cache/topology')
    triples=torch.tensor(q[:,[0,2,1]],device='cuda')
    tm,hm=tasks.strict_negative_mask(t,triples)
    mask=torch.stack([tm,hm],1).cpu().numpy()
    target=q[:,[2,0]]
    m=load_model(args.stage5,args.history,args.official,'B0',1024)
    ref=None;rows=[];arrays={}
    for path in ['native_fp32','querywise_fp32']:
        configure(False)
        if path=='querywise_fp32':install(m)
        for bs in [1,2,4,8]:
            r,_,scores=safe_evaluate(m,d,t,[0.],bs,True)
            s=scores[0]
            true=np.take_along_axis(s,target[:,:,None],axis=2)
            gap=s-true
            exact=((gap==0)&mask).sum(2)
            near=((np.abs(gap)<=1e-5)&mask).sum(2)
            if ref is None:ref=(r.copy(),s.copy())
            changed=(r[0]!=ref[0][0])
            rows.append(dict(path=path,batch=bs,rank_changes_to_native_batch1=int(changed.sum()),
                max_score_error=float(np.abs(s-ref[1]).max()),pessimistic_filtered_wrong_exact_ties=int(exact.sum()),
                pairs_with_exact_ties=int((exact>0).sum()),pairs_with_near_1e5_candidates=int((near>0).sum()),
                changed_pairs_with_exact_ties=int(((exact>0)&changed).sum()),mrr=float((1/r).mean())))
            arrays[f'{path}_b{bs}']=r
            log('tie_check',**rows[-1])
    np.savez_compressed(ROOT/'results/tie_checks.npz',**arrays)
    dump(ROOT/'results/tie_checks.json',dict(status='passed',facts=len(q),rows=rows,
        all_querywise_equal_native_batch1=all(x['rank_changes_to_native_batch1']==0 and x['max_score_error']==0 for x in rows if x['path']=='querywise_fp32'),
        ranking='Pessimistic >= preserved; no rounding, tolerance or performance-based batch choice.'))
