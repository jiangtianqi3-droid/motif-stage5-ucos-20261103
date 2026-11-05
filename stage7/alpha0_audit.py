"""Post-freeze numerical diagnosis; never changes any selected/formal result."""
import types
import gc
import sys
import numpy as np
import torch
from stage7.run import parser
from stage7.util import ROOT,read,dump,sha,log
from stage7.experiment import graph_for
from stage7.numeric_stability import install
from stage7.evaluator import safe_evaluate
from stage6.models import load_model,set_alpha,state_hash
from stage6.proxy_graph import build_graph
from stage6.formal import load_queries


@torch.no_grad()
def main():
    args=parser().parse_args(['formal']+sys.argv[1:])
    frozen=read(ROOT/'frozen.json')
    key='NELLInductive_v1'
    # Full formal run must already exist. No proxy/selection entry is called.
    paths={f:ROOT/'evaluation'/f'{key}__{f}_s1024.npz' for f in ['B0','T-4D']}
    assert all(p.exists() for p in paths.values())
    graph=graph_for(args,key);q,truth=load_queries(args,key,graph)
    d,t,_=build_graph(graph,q,truth,ROOT/'cache/topology')
    b0=install(load_model(args.stage5,args.history,args.official,'B0',1024))
    t4=install(load_model(args.stage5,args.history,args.official,'T-4D',1024))
    before=state_hash(t4)
    a,b=b0.state_dict(),t4.state_dict()
    different=[n for n in a if not torch.equal(a[n],b[n])]
    extra=[n for n in b if n not in a]
    weights=dict(common_tensors=len(a),different_common_tensors=different,extra_t4_tensors=extra,
                 non_gate_weights_identical=not different)
    set_alpha(t4,0.)
    representation_error=0.
    layer_errors=[0.]*len(t4.relation_model.layers)
    # Same canonical relation batches as the frozen evaluator.
    for at in range(0,graph.R,2):
        relq=torch.tensor([at,at+1],device='cuda')
        traces={}
        def capture(label):
            def hook(module,inputs,output):traces[label]=output.detach().clone()
            return hook
        handles=[]
        for label,model in [('B0',b0),('T4',t4)]:
            for i,layer in enumerate(model.relation_model.layers):
                handles.append(layer.linear.register_forward_hook(capture((label,i))))
        r0=b0.relation_model(d.relation_hypergraph,relq)
        r4=t4.relation_model(d.relation_hypergraph,relq)
        for handle in handles:handle.remove()
        representation_error=max(representation_error,float((r0-r4).abs().max()))
        for i in range(len(layer_errors)):
            layer_errors[i]=max(layer_errors[i],float((traces['B0',i]-traces['T4',i]).abs().max()))
    outputs={}
    for label,model in [('B0',b0),('T4_CSR_alpha0',t4)]:
        r,info,s=safe_evaluate(model,d,t,[0.],8,collect_scores=True)
        outputs[label]=(r,s[0])
    original=t4.relation_model.forward
    t4.relation_model.forward=types.MethodType(b0.relation_model.forward.__func__,t4.relation_model)
    try:
        r,info,s=safe_evaluate(t4,d,t,[0.],8,collect_scores=True)
        outputs['T4_official_relation_forward']=(r,s[0])
    finally:t4.relation_model.forward=original
    reference_r,reference_s=outputs['B0']
    rows=[]
    for label,(r,s) in outputs.items():
        family='B0' if label=='B0' else 'T-4D'
        with np.load(paths[family]) as z:archived=z['ranks'][0]
        rows.append(dict(path=label,mrr=float((1/r).mean()),
            rank_changes_to_B0=int((r!=reference_r).sum()),
            score_max_error_to_B0=float(np.abs(s-reference_s).max()),
            rank_changes_to_frozen_family_alpha0=int((r[0]!=archived).sum())))
    assert before==state_hash(t4)
    dump(ROOT/'results/alpha0_relation_kernel_audit.json',dict(status='complete',key=key,facts=len(q),
        frozen_sha256=sha(ROOT/'frozen.json'),weights=weights,rows=rows,
        relation_representation_max_error=representation_error,first_relation_layer_linear_max_error=layer_errors[0],
        relation_linear_max_errors_by_layer=layer_errors,
        no_weight_update=True,no_frozen_outputs_changed=True,
        scope='Post-freeze diagnosis only. CSR alpha0 is compared with official relation forward on the same T4 weights. No inference-path retuning.'))
    log('alpha0_relation_kernel_audit_complete',rows=rows,non_gate_weights_identical=not different)
    del b0,t4,d,t;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
