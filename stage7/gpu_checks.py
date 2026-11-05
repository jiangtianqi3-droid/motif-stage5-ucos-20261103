import gc
import types
import numpy as np
import torch
import torch.nn.functional as F
from stage7.util import ROOT,read,dump,log
from stage7.experiment import SOURCES,graph_for
from stage7.proxy_split import residual
from stage6.proxy_graph import build_graph
from stage6.models import load_model,set_alpha,state_hash
from stage7.numeric_stability import install,configure
from stage7.evaluator import safe_evaluate


@torch.no_grad()
def run(args):
    import sys,os,platform,scipy,torch_geometric,triton
    configure(False)
    dump(ROOT/'results/environment.json',dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,
        pyg=torch_geometric.__version__,triton=triton.__version__,numpy=np.__version__,scipy=scipy.__version__,
        gpu=torch.cuda.get_device_name(0),gpu_total_memory=torch.cuda.get_device_properties(0).total_memory,
        matmul_precision=torch.get_float32_matmul_precision(),matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        warning='Fixed GEMM geometry validated; native sparse kernels are not certified deterministic on every device.'))
    rows=[]
    for key in SOURCES:
        graph=graph_for(args,key)
        with np.load(ROOT/'splits'/f'{key}__p6101.npz') as z:
            arr={k:z[k].copy() for k in z.files}
        rem=residual(graph,arr,0)
        q=arr['facts'][arr['A_selection'][:8]]
        truth=np.c_[graph.edge_index[0],graph.edge_type,graph.edge_index[1]]
        d,t,_=build_graph(rem,q,truth,ROOT/'cache/topology')
        m=install(load_model(args.stage5,args.history,args.official,'T-4D',1024))
        before=state_hash(m)
        rel=m.relation_model
        query=torch.tensor([0,1],device='cuda')
        set_alpha(m,0.)
        zero=rel(d.relation_hypergraph,query).clone()
        gates=[layer.topology_gate for layer in rel.layers]
        for layer in rel.layers: del layer.topology_gate
        plain=rel(d.relation_hypergraph,query)
        for layer,gate in zip(rel.layers,gates): layer.topology_gate=gate
        error=float((plain-zero).abs().max());assert error==0.
        set_alpha(m,.25)
        mod=rel(d.relation_hypergraph,query)
        change=float((mod-zero).abs().max());assert change>0 and torch.isfinite(mod).all()
        ref=None
        batchrows=[]
        for bs in [1,2,4,8,8]:
            r,info,s=safe_evaluate(m,d,t,[0.,.25],bs,True)
            s=np.stack(s)
            if ref is None:ref=(r,s)
            se=float(np.abs(s-ref[1]).max());rc=int((r!=ref[0]).sum())
            batchrows.append(dict(batch=bs,max_score_error=se,rank_changes=rc))
            assert se==0 and rc==0,(key,bs,se,rc)
        # Compare the unchanged Stage5 ranking pipeline on the same canonical
        # relation representations, isolating ranking from numerical path changes.
        from evaluate import evaluate as stage5_evaluate
        set_alpha(m,.25)
        native=rel.forward;memory={}
        for at in range(0,graph.R,2):
            ids=torch.tensor([at,at+1],device='cuda')
            out=native(d.relation_hypergraph,ids)
            memory[at],memory[at+1]=out[0].clone(),out[1].clone()
        def canonical(self,hyper,query):return torch.stack([memory[r] for r in query.cpu().tolist()])
        rel.forward=types.MethodType(canonical,rel)
        try:
            _,standard,_=stage5_evaluate(m,d,t,8,(0.,))
            assert np.array_equal(standard[0],ref[0][1])
        finally:rel.forward=native
        del memory
        # GEMM accumulation precision sensitivity, with all weights unchanged.
        precision=[]
        for dtype in [torch.float64,torch.bfloat16]:
            def precision_linear(self,x):
                def calc(v):
                    return F.linear(v.to(dtype),self.weight.to(dtype),self.bias.to(dtype) if self.bias is not None else None).to(x.dtype)
                return torch.stack([calc(v) for v in x]) if x.ndim>=3 else calc(x)
            for layer in m.entity_model.modules():
                if isinstance(layer,torch.nn.Linear): layer.forward=types.MethodType(precision_linear,layer)
            r,_,s=safe_evaluate(m,d,t,[0.,.25],4,True)
            precision.append(dict(projection_accumulation=str(dtype),propagation='float32',
                 max_score_difference=float(np.abs(np.stack(s)-ref[1]).max()),rank_changes=int((r!=ref[0]).sum())))
        install(m)
        autocast_modes=[]
        if key=='wn18rr_train':
            for dtype in [torch.float16,torch.bfloat16]:
                try:
                    with torch.autocast(device_type='cuda',dtype=dtype):
                        ar,_,_=safe_evaluate(m,d,t,[0.,.25],4,False)
                    autocast_modes.append(dict(mode=str(dtype),status='supported_on_probe',
                        finite_scores_verified=True,
                        rank_changes=int((ar!=ref[0]).sum())))
                except Exception as e:
                    autocast_modes.append(dict(mode=str(dtype),status='unsupported_on_probe',error=repr(e)))
                    gc.collect();torch.cuda.empty_cache()
        assert before==state_hash(m) and all(p.grad is None for p in m.parameters())
        rows.append(dict(key=key,alpha0_plain_error=error,alpha025_relation_change=change,
             batch_rows=batchrows,precision=precision,autocast_modes=autocast_modes,
             no_weight_update=True,standard_stage5_paired_ranking_exact=True))
        log('gpu_checks_passed',key=key,alpha0_error=error,batch_rows=batchrows)
        del m,d,t;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'results/gpu_checks.json',dict(status='passed',rows=rows,
         numeric_policy='query-wise FP32 entity Linear; TF32 disabled; canonical global relation pairs; batch8 with OOM halving; no ties tolerance'))
