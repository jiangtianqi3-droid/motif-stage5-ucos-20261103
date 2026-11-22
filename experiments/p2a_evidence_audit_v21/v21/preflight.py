from pathlib import Path
import platform
import subprocess

import numpy as np
import torch
from src.io_utils import STAGE7, HISTORY
from src.recovery import verify_runtime
from p2.assets import Target, p1_root, P1_SHA, CHECKPOINT
from v21.common import ROOT, OUTPUT, config, verify, identity, read, dump, sha, write_csv


def setup(cfg):
    p1=p1_root()
    verify(read(p1/'audit.json')['commit']==P1_SHA,'unexpected P1 commit')
    verified=verify_runtime()
    checkpoint=HISTORY/'topology-semantic-stage4-20261102/checkpoints/B2_s1024_step1000.pth'
    verify(sha(checkpoint)==CHECKPOINT,'B2 checkpoint mismatch')
    codes={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'v21').glob('*.py'))}
    retrieval_codes={f'{prefix}/{p.name}':sha(p) for prefix in ('motif_pcst_structural_p1/p1',
        'motif_pcst_diagnostic_p2/p2') for p in sorted((ROOT.parent/prefix).glob('*.py'))}
    run_id=identity(dict(config=cfg,code=codes,dependencies=retrieval_codes,checkpoint=CHECKPOINT,p1=p1.name))
    directory=OUTPUT/run_id
    directory.mkdir(parents=True,exist_ok=True)
    view=Target(cfg['experiment']['dataset'],p1,directory)
    # Sampling is solely from P1 cached test row IDs, independently of targets/ranks.
    required=cfg['experiment']['query_count']
    verify(required%2==0 and required<=len(view.indices)*2,'query count requires paired directions and available rows')
    indices=np.random.default_rng(cfg['experiment']['seed']).choice(
        sorted(view.indices),required//2,replace=False).tolist()
    selection=[(int(i),d) for i in indices for d in (0,1)]
    selection_path=ROOT/'manifests'/f'fixed_queries_{run_id}.json'
    selected=[dict(query_id=f'{view.key}:{i}:{d}',fact_index=i,direction=d,
                   query_head=int(view.asset.queries[i,d,0]),query_relation=int(view.asset.queries[i,d,1]))
              for i,d in selection]
    if selection_path.is_file():
        verify(read(selection_path)==selected,'fixed sample changed on restart')
    else:
        dump(selection_path,selected)
    write_csv(OUTPUT/'propagation/fixed_queries.csv',selected)
    batch=view.asset.meta['batch']
    complete_relations={int(view.asset.queries[j,d,1]) for i,_ in selection
        for j in range(i//batch*batch,min((i//batch+1)*batch,len(view.asset.queries))) for d in (0,1)}
    cpu=platform.processor()
    try:
        cpu=subprocess.check_output(['bash','-c',"lscpu | sed -n 's/Model name:[[:space:]]*//p'"],text=True).strip() or cpu
    except subprocess.CalledProcessError:
        pass
    env=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        cpu=cpu,gpu=torch.cuda.get_device_name(0),gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
        cfg=cfg,code=codes,dependency_code=retrieval_codes,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),p1_commit=P1_SHA,p1_run_id=p1.name,
        p2_previous_state=read(ROOT.parent/'motif_pcst_diagnostic_p2/results/status.json')['state'],
        observations=sha(view.asset.bundle),observed_arrays=view.asset.entry['observed_sha256'],
        expected_model_parameters=view.asset.meta['parameter_sha256_before'],runtime=verified,
        run_id=run_id,selection_sha256=identity(selected),llm_api_requests=0)
    dump(directory/'provenance.json',env)
    with np.load(view.asset.history/'B0.npz',allow_pickle=False) as f:
        schema={k:dict(shape=list(f[k].shape),dtype=str(f[k].dtype)) for k in f.files}
    dump(directory/'historical_schema.json',dict(B0_npz=schema,
        candidate_shape=list(view.raw.shape),candidate_kind='reciprocal_raw_rank',candidate_hash=view.asset.meta['feature_sha256']))
    return directory,view,selection,sorted(complete_relations),env
