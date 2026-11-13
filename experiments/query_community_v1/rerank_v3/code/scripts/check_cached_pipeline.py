"""CPU integration check: replay verified B0 scores, exercise corrected selection/ranking."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    from src import pipeline
    from src.util import ROOT,dump,read,identity,source_identity
    def replay(dataset,g,sig,family,work,selections,run):
        for split,indices in selections.items():
            for i in indices:
                for d in (0,1):
                    if not (work/'scores'/split/f'{i}_{d}.npy').exists():raise RuntimeError('Missing verified cached score')
        dump(work/'model_audit.json',dict(training=False,inference_replayed=True,reason='CPU integration only'))
    pipeline.infer=replay
    run=ROOT/'results/cached_integration_smoke'/identity(source_identity())[:12]
    pipeline.worker('NELLInductive_v1','B0',run,smoke=True,skip_llm=True)
    result=read(run/'NELLInductive_v1/B0/completion.json')
    assert result['state']=='smoke_complete'
    print('CACHED_INTEGRATION_PASSED',flush=True)

if __name__=='__main__':main()
