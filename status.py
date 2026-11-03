import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    done=list((ROOT/'results').glob('training_*_s*.json'))
    print('Completed training trajectories:',len(done),'/21')
    p=ROOT/'results/training_status.json'
    if p.exists():
        a=json.loads(p.read_text());print('Current:',a['tag'],'step',a['step'],'/500')
    p=ROOT/'results/training_status_B1worker.json'
    if p.exists() and not (ROOT/'results/B1_worker_complete.json').exists():
        a=json.loads(p.read_text());print('Independent baseline:',a['tag'],'step',a['step'],'/500')
    print('Target result records:',len(list((ROOT/'evaluation').glob('*.json'))),'/275 including reused historical controls')
    for name in ['completion','failure','B1_worker_failure']:
        p=ROOT/'results'/(name+'.json')
        if p.exists():print(name,json.loads(p.read_text()))
    p=ROOT/'logs/pipeline.log'
    if p.exists():
        print('\nLatest progress:')
        print('\n'.join(p.read_text(encoding='utf-8',errors='replace').splitlines()[-6:]))
if __name__=='__main__':main()
