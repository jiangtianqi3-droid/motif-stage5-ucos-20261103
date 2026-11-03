import json,traceback
from common import ROOT,dump,log
from train import run,config
def main():
    assert json.loads((ROOT/'results/operator_tests.json').read_text())['status']=='passed'
    screens=[]
    for freedom in ['gate_only','joint_low_lr']:
        for alpha in [.15,.2,.25,'adaptive']:
            tag='T_'+freedom+'_'+str(alpha).replace('.','p')
            screens.append(run(tag,config(alpha,freedom),1024))
    winners={f:max([r for r in screens if r['config']['freedom']==f],key=lambda r:r['best_source_validation_mrr']) for f in ['gate_only','joint_low_lr']}
    mainwinner=max(winners.values(),key=lambda r:r['best_source_validation_mrr']);formal={}
    for freedom,rec in winners.items():
        rows=[rec]+[run(rec['family'],rec['config'],seed) for seed in [2026,3407]]
        formal['T-4D' if freedom==mainwinner['config']['freedom'] else 'T-4D-alt']={str(r['seed']):r for r in rows}
    cfg=mainwinner['config']
    for family,cc in [('B1-short',config(0.,'joint_low_lr','plain')),('U-Cos',dict(cfg,semantic=True)),('R-Cos-Shuffle',dict(cfg,semantic=True,shuffle=True))]:
        formal[family]={str(seed):run(family,cc,seed) for seed in [1024,2026,3407]}
    dump(ROOT/'analysis/source_choices.json',dict(screens=screens,winners=winners,main_topology_config=mainwinner['config'],formal=formal))
    log('all_training_complete')
    from finalize import main as finish
    finish()
if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(ROOT/'results/failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
