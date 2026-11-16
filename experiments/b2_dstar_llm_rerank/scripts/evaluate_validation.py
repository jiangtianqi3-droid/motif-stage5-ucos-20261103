"""Validation metrics are produced after inference, never supplied to prompts."""
import numpy as np
from common_io import SEEDS, MODES, read
from experiment import score_rows
from offline_eval import truth_index, baseline_rank, changed_rank, metrics

def evaluate(q, truth_facts, R, score_dir, query_dir, selection, confirm):
    truth=truth_index(truth_facts,R)
    ranks={'B2+D*':np.zeros((3,len(q)),dtype=np.int64)}
    ranks.update({mode:np.zeros((3,len(q)),dtype=np.int64) for mode in MODES})
    for j,seed in enumerate(SEEDS):
        for i,score in score_rows(score_dir/str(seed)):
            h,r,t=map(int,q[i]);record=read(query_dir/str(seed)/(f'{h}_{r}.json'))
            ranks['B2+D*'][j,i]=baseline_rank(score,h,r,t,truth)
            for mode in MODES:
                ranks[mode][j,i]=changed_rank(score,h,r,t,truth,record['top30'],record['modes'][mode]['order'])
    result={}
    for group,ids in [('selection_val',selection),('confirm_val',confirm)]:
        queries=np.array([2*i+d for i in ids for d in range(2)],dtype=int)
        result[group]=dict(queries=len(queries),methods={mode:metrics(a[:,queries]) for mode,a in ranks.items()})
    result['selection_performed']=False
    return result
