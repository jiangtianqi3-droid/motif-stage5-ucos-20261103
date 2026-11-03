"""Independent predeclared B1 jobs; identical training, separate progress file."""
import traceback
from common import ROOT,dump
import train
original_dump=train.dump
def progress_dump(path,obj):
    if path.name=='training_status.json':path=path.with_name('training_status_B1worker.json')
    return original_dump(path,obj)
train.dump=progress_dump
try:
    for seed in [1024,2026,3407]:train.run('B1-short',train.config(0.,'joint_low_lr','plain'),seed)
    dump(ROOT/'results/B1_worker_complete.json',dict(status='complete',seeds=[1024,2026,3407]))
except Exception as exc:
    dump(ROOT/'results/B1_worker_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
