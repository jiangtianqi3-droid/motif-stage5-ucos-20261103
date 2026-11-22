"""Recreate audit tables, figures and report solely from verified completed query checkpoints."""
from pathlib import Path
import sys
from datetime import datetime,timezone,timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from v21.common import OUTPUT,config,read,dump,verify
from v21.graph import FactIndex
from v21.audit import collect
from v21.statistics import analyze,figures,make_report
from v21.run import publish
from p2.assets import Target,p1_root

cfg=config()
state=read(OUTPUT/'status.json')
verify(state['state']!='running','finish or stop the worker before regenerating its reports')
folder=Path(state['run_directory'])
proof=read(folder/'smoke_verified.json')
verify(proof['all_checks_passed'],'verified smoke computation is required')
view=Target(cfg['experiment']['dataset'],p1_root(),folder)
records=[read(p) for p in sorted((folder/'query_checkpoints').glob('*.json'))]
verify(bool(records),'no completed query checkpoints')
selection=[(r['fact_index'],r['direction']) for r in records]
index=FactIndex(view.graph)
table=read(view.dir.parents[2]/'source_calibration.json')['table']
_,trace,agg=collect(view,selection,index,cfg,folder,table)
stats=analyze(folder,records,cfg)
figures(folder,stats,trace,agg,cfg)
state.update(state='complete' if len(records)==cfg['experiment']['query_count'] else 'smoke_complete',
             current_phase='report',completed_queries=len(records),last_update=datetime.now(timezone(timedelta(hours=8))).isoformat())
state.pop('error',None)
state['completed_experiments']=list(dict.fromkeys(state['completed_experiments']+
      ['10_query_smoke','smoke_statistics_and_seven_figures']))
make_report(folder,stats,records,trace,agg,cfg,state,proof)
dump(OUTPUT/'status.json',state)
publish(folder)
print(f'Regenerated report and seven figures for {len(records)} verified queries')
