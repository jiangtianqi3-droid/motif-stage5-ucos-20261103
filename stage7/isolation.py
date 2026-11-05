import os
import sys
from pathlib import Path

_enabled=False

def enable():
    global _enabled
    if _enabled:return
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        p=str(args[0]).replace('\\','/').lower()
        name=Path(p).name
        forbidden=('/evaluation/' in p or '/downloads/' in p or ('/results/' in p and '/stage7/' not in p) or
           ('/analysis/' in p and '/stage7/' not in p) or
           (('motif-phase' in p or 'ultra-phase' in p) and '/data/' in p and '/observations/' not in p) or
           name=='target_inputs.json' or 'query_masks' in name or 'group_masks' in name or
           (name.endswith(('.txt','.npy','.npz')) and any(x in name for x in ('valid','test','ranking'))) or
           name=='observed.npz' or (name.endswith('.npy') and 'quer' in name))
        if forbidden:
            raise PermissionError('Stage7 selector blocked label/ranking asset: '+p)
    sys.addaudithook(guard)
    _enabled=True
