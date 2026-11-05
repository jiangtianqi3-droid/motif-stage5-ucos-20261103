"""A process-level access guard, additional to the closed observation data type."""
import os
import sys
from pathlib import Path


def enable_observation_guard():
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = str(args[0]).replace('\\', '/').lower()
        name = Path(path).name
        forbidden = ('/evaluation/' in path or 'query_masks_' in name or 'group_masks_' in name or
                     (name.endswith('.txt') and any(s in name for s in ('valid', 'test'))) or
                     name == 'observed.npz' or name.endswith('__validation.npz') or
                     name == 'target_inputs.json' or
                     (name == 'frozen.json' and '/stage6/' not in path))
        if forbidden:
            raise PermissionError('Observation-only process denied label/ranking asset: ' + path)
    sys.addaudithook(audit)
