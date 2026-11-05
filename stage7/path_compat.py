"""Portable SHA path adapter; frozen sampling/model code stays byte-identical.

The preserved-assets inventory was made on Windows. POSIX treats its backslash
as a filename character. This launcher normalizes separators solely at SHA
file-opening boundaries in formal evaluation and verification.
"""
from pathlib import Path
from datetime import datetime, timezone
import importlib
from stage7.run import parser
from stage7.util import ROOT, sha, read, dump


def portable_sha(path):
    return sha(Path(str(path).replace('\\','/')))


def main():
    args=parser().parse_args()
    assert args.task in ['formal','verify']
    # Independently check the complete historical inventory before opening labels.
    inventory=read(ROOT/'preserved_assets.json')
    for name,expected in inventory.items():
        assert portable_sha(ROOT.parent/name)==expected,name
    dump(ROOT/'results'/('path_compat_'+args.task+'.json'),dict(
        created_utc=datetime.now(timezone.utc).isoformat(),status='passed',
        original_failure='POSIX path stage6\\.gitignore from Windows manifest key',
        normalized_inventory_files=len(inventory),frozen_sha256=sha(ROOT/'frozen.json'),
        adapter_sha256=sha(Path(__file__)),
        scope='Only SHA file path separators; no sampling, model, ranking, or selection change',
        frozen_files_modified=False))
    module=importlib.import_module('stage7.'+args.task)
    module.sha=portable_sha
    module.run(args)


if __name__=='__main__':main()
