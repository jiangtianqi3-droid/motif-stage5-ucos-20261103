"""Windows/WSL entry point preserving the established CUDA environment."""
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if os.name=='nt':
    windows=ROOT.as_posix()
    wsl='/mnt/'+windows[0].lower()+windows[2:]
    raise SystemExit(subprocess.call(['wsl.exe','-d','Ubuntu-24.04','--','bash',
                                     wsl+'/scripts/run.sh','-m','v21.run',*sys.argv[1:]]))
subprocess.run(['bash',str(ROOT/'scripts/run.sh'),'-m','v21.run',*sys.argv[1:]],check=True)
