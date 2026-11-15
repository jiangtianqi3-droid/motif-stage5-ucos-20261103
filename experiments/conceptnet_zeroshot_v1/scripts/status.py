import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
print(json.dumps(json.loads((root/'results/status.json').read_text()),ensure_ascii=False,indent=2))
