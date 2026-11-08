import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'results/status.json'
print(p.read_text(encoding='utf-8') if p.exists() else json.dumps({'state':'not_started'}))
