import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'results/status.json'
print(path.read_text(encoding='utf-8') if path.exists() else json.dumps({'state':'not_started'}))
