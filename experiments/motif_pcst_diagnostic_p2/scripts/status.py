from pathlib import Path
path=Path(__file__).resolve().parents[1]/'results/status.json'
print(path.read_text(encoding='utf-8') if path.is_file() else '{"state":"not_started"}')
