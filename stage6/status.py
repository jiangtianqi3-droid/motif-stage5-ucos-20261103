"""Read-only progress; never loads query labels, rankings or handoff state."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    expected = {'source_diagnostics': 81, 'source_density': 9, 'source_validation': 9,
                'adaptation': 99, 'evaluation': 110}
    status = {k: dict(complete=len(list((root / k).glob('*.json'))), expected=v) for k, v in expected.items()}
    status.update(frozen=(root / 'frozen.json').exists(), report=(root / 'report.md').exists(),
                  verification=(root / 'results/verification.json').exists())
    print(json.dumps(status, indent=2))
    for p in sorted((root / 'logs').glob('*.log')):
        if p.name in ['tests.log', 'source.log', 'source-validation.log', 'adapt.log', 'freeze.log', 'formal.log', 'report.log']:
            lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
            print(p.name, lines[-1] if lines else '(starting)')


if __name__ == '__main__':
    main()
