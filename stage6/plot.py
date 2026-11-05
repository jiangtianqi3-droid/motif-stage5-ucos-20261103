"""Standalone matplotlib renderer; no torch, model or target-selection imports."""
import json
import sys
from pathlib import Path
# This standalone renderer shares a directory with select.py. Avoid shadowing
# Python's standard select extension when matplotlib imports socket/selectors.
sys.path = [p for p in sys.path if Path(p or '.').resolve() != Path(__file__).resolve().parent]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    folder = path.parent
    selected = [r for r in data['points'] if r['alpha'] == .25]
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for i, r in enumerate(selected):
        label = r['key'].replace('Inductive', '').replace('ConceptNet100k_external', 'ConceptNet')
        ax.scatter(r['proxy_gain'], r['formal_gain'], s=45, marker='o', edgecolor='white', linewidth=.5)
        ax.annotate(label, (r['proxy_gain'], r['formal_gain']), xytext=(5, 5 if i % 2 == 0 else -11),
                    textcoords='offset points', fontsize=8)
    ax.axhline(0, color='gray', lw=.8)
    ax.axvline(0, color='gray', lw=.8)
    ax.margins(.22)
    ax.set(xlabel='Proxy selection MRR gain (alpha=.25 vs 0)', ylabel='Formal test MRR gain (alpha=.25 vs 0)',
           title='H1: gain prediction on 11 historically exposed graphs')
    fig.savefig(folder / 'proxy_vs_formal.png', dpi=180)
    fig.savefig(folder / 'proxy_vs_formal.svg')
    plt.close(fig)
    methods = ['B0', 'B1', 'B2', 'reference', 'fixed', 'random', 'joint_mean', 'topology_lcb', 'entity_lcb', 'joint_lcb']
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    for ax, group in zip(axes, ['locked', 'historical', 'external']):
        vals = [next(r['mrr'] for r in data['macros'] if r['group'] == group and r['method'] == m) for m in methods]
        ax.barh(methods, vals, color=['#627B8C' if m != 'joint_lcb' else '#B84A35' for m in methods])
        ax.set_xlim(0, max(vals) * 1.23)
        for y, value in enumerate(vals):
            ax.text(value + max(vals) * .015, y, f'{value:.5f}', va='center', fontsize=8)
        ax.invert_yaxis()
        ax.set(title=group, xlabel='Macro MRR')
    fig.savefig(folder / 'macro_comparison.png', dpi=180)
    fig.savefig(folder / 'macro_comparison.svg')
    plt.close(fig)
    ci = [next(r for r in data['bootstrap'] if r['key'] == 'macro_' + group and
               r['method'] == 'joint_lcb' and r['comparator'] == 'B0') for group in ['locked', 'historical', 'external']]
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    values = [r['delta'] for r in ci]
    ax.errorbar(values, range(3), xerr=[[r['delta'] - r['lo'] for r in ci], [r['hi'] - r['delta'] for r in ci]],
                fmt='o', capsize=4, color='#B84A35')
    ax.axvline(0, color='gray', lw=.8)
    ax.set_yticks(range(3), ['locked', 'historical', 'external'])
    ax.invert_yaxis()
    ax.set(xlabel='Joint TSDC MRR minus current B0', title='95% paired triple-bootstrap intervals (5,000 resamples)')
    fig.savefig(folder / 'macro_delta_ci.png', dpi=180)
    fig.savefig(folder / 'macro_delta_ci.svg')
    plt.close(fig)
    (folder / 'environment.json').write_text(json.dumps(dict(python=sys.version, matplotlib=matplotlib.__version__), indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=Path)
    run(parser.parse_args().inputs)
