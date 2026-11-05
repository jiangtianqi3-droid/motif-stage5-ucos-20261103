import argparse
from stage6.util import ROOT, runtime_path, read, log
from stage6.protocol import register


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['tests', 'source', 'source-validation', 'adapt', 'freeze', 'formal', 'report', 'all'])
    p.add_argument('--stage5', type=runtime_path, required=True)
    p.add_argument('--history', type=runtime_path, required=True)
    p.add_argument('--source-root', type=runtime_path, required=True)
    p.add_argument('--official', type=runtime_path, required=True)
    p.add_argument('--observations', type=runtime_path, default=ROOT / 'observations')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--plot-python', type=runtime_path, default=runtime_path('E:/Python/Python39/python.exe'),
                   help='Existing Python with matplotlib, used only for post-hoc figures; may be Windows Python from WSL')
    return p.parse_args()


def main():
    a = arguments()
    p = register()
    command = a.command
    if command == 'all':
        import subprocess
        import sys
        for step in ['tests', 'source', 'source-validation', 'adapt', 'freeze', 'formal', 'report']:
            subprocess.run([sys.executable, '-u', '-m', 'stage6.run', step] + sys.argv[2:], check=True)
        return
    if command in ('tests', 'all'):
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        from stage6.util import dump
        dump(ROOT / 'results/core_tests.json', dict(run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped)))
        assert result.wasSuccessful()
        from stage6.numeric_tests import run
        run(a)
    if command in ('source', 'adapt', 'all'):
        from stage6.adapt import run_one
        from stage6.isolation import enable_observation_guard
        enable_observation_guard()
        if command in ('source', 'all'):
            for key in ['wn18rr_train', 'codex_medium_train', 'fb15k237_train']:
                for rate in p['source_rates']:
                    for split_seed in p['proxy_seeds']:
                        run_one(a, key, rate, split_seed, source=True)
        if command in ('adapt', 'all'):
            for _, key in p['targets']:
                for split_seed in p['proxy_seeds']:
                    run_one(a, key, p['target_rate'], split_seed)
        # A guarded process cannot load formal labels. Exit and use a new process.
        if command == 'all':
            from stage6.adapt import freeze
            freeze(a)
            log('adaptation_finished', next='run formal in a new unguarded process')
            return
    if command == 'freeze':
        from stage6.adapt import freeze
        freeze(a)
    if command == 'source-validation':
        from stage6.source_validation import run
        run(a)
    if command == 'formal':
        from stage6.formal import run
        run(a)
    if command == 'report':
        from stage6.analysis import run
        run(a)


if __name__ == '__main__':
    main()
