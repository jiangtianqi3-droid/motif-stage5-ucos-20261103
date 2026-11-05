import argparse
from pathlib import Path
from stage7.util import ROOT


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('task', choices=['audit', 'numeric', 'gpu-checks', 'tie-checks', 'prepare', 'mechanism', 'source-validation', 'protocol', 'adapt', 'freeze', 'formal', 'analyze', 'verify', 'report'])
    p.add_argument('--stage5', type=Path, default=ROOT.parent)
    p.add_argument('--history', type=Path, default=ROOT.parents[1]/'topology-semantic-stage4-20261102')
    p.add_argument('--official', type=Path, default=ROOT.parents[2]/'motif-phase3-work/MOTIF/ckpts/motif_3g.pth')
    p.add_argument('--observations', type=Path, default=ROOT.parent/'stage6/observations')
    p.add_argument('--source-root', type=Path, default=ROOT.parents[1]/'topology-stage2-20261102')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--keys', nargs='*')
    p.add_argument('--training-seeds', type=int, nargs='*')
    p.add_argument('--proxy-seeds', type=int, nargs='*')
    return p


def main():
    args = parser().parse_args()
    (ROOT/'results').mkdir(exist_ok=True)
    if args.task == 'numeric':
        from stage7.numeric_probe import run
    elif args.task == 'source-validation':
        from stage7.experiment import run_validation as run
    else:
        import importlib
        run = importlib.import_module('stage7.'+{'audit':'audit', 'gpu-checks':'gpu_checks', 'tie-checks':'tie_checks', 'prepare':'prepare', 'mechanism':'experiment',
           'protocol':'protocol', 'adapt':'experiment', 'freeze':'freeze', 'formal':'formal',
           'analyze':'analysis', 'verify':'verify', 'report':'report_writer'}[args.task]).run
    run(args)


if __name__ == '__main__':
    main()
