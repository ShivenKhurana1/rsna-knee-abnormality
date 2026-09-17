import argparse
import json

from .common import config
from .runner import prepare, train, merge, infer
from .evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description='V16 dynamic MRI pipeline')
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Audit groups, freeze folds and cache real DICOMs')
    prep.add_argument('--data-root', required=True)
    prep.add_argument('--work', required=True)
    prep.add_argument('--config', required=True)
    prep.add_argument('--report-labels')
    prep.add_argument('--groups')
    fit = sub.add_parser('train', help='Train one fold; resumes exact matching checkpoint')
    fit.add_argument('--work', required=True)
    fit.add_argument('--fold', type=int, required=True)
    fit.add_argument('--seed', type=int)
    fit.add_argument('--arm', choices=['none', 'raw', 'platt'])
    combine = sub.add_parser('merge', help='Verify every fold and assemble complete OOF')
    combine.add_argument('--work', required=True)
    combine.add_argument('--seed', type=int, required=True)
    combine.add_argument('--arm', choices=['none', 'raw', 'platt'], required=True)
    predict = sub.add_parser('infer', help='Offline prediction using trained V16 checkpoints')
    predict.add_argument('--data-root', required=True)
    predict.add_argument('--checkpoints', nargs='+', required=True)
    predict.add_argument('--output', default='submission.csv')
    predict.add_argument('--cache', required=True)
    predict.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    predict.add_argument('--encoder-code', help='Attached local DINOv2 repo, for native Meta checkpoints')
    predict.add_argument('--baseline-csv')
    predict.add_argument('--blend-weight', type=float)
    predict.add_argument('--max-seconds', type=float, default=28800)
    score = sub.add_parser('evaluate', help='Paired expert-label group-bootstrap comparison')
    score.add_argument('--work', required=True)
    score.add_argument('--baseline-oof', required=True)
    score.add_argument('--candidate-oof', required=True)
    score.add_argument('--output', required=True)
    score.add_argument('--weight', type=float, default=1.)
    score.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.data_root, args.work, config(args.config), args.report_labels, args.groups)
    elif args.command == 'train':
        result = train(args.work, args.fold, args.seed, args.arm)
    elif args.command == 'merge':
        result = merge(args.work, args.seed, args.arm)
    elif args.command == 'evaluate':
        result = evaluate(args.work, args.baseline_oof, args.candidate_oof, args.output,
                          args.weight, args.bootstrap)
    else:
        result = infer(args.data_root, args.checkpoints, args.output, args.cache, args.device,
                       args.baseline_csv, args.blend_weight, args.encoder_code, args.max_seconds)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
