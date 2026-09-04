"""Paired common-cohort comparison of two report-derived label sources.

The primary metric retains 0.5 unknowns so both sources are scored on the same
58 expert studies. Addressed-only AUCs are not compared as a headline because
different sources can choose different, easier subsets of cases.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transfer_audit import TARGETS, UID, auc, require_unique_ids


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def align_source(path, ids, name):
    frame = pd.read_csv(path, dtype={UID: str})
    require_unique_ids(frame, name)
    if not set(TARGETS).issubset(frame.columns):
        raise ValueError(f'{name} is missing target columns')
    missing = set(ids) - set(frame[UID])
    if missing:
        raise ValueError(f'{name} is missing {len(missing)} gold study IDs')
    values = frame.set_index(UID).loc[ids, TARGETS].to_numpy(float)
    if not (np.isfinite(values) & (values >= 0) & (values <= 1)).all():
        raise ValueError(f'{name} values must be finite in [0,1]')
    return values


def macro(y, prediction):
    values = np.array([auc(y[:, j], prediction[:, j]) for j in range(len(TARGETS))])
    return float(np.nanmean(values)), values


def compare(gold_path, baseline_path, candidate_path, bootstrap=5000, seed=1400):
    gold = pd.read_csv(gold_path, dtype={UID: str})
    require_unique_ids(gold, 'gold')
    y = gold[TARGETS].to_numpy(float)
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError('Gold labels must be fully binary')
    ids = gold[UID].tolist()
    baseline = align_source(baseline_path, ids, 'baseline')
    candidate = align_source(candidate_path, ids, 'candidate')
    base_macro, base_targets = macro(y, baseline)
    cand_macro, cand_targets = macro(y, candidate)
    if not (np.isfinite(base_targets).all() and np.isfinite(cand_targets).all()):
        raise ValueError('Every target must have both expert classes')
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        idx = rng.integers(0, len(ids), len(ids))
        base_draw, base_per = macro(y[idx], baseline[idx])
        cand_draw, cand_per = macro(y[idx], candidate[idx])
        if np.isfinite(base_per).all() and np.isfinite(cand_per).all():
            draws.append(cand_draw - base_draw)
    if len(draws) < bootstrap * 0.8:
        raise ValueError('Too many bootstrap draws lack both classes for all targets')
    draws = np.asarray(draws)
    delta = cand_macro - base_macro
    return {
        'scope': ('Paired common-cohort label-source ranking comparison. Unknown 0.5 values '
                  'are retained; this does not prove an image model will inherit the gain.'),
        'studies': len(ids), 'baseline_macro_auc': base_macro,
        'candidate_macro_auc': cand_macro, 'macro_delta': delta,
        'macro_delta_ci95': np.quantile(draws, [.025, .975]).tolist(),
        'bootstrap_standard_error': float(draws.std(ddof=1)),
        'probability_better': float((draws > 0).mean()),
        'addressed_cells': {'baseline': int((baseline != 0.5).sum()),
                            'candidate': int((candidate != 0.5).sum())},
        'targets': {target: {
            'baseline_auc': float(base_targets[j]),
            'candidate_auc': float(cand_targets[j]),
            'delta': float(cand_targets[j] - base_targets[j]),
            'baseline_addressed': int((baseline[:, j] != 0.5).sum()),
            'candidate_addressed': int((candidate[:, j] != 0.5).sum()),
        } for j, target in enumerate(TARGETS)},
        'input_sha256': {'gold': digest(gold_path), 'baseline': digest(baseline_path),
                         'candidate': digest(candidate_path)},
        'bootstrap_seed': seed, 'bootstrap_requested': bootstrap,
        'promotion_note': ('A positive label-source delta is necessary but not sufficient. '
                           'Promotion still requires paired image-model OOF improvement.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=1400)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.gold, args.baseline, args.candidate, args.bootstrap, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f"paired label delta {result['macro_delta']:+.4f}, CI95 {result['macro_delta_ci95']}")


if __name__ == '__main__':
    main()
