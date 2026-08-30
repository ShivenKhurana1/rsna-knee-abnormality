"""Compare paired, held-out prediction CSVs; never fit blend weights or train models.

Requires keyed labels plus baseline and candidate predictions on exactly the same
studies. Results establish only the supplied evaluation cohort's performance.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'


def aligned_inputs(labels, baseline, candidate):
    for name, frame in [('labels', labels), ('baseline', baseline), ('candidate', candidate)]:
        if not set([UID] + TARGETS).issubset(frame.columns):
            raise ValueError(f'{name}: required UID/target columns missing')
        if frame[UID].isna().any() or frame[UID].duplicated().any() or frame.empty:
            raise ValueError(f'{name}: empty input, missing IDs or duplicate IDs')
        if set(frame[UID]) != set(labels[UID]):
            raise ValueError(f'{name}: study coverage differs; partial scoring is not allowed')
    y = labels[TARGETS].to_numpy(float)
    if not np.all(np.isnan(y) | (y == 0) | (y == 1)):
        raise ValueError('Labels must be binary 0/1 or missing, not soft probabilities')
    predictions = []
    for frame in [baseline, candidate]:
        p = frame.set_index(UID).loc[labels[UID], TARGETS].to_numpy(float)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Predictions must be finite and in [0,1]')
        predictions.append(p)
    return y, *predictions


def auc(y, p):
    valid = np.isfinite(y)
    y, p = np.asarray(y)[valid], np.asarray(p)[valid]
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    if not positives or not negatives:
        return np.nan
    ranks = pd.Series(p).rank(method='average').to_numpy()
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def compare(labels, baseline, candidate, bootstrap=2000, seed=1300):
    if bootstrap < 1:
        raise ValueError('bootstrap must be positive')
    y, base, cand = aligned_inputs(labels, baseline, candidate)
    base_auc = np.array([auc(y[:, j], base[:, j]) for j in range(len(TARGETS))])
    cand_auc = np.array([auc(y[:, j], cand[:, j]) for j in range(len(TARGETS))])
    if not np.isfinite(base_auc).all():
        raise ValueError('Every target needs positive and negative held-out labels for twelve-target macro AUC')
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(bootstrap):
        # Paired study bootstrap: exactly the same sampled rows for both models
        # and all targets. Never independently resample the two prediction sets.
        ids = rng.integers(0, len(y), len(y))
        delta = np.array([auc(y[ids, j], cand[ids, j]) - auc(y[ids, j], base[ids, j])
                          for j in range(len(TARGETS))])
        if np.isfinite(delta).all():
            deltas.append(delta)
    if len(deltas) < max(1, int(.8 * bootstrap)):
        raise ValueError('Too many bootstrap replicates lack both classes; cohort is too small/imbalanced')
    deltas = np.stack(deltas)
    macro_ci = np.quantile(deltas.mean(axis=1), [.025, .975]).tolist()
    target_ci = np.quantile(deltas, [.025, .975], axis=0)
    delta = float(np.mean(cand_auc - base_auc))
    return {
        'scope': 'Supplied held-out labels only; not Kaggle leaderboard performance or proof of training exclusion.',
        'studies': len(y), 'bootstrap_unit': 'study', 'bootstrap_seed': seed,
        'bootstrap_requested': bootstrap, 'bootstrap_valid_all_targets': len(deltas),
        'ci_note': 'Paired percentile 95% CI; replicates missing either class in any target are excluded. Not selection-adjusted.',
        'baseline_macro_auc': float(base_auc.mean()), 'candidate_macro_auc': float(cand_auc.mean()),
        'macro_delta': delta, 'macro_delta_ci95': macro_ci,
        'point_estimate_gain_ge_0_01': delta >= .01,
        'point_estimate_gain_ge_0_02': delta >= .02,
        'ci_lower_bound_ge_0_01': macro_ci[0] >= .01,
        'ci_lower_bound_ge_0_02': macro_ci[0] >= .02,
        'targets': {target: {'n_labeled': int(np.isfinite(y[:, j]).sum()),
            'positive': int(np.nansum(y[:, j])), 'baseline_auc': float(base_auc[j]),
            'candidate_auc': float(cand_auc[j]), 'delta': float(cand_auc[j] - base_auc[j]),
            'delta_ci95': target_ci[:, j].tolist()} for j, target in enumerate(TARGETS)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['labels', 'baseline', 'candidate', 'output']:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1300)
    args = parser.parse_args()
    inputs = [args.labels, args.baseline, args.candidate]
    if args.output.resolve() in [p.resolve() for p in inputs] or args.output.exists():
        raise ValueError('Output must be a new file, not an input or existing report')
    frames = [pd.read_csv(p, dtype={UID: str}) for p in inputs]
    result = compare(*frames, bootstrap=args.bootstrap, seed=args.seed)
    result['input_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in zip(['labels', 'baseline', 'candidate'], inputs)}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"Paired macro AUC: {result['baseline_macro_auc']:.6f} -> {result['candidate_macro_auc']:.6f}")
    print(f"Delta {result['macro_delta']:+.6f}; 95% CI {result['macro_delta_ci95']}")
    print('This is not a measured Kaggle leaderboard gain.')


if __name__ == '__main__':
    main()
