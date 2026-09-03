"""Automated, paired comparison of the baseline-arm vs. auxiliary-arm OOF
predictions against expert gold labels. Same statistical standard as v14's
diagnostics.py: grouped paired bootstrap, no unpaired comparisons, explicit
scope caveats rather than a bare number.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from contract import TARGETS, UID


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def align(frame, ids):
    if frame[UID].isna().any() or frame[UID].duplicated().any():
        raise ValueError('Missing or duplicate study IDs')
    if set(frame[UID]) != set(ids):
        raise ValueError('Prediction coverage does not exactly match the requested study IDs; '
                         'partial scoring is prohibited')
    return frame.set_index(UID).loc[list(ids)].reset_index()


def auc(y, p):
    valid = np.isfinite(y)
    y, p = y[valid], p[valid]
    pos, neg = np.sum(y == 1), np.sum(y == 0)
    if not pos or not neg:
        return np.nan
    return float((rankdata(p)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def aucs(y, p):
    return np.array([auc(y[:, j], p[:, j]) for j in range(len(TARGETS))])


def compare(labels_path, baseline_oof_path, auxiliary_oof_path, bootstrap=2000, seed=1400):
    labels = pd.read_csv(labels_path, dtype={UID: str})
    baseline = pd.read_csv(baseline_oof_path, dtype={UID: str})
    auxiliary = pd.read_csv(auxiliary_oof_path, dtype={UID: str})
    ids = labels[UID].tolist()
    baseline = align(baseline, ids)
    auxiliary = align(auxiliary, ids)
    y = labels.set_index(UID).loc[ids, TARGETS].to_numpy(float)
    b = baseline[TARGETS].to_numpy(float)
    a = auxiliary[TARGETS].to_numpy(float)
    if not (np.isin(y, [0.0, 1.0])).all():
        raise ValueError('Labels must be strictly binary for this comparison')

    ba, aa = aucs(y, b), aucs(y, a)
    rng = np.random.default_rng(seed)
    n = len(ids)
    draws = []
    for _ in range(bootstrap):
        idx = rng.integers(0, n, n)
        d = aucs(y[idx], a[idx]) - aucs(y[idx], b[idx])
        if np.isfinite(d).all():
            draws.append(d)
    if len(draws) < 0.5 * bootstrap:
        raise ValueError('Too many resamples lacked both classes; cohort too small/imbalanced for this comparison')
    draws = np.asarray(draws)
    macro_ci = np.quantile(draws.mean(axis=1), [.025, .975])
    target_ci = np.quantile(draws, [.025, .975], axis=0)

    return {
        'scope': ('Paired comparison on the SAME 58-study gold cohort that also selected the '
                  'auxiliary-loss policy (see labels.py). This is exploratory, not independent '
                  'confirmation, and not a hidden-test-set estimate.'),
        'cohort_studies': n,
        'baseline_macro_auc': float(ba.mean()), 'auxiliary_macro_auc': float(aa.mean()),
        'macro_delta': float((aa - ba).mean()), 'macro_delta_ci95': macro_ci.tolist(),
        'bootstrap_seed': seed, 'valid_replicates': len(draws), 'requested_replicates': bootstrap,
        'targets': {t: {'baseline_auc': float(ba[j]), 'auxiliary_auc': float(aa[j]),
                        'delta': float(aa[j] - ba[j]), 'delta_ci95': target_ci[:, j].tolist()}
                   for j, t in enumerate(TARGETS)},
        'input_sha256': {'labels': digest(labels_path), 'baseline_oof': digest(baseline_oof_path),
                         'auxiliary_oof': digest(auxiliary_oof_path)},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--baseline-oof', type=Path, required=True)
    p.add_argument('--auxiliary-oof', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=1400)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.labels, args.baseline_oof, args.auxiliary_oof, args.bootstrap, args.seed)
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(f"macro delta {result['macro_delta']:+.6f}, CI95 {result['macro_delta_ci95']}")
    for t in TARGETS:
        d = result['targets'][t]
        print(f"  {t:18s} base={d['baseline_auc']:.4f} aux={d['auxiliary_auc']:.4f} "
              f"delta={d['delta']:+.4f} ci95={d['delta_ci95']}")


if __name__ == '__main__':
    main()
