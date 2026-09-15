"""Train the residual-specialist head on genuine frozen DINOv2 image features (not the
pipeline's own fused predictions -- that was the flaw diagnostics.py's first attempt
exposed) and validate it on the untouched 58 gold studies.

Training rows use a neutral 0.5 baseline, not V13's own output on those studies: the
pool studies were very likely in the CoAtNets' own training set, so their existing
predictions cannot be trusted as a training signal. The learned weights are offset-
invariant, so applying them at predict time against gold's REAL, unleaked V13 baseline
is the correct use of the same head.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from diagnostics import TARGETS, UID, align, compare, require_changed_rankings, write_new
from residual_specialist import crossfit, fit, predict


def load_npz_features(path, ids):
    with np.load(path, allow_pickle=False) as f:
        raw_ids, raw = f[UID].astype(str), f['features']
    index = {u: i for i, u in enumerate(raw_ids)}
    missing = [u for u in ids if u not in index]
    if missing:
        raise ValueError(f'{len(missing)} requested study ids missing from {path}, e.g. {missing[:3]}')
    return raw[[index[u] for u in ids]].astype(float)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool-features', type=Path, required=True)
    p.add_argument('--pseudo-labels', type=Path, required=True)
    p.add_argument('--gold-features', type=Path, required=True)
    p.add_argument('--gold-labels', type=Path, required=True)
    p.add_argument('--gold-baseline', type=Path, required=True, help='real, unleaked V13 output on the gold studies')
    p.add_argument('--regularization', type=float, default=1.0)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()

    with np.load(args.pool_features, allow_pickle=False) as f:
        pool_ids = list(f[UID].astype(str))
    x_pool = load_npz_features(args.pool_features, pool_ids)

    raw_pseudo = pd.read_csv(args.pseudo_labels, dtype={UID: str})
    pseudo = align(raw_pseudo.loc[raw_pseudo[UID].isin(pool_ids)].copy(), pool_ids, labels=True)
    y_pool = pseudo[TARGETS].to_numpy(float)
    baseline_pool = np.full_like(y_pool, 0.5)
    groups = np.array(pool_ids)

    oof, _ = crossfit(x_pool, baseline_pool, y_pool, groups, folds=args.folds, regularization=args.regularization)
    oof_auc = {}
    for j, t in enumerate(TARGETS):
        mask = np.isfinite(y_pool[:, j])
        pos, neg = (y_pool[mask, j] == 1).sum(), (y_pool[mask, j] == 0).sum()
        if pos < 5 or neg < 5:
            oof_auc[t] = None
            continue
        r = rankdata(oof[mask, j])
        oof_auc[t] = float((r[y_pool[mask, j] == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))

    head = fit(x_pool, baseline_pool, y_pool, args.regularization)

    with np.load(args.gold_features, allow_pickle=False) as f:
        gold_ids = list(f[UID].astype(str))
    x_gold = load_npz_features(args.gold_features, gold_ids)
    baseline_gold = align(pd.read_csv(args.gold_baseline, dtype={UID: str}), gold_ids)
    b_gold = baseline_gold[TARGETS].to_numpy(float)
    candidate_values = predict(head, x_gold, b_gold)
    candidate = pd.DataFrame(candidate_values, columns=TARGETS)
    candidate.insert(0, UID, gold_ids)

    labels = pd.read_csv(args.gold_labels, dtype={UID: str})
    try:
        require_changed_rankings(baseline_gold, candidate)
        gate = 'PASS_CHANGED_RANKINGS'
    except ValueError as exc:
        gate = 'FAIL: ' + str(exc)
    result = compare(labels, baseline_gold, candidate, bootstrap=2000, seed=1400)
    result['output_difference_gate'] = gate
    result['pool_studies'] = len(pool_ids)
    result['pool_supported_targets'] = [t for j, t in enumerate(TARGETS) if head['supported'][j]]
    result['pool_oof_auc_vs_pseudo_labels'] = oof_auc
    result['feature_source'] = 'frozen DINOv2-small pooled (CLS+mean-token) features, 768-dim'
    result['scope'] = ('58 gold studies, real expert labels. Pool baseline was a neutral 0.5, never '
                       'V13\'s own leaked output. This is a genuine held-out evaluation of the head.')
    write_new(args.report, result)
    print(gate)
    print(f"macro delta {result['macro_delta']:+.6f}, CI95 {result['macro_delta_ci95']}")
    print(f"baseline macro AUC {result['baseline_macro_auc']:.4f} -> candidate {result['candidate_macro_auc']:.4f}")
    for t in TARGETS:
        d = result['targets'][t]
        print(f"  {t:18s} base={d['baseline_auc']:.4f} cand={d['candidate_auc']:.4f} delta={d['delta']:+.4f}"
              f" oof_vs_pseudo={oof_auc[t]}")


if __name__ == '__main__':
    main()
