"""Train target-specific residual heads on supplied frozen MRI features.

Adds image information to a baseline, not just calibration/blend weights. This
is a feature-level research candidate: no feature extractor or trained weights
are supplied. Baseline and features must themselves be genuinely held out.
Missing labels are masked. Scaling and heads are fitted within each group fold.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold

from diagnostics import UID, TARGETS, align, digest, write_new


def fit(x, baseline, y, regularization=1.0):
    x, baseline, y = map(lambda a: np.asarray(a, dtype=float), (x, baseline, y))
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Features must be nonempty and finite')
    if baseline.shape != (len(x), 12) or y.shape != baseline.shape:
        raise ValueError('Baseline, label and feature shapes differ')
    if not (np.isfinite(baseline) & (baseline >= 0) & (baseline <= 1)).all():
        raise ValueError('Invalid baseline')
    if not (np.isnan(y) | (y == 0) | (y == 1)).all() or regularization <= 0:
        raise ValueError('Expected binary/missing labels and positive regularization')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.)
    z = (x - mean) / scale
    offset = logit(np.clip(baseline, 1e-5, 1 - 1e-5))
    weights, intercept, supported = np.zeros((x.shape[1], 12)), np.zeros(12), np.zeros(12, bool)
    for j in range(12):
        mask = np.isfinite(y[:, j])
        if np.sum(y[mask, j] == 1) < 5 or np.sum(y[mask, j] == 0) < 5:
            continue  # no evidence: exact baseline fallback for this target
        a, t, o = z[mask], y[mask, j], offset[mask, j]

        def objective(theta):
            w, bias = theta[:-1], theta[-1]
            logits = o + a @ w + bias
            loss = np.mean(np.logaddexp(0, logits) - t * logits) + regularization * (w @ w) / 2
            error = (expit(logits) - t) / len(t)
            grad = np.r_[a.T @ error + regularization * w, error.sum()]
            return loss, grad

        result = minimize(objective, np.zeros(x.shape[1] + 1), method='L-BFGS-B', jac=True,
                          options={'maxiter': 400, 'ftol': 1e-10})
        if not result.success:
            raise RuntimeError(f'{TARGETS[j]} fit did not converge: {result.message}')
        weights[:, j], intercept[j], supported[j] = result.x[:-1], result.x[-1], True
    return {'mean': mean, 'scale': scale, 'weights': weights, 'intercept': intercept,
            'supported': supported, 'regularization': np.array(regularization),
            'targets': np.array(TARGETS)}


def predict(model, x, baseline):
    x, baseline = np.asarray(x, float), np.asarray(baseline, float)
    if x.ndim != 2 or x.shape[1] != len(model['mean']) or baseline.shape != (len(x), 12):
        raise ValueError('Inference shapes do not match fitted head')
    if list(model['targets']) != TARGETS or not np.isfinite(x).all():
        raise ValueError('Feature/target contract mismatch')
    if not (np.isfinite(baseline) & (baseline >= 0) & (baseline <= 1)).all():
        raise ValueError('Invalid baseline')
    residual = ((x - model['mean']) / model['scale']) @ model['weights'] + model['intercept']
    pred = expit(logit(np.clip(baseline, 1e-5, 1 - 1e-5)) + residual)
    pred[:, ~model['supported']] = baseline[:, ~model['supported']]
    return pred


def crossfit(x, baseline, y, groups, folds=5, regularization=1.0):
    if len(groups) != len(x) or pd.isna(groups).any() or len(set(groups)) < folds or folds < 2:
        raise ValueError('Need nonmissing, aligned groups and enough distinct groups')
    out, assignment = np.empty_like(baseline, dtype=float), np.full(len(x), -1)
    for k, (train, valid) in enumerate(GroupKFold(folds).split(x, groups=groups)):
        head = fit(x[train], baseline[train], y[train], regularization)
        out[valid] = predict(head, x[valid], baseline[valid])
        assignment[valid] = k
    return out, assignment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'baseline', 'labels', 'groups', 'out'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--regularization', type=float, default=1.0)
    p.add_argument('--folds', type=int, default=5)
    args = p.parse_args()
    labels = align(pd.read_csv(args.labels, dtype={UID: str}), labels=True)
    base = align(pd.read_csv(args.baseline, dtype={UID: str}), labels[UID])
    with np.load(args.features, allow_pickle=False) as f:
        ids, raw = f[UID].astype(str), f['features']
    if raw.ndim != 2 or len(raw) != len(ids) or len(set(ids)) != len(ids) or set(ids) != set(labels[UID]):
        raise ValueError('Feature IDs/shape do not cover the label table exactly')
    index = {uid: i for i, uid in enumerate(ids)}
    x = raw[[index[uid] for uid in labels[UID]]].astype(float)
    g = pd.read_csv(args.groups, dtype=str)
    if g[UID].duplicated().any() or set(g[UID]) != set(labels[UID]):
        raise ValueError('Group table coverage mismatch')
    groups = g.set_index(UID).loc[labels[UID], 'GroupID'].to_numpy()
    args.out.mkdir(parents=True, exist_ok=False)
    b, y = base[TARGETS].to_numpy(float), labels[TARGETS].to_numpy(float)
    pred, folds = crossfit(x, b, y, groups, args.folds, args.regularization)
    frame = pd.DataFrame(pred, columns=TARGETS)
    frame.insert(0, UID, labels[UID].to_numpy())
    frame.to_csv(args.out / 'specialist_oof.csv', index=False)
    pd.DataFrame({UID: labels[UID], 'GroupID': groups, 'fold': folds}).to_csv(args.out / 'folds.csv', index=False)
    model = fit(x, b, y, args.regularization)
    np.savez_compressed(args.out / 'residual_head.npz', **model)
    write_new(args.out / 'training_receipt.json', {
        'status': 'UNCONFIRMED_RESEARCH_MODEL', 'measured_kaggle_auc': None,
        'regularization': args.regularization, 'folds': args.folds,
        'training_ids': labels[UID].tolist(), 'training_groups': sorted(set(groups)),
        'input_sha256': {k: digest(getattr(args, k)) for k in ('features', 'baseline', 'labels', 'groups')},
        'model_sha256': digest(args.out / 'residual_head.npz'),
        'caveat': 'Head OOF only. Encoder and baseline training exclusion must be independently verified. Not a deployment approval.',
    })
    print('Saved cross-fitted research predictions and final head; no accuracy improvement claimed.')


if __name__ == '__main__':
    main()
