import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from .common import check_ids, fingerprint


def auc(y, p):
    observed = np.isfinite(y)
    y, p = y[observed], p[observed]
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if not pos or not neg:
        return None
    return float((rankdata(p)[y == 1].sum()-pos*(pos+1)/2)/(pos*neg))


def metrics(y, p, targets):
    if y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError('Invalid metric matrix')
    values = [auc(y[:, j], p[:, j]) for j in range(y.shape[1])]
    return {'macro_auc': float(np.mean(values)) if all(v is not None for v in values) else None,
            'targets': {t: {'auc': values[j], 'positive': int((y[:, j] == 1).sum()),
                             'negative': int((y[:, j] == 0).sum()), 'unknown': int(np.isnan(y[:, j]).sum())}
                        for j, t in enumerate(targets)},
            'all_targets_scorable': all(v is not None for v in values)}


def load_groups(path, ids, id_col):
    if path is None:
        return np.asarray(ids), 'Study-only grouping; patient separation is unverified'
    frame = pd.read_csv(path, dtype=str)
    actual = check_ids(frame, id_col)
    if set(actual) != set(ids) or 'group_id' not in frame or frame.group_id.isna().any() or frame.group_id.str.strip().eq('').any():
        raise ValueError('Groups CSV must cover all studies with nonempty group_id')
    return frame.set_index(id_col).loc[ids, 'group_id'].to_numpy(), 'User-supplied patient/duplicate grouping; verify identity provenance'


def make_folds(ids, y, groups, k, seed):
    """Group-preserving greedy multilabel count allocation, with support audits."""
    if not (np.isnan(y) | (y == 0) | (y == 1)).all():
        raise ValueError('Expert targets must be binary or missing')
    groups = np.asarray(groups)
    names, inverse = np.unique(groups, return_inverse=True)
    if len(names) < k:
        raise ValueError('Fewer independent groups than folds')
    features = np.c_[y == 1, y == 0, np.isfinite(y).any(1), np.ones(len(y))].astype(float)
    counts = np.stack([features[inverse == i].sum(0) for i in range(len(names))])
    totals = counts.sum(0)
    scale = np.maximum(totals/k, 1)
    active = totals > 0
    rng = np.random.default_rng(seed)
    ties = rng.random(len(names))
    rarity = ((counts[:, :y.shape[1]] > 0)/np.maximum(totals[:y.shape[1]], 1)).sum(1)
    order = sorted(range(len(names)), key=lambda i: (-rarity[i], -counts[i, -1], ties[i]))
    allocation = np.zeros((k, counts.shape[1]))
    assignment = np.full(len(names), -1)
    tie_order = rng.permutation(k).tolist()
    for step, g in enumerate(order):
        candidates = [tie_order[step]] if step < k else tie_order
        def cost(f):
            before = ((allocation[f]-totals/k)/scale)**2
            after = ((allocation[f]+counts[g]-totals/k)/scale)**2
            return (after-before)[active].sum()
        f = min(candidates, key=cost)
        allocation[f] += counts[g]
        assignment[g] = f
    fold = assignment[inverse]
    audit = []
    for f in range(k):
        tr, va = fold != f, fold == f
        assert not set(groups[tr]) & set(groups[va])
        audit.append({'fold': f, 'train_studies': int(tr.sum()), 'val_studies': int(va.sum()),
                      'positive': (y[va] == 1).sum(0).tolist(), 'negative': (y[va] == 0).sum(0).tolist(),
                      'train_scorable': ((y[tr] == 1).any(0) & (y[tr] == 0).any(0)).tolist(),
                      'val_scorable': ((y[va] == 1).any(0) & (y[va] == 0).any(0)).tolist()})
    return fold, {'method': 'group-preserving multilabel greedy allocation; approximate balance',
                  'k': k, 'seed': seed, 'audit': audit,
                  'assignment_hash': fingerprint(dict(zip(ids, fold.tolist())))}


def supervision(y, report, groups, train_idx, cfg, targets, ids):
    """Calibrators and raw-source transfer gate see training-fold expert cells ONLY."""
    expert_mask = np.isfinite(y)
    addressed = np.isfinite(report) & (report != cfg['silent_value'])
    aux_mask = addressed & ~expert_mask
    weights = np.zeros(y.shape[1], np.float32)
    calibrated = report.copy()
    policies = {}
    train_mask = np.zeros(len(y), bool)
    train_mask[train_idx] = True
    for j, target in enumerate(targets):
        selected = np.flatnonzero(train_mask & expert_mask[:, j] & addressed[:, j])
        truth, raw = y[selected, j], report[selected, j]
        policy = {'fit_ids': [ids[i] for i in selected], 'weight': 0., 'calibrator': 'identity',
                  'reason': 'disabled' if cfg['auxiliary'] == 'none' else 'insufficient_support'}
        policies[target] = policy
        if cfg['auxiliary'] == 'none' or min((truth == 1).sum(), (truth == 0).sum()) < 3:
            continue
        unique = np.unique(groups[selected])
        by_group = [np.flatnonzero(groups[selected] == g) for g in unique]
        rng = np.random.default_rng(cfg['fold_seed'] + j)
        draws = []
        for _ in range(cfg['policy_bootstrap']):
            sample = np.concatenate([by_group[g] for g in rng.integers(len(unique), size=len(unique))])
            value = auc(truth[sample], raw[sample])
            if value is not None:
                draws.append(value)
        if len(draws) < .8 * cfg['policy_bootstrap']:
            policy['reason'] = 'unreliable_group_bootstrap'
            continue
        lo = float(np.quantile(draws, .025))
        prevalence = truth.mean()
        skill = 1 - float(np.mean((raw-truth)**2)) / float(prevalence*(1-prevalence))
        policy.update({'auc_lower95': lo, 'raw_brier_skill': skill, 'fit_group_count': len(unique)})
        if lo > .5 and skill > 0:
            weight = min(skill, cfg['aux_cap'])
            support = min(len(np.unique(groups[selected][truth == 1])), len(np.unique(groups[selected][truth == 0])))
            if support < 10:
                weight *= .5
            weights[j] = weight
            policy.update({'weight': float(weight), 'reason': 'raw_source_gate_passed'})
        else:
            policy['reason'] = 'raw_source_gate_failed'
        if cfg['auxiliary'] == 'platt':
            z = np.log(np.clip(raw, 1e-4, 1-1e-4)/(1-np.clip(raw, 1e-4, 1-1e-4)))
            try:
                model = LogisticRegression(C=1., solver='lbfgs', max_iter=1000).fit(z[:, None], truth)
                indices = np.flatnonzero(addressed[:, j])
                p = np.clip(report[indices, j], 1e-4, 1-1e-4)
                calibrated[indices, j] = model.predict_proba(np.log(p/(1-p))[:, None])[:, 1]
                policy.update({'calibrator': 'platt', 'coef': float(model.coef_[0, 0]),
                               'intercept': float(model.intercept_[0])})
            except (ValueError, FloatingPointError) as exc:
                policy['calibration_error'] = str(exc)
    aux_mask &= weights[None] > 0
    # Do not turn the raw-policy gate into an in-sample calibrated gate.
    return calibrated.astype(np.float32), aux_mask, weights, policies
