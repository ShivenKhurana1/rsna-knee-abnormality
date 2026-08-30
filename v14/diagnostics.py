"""Keyed prediction diagnostics and paired, optionally grouped AUC confirmation.

No label fitting occurs here. Confidence intervals are not selection-adjusted;
evaluate one frozen candidate on a confirmation cohort not used for selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def align(frame, ids=None, labels=False):
    if frame.empty or not set([UID] + TARGETS).issubset(frame):
        raise ValueError('Empty table or missing study/target columns')
    if frame[UID].isna().any() or frame[UID].duplicated().any():
        raise ValueError('Missing or duplicate study IDs')
    if not frame[UID].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError('Study IDs must be nonempty strings; read CSV with dtype=str for IDs')
    if ids is not None:
        if len(ids) != len(set(ids)) or set(ids) != set(frame[UID]):
            raise ValueError('Study coverage mismatch; partial scoring is prohibited')
        frame = frame.set_index(UID).loc[list(ids)].reset_index()
    values = frame[TARGETS].to_numpy(float)
    good = np.isnan(values) | (values == 0) | (values == 1) if labels else (
        np.isfinite(values) & (values >= 0) & (values <= 1))
    if not good.all():
        raise ValueError('Labels must be binary or missing' if labels else 'Invalid probabilities/ranks')
    return frame[[UID] + TARGETS].copy()


def auc(y, p):
    valid = np.isfinite(y)
    y, p = y[valid], p[valid]
    pos, neg = np.sum(y == 1), np.sum(y == 0)
    if not pos or not neg:
        return np.nan
    return float((rankdata(p)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def aucs(y, p):
    return np.array([auc(y[:, j], p[:, j]) for j in range(len(TARGETS))])


def prediction_delta(baseline, candidate):
    baseline = align(baseline)
    candidate = align(candidate, baseline[UID])
    b, c = (x[TARGETS].to_numpy(float) for x in (baseline, candidate))
    rb, rc = (rankdata(x, axis=0) for x in (b, c))
    return {
        'studies': len(b), 'values_exactly_equal': bool(np.array_equal(b, c)),
        'all_target_rankings_equal': bool(np.array_equal(rb, rc)),
        'targets': {t: {
            'max_absolute_change': float(np.max(np.abs(c[:, j] - b[:, j]))),
            'mean_absolute_change': float(np.mean(np.abs(c[:, j] - b[:, j]))),
            'rank_changed_rows': int(np.sum(rb[:, j] != rc[:, j])),
            'baseline_unique_values': int(len(np.unique(b[:, j]))),
            'candidate_unique_values': int(len(np.unique(c[:, j]))),
        } for j, t in enumerate(TARGETS)},
        'interpretation': 'Identical ranks imply identical AUC on any fixed labels. Changed ranks do not imply improvement.',
    }


def require_changed_rankings(baseline, candidate):
    """A candidate must change same-study rankings; this is NOT an accuracy gate."""
    result = prediction_delta(baseline, candidate)
    if result['values_exactly_equal']:
        raise ValueError('Candidate is numerically identical to the control')
    if result['all_target_rankings_equal']:
        raise ValueError('Candidate values differ but all rankings are unchanged; AUC cannot improve')
    return result


def compare(labels, baseline, candidate, bootstrap=2000, seed=1400, groups=None):
    labels = align(labels, labels=True)
    baseline, candidate = (align(x, labels[UID]) for x in (baseline, candidate))
    y, b, c = (x[TARGETS].to_numpy(float) for x in (labels, baseline, candidate))
    ba, ca = aucs(y, b), aucs(y, c)
    if not np.isfinite(ba).all():
        raise ValueError('All 12 targets need positive and negative labels; no silent macro target dropping')
    if bootstrap < 100:
        raise ValueError('Use at least 100 paired bootstrap replicates')
    if groups is None:
        clusters = [np.array([i]) for i in range(len(y))]
    else:
        if len(groups) != len(y) or pd.isna(groups).any():
            raise ValueError('Group IDs must be aligned and nonmissing')
        codes, _ = pd.factorize(groups)
        clusters = [np.flatnonzero(codes == k) for k in np.unique(codes)]
    if len(clusters) < 20:
        raise ValueError('Fewer than 20 independent resampling units; collect more confirmation data')
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        idx = np.concatenate([clusters[k] for k in rng.integers(0, len(clusters), len(clusters))])
        d = aucs(y[idx], c[idx]) - aucs(y[idx], b[idx])
        if np.isfinite(d).all():
            draws.append(d)
    if len(draws) < .8 * bootstrap:
        raise ValueError('Too many resamples lack both classes; confirmation cohort is insufficient')
    draws = np.asarray(draws)
    ci = np.quantile(draws.mean(axis=1), [.025, .975])
    target_ci = np.quantile(draws, [.025, .975], axis=0)
    return {
        'scope': 'Supplied cohort only. Not hidden-test performance; not proof of training exclusion.',
        'baseline_macro_auc': float(ba.mean()), 'candidate_macro_auc': float(ca.mean()),
        'macro_delta': float((ca - ba).mean()), 'macro_delta_ci95': ci.tolist(),
        'point_gain_at_least_0_02': bool((ca - ba).mean() >= .02),
        'ci_lower_bound_at_least_0_02': bool(ci[0] >= .02),
        'bootstrap_unit': 'group' if groups is not None else 'study',
        'independent_units': len(clusters), 'valid_replicates': len(draws),
        'requested_replicates': bootstrap, 'seed': seed,
        'ci_caveat': 'Paired percentile intervals; degenerate replicates excluded; not corrected for candidate selection.',
        'targets': {t: {'baseline_auc': float(ba[j]), 'candidate_auc': float(ca[j]),
            'delta': float(ca[j] - ba[j]), 'delta_ci95': target_ci[:, j].tolist(),
            'positive': int(np.sum(y[:, j] == 1)), 'negative': int(np.sum(y[:, j] == 0)),
            'missing': int(np.isnan(y[:, j]).sum()),
            'perfect_target_max_macro_gain': float((1 - ba[j]) / len(TARGETS))}
            for j, t in enumerate(TARGETS)},
        'prediction_changes': prediction_delta(baseline, candidate),
    }


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--labels', type=Path)
    p.add_argument('--groups', type=Path, help='CSV with StudyInstanceUID,GroupID')
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args()
    frames = [pd.read_csv(x, dtype={UID: str}) for x in (args.baseline, args.candidate)]
    result = prediction_delta(*frames)
    if args.labels:
        labels = pd.read_csv(args.labels, dtype={UID: str})
        groups = None
        if args.groups:
            g = pd.read_csv(args.groups, dtype=str)
            if g[UID].duplicated().any() or set(g[UID]) != set(labels[UID]):
                raise ValueError('Group table coverage mismatch')
            groups = g.set_index(UID).loc[labels[UID], 'GroupID'].to_numpy()
        result = compare(labels, *frames, bootstrap=args.bootstrap, groups=groups)
    elif args.groups:
        raise ValueError('--groups requires --labels')
    result['sha256'] = {k: digest(v) for k in ('baseline', 'candidate', 'labels', 'groups')
                        if (v := getattr(args, k)) is not None}
    write_new(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'targets'}, indent=2))


if __name__ == '__main__':
    main()
