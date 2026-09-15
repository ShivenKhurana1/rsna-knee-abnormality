"""Paired OOF evaluation of adding Family A to a frozen existing ensemble.

This estimates development-cohort discrimination change. It never calls the
result an actual leaderboard gain: public/private transport remains unknown
until Kaggle scores a frozen submission.
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


def align(frame, ids, name):
    if UID not in frame or frame[UID].isna().any() or frame[UID].duplicated().any():
        raise ValueError(f'{name}: missing or duplicate study IDs')
    if set(frame[UID]) != set(ids):
        raise ValueError(f'{name}: coverage differs from labels; partial scoring is prohibited')
    values = frame.set_index(UID).loc[ids, TARGETS].to_numpy(float)
    if not (np.isfinite(values) & (values >= 0) & (values <= 1)).all():
        raise ValueError(f'{name}: predictions must be finite probabilities in [0,1]')
    return values


def labels_from_frame(frame, source_kind):
    if UID not in frame or frame[UID].isna().any() or frame[UID].duplicated().any():
        raise ValueError('labels: missing or duplicate study IDs')
    raw = frame[TARGETS].to_numpy(float)
    if source_kind == 'binary':
        if not (np.isnan(raw) | (raw == 0) | (raw == 1)).all():
            raise ValueError('binary labels must be 0, 1, or missing')
        return raw
    if source_kind == 'report-soft':
        if not (np.isfinite(raw) & (raw >= 0) & (raw <= 1)).all():
            raise ValueError('report-soft labels must be finite in [0,1]')
        return np.where(raw == 0.5, np.nan, (raw > 0.5).astype(float))
    raise ValueError("source_kind must be 'binary' or 'report-soft'")


def auc(y, prediction):
    valid = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[valid], prediction[valid]
    positive, negative = int((y == 1).sum()), int((y == 0).sum())
    if not positive or not negative:
        return np.nan
    ranks = rankdata(prediction)
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) /
                 (positive * negative))


def macro_auc(y, prediction):
    values = np.array([auc(y[:, j], prediction[:, j]) for j in range(len(TARGETS))])
    return float(np.nanmean(values)), values


def column_ranks(values):
    return np.column_stack([rankdata(values[:, j]) / len(values) for j in range(values.shape[1])])


def evaluate(labels_path, baseline_path, family_paths, source_kind='binary',
             weights=(0.10,), bootstrap=5000, seed=1500, public_baseline=0.935):
    labels = pd.read_csv(labels_path, dtype={UID: str})
    ids = labels[UID].tolist()
    y = labels_from_frame(labels, source_kind)
    baseline = align(pd.read_csv(baseline_path, dtype={UID: str}), ids, 'baseline')
    family_by_seed = [align(pd.read_csv(path, dtype={UID: str}), ids, f'family[{i}]')
                      for i, path in enumerate(family_paths)]
    if len(family_by_seed) < 2:
        raise ValueError('At least two independently seeded Family A OOF files are required')

    baseline_rank = column_ranks(baseline)
    family_mean_rank = column_ranks(np.mean(family_by_seed, axis=0))
    baseline_macro, baseline_targets = macro_auc(y, baseline_rank)
    if not np.isfinite(baseline_targets).all():
        raise ValueError('Every target needs both classes; silent target dropping is prohibited')
    rng = np.random.default_rng(seed)
    results = {}
    for weight in weights:
        if not 0 < weight < 1:
            raise ValueError('Every Family A blend weight must be strictly between 0 and 1')
        candidate = (1 - weight) * baseline_rank + weight * family_mean_rank
        candidate_macro, candidate_targets = macro_auc(y, candidate)
        if not np.isfinite(candidate_targets).all():
            raise ValueError('Candidate is not scorable on every target')
        draws = []
        for _ in range(bootstrap):
            idx = rng.integers(0, len(ids), len(ids))
            _, base_targets_draw = macro_auc(y[idx], baseline_rank[idx])
            _, candidate_targets_draw = macro_auc(y[idx], candidate[idx])
            if np.isfinite(base_targets_draw).all() and np.isfinite(candidate_targets_draw).all():
                draws.append(float((candidate_targets_draw - base_targets_draw).mean()))
        if len(draws) < bootstrap * 0.8:
            raise ValueError('Too many bootstrap replicates were unscorable')
        draws = np.asarray(draws)
        delta = candidate_macro - baseline_macro
        ci = np.quantile(draws, [.025, .975]).tolist()
        results[f'{weight:.6f}'] = {
            'family_weight': float(weight),
            'baseline_macro_auc': baseline_macro,
            'candidate_macro_auc': candidate_macro,
            'macro_delta': delta,
            'macro_delta_ci95': ci,
            'bootstrap_standard_error': float(draws.std(ddof=1)),
            'probability_better': float((draws > 0).mean()),
            'conditional_projected_public_score': public_baseline + delta,
            'conditional_projected_public_score_ci95': [public_baseline + ci[0],
                                                        public_baseline + ci[1]],
            'reaches_0_950_point_estimate': bool(public_baseline + delta >= 0.950),
            'reaches_0_950_ci_lower': bool(public_baseline + ci[0] >= 0.950),
            'targets': {target: {
                'baseline_auc': float(baseline_targets[j]),
                'candidate_auc': float(candidate_targets[j]),
                'delta': float(candidate_targets[j] - baseline_targets[j]),
            } for j, target in enumerate(TARGETS)},
        }

    primary = results[f'{float(weights[0]):.6f}']
    seed_deltas = []
    for prediction in family_by_seed:
        candidate = (1 - weights[0]) * baseline_rank + weights[0] * column_ranks(prediction)
        candidate_macro, _ = macro_auc(y, candidate)
        seed_deltas.append(candidate_macro - baseline_macro)
    seed_sd = float(np.std(seed_deltas, ddof=1))
    seed_se_mean = seed_sd / np.sqrt(len(seed_deltas))
    bootstrap_se = primary['bootstrap_standard_error']
    combined_se = float(np.sqrt(bootstrap_se ** 2 + seed_se_mean ** 2))
    combined_delta_ci = [primary['macro_delta'] - 1.96 * combined_se,
                         primary['macro_delta'] + 1.96 * combined_se]
    rounding_half_unit = 0.0005
    return {
        'scope': ('Paired OOF development estimate for a frozen rank blend. Conditional public-score '
                  'projections assume one-for-one transfer and are not actual leaderboard scores.'),
        'source_kind': source_kind,
        'cohort_studies': len(ids),
        'public_baseline_recorded': public_baseline,
        'gain_required_for_0_950': 0.950 - public_baseline,
        'primary_weight': float(weights[0]),
        'primary_result': primary,
        'all_predeclared_weights': results,
        'primary_seed_deltas': seed_deltas,
        'primary_seed_delta_sd_descriptive': seed_sd,
        'error_propagation': {
            'formula': ('SE_quantified ≈ sqrt(SE_paired_bootstrap^2 + '
                        '(SD_training_seed/sqrt(n_seeds))^2). This omits transport shift.'),
            'paired_bootstrap_se': bootstrap_se,
            'training_seed_se_of_mean_descriptive': float(seed_se_mean),
            'combined_se_approx': combined_se,
            'combined_delta_ci95_normal_approx': combined_delta_ci,
            'conditional_public_score_ci95_normal_approx':
                [public_baseline + combined_delta_ci[0], public_baseline + combined_delta_ci[1]],
            'conditional_public_score_ci95_including_baseline_display_rounding':
                [public_baseline - rounding_half_unit + combined_delta_ci[0],
                 public_baseline + rounding_half_unit + combined_delta_ci[1]],
            'leaderboard_rounding': {
                'assumption': 'scores displayed to nearest 0.001',
                'baseline_0_935_latent_interval': [0.9345, 0.9355],
                'candidate_0_950_latent_interval': [0.9495, 0.9505],
                'displayed_gain_0_015_latent_interval': [0.0140, 0.0160],
                'latent_score_at_least_0_950_requires_displayed': 0.951,
            },
            'warning': ('The numerical interval is not a leaderboard prediction interval because '
                        'validation-to-public transport and public/private shift are unquantified.'),
        },
        'error_budget': {
            'cohort_sampling': 'quantified by paired study bootstrap',
            'training_seed': ('descriptive SD only; two seeds are insufficient for a reliable '
                              'variance component'),
            'policy_selection': ('not adjusted if the 58 gold cases selected the auxiliary policy'),
            'validation_to_public_transport': 'unquantified',
            'public_to_private_shift': 'unquantified',
            'actual_proof': 'requires a scored frozen Kaggle submission at or above 0.950',
        },
        'input_sha256': {
            'labels': digest(labels_path), 'baseline': digest(baseline_path),
            'family_by_seed': [digest(path) for path in family_paths],
        },
        'bootstrap_seed': seed,
        'bootstrap_requested_replicates': bootstrap,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--baseline-oof', type=Path, required=True)
    parser.add_argument('--family-oof', type=Path, nargs='+', required=True)
    parser.add_argument('--source-kind', choices=['binary', 'report-soft'], default='binary')
    parser.add_argument('--weights', type=float, nargs='+', default=[0.10, 0.05, 0.15])
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=1500)
    parser.add_argument('--public-baseline', type=float, default=0.935)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.labels, args.baseline_oof, args.family_oof, args.source_kind,
                      tuple(args.weights), args.bootstrap, args.seed, args.public_baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    primary = result['primary_result']
    print(f"primary blend delta {primary['macro_delta']:+.6f}, "
          f"CI95 {primary['macro_delta_ci95']}")
    print(f"conditional public score {primary['conditional_projected_public_score']:.6f}; "
          "actual leaderboard transfer remains unmeasured")


if __name__ == '__main__':
    main()
