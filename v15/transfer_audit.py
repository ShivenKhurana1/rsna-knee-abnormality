"""Stage L0 target-transfer audit: does report-derived soft evidence rank-order
expert image labels well enough to justify capped auxiliary supervision?

AUC answers a ranking question only -- it is not an agreement rate, not an error
rate, and does not by itself certify calibration. This script deliberately keeps
those questions separate:

  * auc_all_retain_silent   -- AUC treating every study's raw soft value as the
                               prediction, including report-silent 0.5 cells.
  * auc_addressed_only      -- AUC restricted to studies where the report source
                               actually addressed the target (value != 0.5).
  * silent_expert_prevalence -- among studies the report was silent on, what
                               fraction are expert-positive. This tests whether
                               silence behaves like an implicit negative; it does
                               NOT establish that silence *causes* any AUC gap
                               elsewhere, since silent and addressed cases can
                               differ in other ways too.
  * brier_addressed_only    -- mean squared error of the addressed soft values
                               against expert truth: a calibration diagnostic,
                               not a ranking one.

Every AUC is bootstrap-resampled at the study level with a percentile CI, and
every count is reported so wide intervals from small positive/negative counts
are visible rather than hidden behind a single point estimate. This produces
evidence for Stage L0 of the V15 label-supervision gate; it does not itself
decide which targets are promoted to training -- that call also needs the
grouped-fold validation the plan requires before any target/source pair is
used as an auxiliary loss.
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
MIN_CLASS_COUNT_FOR_AUC = 3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_unique_ids(frame, name):
    if frame[UID].isna().any():
        raise ValueError(f'{name}: missing study IDs')
    if frame[UID].duplicated().any():
        dupes = frame.loc[frame[UID].duplicated(), UID].unique()[:3].tolist()
        raise ValueError(f'{name}: duplicate study IDs, e.g. {dupes}')


def load_aligned(gold_path, source_path):
    gold = pd.read_csv(gold_path, dtype={UID: str})
    source = pd.read_csv(source_path, dtype={UID: str})
    require_unique_ids(gold, 'gold labels')
    require_unique_ids(source, 'report source')
    if not set(TARGETS).issubset(gold.columns) or not set(TARGETS).issubset(source.columns):
        raise ValueError('Both tables must carry all twelve targets')
    missing = set(gold[UID]) - set(source[UID])
    if missing:
        raise ValueError(f'{len(missing)} gold studies absent from report source, e.g. {sorted(missing)[:3]}')
    gold = gold.set_index(UID)
    source = source.set_index(UID).loc[gold.index]
    y = gold[TARGETS].to_numpy(float)
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError('Gold labels must be strictly binary; this audit requires a fully labeled cohort')
    p = source[TARGETS].to_numpy(float)
    if not (np.isfinite(p) & (p >= 0) & (p <= 1)).all():
        raise ValueError('Report-derived values must be finite probabilities in [0, 1]')
    return gold.index.to_numpy(), y, p


def auc(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    valid = np.isfinite(y)
    y, p = y[valid], p[valid]
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos < MIN_CLASS_COUNT_FOR_AUC or neg < MIN_CLASS_COUNT_FOR_AUC:
        return float('nan')
    return float((rankdata(p)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def brier(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    valid = np.isfinite(y)
    if not valid.any():
        return None
    return float(np.mean((p[valid] - y[valid]) ** 2))


def brier_skill(y, p):
    """Brier score is not comparable across targets of differing prevalence: a
    target that is 90% negative scores well by predicting 0.1 everywhere. Report
    skill relative to the prevalence-only baseline (always predict the base rate)
    instead of the raw score, so 0 means no better than knowing prevalence alone
    and 1 means perfect."""
    y = np.asarray(y, float)
    valid = np.isfinite(y)
    if not valid.any():
        return {'brier': None, 'prevalence_baseline_brier': None, 'skill_score': None}
    y = y[valid]
    prevalence = float(y.mean())
    baseline_brier = float(np.mean((prevalence - y) ** 2))
    model_brier = brier(y, p[valid] if hasattr(p, '__len__') else p)
    if baseline_brier <= 1e-12:
        return {'brier': model_brier, 'prevalence_baseline_brier': baseline_brier, 'skill_score': None}
    return {'brier': model_brier, 'prevalence_baseline_brier': baseline_brier,
            'skill_score': float(1 - model_brier / baseline_brier)}


def bootstrap_auc(y, p, n=2000, seed=1400):
    y, p = np.asarray(y, float), np.asarray(p, float)
    point = auc(y, p)
    if not np.isfinite(point):
        return {'point': None, 'ci95': None, 'valid_replicates': 0, 'requested_replicates': n,
                'note': 'fewer than %d cases in one class; AUC undefined' % MIN_CLASS_COUNT_FOR_AUC}
    rng = np.random.default_rng(seed)
    n_obs = len(y)
    draws = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        a = auc(y[idx], p[idx])
        if np.isfinite(a):
            draws.append(a)
    result = {'point': point, 'valid_replicates': len(draws), 'requested_replicates': n}
    if len(draws) < 0.5 * n:
        result['ci95'] = None
        result['note'] = 'fewer than half the resamples had both classes; CI unreliable at this n'
    else:
        result['ci95'] = np.quantile(draws, [.025, .975]).tolist()
    return result


def audit_target(y, p, silent_value=0.5, bootstrap=2000, seed=1400):
    addressed = p != silent_value
    pos_addr = int((y[addressed] == 1).sum())
    neg_addr = int((y[addressed] == 0).sum())
    pos_silent = int((y[~addressed] == 1).sum())
    neg_silent = int((y[~addressed] == 0).sum())
    return {
        'n_total': int(len(y)),
        'n_addressed': int(addressed.sum()),
        'n_silent': int((~addressed).sum()),
        'addressed_positive': pos_addr, 'addressed_negative': neg_addr,
        'silent_positive': pos_silent, 'silent_negative': neg_silent,
        'addressed_expert_prevalence': (pos_addr / (pos_addr + neg_addr)) if (pos_addr + neg_addr) else None,
        'silent_expert_prevalence': (pos_silent / (pos_silent + neg_silent)) if (pos_silent + neg_silent) else None,
        'auc_all_retain_silent': bootstrap_auc(y, p, bootstrap, seed),
        'auc_addressed_only': bootstrap_auc(y[addressed], p[addressed], bootstrap, seed),
        'brier_addressed_only': brier_skill(y[addressed], p[addressed]),
        'small_sample_warning': min(pos_addr, neg_addr) < 10,
    }


def run(gold_path, source_path, bootstrap=2000, seed=1400):
    ids, y, p = load_aligned(gold_path, source_path)
    targets = {t: audit_target(y[:, j], p[:, j], bootstrap=bootstrap, seed=seed) for j, t in enumerate(TARGETS)}
    return {
        'scope': ('Stage L0 transfer audit. AUC is a ranking diagnostic, not an agreement or '
                  'error rate. Point estimates from <=58 studies carry wide uncertainty; treat '
                  'ci95=None as "cannot be estimated reliably at this n", not as zero evidence. '
                  'Brier is reported only as a skill score against the per-target prevalence '
                  'baseline, since raw Brier is not comparable across targets of differing '
                  'prevalence. NON-INDEPENDENCE: if this audit\'s own output is used to choose '
                  'which target/source pairs get auxiliary-loss weight, any later evaluation of '
                  'that policy on this same 58-study cohort is exploratory, not independent '
                  'confirmation -- the cohort selected the policy it is then used to score.'),
        'cohort_studies': int(len(ids)),
        'input_sha256': {'gold_labels': digest(gold_path), 'report_source': digest(source_path)},
        'bootstrap_seed': seed, 'bootstrap_requested_replicates': bootstrap,
        'targets': targets,
        'decision_note': ('This audit alone does not select target/source pairs for training. '
                          'Promotion additionally requires the grouped-fold auxiliary-loss '
                          'ablation in Run 4 of the V15 plan to show a cross-fitted expert-label '
                          'gain, not just a favorable audit AUC.'),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gold-labels', type=Path, required=True)
    p.add_argument('--report-source', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=1400)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = run(args.gold_labels, args.report_source, args.bootstrap, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    header = f'{"target":18s} {"n_addr":>6s} {"pos/neg":>9s} {"auc_addr":>10s} {"ci95":>22s} {"brier_skill":>11s} {"small_n":>7s}'
    print(header)
    for t in TARGETS:
        d = result['targets'][t]
        a = d['auc_addressed_only']
        ci = f"[{a['ci95'][0]:.3f},{a['ci95'][1]:.3f}]" if a['ci95'] else 'n/a'
        auc_str = f"{a['point']:.4f}" if a['point'] is not None else 'n/a'
        skill = d['brier_addressed_only']['skill_score']
        brier_str = f"{skill:.4f}" if skill is not None else 'n/a'
        print(f"{t:18s} {d['n_addressed']:6d} {d['addressed_positive']}/{d['addressed_negative']:>7d} "
              f"{auc_str:>10s} {ci:>22s} {brier_str:>11s} {'yes' if d['small_sample_warning'] else 'no':>7s}")


if __name__ == '__main__':
    main()
