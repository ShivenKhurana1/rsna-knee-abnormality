"""Label construction: expert gold labels are the primary objective; a capped,
evidence-gated auxiliary target from report-derived soft labels is optional.

Corrections applied per review, encoded here rather than left as prose policy:

  * Report-silent cells (raw value == 0.5) are EXCLUDED from the auxiliary loss
    (mask weight 0), never filled in as a hard negative. "Masked" and "filled as
    negative" are different treatments; Run 4 uses the former exclusively.
  * A target gets nonzero auxiliary weight only if its transfer_audit.py
    addressed-only AUC bootstrap CI lower bound exceeds 0.5 AND its Brier skill
    score (vs. the prevalence-only baseline) is positive. AUC alone is not
    sufficient: MCL has AUC 0.958 but skill only 0.11 (near-chance calibration
    despite near-perfect ranking); Lateral OA/Synovitis/Contusion have AUC CI
    lower bounds above 0.5 but NEGATIVE skill and are excluded despite passing
    on AUC alone.
  * A target flagged small_sample_warning (fewer than 10 studies in the smaller
    addressed class) has its weight halved rather than trusted at face value.
  * This policy is derived from the same 58 gold studies Run 4 later scores
    against. Any Run 4 result on that same 58-study cohort is exploratory, not
    independent confirmation of the policy that selected it.
"""
import json
from pathlib import Path

import numpy as np

from contract import TARGETS, UID

MAX_AUX_WEIGHT = 0.3
SMALL_SAMPLE_WEIGHT_MULTIPLIER = 0.5
SILENT_VALUE = 0.5


def load_transfer_policy(transfer_audit_report_path):
    """Derive {target: {'weight': float, 'use_aux': bool, 'reason': str}} from a
    transfer_audit.py report. Every decision carries its numeric justification
    so the policy is auditable, not a hand-picked list."""
    report = json.loads(Path(transfer_audit_report_path).read_text(encoding='utf-8'))
    return policy_from_audit(report)


def policy_from_audit(report):
    """Convert an in-memory transfer audit to the capped auxiliary policy."""
    policy = {}
    for t in TARGETS:
        d = report['targets'][t]
        auc_ci = d['auc_addressed_only']['ci95']
        skill = d['brier_addressed_only']['skill_score']
        if auc_ci is None or skill is None:
            policy[t] = {'weight': 0.0, 'use_aux': False,
                         'reason': 'insufficient evidence: AUC CI or Brier skill unavailable at this n'}
            continue
        ci_lower = auc_ci[0]
        if ci_lower <= 0.5 or skill <= 0.0:
            policy[t] = {'weight': 0.0, 'use_aux': False,
                         'reason': f'gate failed: AUC CI lower={ci_lower:.3f}, skill={skill:.3f}'}
            continue
        weight = min(skill, MAX_AUX_WEIGHT)
        halved = bool(d['small_sample_warning'])
        if halved:
            weight *= SMALL_SAMPLE_WEIGHT_MULTIPLIER
        policy[t] = {'weight': float(weight), 'use_aux': True,
                     'reason': f'gate passed: AUC CI lower={ci_lower:.3f}, skill={skill:.3f}'
                               + (', weight halved (small sample)' if halved else '')}
    return policy


def crossfit_transfer_policy(ids, train_idx, gold_labels, report_source,
                             bootstrap=1000, seed=1400):
    """Derive a policy using training-fold expert cases only.

    This closes the policy-selection leak that occurs when a policy selected on
    all 58 gold cases is evaluated on OOF predictions for those same cases.
    """
    try:
        from transfer_audit import audit_target
    except ModuleNotFoundError:  # local test invocation: transfer_audit.py lives in the parent v15/ dir
        import sys
        from pathlib import Path
        parent = str(Path(__file__).resolve().parent.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from transfer_audit import audit_target

    ids = np.asarray(ids, dtype=str)
    train_ids = set(ids[np.asarray(train_idx, dtype=int)])
    gold = gold_labels.set_index(UID) if UID in gold_labels.columns else gold_labels
    report = report_source.set_index(UID) if UID in report_source.columns else report_source
    if gold.index.duplicated().any() or report.index.duplicated().any():
        raise ValueError('Duplicate study IDs cannot be used for cross-fitted policy selection')
    selected = [uid for uid in gold.index.astype(str) if uid in train_ids]
    if not selected:
        raise ValueError('No training-fold expert cases available for policy selection')
    if not set(selected).issubset(set(report.index.astype(str))):
        raise ValueError('Training-fold expert cases are missing from report source')
    y = gold.loc[selected, TARGETS].to_numpy(float)
    p = report.loc[selected, TARGETS].to_numpy(float)
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError('Cross-fitted policy requires fully binary expert labels')
    targets = {t: audit_target(y[:, j], p[:, j], bootstrap=bootstrap,
                               seed=seed + j)
               for j, t in enumerate(TARGETS)}
    audit = {'cohort_studies': len(selected), 'bootstrap': bootstrap,
             'seed': seed, 'targets': targets}
    return policy_from_audit(audit), audit


def zero_policy():
    """The baseline-arm policy: auxiliary loss disabled for every target."""
    return {t: {'weight': 0.0, 'use_aux': False, 'reason': 'baseline arm: auxiliary loss disabled'}
            for t in TARGETS}


def build_supervision(ids, gold_labels, report_source, policy, silent_value=SILENT_VALUE):
    """Return (y_expert, expert_mask, y_aux, aux_mask, aux_weight) aligned to `ids`.

    y_expert/expert_mask: expert gold labels where the study is in the gold
    cohort, NaN/False elsewhere. This is the only signal Run 4 evaluates against.

    y_aux/aux_mask: report-derived soft value, masked out (never filled as a
    negative) wherever the report was silent OR the target failed the transfer
    policy gate. Fold-safety (excluding validation-fold rows from any loss) is
    the training loop's responsibility, not this function's -- these arrays are
    fold-agnostic raw supervision.

    aux_weight: one capped scalar per target from the policy, constant across
    studies.
    """
    ids = list(ids)
    n = len(ids)
    y_expert = np.full((n, len(TARGETS)), np.nan)
    expert_mask = np.zeros((n, len(TARGETS)), dtype=bool)
    if gold_labels is not None and len(gold_labels):
        gold = gold_labels.set_index(UID) if UID in gold_labels.columns else gold_labels
        if gold.index.duplicated().any():
            raise ValueError('Duplicate study IDs in gold labels')
        for i, uid in enumerate(ids):
            if uid in gold.index:
                row = gold.loc[uid, TARGETS].to_numpy(float)
                y_expert[i] = row
                expert_mask[i] = np.isfinite(row)

    report = report_source.set_index(UID) if UID in report_source.columns else report_source
    if report.index.duplicated().any():
        raise ValueError('Duplicate study IDs in report source')
    missing = [u for u in ids if u not in report.index]
    if missing:
        raise ValueError(f'{len(missing)} studies missing from report source, e.g. {missing[:3]}')
    raw = report.loc[ids, TARGETS].to_numpy(float)
    if not (np.isfinite(raw) & (raw >= 0) & (raw <= 1)).all():
        raise ValueError('Report-derived values must be finite probabilities in [0, 1]')

    addressed = raw != silent_value
    use_aux = np.array([policy.get(t, {}).get('use_aux', False) for t in TARGETS])
    # Expert truth takes precedence. Report supervision is used only for cells
    # that do not already carry an expert label, avoiding contradictory double
    # supervision on the 58 gold studies.
    aux_mask = addressed & use_aux[None, :] & ~expert_mask
    y_aux = np.where(aux_mask, raw, np.nan)
    aux_weight = np.array([policy.get(t, {}).get('weight', 0.0) for t in TARGETS], dtype=float)
    return y_expert, expert_mask, y_aux, aux_mask, aux_weight
