"""Paired group-bootstrap evaluation; never projects a leaderboard score."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .common import aligned, atomic_json, check_ids, sha
from .validation import metrics


def evaluate(work, baseline, candidate, output, weight=1., bootstrap=2000, seed=1700):
    if not 0 < weight <= 1 or bootstrap < 20:
        raise ValueError('Invalid candidate weight or bootstrap count')
    work = Path(work)
    contract = json.loads((work/'schema.json').read_text())
    labels = pd.read_csv(work/'expert_labels.csv', dtype={contract['id_col']: str})
    ids = check_ids(labels, contract['id_col'])
    y = aligned(labels, ids, contract)
    b = aligned(pd.read_csv(baseline, dtype={contract['id_col']: str}), ids, contract, probabilities=True)
    c = aligned(pd.read_csv(candidate, dtype={contract['id_col']: str}), ids, contract, probabilities=True)
    c = (1-weight)*b + weight*c
    base_metrics, cand_metrics = metrics(y, b, contract['targets']), metrics(y, c, contract['targets'])
    if not base_metrics['all_targets_scorable']:
        raise ValueError('Every target needs both expert classes; cannot compute full macro AUC')
    group_frame = pd.read_csv(work/'folds.csv', dtype={contract['id_col']: str, 'group_id': str})
    group = group_frame.set_index(contract['id_col']).loc[ids, 'group_id'].to_numpy()
    gold = np.isfinite(y).any(1)
    y, b, c, group = y[gold], b[gold], c[gold], group[gold]
    names = np.unique(group)
    indices = [np.flatnonzero(group == g) for g in names]
    rng, draws = np.random.default_rng(seed), []
    for _ in range(bootstrap):
        draw = np.concatenate([indices[i] for i in rng.integers(len(names), size=len(names))])
        bm = metrics(y[draw], b[draw], contract['targets'])['macro_auc']
        cm = metrics(y[draw], c[draw], contract['targets'])['macro_auc']
        if bm is not None and cm is not None:
            draws.append(cm-bm)
    valid = len(draws) >= .8*bootstrap
    result = {'scope': 'Paired development estimate against expert labels; no leaderboard projection',
              'baseline_oof_provenance': 'CSV inputs must be independently audited as held out; filename is not evidence',
              'candidate_weight': weight, 'baseline': base_metrics, 'candidate': cand_metrics,
              'delta': cand_metrics['macro_auc']-base_metrics['macro_auc'],
              'delta_ci95': np.quantile(draws, [.025, .975]).tolist() if valid else None,
              'bootstrap_requested': bootstrap, 'bootstrap_valid': len(draws),
              'interval_usable': valid, 'groups_resampled': len(names),
              'baseline_sha256': sha(baseline), 'candidate_sha256': sha(candidate)}
    atomic_json(output, result)
    return result
