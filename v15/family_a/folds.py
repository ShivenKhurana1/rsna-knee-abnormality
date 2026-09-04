"""Deterministic grouped fold assignment.

The repository's own PatientID audit found singleton groups in the available
metadata (no verified repeat-patient linkage), so grouping by patient identity
cannot currently be claimed here. This groups by StudyInstanceUID -- i.e. every
study is its own group -- which is a weaker guarantee than true patient
separation and is labeled as such in every receipt this module contributes to,
rather than silently presented as patient-safe splitting.
"""
import numpy as np


def assign_folds(ids, k=5, seed=1400, priority_ids=None):
    if k < 2:
        raise ValueError('Need at least 2 folds')
    ids = list(ids)
    n = len(ids)
    if n < k:
        raise ValueError(f'Need at least as many studies ({n}) as folds ({k})')
    if len(set(ids)) != n:
        raise ValueError('Duplicate study IDs cannot be assigned to folds')
    priority_ids = set() if priority_ids is None else set(priority_ids)
    unknown = priority_ids - set(ids)
    if unknown:
        raise ValueError(f'Priority set contains {len(unknown)} unknown study IDs')
    rng = np.random.default_rng(seed)
    priority = np.array([i for i, uid in enumerate(ids) if uid in priority_ids], dtype=int)
    ordinary = np.array([i for i, uid in enumerate(ids) if uid not in priority_ids], dtype=int)
    fold_of = np.full(n, -1, dtype=int)
    fold_order = rng.permutation(k)
    for j, row in enumerate(rng.permutation(priority)):
        fold_of[row] = fold_order[j % k]
    counts = np.bincount(fold_of[fold_of >= 0], minlength=k)
    tie_order = {fold: rank for rank, fold in enumerate(fold_order)}
    for row in rng.permutation(ordinary):
        fold = min(range(k), key=lambda f: (counts[f], tie_order[f]))
        fold_of[row] = fold
        counts[fold] += 1
    return {
        'fold_assignment': dict(zip(ids, fold_of.tolist())),
        'k': k, 'seed': seed, 'n_studies': n,
        'priority_studies_balanced': len(priority_ids),
        'grouping_caveat': ('Grouped by StudyInstanceUID, one study per group. Patient-level '
                            'grouping was not used: the repository\'s metadata audit found only '
                            'singleton patient groups, so patient-safe separation is unverified.'),
    }
