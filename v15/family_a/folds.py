"""Deterministic grouped fold assignment.

The repository's own PatientID audit found singleton groups in the available
metadata (no verified repeat-patient linkage), so grouping by patient identity
cannot currently be claimed here. This groups by StudyInstanceUID -- i.e. every
study is its own group -- which is a weaker guarantee than true patient
separation and is labeled as such in every receipt this module contributes to,
rather than silently presented as patient-safe splitting.
"""
import numpy as np


def assign_folds(ids, k=5, seed=1400):
    if k < 2:
        raise ValueError('Need at least 2 folds')
    ids = list(ids)
    n = len(ids)
    if n < k:
        raise ValueError(f'Need at least as many studies ({n}) as folds ({k})')
    if len(set(ids)) != n:
        raise ValueError('Duplicate study IDs cannot be assigned to folds')
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold_of = np.empty(n, dtype=int)
    fold_of[order] = np.arange(n) % k
    return {
        'fold_assignment': dict(zip(ids, fold_of.tolist())),
        'k': k, 'seed': seed, 'n_studies': n,
        'grouping_caveat': ('Grouped by StudyInstanceUID, one study per group. Patient-level '
                            'grouping was not used: the repository\'s metadata audit found only '
                            'singleton patient groups, so patient-safe separation is unverified.'),
    }
