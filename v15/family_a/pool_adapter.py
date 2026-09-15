"""Expose non-gold competition training studies through the test-data contract.

This is a cache-building adapter only. It never reads finding labels, trains a
model, or creates a leaderboard submission.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def select_pool(train, n, stride=None):
    gold = train[TARGETS].notna().all(axis=1)
    pool = train.loc[~gold, [UID]].drop_duplicates().sort_values(UID).reset_index(drop=True)
    if len(pool) < n:
        raise ValueError(f'Non-gold pool has only {len(pool)} studies, fewer than requested {n}')
    stride = stride or max(1, len(pool) // n)
    ids = pool[UID].iloc[::stride].iloc[:n].tolist()
    if len(ids) != n:
        raise ValueError(f'Stride sampling produced {len(ids)} ids, expected {n}')
    return ids


def prepare_pool(real, work, n, temporary_parent=None, stride=None):
    real, work = Path(real), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if (work / 'pool_audit.json').exists():
        raise RuntimeError('Existing pool artifacts: use a fresh session/work directory')
    train = pd.read_csv(real / 'train.csv', dtype={UID: str})
    if train[UID].isna().any() or train[UID].duplicated().any():
        raise ValueError('Invalid training study identities')
    ids = select_pool(train, n, stride)
    series = pd.read_csv(real / 'train_series.csv',
                         dtype={UID: str, 'SeriesInstanceUID': str})
    selected = series.loc[series[UID].isin(ids)].copy()
    if set(selected[UID]) != set(ids):
        raise ValueError('Pool study missing series metadata')
    for uid in ids:
        if '/' in uid or '\\' in uid or uid in ('.', '..'):
            raise ValueError('Unsafe study identifier')
        if not (real / 'train_series' / uid).is_dir():
            raise FileNotFoundError('Pool study image folder absent: ' + uid)
    if temporary_parent is not None:
        Path(temporary_parent).mkdir(parents=True, exist_ok=True)
    pseudo_parent = Path(tempfile.mkdtemp(prefix='v15-pool-', dir=temporary_parent))
    pseudo = pseudo_parent / 'rsna-knee-abnormality-detection'
    (pseudo / 'test_series').mkdir(parents=True)
    pd.DataFrame({UID: ids}).to_csv(pseudo / 'test.csv', index=False)
    selected.to_csv(pseudo / 'test_series.csv', index=False)
    pd.DataFrame({UID: ids}).assign(**{t: .5 for t in TARGETS}).to_csv(
        pseudo / 'sample_submission.csv', index=False)
    train[[UID]].to_csv(pseudo / 'train.csv', index=False)
    for uid in ids:
        os.symlink((real / 'train_series' / uid).resolve(),
                   pseudo / 'test_series' / uid, target_is_directory=True)
    pd.DataFrame({UID: ids}).to_csv(work / 'pool_study_ids.csv', index=False)
    audit = {
        'status': 'EXPLORATORY_NOT_INDEPENDENT_CONFIRMATION',
        'studies': len(ids),
        'cohort_definition': (f'Deterministic stride sample of {len(ids)} train.csv studies '
                              'missing at least one of the 12 expert targets.'),
        'source_train_csv_sha256': hashlib.sha256((real / 'train.csv').read_bytes()).hexdigest(),
        'labels_read': False,
        'training_enabled': False,
    }
    (work / 'pool_audit.json').write_text(
        json.dumps(audit, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('POOL AUDIT BEFORE CACHE:', json.dumps(audit), flush=True)
    return pseudo, work
