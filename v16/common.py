import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def config(path):
    defaults = json.loads(Path(__file__).with_name('config.json').read_text())
    supplied = json.loads(Path(path).read_text())
    if set(supplied) - set(defaults):
        raise ValueError(f'Unknown configuration keys: {set(supplied) - set(defaults)}')
    defaults.update(supplied)
    for key in ('image_size', 'max_centres', 'max_series', 'epochs', 'batch_size', 'encoder_chunk', 'hidden'):
        if not isinstance(defaults[key], int) or defaults[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not defaults['offsets'] or not all(isinstance(x, int) for x in defaults['offsets']):
        raise ValueError('offsets must be nonempty integers')
    if defaults['encoder'] not in ('dinov2', 'cnn', 'tiny'):
        raise ValueError('encoder must be dinov2, cnn, or tiny (tests only)')
    if defaults['auxiliary'] not in ('none', 'raw', 'platt'):
        raise ValueError('auxiliary must be none, raw, or platt')
    if defaults['expert_loss'] not in ('bce', 'asl'):
        raise ValueError('expert_loss must be bce or asl')
    if not 0 <= defaults['series_dropout'] < 1 or not 0 < defaults['expert_fraction'] < 1:
        raise ValueError('Invalid sampling/dropout fractions')
    if not 0 <= defaults['aux_cap'] <= 1 or defaults['folds'] < 2 or defaults['policy_bootstrap'] < 20:
        raise ValueError('Invalid auxiliary cap, fold count, or bootstrap count')
    lo, hi = defaults['percentiles']
    if not 0 <= lo < hi <= 100 or defaults['crop_mm'] <= 0:
        raise ValueError('Invalid intensity/crop configuration')
    return defaults


def schema(root, id_col):
    columns = pd.read_csv(Path(root) / 'sample_submission.csv', nrows=0).columns.tolist()
    if id_col not in columns:
        raise ValueError(f'Submission schema lacks {id_col}')
    targets = [c for c in columns if c != id_col]
    if not targets:
        raise ValueError('No finding columns')
    return {'id_col': id_col, 'targets': targets}


def read_studies(root, split, contract):
    frame = pd.read_csv(Path(root) / f'{split}.csv', dtype={contract['id_col']: str})
    check_ids(frame, contract['id_col'])
    return frame


def check_ids(frame, id_col):
    if id_col not in frame or frame[id_col].isna().any():
        raise ValueError('Missing study IDs')
    ids = frame[id_col].astype(str)
    if ids.duplicated().any() or ids.str.strip().eq('').any():
        raise ValueError('Duplicate or empty study IDs')
    return ids.tolist()


def child(root, uid):
    root = Path(root).resolve()
    if not uid or '/' in uid or '\\' in uid or uid in ('.', '..'):
        raise ValueError('Invalid UID path component')
    path = (root / uid).resolve()
    if not path.is_relative_to(root):
        raise ValueError('UID path escaped root')
    return path


def aligned(frame, ids, contract, probabilities=False):
    id_col, targets = contract['id_col'], contract['targets']
    actual = check_ids(frame, id_col)
    if set(actual) != set(ids):
        raise ValueError('Study coverage differs from the expected cohort')
    frame = frame.assign(**{id_col: frame[id_col].astype(str)})
    values = frame.set_index(id_col).loc[ids, targets].to_numpy(float)
    valid = np.isfinite(values) & (values >= 0) & (values <= 1)
    if not (valid if probabilities else (valid | np.isnan(values))).all():
        raise ValueError('Invalid probabilities/labels')
    return values


def write_predictions(path, ids, values, contract):
    if values.shape != (len(ids), len(contract['targets'])):
        raise ValueError('Prediction shape differs from schema')
    frame = pd.DataFrame(values, columns=contract['targets'])
    frame.insert(0, contract['id_col'], ids)
    aligned(frame, ids, contract, probabilities=True)
    atomic_csv(path, frame)
    return frame
