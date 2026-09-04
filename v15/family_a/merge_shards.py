"""Merge independently executed Family A fold shards into one exact OOF arm.

Every shard is checked against the deterministic fold map, input hashes, model
configuration, seed, and study coverage. A missing, duplicated, or foreign fold
is fatal; this script never fills gaps or silently averages overlapping rows.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from contract import TARGETS, UID
from folds import assign_folds


CONFIG_KEYS = ('arm', 'seed', 'fold_seed', 'k', 'epochs_requested', 'head_lr',
               'backbone_lr', 'weight_decay', 'batch_size', 'policy_selection',
               'policy_bootstrap', 'policy_seed', 'expert_fraction',
               'fold_assignment_sha256')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ids(path, name):
    frame = pd.read_csv(path, dtype={UID: str})
    if UID not in frame or frame[UID].isna().any() or frame[UID].duplicated().any():
        raise ValueError(f'{name} must contain unique, nonmissing {UID} values')
    return frame[UID].tolist()


def resolve_prediction(receipt_path, receipt):
    recorded = Path(receipt['partial_oof_path'])
    candidates = [recorded, Path(receipt_path).parent / recorded.name]
    prediction = next((path for path in candidates if path.is_file()), None)
    if prediction is None:
        raise FileNotFoundError(f'Prediction file for {receipt_path} was not found')
    if digest(prediction) != receipt['partial_oof_sha256']:
        raise ValueError(f'Prediction hash differs from receipt: {prediction}')
    return prediction


def merge_partial_receipts(receipt_paths, ids_path, gold_ids_path, out_dir):
    receipt_paths = [Path(path) for path in receipt_paths]
    if not receipt_paths:
        raise ValueError('At least one shard receipt is required')
    ids = read_ids(ids_path, 'all-study table')
    gold_ids = read_ids(gold_ids_path, 'gold-study table')
    if not set(gold_ids).issubset(set(ids)):
        raise ValueError('Gold IDs are not a subset of all study IDs')
    receipts = [json.loads(path.read_text(encoding='utf-8')) for path in receipt_paths]
    if any(receipt.get('status') != 'COMPLETE_FOLD_SHARD_NOT_COMPLETE_ARM'
           for receipt in receipts):
        raise ValueError('Every input must be a completed partial-arm receipt')
    expected_config = {key: receipts[0].get(key) for key in CONFIG_KEYS}
    for receipt in receipts[1:]:
        actual = {key: receipt.get(key) for key in CONFIG_KEYS}
        if actual != expected_config:
            raise ValueError('Shard model/fold configuration mismatch')

    k = int(expected_config['k'])
    fold_info = assign_folds(ids, k=k, seed=int(expected_config['fold_seed']),
                             priority_ids=gold_ids)
    expected_fold_hash = hashlib.sha256(json.dumps(
        fold_info['fold_assignment'], sort_keys=True).encode()).hexdigest()
    if expected_fold_hash != expected_config['fold_assignment_sha256']:
        raise ValueError('Shard fold map does not match supplied study IDs')

    seen_folds = set()
    pieces = []
    source_records = []
    for receipt_path, receipt in zip(receipt_paths, receipts):
        selected = set(map(int, receipt['selected_folds']))
        if not selected or seen_folds.intersection(selected):
            raise ValueError('Shard folds are empty or overlap')
        prediction_path = resolve_prediction(receipt_path, receipt)
        frame = pd.read_csv(prediction_path, dtype={UID: str})
        if UID not in frame or frame[UID].isna().any() or frame[UID].duplicated().any():
            raise ValueError(f'Invalid prediction identities in {prediction_path}')
        expected_ids = {uid for uid in ids
                        if fold_info['fold_assignment'][uid] in selected}
        if set(frame[UID]) != expected_ids:
            raise ValueError(f'Prediction coverage does not equal declared folds: {prediction_path}')
        values = frame[TARGETS].to_numpy(float)
        if not (np.isfinite(values) & (values >= 0) & (values <= 1)).all():
            raise ValueError(f'Invalid probabilities in {prediction_path}')
        pieces.append(frame[[UID] + TARGETS])
        source_records.append({'receipt': str(receipt_path),
                               'receipt_sha256': digest(receipt_path),
                               'prediction': str(prediction_path),
                               'prediction_sha256': digest(prediction_path),
                               'folds': sorted(selected)})
        seen_folds.update(selected)
    if seen_folds != set(range(k)):
        raise ValueError(f'Incomplete folds: got {sorted(seen_folds)}, need 0..{k - 1}')

    merged = pd.concat(pieces, ignore_index=True)
    if merged[UID].duplicated().any() or set(merged[UID]) != set(ids):
        raise ValueError('Merged shards do not cover every study exactly once')
    merged = merged.set_index(UID).loc[ids].reset_index()
    gold = merged.set_index(UID).loc[gold_ids].reset_index()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arm = expected_config['arm']
    all_path = out_dir / f'{arm}_all_oof.csv'
    gold_path = out_dir / f'{arm}_oof.csv'
    with all_path.open('x', encoding='utf-8') as stream:
        merged.to_csv(stream, index=False)
    with gold_path.open('x', encoding='utf-8') as stream:
        gold.to_csv(stream, index=False)
    result = {
        'status': 'COMPLETE_ARM_MERGED_FROM_SHARDS', **expected_config,
        'selected_folds': sorted(seen_folds), 'n_studies_total': len(merged),
        'n_studies_scored': len(gold), 'sources': source_records,
        'all_oof_sha256': digest(all_path), 'gold_oof_sha256': digest(gold_path),
    }
    receipt_out = out_dir / f'{arm}_receipt.json'
    with receipt_out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return merged, gold, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', type=Path, nargs='+', required=True)
    parser.add_argument('--ids', type=Path, required=True)
    parser.add_argument('--gold-ids', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    _, _, result = merge_partial_receipts(args.receipts, args.ids, args.gold_ids, args.out_dir)
    print(f"merged {result['arm']} seed {result['seed']} across {result['k']} folds")


if __name__ == '__main__':
    main()
