"""Build conservative report labels from three independently published sources.

A target is emitted as 0/1 only when all three sources address it and agree.
Every disagreement, refusal, or missing-source cell becomes 0.5 (unknown), never
a hard negative. Run transfer_audit.py on the result before using it for image
training; consensus is a hypothesis about precision, not assumed ground truth.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transfer_audit import TARGETS, UID, require_unique_ids


def pilkwang_values(frame, target):
    column = target + '__verdict'
    if column not in frame:
        raise ValueError(f'Pilkwang source is missing {column}')
    raw = frame[column]
    invalid = set(raw.dropna().astype(str).unique()) - {'YES', 'NO', 'UNK'}
    if invalid:
        raise ValueError(f'Unexpected Pilkwang verdicts for {target}: {sorted(invalid)}')
    return raw.map({'YES': 1.0, 'NO': 0.0, 'UNK': np.nan})


def steven_values(frame, target):
    if target not in frame:
        raise ValueError(f'Steven source is missing {target}')
    raw = pd.to_numeric(frame[target], errors='coerce')
    if not (raw.dropna().between(0, 1)).all():
        raise ValueError(f'Steven values outside [0,1] for {target}')
    return pd.Series(np.where(raw == 0.5, np.nan,
                             np.where(raw > 0.5, 1.0, 0.0)), index=frame.index)


def lixin_values(frame, target):
    if target not in frame:
        raise ValueError(f'Lixin source is missing {target}')
    raw = pd.to_numeric(frame[target], errors='coerce')
    if not raw.dropna().between(0, 1).all():
        raise ValueError(f'Lixin values outside [0,1] for {target}')
    # This source is graded. Equality with the two binary sources below is
    # deliberately exact, so only its maximally certain 0/1 cells can become
    # unanimous; intermediate values remain unknown rather than thresholded.
    return raw.astype(float)


def build_consensus(pilkwang, steven, lixin, universe_ids=None):
    sources = [('pilkwang', pilkwang), ('steven', steven), ('lixin', lixin)]
    for name, frame in sources:
        require_unique_ids(frame, name)
    indexed = [frame.set_index(UID) for _, frame in sources]
    if universe_ids is None:
        ids = sorted(set().union(*(set(frame.index.astype(str)) for frame in indexed)))
    else:
        ids = list(map(str, universe_ids))
        if len(ids) != len(set(ids)):
            raise ValueError('Universe contains duplicate study IDs')
    aligned = [frame.reindex(ids) for frame in indexed]
    out = pd.DataFrame({UID: ids})
    summary = {}
    for target in TARGETS:
        a = pilkwang_values(aligned[0], target)
        b = steven_values(aligned[1], target)
        c = lixin_values(aligned[2], target)
        known = a.notna() & b.notna() & c.notna()
        unanimous = known & a.eq(b) & b.eq(c)
        value = a.where(unanimous, 0.5).to_numpy(float)
        out[target] = value
        summary[target] = {
            'unanimous_addressed': int(unanimous.sum()),
            'unknown_or_disagreed': int((~unanimous).sum()),
            'positive': int((value == 1).sum()),
            'negative': int((value == 0).sum()),
        }
    return out, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilkwang', type=Path, required=True)
    parser.add_argument('--steven', type=Path, required=True)
    parser.add_argument('--lixin', type=Path, required=True)
    parser.add_argument('--ids', type=Path,
                        help='Optional CSV defining exact output identity/order')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    frames = [pd.read_csv(path, dtype={UID: str})
              for path in (args.pilkwang, args.steven, args.lixin)]
    universe = None if args.ids is None else pd.read_csv(
        args.ids, dtype={UID: str})[UID].tolist()
    consensus, summary = build_consensus(*frames, universe_ids=universe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        consensus.to_csv(stream, index=False)
    with args.summary.open('x', encoding='utf-8') as stream:
        json.dump({'studies': len(consensus), 'targets': summary}, stream,
                  indent=2, allow_nan=False)
        stream.write('\n')
    print(f'wrote {len(consensus)} conservative consensus rows to {args.out}')


if __name__ == '__main__':
    main()
