"""Threshold soft report-derived LLM labels into hard pseudo-labels for residual_specialist.py.

Source: https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels (CC0).
0.5 in that dataset means "the report does not address this finding" -- not a coin flip.
Masking (not filtering): a cell becomes a hard 0/1 pseudo-label only when the LLM's
probability clears --hi/--lo; everything else, including genuine 0.5 cells, is left NaN
and excluded from training loss by residual_specialist.py's own masking. Gold-labeled
studies are dropped from the pool entirely -- they are the confirmation cohort, never
training data, matching confirm.py's exclusion-manifest rule.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics import TARGETS, UID, write_new


def threshold(raw, gold_ids, lo=0.1, hi=0.9):
    if raw[UID].isna().any() or raw[UID].duplicated().any():
        raise ValueError('Duplicate or missing study IDs in the LLM label source')
    if not set(TARGETS).issubset(raw.columns):
        raise ValueError('LLM label source is missing one or more of the 12 targets')
    if lo >= 0.5 or hi <= 0.5:
        raise ValueError('--lo must be below 0.5 and --hi must be above 0.5')
    values = raw[TARGETS].to_numpy(float)
    if not (np.isfinite(values) & (values >= 0) & (values <= 1)).all():
        raise ValueError('LLM label source must be finite probabilities in [0, 1]')
    hard = np.where(values <= lo, 0.0, np.where(values >= hi, 1.0, np.nan))
    pseudo = pd.DataFrame(hard, columns=TARGETS)
    pseudo.insert(0, UID, raw[UID].to_numpy())
    pseudo = pseudo.loc[~pseudo[UID].isin(gold_ids)].reset_index(drop=True)
    return pseudo


def report(pseudo):
    counts = {}
    for t in TARGETS:
        col = pseudo[t]
        counts[t] = {'positive': int((col == 1).sum()), 'negative': int((col == 0).sum()),
                     'masked': int(col.isna().sum())}
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llm-labels', type=Path, required=True)
    p.add_argument('--gold-ids', type=Path, required=True, help='CSV with a StudyInstanceUID column to exclude')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--lo', type=float, default=0.1)
    p.add_argument('--hi', type=float, default=0.9)
    args = p.parse_args()
    raw = pd.read_csv(args.llm_labels, dtype={UID: str})
    gold_ids = set(pd.read_csv(args.gold_ids, dtype={UID: str})[UID])
    pseudo = threshold(raw, gold_ids, args.lo, args.hi)
    pseudo.to_csv(args.out, index=False)
    counts = report(pseudo)
    write_new(args.report, {
        'source': str(args.llm_labels), 'lo': args.lo, 'hi': args.hi,
        'studies_in_source': len(raw), 'gold_studies_excluded': len(gold_ids),
        'pseudo_pool_studies': len(pseudo), 'counts': counts,
    })
    print(pd.DataFrame(counts).T)


if __name__ == '__main__':
    main()
