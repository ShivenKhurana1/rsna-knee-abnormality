# Three-source consensus label audit

## Why this is the next label experiment

The competition host states that image-derived expert labels are authoritative
and that ambiguous findings were graded negative. A report reader can therefore
be confidently correct about the report while still teaching the wrong target.
The current Family A source is already strong for several targets, but no single
report reader resolves this report-to-image gap.

A public notebook titled `Report Pseudo-Labels Add +0.022 AUC (4/5)` provides a
useful hypothesis: keep only cells on which three independently published label
sources agree. Its public source (version 1, SHA-256
`3bccc03cbd5eb9be6bf4005099eacda6bb79be27cd27b6e82f53d679f01f895b`) is not
proof of `+0.022`, however:

- each fold silently drops targets having one class, so fold macro AUCs do not
  score the same target set;
- it averages five small fold-level macro AUCs instead of pooling exact OOF
  predictions over all 58 studies;
- it uses one random 250-study draw from 2,570 eligible cases and one seed;
- it reports no paired uncertainty interval;
- it evaluates a frozen-feature MLP, not the deployed ensemble or Family A.

The defensible reuse is the consensus construction, followed by our existing
common-cohort audit and then a separate image-model ablation.

## Reproduce the label-only test

Run this where the three public label CSVs and the real gold CSV are mounted:

```bash
python v15/consensus_labels.py \
  --pilkwang /path/to/report_labels_v2.csv \
  --steven /path/to/llm_labels_v2.csv \
  --lixin /path/to/labels_llm_gpt56sol.csv \
  --ids /path/to/train.csv \
  --out /path/to/consensus_labels.csv \
  --summary /path/to/consensus_summary.json

python v15/transfer_audit.py \
  --gold-labels /path/to/gold_labels.csv \
  --report-source /path/to/consensus_labels.csv \
  --bootstrap 5000 --seed 1400 \
  --out /path/to/consensus_transfer_audit.json

python v15/compare_label_sources.py \
  --gold /path/to/gold_labels.csv \
  --baseline /path/to/llm_labels_v2.csv \
  --candidate /path/to/consensus_labels.csv \
  --bootstrap 5000 --seed 1400 \
  --out /path/to/consensus_vs_steven.json
```

Every disagreement, refusal, or missing-source cell becomes `0.5` and is never
coerced to negative. The paired comparison retains those `0.5` values so both
sources are scored on the identical 58-study cohort.

## Predeclared promotion rule

Do not allocate a second full Family A run to consensus labels unless all of the
following hold:

1. paired macro delta is at least `+0.010`;
2. bootstrap probability of improvement is at least 90%;
3. at least 8 of 12 target deltas are nonnegative;
4. consensus retains at least half as many addressed cells as the single source;
5. no target declines by more than `0.05` AUC.

Passing this rule only authorizes a paired image-model experiment. It does not
add the label delta to the leaderboard score and does not authorize submission.

## Current status

The three public source datasets were downloaded directly from Kaggle and the
coverage-only construction completed over 4,407 studies. Exact source and
output hashes plus per-target coverage are recorded in
`consensus_coverage_report.json`.

Consensus retains 59%-97% of the current single-source addressed cells for 11
targets. Synovitis is the exception: only 607/3,989 cells (15.2%) survive
three-way exact agreement. That may remove noise or simply discard signal; it
must not be guessed from coverage.

The expert `gold_labels.csv` is absent on this machine and Kaggle's competition
metadata endpoint returned HTTP 401 without an accepted-session credential.
Therefore the correctness comparison remains
`PENDING_EXPERT_TRANSFER_AUDIT`; the coverage result is not a positive or
negative image-training result.
