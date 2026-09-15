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

## Result: rejected by the expert-label gate

The three public source datasets were downloaded directly from Kaggle and the
coverage-only construction completed over 4,407 studies. Exact source and
output hashes plus per-target coverage are recorded in
`consensus_coverage_report.json`.

Consensus retains 59%-97% of the current single-source addressed cells for 11
targets. Synovitis is the exception: only 607/3,989 cells (15.2%) survive
three-way exact agreement. That may remove noise or simply discard signal; it
must not be guessed from coverage.

The 58 expert rows were reconstructed from the public gold-v3 table by selecting
the studies marked `has_labels=1` in the grouped-fold metadata. The resulting
CSV has SHA-256
`a3ca7df32d43ee7c683091162b1839ea8308d802bea3f2b437f4908d9cbcb079`, an
exact byte-for-byte match to the original `gold_labels.csv` hash recorded before
that local file became unavailable. The expert rows were kept outside the repo;
only aggregate results are recorded here.

On the predeclared common 58-study cohort, retaining `0.5` unknowns for both
sources, the existing Steven source scores macro AUC `0.8873495` and exact
three-way consensus scores `0.8355313`. The paired delta is `-0.0518182` with
bootstrap 95% CI `[-0.0737008, -0.0311104]`, 5,000 study-level replicates,
seed 1400, and bootstrap probability of improvement `0.000`. Consensus retains
431 addressed cells versus 550 for the baseline on this cohort.

Only 2/12 target deltas are nonnegative (Lateral Meniscus `+0.0031`, Fracture
`+0.0153`). The largest declines are Effusion `-0.1689`, Synovitis `-0.0980`,
PF OA `-0.0792`, and Contusion `-0.0749`. Consequently, consensus fails four of
the five promotion criteria: positive delta, probability of improvement,
8/12 nonnegative targets, and no decline worse than `0.05`. It passes only the
minimum addressed-cell retention criterion.

**Decision: `REJECT_EXACT_THREE_SOURCE_CONSENSUS`.** Do not spend GPU time on a
consensus-label Family A arm and do not add the public notebook's claimed
`+0.022` to any score forecast. A different label combination must clear this
same paired expert gate before it is eligible for image-model training.

### Exploratory convex-blend follow-up

A grid of global convex weights over Steven, Pilkwang, and Lixin was also tested
to determine whether unanimity was merely too destructive. Choosing weights on
all 58 studies gives an apparent `+0.0057363` macro AUC at weights
`[0.4, 0.1, 0.5]`, but that number is selection-biased. In the valid nested
check, each held-out grouped fold receives weights selected using only the other
four folds. The pooled cross-fitted delta is `-0.0096559`; its 5,000-replicate
study bootstrap CI is `[-0.0277010, +0.0085189]`, with probability of
improvement `0.1402`. Allowing separate per-target weights is also negative
(`-0.0106331`, CI `[-0.0318853, +0.0097435]`, probability better `0.1652`).

This reversal demonstrates why the direct 58-study optimum is not actionable.
Neither exact consensus nor a data-selected convex label blend is eligible for
the GPU queue. The current single-source, fold-local transfer-gated Family A
experiment remains the next valid image-model test.
