# V11 vs V13: actual findings and V14 status

Inspected 2026-08-30 in the user's authenticated Kaggle browser. These are the
exact linked versions, not assumed equivalents of the local notebook files.

| Check | Result |
|---|---|
| V11 script version | 345726369; displayed public AUC 0.935 |
| V13 script version | 345967510; displayed public AUC 0.935 |
| Visible output coverage | Same 3 studies × 12 targets |
| Final CSV values changed | **0 of 36** |
| Final per-target rankings changed | **0 of 12** |
| Intermediate CoAtNet-blend values changed | **6 of 36**, using logged precision |
| Intermediate affected targets | Lateral Meniscus, Lateral OA, Contusion |
| Completed CoAtNet arms | All 3 in both visible runs |
| V13 real-prediction replay error | **0.0 maximum absolute error** |
| V14 +0.020 AUC demonstrated | **No** |

Sources: [V11 output](https://www.kaggle.com/code/seanzhang2445/rsna-knee-ensemble-v11/output?scriptVersionId=345726369&select=submission.csv),
[V13 output](https://www.kaggle.com/code/seanzhang2445/rsna-knee-ensemble-v13-input-contracts/output?scriptVersionId=345967510&select=submission.csv),
[V11 logs](https://www.kaggle.com/code/seanzhang2445/rsna-knee-ensemble-v11/log?scriptVersionId=345726369),
[V13 logs](https://www.kaggle.com/code/seanzhang2445/rsna-knee-ensemble-v13-input-contracts/log?scriptVersionId=345967510).

## What explains the identical visible outputs?

The CoAtNet branch really changed. For example, its Lateral Meniscus outputs in
study order changed from approximately `[0.666667, 0.666667, 0.166667]` to
`[0.666667, 0.500000, 0.333333]`. The final submission nevertheless remains
`[0.6666666666666666, 1.0, 0.3333333333333333]` for that target in both versions.
Intermediate changes were erased by the final fusion/ranking on these cases.
This is a measured mechanism, not an assumption that the new models failed to run.

V13 still uses the same three trained CoAtNets, equal arm weights and a 50/50
outer blend. Input-contract corrections are not learning/training; no specialist
for an undertrained finding was added. More precision or different probabilities
alone cannot improve AUC unless positive-versus-negative orderings improve.

Important limits: these are **three visible test studies**, not the hidden scoring
cohort. Exact hidden-score deltas and hidden CSV equality cannot be inferred.
The original file bytes were not obtained: all displayed cells were captured at
their full numeric precision and compared by UID. This establishes numerical
output equality, not identical CSV encodings/newlines. Do not perturb the CSV just
to make the files differ; that would not demonstrate learning or better diagnosis.

The remote V11 uses shared resized-window caching and has no V11 audit receipt;
today's local V11 contains later audit changes. The remote V13 receipt's parent,
builder and input-contract hashes match the local manifest. We did not assert an
exact source hash for the executed remote notebooks.

Follow-up source inspection retrieved all 33 published V13 code cells: 32 match
the local V13 AST exactly. The remaining runtime cell already uses full-precision
CSV writing and `equal_nan=True` on Kaggle, unlike the local copy. The V14
validation builder now reproduces the published CSV precision. See
`published_v13_source_audit.json` and `KAGGLE_VALIDATION.md` for the audit and
private labeled benchmark history. This mismatch is not a published-V13 accuracy
defect and must not be presented as the cause of its 0.935 score.

## Real V14 ablation replay, with no extra GPU run

Loaded V13's actual Stage-3 and three raw-arm prediction tables. Reconstructed its
final output exactly, then changed the included CoAtNet arms. All inputs were
aligned by study UID. The V13 submission was not overwritten.

| Ablation | Final numeric cells different from V13 |
|---|---:|
| MaxSpan only | 6/36 |
| Remove MaxSpan | 2/36 |
| Native384 only | 6/36 |
| Remove Native384 | 10/36 |
| WideDense only | 6/36 |
| Remove WideDense | 8/36 |

**None is an accuracy winner:** there are no labels for these visible rows.
Candidate selection based on these differences would be unjustified. The purpose
is to establish that the diagnostic pipeline reproduces the baseline and detects
actual surviving output changes. Full results: `actual_ablation_replay.json`.

## Other checks actually run

The local source audit confirms unchanged checkpoint lists/equal weights and
28 unchanged Stage-1-to-3 code cells in the local V11/V13 sources after version
name normalization. This source comparison is separate from the remote-run evidence.

The V14 regression suite has 17 passing tests. The inherited V13 and comparison
suites also pass; see `verification.json` for exact commands, counts and outputs.
Tests include exact V13 rank-helper agreement including float32 CSV round trips,
UID alignment, ties, missing labels, paired/grouped bootstrap, no target omission,
training/confirmation overlap rejection, group-held-out head fitting, and
byte-preservation of the control during ablation export.

Synthetic mechanism/performance checks (not knee accuracy):

- 4,349 artificial studies, 3 arms: CPU rank fusion median about 0.034 seconds
  in this run, seven repetitions. Not model inference time.
- AUC implementation differs from scikit-learn by at most `1.11e-16` in the test.
- Squaring synthetic probabilities changes them by up to 0.25 while leaving
  all final rank-fused outputs unchanged.
- A new-feature residual head learns an intentionally omitted synthetic signal
  in five group folds. Its artificial AUC increase is recorded in
  `local_benchmark.json`, explicitly NOT evidence for a knee or Kaggle gain.

## What is and is not delivered

Delivered: a runnable V14 control/ablation notebook; a trainable residual-head
research component; real-output replay; an evaluation/confirmation toolkit; and
the experiment plan in `PLAN.md`. Existing V11/V13 submissions are preserved.

Not delivered: newly trained MRI specialists, a full end-to-end specialist image
extractor, independent labeled confirmation, a new Kaggle GPU benchmark, or a
0.955 submission. The default V14 notebook intentionally retains V13 predictions
until a changed candidate has real evidence. This limitation is explicit in its
title text, manifest, results and README; it is not a claimed model improvement.

The next useful run is an explicitly labeled, provenance-audited validation
experiment using Kaggle GPUs—not another three-row submission. The historical
58 gold cases can support exploratory error analysis if training exclusions are
verified, but have already been inspected and cannot become pristine confirmation
data again. A credible +0.020 claim needs independent data and materially better
image information/supervision, as laid out in the plan.
