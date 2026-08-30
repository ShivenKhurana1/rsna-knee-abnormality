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

The V14 regression suite now has 25 passing tests (including eight gold-run tests).
The inherited V13 and comparison
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

## Completed private GPU benchmark: 58 labeled studies

[Kaggle Version 3 / 346064640](https://www.kaggle.com/code/seanzhang2445/rsna-knee-v14-gold-benchmark?scriptVersionId=346064640)
completed successfully in 743.8 seconds (12m 24s), T4 x2, internet off. All 24
Stage-1 members, DINOv3/Rad stages and three CoAtNet arms completed. Nothing was
submitted to the competition. This is a separate labeled experiment, not the
hidden scoring run behind the displayed public 0.935.

Downloaded 21 original output files through Kaggle's offered Download links,
not the truncated CSV preview. All prediction tables contain the same 58 unique
IDs and 12 targets. Fifteen distinct CSV hashes agree with the saved receipts.
`gold_independent_verification.json` independently replays the control and every
saved candidate (maximum numerical error <=1.11e-16), checks baseline AUC against
scikit-learn (<=1.11e-16), and reruns all six paired 2,000-resample comparisons.
Local AUC/delta/interval results agree exactly with Kaggle's saved report.

### Actual output differences and AUC

All six candidates change rankings in **all 12 targets**. Counts below compare
the actual saved candidate files with the V13-recipe control on this same cohort.
The unchanged `v13_control.csv` and `v13_gold_predictions.csv` are byte-identical,
as required for the control. No random noise, altered labels, or cosmetic-only
probability changes were added. Changing an ensemble recipe is not new training.

| Candidate (Stage 3 retained in each blend) | Changed values / 696 | Macro AUC | Delta vs control | Exploratory 95% paired interval |
|---|---:|---:|---:|---:|
| V13 control / default V14 recipe | 0 | 0.959448 | 0 | — |
| Native384 only (`only_arm_0`) | 550 | 0.956420 | -0.003029 | [-0.008839, +0.001789] |
| Remove Native384 (`without_arm_0`) | 468 | 0.958271 | -0.001177 | [-0.004203, +0.001810] |
| WideDense only (`only_arm_1`) | 533 | 0.955072 | -0.004377 | [-0.009413, +0.000318] |
| Remove WideDense (`without_arm_1`) | 450 | 0.960671 | +0.001223 | [-0.001833, +0.004621] |
| MaxSpan only (`only_arm_2`) | 530 | 0.959354 | -0.000094 | [-0.005027, +0.004309] |
| Remove MaxSpan (`without_arm_2`) | 460 | 0.956945 | -0.002503 | [-0.007380, +0.001051] |

The largest point gain is only 6.1% of the requested +0.020. Every interval spans
zero. **No candidate passes the improvement requirement; none is promoted.**
Intervals are study-level, not patient-grouped, and not selection-adjusted.

### Which findings deserve error review?

| Target | Positive / negative | Control AUC |
|---|---:|---:|
| ACL | 24 / 34 | 0.981005 |
| MCL | 9 / 49 | 0.981859 |
| Medial Meniscus | 26 / 32 | 0.990385 |
| Lateral Meniscus | 23 / 35 | 0.929814 |
| Medial OA | 15 / 43 | 0.978295 |
| Lateral OA | 11 / 47 | 0.916828 |
| PF OA | 21 / 37 | 0.890605 |
| Effusion | 35 / 23 | 0.993789 |
| Synovitis | 27 / 31 | 0.924731 |
| Baker's | 12 / 46 | 0.986413 |
| Contusion | 19 / 39 | 0.973684 |
| Fracture | 18 / 40 | 0.965972 |

Prioritize PF OA, lateral OA, synovitis and lateral meniscus for image/label/
coverage review. This establishes relative error rates on this cohort, **not**
that those targets received less training. MCL's nine positives illustrate why
small-cohort estimates are fragile even when the point AUC is high.

### Exposure warning and the tempting Stage-1 shortcut

The checkpoint audit observed 57 loads from 34 unique paths and found **zero
explicit training/validation ID lists**. CoAtNet source excludes these gold
studies from gradient training but uses their AUC for checkpoint selection.
Exposure of the other families and calibration remains unknown. Stripping labels
from inference inputs does not undo prior training or selection exposure.

| Cumulative stage | Gold macro AUC |
|---|---:|
| Stage 1 | 0.995752 |
| Stage 2 | 0.959723 |
| Stage 3 | 0.973334 |
| Stage 4 / final control | 0.959448 |

Stage 1 is +0.036303 above the final control **on these potentially exposed gold
studies**. That is a diagnostic flag, not an honest +0.02 solution. It could
reflect training exposure and/or fusion effects; this run cannot distinguish
them. Do not ship Stage-1-only on this evidence. Audit its actual training IDs
and measure stage transitions on an eligible held-out cohort first.

All findings are `EXPLORATORY_GOLD_SELECTED_NOT_CONFIRMATION`.
`gold_run_audit.json` records receipt checks, completion and stage metrics.
Per-study originals stay git-ignored under `private_artifacts/gold_346064640/`;
aggregate reports do not include study IDs or temporary download credentials.

## What is and is not delivered

Delivered: a runnable V14 control/ablation notebook; a trainable residual-head
research component; real-output replay; a completed private MRI GPU benchmark
with independently verified CSV differences, AUCs and uncertainty; an
evaluation/confirmation toolkit; and the updated experiment plan in `PLAN.md`.
Existing V11/V13 submissions are preserved.

Not delivered: newly trained MRI specialists, a full end-to-end specialist image
extractor, independent labeled confirmation, or a
0.955 submission. The default V14 notebook intentionally retains V13 predictions
until a changed candidate has real evidence. This limitation is explicit in its
title text, manifest, results and README; it is not a claimed model improvement.

The authorized exploratory experiment is complete and did not establish the
requested gain. Further gold-set blend tuning cannot provide independent proof.
Next audit the Stage-1 exposure/transition discrepancy, establish an eligible
held-out cohort, and train/evaluate a materially different specialist under
patient-separated splits. A credible +0.020 claim needs that evidence, as laid
out in the plan.
