# Private gold validation run

User authorized a private GPU validation run, not leaderboard submission.

Notebook: [RSNA Knee V14 Gold Benchmark](https://www.kaggle.com/code/seanzhang2445/rsna-knee-v14-gold-benchmark)

The separate `rsna-knee-v14-gold-validation.ipynb` wraps frozen V13 inference;
it is not the production `rsna-knee-ensemble-v14.ipynb`. It exposes every fully
expert-labeled training study as a pseudo-test study, stripping labels and reports
from those inputs. It records checkpoint metadata during normal loading, exports
all four stage predictions and six predeclared CoAtNet ablations, and evaluates
macro/per-target ROC-AUC with paired exploratory bootstrap intervals when feasible.
It requires 24 completed stage-1 members, completed DINOv3/Rad stages, and all three
CoAtNet arms before evaluating the full-control comparison. It does not train or
promote a candidate. Outputs remain under `/kaggle/working/v14_validation/`.

## Training-exposure audit

The [CoAtNet author's training source](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model)
excludes fully labeled gold IDs from gradient training, but uses gold AUC for
epoch selection, top-k checkpoints and SWA evaluation/selection. This supports
the intended split, not independent proof of the IDs used by each published
checkpoint. Gold is therefore **not an untouched confirmation set**.

Training and calibration exposure of the other loaded families is unverified.
The embedded Stage-3 calibration payload contains `mean`, `scale`, `coef`,
`intercept`, `gate`, `protocol_columns`, and `groups`, but no training-study IDs
or split provenance. SHA-256 of its encoded payload:
`025039597077cce896ef43b35961b5e8de4c05fdf2de5a87f33676055debe17a`.
The run records explicit training/validation/gold ID sets when present in loaded
checkpoint metadata. A fold number or absent ID list is never treated as proof
of exclusion. Study-level bootstrap assumes independent studies; patient grouping
has not been verified. Intervals are not corrected for candidate selection.

No score on this cohort establishes a public/private leaderboard gain, and
`plus_0_02_verified` remains false regardless of an exploratory positive delta.

## Execution history

- Version 1, scriptVersionId `346060671`: failed after 21.1 seconds, before
  inference, because `/kaggle/temp` did not exist. No model AUC was produced.
  The traceback is retained privately. Adapter now creates its temporary parent;
  the regression test starts with an absent parent directory.
- Version 2, scriptVersionId `346061261`: launched with the startup fix.
  Uploaded notebook SHA-256:
  `936b287dbfa3507af0eb85b1775e48309b179a4cd6db01be59e9567a36ee0735`.
  Confirmed 58 studies, both CUDA probes passed, and inference started.
  At 542.5 seconds, the first CoAtNet finished all 58 studies but the local
  V13 atomic-CSV guard rejected its float32 serialization round trip. The arm
  was dropped. This confounds a full-baseline comparison, so the run was cancelled
  before spending more GPU time on further affected arms. No full-model AUC claim.

The failure is reproduced locally using the frozen local V13 CSV writer and a
58x12 float32 matrix. Writing 17 significant digits fixes it while preserving
the original float32 values exactly after reload. The guard's strict tolerance
is retained. V14 builders now include that fix; V11/V13 files are unchanged.
Published-source audit subsequently confirmed all 33 rendered code cells were
available. 32 are AST-identical to the local V13 source. The sole changed cell
already has `float_format='%.17g'` and `equal_nan=True` in the published version.
Thus this was a **local-source/benchmark mismatch, not a defect discovered in
published V13 or a cause of its 0.935 score**. The corrected benchmark matches
the published precision setting, retaining stricter NaN rejection for outputs
that are already required to be finite. See `published_v13_source_audit.json`.
The updated regression suite has 24 passing tests.

- Version 3, scriptVersionId `346064640`: launched with the full-precision CSV guard fix. Uploaded notebook
  SHA-256 `f5338973ae8ef1316209fb23dc9e9dbc95f30cc739f4c035958127b008141c98`.
  Model weights, cohort, blend settings and candidate definitions are unchanged.
  **Completed successfully in 743.8 seconds (12m 24s)**. All 24 Stage-1 members,
  Stage 2/3 and all three CoAtNet arms completed. The original notebook remains
  private and no competition submission was made.

Historical local checks progressed from 21 to 22 tests after the startup and
output-difference checks, then to 24 after the precision regression and source
audit. Generated cells compile. Version 3 includes the corrected writer and
the difference-gate helper; its exact hash is recorded above. Subsequent local
download-verification additions did not change the uploaded notebook or recipe.
The final suite has 25 tests. These validate software, not MRI accuracy.

Two unused, zero-version setup drafts created during this task were deleted:
`RSNA Knee V14 Gold Validation - Exploratory` and
`Fork of RSNA Knee Ensemble V13 Input Contra 1a494f`. Their code is reproducible
from the local files; Kaggle's draft deletion itself is irreversible. Neither had
run, and the actual benchmark, V11, and V13 were not deleted or altered.

## Completed output verification

Original files were retrieved using the exact Download links offered by Kaggle's
UI (21 files). CSV preview truncation was not accepted as complete data. Temporary
download signatures were not retained. Fifteen distinct CSVs match run-receipt
hashes, all required tables have 58 aligned studies, and all six actual saved
candidate files replay within 1.11e-16. Independent AUC agrees with scikit-learn
within 1.11e-16. All six 2,000-resample paired deltas and intervals agree exactly
with the saved Kaggle report. See `gold_independent_verification.json` and
`gold_run_audit.json`. The former includes the sanitized source/hash manifest.

Control macro AUC: **0.9594482823764294**. Largest fixed-candidate point gain:
remove WideDense (`without_arm_1`), AUC **0.9606708681941164**, delta
**+0.0012225858176867**, exploratory 95% interval
**[-0.0018327998964697, +0.0046211809197396]**. Its saved CSV changes 450/696
numeric values and rankings in all 12 targets. All six candidates change rankings;
none has a positive 95% lower bound or a +0.020 point gain. No promotion occurred.

The audit observed 57 loads / 34 unique checkpoint paths and zero explicit ID
lists. Stage-1 gold AUC 0.995752 is not held-out proof of superiority: exposure is
unknown. Source-documented CoAtNet gold checkpoint selection also prevents calling
this independent confirmation. Lowest control AUCs are PF OA, lateral OA,
synovitis, and lateral meniscus; these inform error review, not assumptions about
training counts.

Final local regression suite: **25 tests passed**. The bundled analysis Python
lacked scikit-learn; the existing project Python was used for independent scoring
and regression tests. The production V14 default remains the unchanged control;
candidate outputs are separate diagnostic files. Full results and next experiment
criteria are in `RESULTS.md` and `PLAN.md`.
