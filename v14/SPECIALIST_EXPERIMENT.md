# Patient-held-out specialist experiment

Requested 2026-08-30: actual new specialist training and patient-held-out evaluation,
minimizing operational/regression risk while pursuing the largest defensible gain.
No guarantee of improvement and no leaderboard submission authorization inferred.

## Recorded starting evidence

Private Kaggle run 346064640 completed all stages on 58 expert-labeled studies in
12m 24s. V13-recipe control macro AUC was 0.959448. Removing WideDense was the best
of six fixed ablations: 0.960671, delta +0.001223, exploratory 95% interval
[-0.001833, +0.004621]. Its saved CSV changed 450/696 values. All six candidates
changed rankings in all 12 findings. None established +0.020. See `RESULTS.md`.

Gold was already used for checkpoint selection. Across 34 loaded checkpoint
paths, no explicit train/validation ID lists were found. Stage-1 gold AUC 0.995752
does not establish held-out superiority. Do not promote it on this evidence.

## First gates: before training

1. Establish a de-identified study-to-patient map from a trustworthy source.
   `StudyInstanceUID` is a study identifier, not automatically a patient identifier.
   Distinct anonymous IDs alone do not prove distinct people. Do not reconstruct
   identities from reports/demographics or relabel study folds as patient folds.
2. Keep every study, knee, series and augmentation from a patient in one partition.
   Verify no duplicates/overlap before feature extraction or model fitting.
3. Establish exclusion for every learned component used in the comparison:
   encoders, baseline heads, specialist heads, calibration, and model selection.
   A new head's cross-fitting does not repair exposure in frozen V13 predictions.
4. A fair comparison to exact deployed V13 needs an external excluded cohort or
   its complete verified exclusions. Retraining a clean reference instead changes
   the comparator; report that explicitly, never call it V13.
5. The 58 inspected gold cases can be development data, not a newly untouched
   confirmation set. Weak report labels remain weak/soft, not expert ground truth.

## Minimal-risk first training candidate, contingent on those gates

- Preserve V13 and all previous run artifacts; private outputs, no auto-promotion.
- Focus error analysis on PF OA, lateral OA, synovitis and lateral meniscus.
- First use a frozen encoder with verified pretraining provenance and separate
  plane/region pools, then fit strongly regularized target-specific residual heads.
  Do not fine-tune all existing ensemble weights as the first experiment.
- Fit normalization, hyperparameters, target selection and early stopping only on
  training/inner-development patients. Retain an outer patient-held-out evaluation.
- If regional frozen features fail, consider bounded encoder fine-tuning only
  after image review supports a representation bottleneck and timing fits quota.
- Hold unaffected targets at their baseline outputs. Predeclare candidate settings
  and regression margins before scoring a new confirmation set. Low operational
  risk does not imply a guaranteed nonnegative AUC delta.
- Report all 12 target counts/AUCs, same-patient paired bootstrap, changed rankings,
  calibration checks and runtime. Freeze one recipe before final confirmation.

## Current status

The CPU patient-tag audit completed successfully. No new specialist has been
trained by this experiment and no actual patient-held-out gain is claimed.
The user subsequently approved proceeding with a clean-reference experiment and
provisional PatientID grouping. This does not waive the limitations on claims.

The competition's published CSV schema supplies study/series IDs, not a patient
map. One original training DICOM was downloaded privately and inspected without
decoding pixels: PatientID is nonempty, is not StudyInstanceUID, and is not a
generic placeholder. IssuerOfPatientID is absent in that sample. This is a
feasibility sample, not proof of cohort-wide patient identity preservation.
Source: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data

`rsna-knee-v14-patient-audit.ipynb` was uploaded to a separate private notebook
with only the competition attached, CPU acceleration (None) and internet off.
It checks the first/last header in every training series, flags inconsistent or
missing patient IDs, and exports salted-hash grouping tokens, never raw PatientID
values. Its four synthetic DICOM tests pass. Uploaded SHA-256:
`d2cf3130f1f8b9003058a12e4822dd322f8e930f059502a1dcfba95875628f05`.
Exact completed run:
https://www.kaggle.com/code/seanzhang2445/rsna-knee-v14-patient-audit?scriptVersionId=346132000

In 241.4 seconds it checked **48,742 headers from 24,371 series across all 4,407
training studies**. All studies had consistent, nonmissing tags. There were
**4,407 distinct PatientID-derived groups, zero multi-study groups**, and no
recorded issues. All 58 gold studies have grouping tokens. This proves usable tag
coverage, not that the anonymizer preserved identity across repeat examinations.
Every group is a singleton, so group splitting these tokens currently produces
the same partitions as study splitting; calling that confirmed patient exclusion
would overstate the evidence.

All three original output files were downloaded privately. Local recomputation
verified the map's SHA-256 against the remote receipt, exact agreement with all
4,407 private audit records, UID uniqueness, nonmissing hashed groups, header/
series totals, and coverage of all 58 previously downloaded gold IDs.
Aggregate evidence: `patient_audit_result.json`. Raw IDs were never exported;
the hashed study mapping stays in git-ignored `private_artifacts/`.

All **29 local tests passed**, including four synthetic DICOM audit tests. These
are software checks, not new model-accuracy measurements.

An existing account artifact, `RSNA Knee V14 Specialist Pool` version 346076064,
was also inspected read-only. It completed V13-style prediction exports on 500
studies in 2h 1m 6s. Its logs say feature export only; this is not evidence that a
new specialist was trained or that its underlying encoders excluded those
patients. It was not modified, rerun, or claimed as work performed in this turn.

Decision received: proceed with an explicitly named clean reference, not exact
V13, using the audited PatientID groups provisionally. Benchmark measured gains
and potential, without turning an exploratory result into a leaderboard forecast.

## What is needed to resume actual training

For a confirmed patient-held-out claim, obtain a source-backed guarantee that
the supplied PatientID is stable across examinations, or an authoritative
de-identified study-to-patient mapping. No raw personal identities are needed.

For a fair comparison to exact V13, supply verified excluded labeled cases or
complete exposure records for all its learned components. If those cannot be
obtained, the recommended alternative is an explicitly named clean reference:
an external-pretrained encoder with no RSNA cohort fitting, plus new reference
and regional specialist heads trained under the same group split. Until patient
linkage is confirmed, any such run is **provisional PatientID-grouped research**,
not verified patient-independent evidence and not proof of beating exact V13.

The first candidate should use frozen features and a small predeclared inner-fold
regularization search, keep PF OA/lateral OA/synovitis/lateral-meniscus heads
separate, preserve the other reference outputs, and export model/feature hashes,
partition manifests, all-target metrics and failed candidates. Gold can support
exploratory cross-fitted evaluation, not untouched confirmation after prior use.
Do not replace production V13, submit to Kaggle, or claim +0.020 automatically.

## Approved first pilot: fixed before new training

`clean_specialist.py`, `clean_features.py`, and
`rsna-knee-v14-clean-specialist.ipynb` implement the bounded pilot:

- 58 expert-labeled MRI studies, no weak/report-derived labels and no V13
  predictions/weights. This is a small-sample learning experiment, not a complete
  retraining of the original ensemble on all 4,407 studies.
- Frozen external `facebook/dinov2-small`, revision
  `ed25f3a31f01632728cabb09d1542f84ab7b0056`. Source/model hashes are recorded.
  Public pretraining provenance is known; exact image-level membership is not
  independently audited. No cohort adaptation of the encoder is performed.
- Up to two series per sagittal/coronal/axial plane, 12 physically ordered slices
  per series sampled over its central 10–90% span. Full FOV, aspect-preserving
  336-pixel input, percentile intensity scaling, explicit MONOCHROME1 handling.
- Reference: per-plane global CLS mean/max. Specialists: the same global features
  plus mean/max pools of four image-coordinate patch quadrants per plane. These
  are spatial features, not claims of verified anatomical segmentation.
- Five label-independent outer group folds; three inner folds. All scaling,
  regularization and blend selection fit only inner/outer training patients.
  New L2 logistic heads are fitted and saved, using an exact row-space solver.
- L2 grid: 1, 0.1, 0.01. Blend grid: 0, 0.25, 0.5, 1; nonzero blend must improve
  inner AUC by at least 0.01 without worsening inner BCE by more than 0.01.
  Other eight targets stay exactly at the clean reference outputs.
- Primary: inner-selected blend. Prespecified secondary benchmarks: fixed 25%
  specialist and regional-only predictions on the four focus targets. Report all
  three; do not choose a winner after seeing outer labels.
- Export all outer models, OOF predictions, folds, feature/source hashes,
  extraction timing, target support, AUC/BCE/Brier and paired 2,000-draw intervals.
  Intervals resample fixed OOF predictions and omit training/fold uncertainty.
- Six new software tests pass, including outer-label perturbation, exact
  unaffected-target preservation, weight reload and direct-primal equivalence.

Sources: https://huggingface.co/facebook/dinov2-small and
https://huggingface.co/docs/transformers/model_doc/dinov2 (global/patch features);
https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
(nested model selection). There is no defensible numeric forecast of this
pilot's gain on the hidden leaderboard before measuring transfer to exact V13.

### Execution status of the approved pilot

All 35 local software tests pass. A separate private Kaggle draft named
`RSNA Knee V14 Clean Specialist` has the competition and patient-audit output
attached, T4 x2 selected, and internet enabled to retrieve the pinned public
encoder. No saved GPU training run has been started.

The first draft import succeeded. A security check blocked uploading the final
preflight-optimized notebook build, because this transfers local source code to
Kaggle. Do not bypass that rejection by pasting code or executing another upload
path. Explicit user approval for this source-code transfer is required before
continuing. The upload notebook contains program source and source hashes, not
embedded MRI images, reports, raw PatientIDs, credentials, or the private group
mapping. The latter is read from the already-private Kaggle audit input at run time.

Final local build awaiting upload:
`rsna-knee-v14-clean-specialist.ipynb`, SHA-256
`987138b89010a70f99c91667022383fa2190eab164cabd607b7bb42c83e14800`.
`clean_specialist_build.json` records the frozen recipe and source hashes.
`verify_clean_specialist.py` is ready to reload saved fold models, verify the
partitions, reconstruct OOF predictions, optionally retrain heads locally, and
independently recompute all paired metrics. No new real-data accuracy result or
numeric leaderboard forecast exists yet.

Explicit approval received in the next user message: upload this notebook source
to the user's private Kaggle account and run the GPU benchmark. Proceed with the
approved transfer; preserve private visibility and do not submit to competition.

The approved final upload succeeded and a private T4 x2 Save & Run All job was
started (version 1, "Clean heads nested grouped pilot - no V13"). Status observed:
Running. This is not yet a completed accuracy result.
