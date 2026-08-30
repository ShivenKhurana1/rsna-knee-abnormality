# V14 improvement plan: target +0.020, no guaranteed score

## Success criterion

Increase V13's exact macro ROC-AUC by at least 0.020 on an independent,
expert-image-labeled cohort; report all 12 targets and paired patient/group
bootstrap uncertainty. Require the 95% lower bound >=0.020 before calling the
improvement statistically supported. A hidden Kaggle score remains unknown
until the platform evaluates the frozen notebook. Never sum gains measured
against different baselines or on different cohorts.

0.935 -> 0.955 is a 30.8% reduction in `1 - AUC` (pair-ranking error), NOT a 2%
increase in diagnostic accuracy. Across 12 targets it needs 0.24 total AUC points:
four +0.060 gains or eight +0.030 gains are illustrative budgets, not forecasts.

## 1. Establish the actual baseline before optimization

Pin the user-linked versions V11/345726369 and V13/345967510. Verify displayed
scores, input attachments, completion logs, saved prediction values and hashes.
The current local V11 need not be the uploaded V11: distinguish them explicitly.
Compare all output studies by UID, not row order. Report exact-value differences,
per-target ordering changes, constant outputs and missing/failed members. Both
saved notebook outputs cover only three visible studies, not hidden submission
predictions. Never treat their comparison as an accuracy benchmark.

## 2. Build a validation set that can answer the question

Inventory each target's positive/negative/missing counts, label source, scanner,
protocol, plane coverage and patient/site groups. Trace train/validation exposure
for every encoder and head. Keep train/development/confirmation patients separate.
Use report-derived labels for development with an explicit weak-label tag; do not
call them expert labels or convert soft probabilities to hard truth to pass a gate.

The historical 58 gold cases have already been inspected in this project and are
not a pristine final confirmation set. Rare positives make bootstrap estimates
unstable; determine sample needs from observed class frequencies and paired
prediction variance. No unsupported universal sample-size guarantee. Full-trained
CoAtNets require a genuinely external held-out cohort or true fold retraining.

## 3. Localize the bottleneck across the twelve findings

Run one frozen V13 inference per cohort and retain raw family predictions, slot
coverage, slice spacing, laterality decisions, decode failures and arm completion.
Use same-run leave-one-out outputs to measure whether each CoAtNet has marginal
value. Evaluate additions conditional on V13, not by their standalone AUC.

Priority is determined by measured target deficit and uncertainty, NOT the worst
target from a failed unrelated reconstruction. Review errors by target category:

| Findings | Testable model/supervision experiment |
|---|---|
| ACL, MCL | Ligament-focused crops; preserve continuity and plane coverage; audit severity ambiguity |
| Medial/lateral meniscus | Higher-resolution compartment features; preserve adjacent-slice evidence |
| Medial/lateral/PF OA | Compartment-specific cartilage features; audit severity-label disagreement |
| Effusion, synovitis, Baker's | Distinguish location/severity labels and fluid-sensitive coverage |
| Contusion, fracture | Preserve marrow/cortical detail; audit acute/chronic and uncertain mentions |

These are research hypotheses, not clinical labeling rules. Obtain the current
host definitions and expert review before rewriting labels. Do not impose new
cropping/pooling on existing checkpoints without reproducing their training contract.

## 4. Train one materially different feature/specialist candidate

First extract frozen MRI features from an eligible, reproducible encoder with
features outside the current global pooled representation: separate per-plane/
sequence/compartment pools and missing-slot masks. Do not download/execute unknown
pickle checkpoints; verify provenance and competition eligibility.

The implemented residual head adds target-specific image-feature corrections to
the baseline score. This is the cheapest test of useful information in those
features. Fit within grouped folds; uncertainty labels are masked. Audit whether
high residuals correspond to real visual errors or label noise. Freeze feature
definition and regularization before confirmation. If global frozen features
cannot resolve the misses, escalate to a series-aware trainable image encoder,
then compartment-specific heads; this is a NEW training project, not another
blend-weight edit. No trained specialist has been produced yet.

Read-only timing history suggests old training was storage-bound. Benchmark
100-200 representative studies on local scratch cache with a checksum-backed
manifest, pinned preprocessing, fp16 T4 and staged reads. Measure actual read vs
forward/backward time before planning GPU hours. Do not extrapolate an epoch
from a three-study inference run. Do not assume all Kaggle training is impossible
because an old network-storage pipeline was slow.

## 5. Predeclared experiment ladder and stop rules

1. Reproduce V13 output and targets exactly. Stop on unknown preprocessing,
   unmatched rows, nonfinite values or incomplete inference.
2. Screen the fixed same-run arm ablations on development data. They identify
   redundancy; historical evidence does not justify expecting +0.020 from them.
3. Test one new-feature residual candidate against V13 on grouped development
   folds. Reject a candidate that only improves its own training predictions.
4. Add targeted fine-tuning only where image review and coverage/label audits
   support the bottleneck. Verify two fixed seeds to detect seed-only gains.
5. Freeze ONE complete recipe (all model/feature/preprocessing hashes), then
   evaluate the untouched confirmation cohort once. If the +0.020 gate fails,
   report that failure; do not rename a smaller gain as success.
6. Benchmark the frozen end-to-end notebook at representative cohort size under
   actual offline limits. Record peak memory, seconds/study and complete-member
   coverage; preserve the control if the new recipe cannot finish.
7. Ask for/receive authorization for the external upload/run or final submission
   as required by the execution surface. Never consume paid resources implicitly.

## Evidence rules

`audit_and_benchmark.py` tests source and synthetic mechanics, not knee accuracy.
`residual_specialist.py` supplies research code, not a trained clinical model.
`diagnostics.py` reports real AUC only when supplied with real held-out labels.
`confirm.py` refuses declared training/selection overlaps and weak labels, but
cannot prove that manifests are complete or truthful. Independent audit remains
necessary. The old `oof_harness/tier1_analysis.py` chooses `best_dw` on the same
cohort later bootstrapped; its intervals are not selection-adjusted and should not
be treated as independent confirmation of the selected weight.

## Current remaining work

Real images/features, trustworthy train/selection exclusion records and a suitable
held-out labeled cohort are needed for training/accuracy benchmarking. Authenticated
Kaggle access permits inspecting existing artifacts; it does not by itself supply
hidden-test labels. No proposed approach guarantees a 0.955 public/private score.

### Authorized follow-up, 2026-08-30

The user authorized private Kaggle GPU diagnostics, without leaderboard submission.
`KAGGLE_VALIDATION.md` tracks the exact runs and the source-parity correction.
The 58 fully expert-labeled studies are now available for exploratory same-cohort
evaluation; CoAtNet source explicitly reuses gold AUC for checkpoint selection.
This is not independent confirmation. The predeclared experiments are three
single-CoAtNet candidates and three leave-one-CoAtNet-out candidates, all fused
with the same V13 Stage-3 output. No label-fitted weights are being promoted.

Output-difference requirement: compare exactly aligned study IDs, preserve V13
as the control, and reject a candidate that is numerically identical or only
changes probabilities while preserving every ranking. Independently replay from
saved raw predictions and validate AUC against scikit-learn. A changed CSV alone
does not pass the improvement gate. Gold findings identify error-analysis
priorities; they do not by themselves establish which targets were undertrained.
