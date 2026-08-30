# V14: completed diagnostics; +0.020 not demonstrated

The requested **V13 +0.020 macro-AUC has not been achieved or verified**. No hidden-test
score can be guaranteed. This folder deliberately separates runnable diagnostics,
a trainable model component, synthetic tests, and actual knee-MRI accuracy evidence.

The private Kaggle benchmark completed on 58 labeled studies: control AUC
**0.959448**, best fixed arm ablation **0.960671** (delta **+0.001223**, exploratory
95% interval **[-0.001833, +0.004621]**). All six saved candidate CSVs change
rankings; the best changes **450/696** values. None meets the improvement gate.
See `RESULTS.md`, `gold_independent_verification.json`, and `gold_run_audit.json`.
These gold studies were used for checkpoint selection and training exclusion
is unverified across the full ensemble. This is not a new leaderboard score.

## Deliverables

- `rsna-knee-v14-clean-specialist.ipynb`: newly implemented private clean-reference
  training pilot, not exact V13. Frozen external DINOv2 features, new general and
  regional heads, five outer/three inner provisional PatientID-grouped folds.
  The approved private GPU run is version **346135806**; feature extraction and
  all five outer-fold fits have completed, with scoring/verification in progress.
  See `SPECIALIST_EXPERIMENT.md` and
  `clean_specialist_build.json`; verification code is `verify_clean_specialist.py`.
- `SPECIALIST_EXPERIMENT.md`: recorded benchmark findings and the new specialist
  experiment's data/provenance gates. CPU audit version 346132000 checked 48,742
  DICOM headers across 4,407 studies; see `patient_audit_result.json`. Patient-tag
  coverage passed, but repeat-patient identity preservation and V13 exclusions
  remain unverified. Clean-reference training uses these groups provisionally.
- `rsna-knee-v14-gold-validation.ipynb`: separate private, exploratory labeled
  validation run on Kaggle; see `KAGGLE_VALIDATION.md` for execution history and
  training-exposure caveats. It must not be submitted to the leaderboard.
- `rsna-knee-ensemble-v14.ipynb`: self-contained Kaggle V13 control, with same-run
  single-arm/leave-one-CoAtNet-out exports. Default `submission.csv` is still V13;
  this notebook is NOT an improved submission masquerading as a new version.
- `residual_specialist.py`: actual new target-specific residual logistic heads on
  supplied frozen MRI features. Each target gets its own coefficients; missing
  labels are masked; standardization and fitting occur inside group-held-out folds.
  A target with insufficient positive/negative examples falls back exactly to its
  baseline. No real feature extractor/weights/data are bundled, and the head is
  not yet integrated into end-to-end DICOM inference.
- `diagnostics.py`: exact numerical/rank comparison, all 12 AUCs, target-level
  headroom, paired study/group bootstrap intervals. Does not tune candidates.
  `require_changed_rankings` rejects an identical candidate and cosmetic numeric
  changes that preserve every ordering. Passing it does not establish better AUC.
- `confirm.py`: separate confirmation gate requiring expert image labels,
  training/selection exclusion manifests and a frozen candidate artifact hash.
  Completeness/timing of declarations still need independent audit.
- `audit_and_benchmark.py`: reproducible source audit, numerical correctness and
  synthetic performance tests. Synthetic gains are NOT evidence of a knee-model gain.
- `PLAN.md`: concrete experiments, acceptance criteria and stop rules.

## Why V13 can still display 0.935

V13 retained the three CoAtNet checkpoints and equal arm weights, with a 50/50
CoAtNet/transformer outer rank blend. It corrected the Native384 cache resolution
and WideDense slice-span/window-count contracts. It did not retrain weak targets.
Input fixes can change probabilities without changing their ordering; repeated
rank fusion discards such changes. Changes that survive fusion may be too small
to affect a three-decimal score, or gains/losses across targets can cancel.
These mechanisms must be distinguished using the exact run artifacts, not guessed.

### Actual authenticated comparison, 2026-08-30

Inspected V11 **345726369** and V13 **345967510** in the signed-in Kaggle browser.
Both display public score **0.935**. Both visible runs completed all three CoAtNets.
Their final CSV tables contain the same three UIDs and **all 36 numeric values are
exactly equal at the full precision displayed by Kaggle**. This is numerical
table equality; downloaded original file-byte equality has not been verified.

The intermediate CoAtNet blends are NOT identical: six logged cells changed,
in Lateral Meniscus, Lateral OA and Contusion. Final fusion erased the differences
on these rows. See `actual_output_comparison.json` and
`actual_pipeline_comparison.json`. Hidden-test predictions remain unavailable;
these three cases cannot establish why the exact hidden AUC did/didn't move.

The remote V11 log includes shared resized-window caching and lacks the local
V11 audit receipt. Do not equate that uploaded source with today's local V11.
The remote V13 receipt's builder/contracts/parent hashes match the local build
metadata, though that does not by itself hash every executed remote cell.

The real V13 Stage-3 and three raw-arm tables were also captured. Replaying them
through V14's control fusion gives **maximum absolute error 0.0** against the
actual saved final output. `actual_ablation_replay.json` records six label-blind
ablations. This is real prediction replay, NOT a new GPU run or evidence of gain.

The repository's `r=0.7258` is agreement of a separate, failed DINO reconstruction
with reference predictions on 12 studies. Squaring it gives `0.5268`, not “53%
perfect” for V13. Pearson correlation squared is also not generally the same as
`r2_score(y_true, y_pred)`. This competition ranks 12 binary findings by macro ROC-AUC.

See the [ROC-AUC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
and [R² documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).

## Local verification

Requires Python, NumPy, pandas, SciPy and scikit-learn. No GPU or medical data is
needed for the tests below; that is precisely why they cannot establish clinical accuracy.

```powershell
python v14/build_v14.py
python -m unittest discover -s v14 -p "test_*.py" -v
python v14/audit_and_benchmark.py --output v14/local_benchmark.json
python v14/verify.py --output v14/verification.json
```

Reports use exclusive creation: choose a new output filename for subsequent runs.

Reproduce the saved-run analyses locally (private captures are git-ignored):

```powershell
python v14/compare_saved_outputs.py --v11 v14/private_artifacts/v11_submission_dom.json --v13 v14/private_artifacts/v13_submission_dom.json --output v14/output_comparison_rerun.json
python v14/analyze_saved_runs.py --artifacts v14/private_artifacts --output v14/pipeline_comparison_rerun.json
python v14/replay_saved_predictions.py --artifacts v14/private_artifacts --output v14/ablation_replay_rerun.json
python v14/verify_gold_artifacts.py --captures v14/private_artifacts/gold_346064640 --script-version 346064640 --downloaded --output v14/gold_verification_rerun.json
python v14/audit_gold_downloads.py --downloads v14/private_artifacts/gold_346064640 --script-version 346064640 --output v14/gold_audit_rerun.json
```

## Compare actual outputs

```powershell
python v14/diagnostics.py --baseline path/to/v11.csv --candidate path/to/v13.csv --output v14/csv_difference.json
python v14/diagnostics.py --labels path/to/heldout_labels.csv --baseline path/to/v13.csv --candidate path/to/v14.csv --groups path/to/groups.csv --output v14/heldout_report.json
```

Prediction/label CSVs require `StudyInstanceUID` plus the exact 12 target columns.
IDs are strings, including leading zeros. Labels must be binary or missing, never
thresholded weak probabilities. Every target must have both classes. Group files
require `StudyInstanceUID,GroupID`; use patient identity when multiple studies
belong to a patient. Missingness and rare positives are reported, not hidden.

## Train the specialist research component

Supply an NPZ containing Unicode `StudyInstanceUID` and a finite `features` array
of shape N×D, plus keyed baseline, labels and groups CSVs:

```powershell
python v14/residual_specialist.py --features data/features.npz --baseline data/v13_oof.csv --labels data/labels.csv --groups data/groups.csv --out v14/training_runs/experiment01
```

This produces `specialist_oof.csv`, `folds.csv`, `residual_head.npz`, and a training
receipt. Cross-fitting the head does NOT repair leakage from an encoder or V13
baseline trained on those validation patients. Full-trained CoAtNets cannot be
scored on their training cohort and relabeled OOF. Also audit cross-fold model
dependencies before interpreting stacked OOF results; independent confirmation
is required. The default regularization is fixed in advance; changing it uses
development data only. Inference API: `predict(model, features, baseline)`.

## Confirm one frozen candidate

Prepare an evidence JSON with:

```json
{
  "label_quality": "expert_image",
  "candidate_frozen_before_confirmation": true,
  "confirmation_labels_used_for_selection": false,
  "all_component_training_and_selection_ids_included": true,
  "provenance_notes": "Describe every encoder/head/base model, exclusions, dates and freeze.",
  "exclusion_manifests": [{"path": "training_and_selection_ids.csv", "sha256": "REAL_SHA256"}],
  "frozen_candidate_artifact": {"path": "frozen_recipe_or_model.json", "sha256": "REAL_SHA256"}
}
```

Exclusion CSVs contain `StudyInstanceUID,GroupID` for all supervised training and
selection exposures of every baseline/candidate component. Paths resolve relative
to the evidence JSON. The frozen artifact should pin all model/preprocessing hashes.

```powershell
python v14/confirm.py --labels data/confirmation_labels.csv --baseline data/v13_confirmation.csv --candidate data/v14_confirmation.csv --groups data/confirmation_groups.csv --evidence data/evidence.json --output v14/confirmation_result.json
```

The strict gate requires the paired 95% CI lower bound to exceed +0.020 on that
confirmation cohort. Passing is cohort evidence, never a hidden-test guarantee.
Do not repeatedly test candidates on this cohort and retain only the winner.

## Kaggle run

Import the notebook in the ZIP, attach the metadata-listed V13 inputs, use T4 x2,
internet off. It retains V13's current-run recovery and input contracts. The
additional final cell exports cheap CPU ablations into
`v14_diagnostics/<run_id>/ablations/`. It requires the reconstructed V13 output
to match the saved baseline before exporting. The submission is byte-preserved.
No final competition submission is made by any script here.

Three visible test studies can confirm execution and output differences but not
AUC or hidden-test runtime. For real validation, use an audited held-out image
cohort and matching inference inputs; this package does not silently substitute
the training images for test images.

Private browser-extracted artifacts live under git-ignored `private_artifacts/`.
Original model/community attribution remains in the notebook and root README.
