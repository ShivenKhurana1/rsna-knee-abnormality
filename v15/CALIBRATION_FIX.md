# Fold-local calibration fix for report auxiliary labels

## Outcome

Strict source consensus and convex source blending failed the paired expert
audit. The useful remaining label intervention is **fold-local Platt
calibration of the existing Steven soft labels**. It changes calibration, not
ranking, and therefore addresses the quantity consumed by the auxiliary BCE
loss without pretending to create new image evidence.

This is a change specification, not an implemented or leaderboard-proven gain.

## Evidence that authorizes an image-model ablation

The 58 expert rows were reconstructed with exact SHA-256
`a3ca7df32d43ee7c683091162b1839ea8308d802bea3f2b437f4908d9cbcb079`.
Using the existing five grouped folds, each validation row was transformed by a
calibrator fitted only on expert rows from the other four folds.

- addressed-cell macro Brier before calibration: `0.1790975`;
- addressed-cell macro Brier after calibration: `0.1346528`;
- paired delta (after minus before): `-0.0444446`;
- 5,000-study-bootstrap 95% CI: `[-0.0623821, -0.0259291]`;
- bootstrap probability that calibration lowers Brier: `1.000`;
- every one of the 12 target Brier scores improved out of fold.

This result is exploratory because the intervention was investigated after
examining this cohort. Cross-fitting prevents row leakage, but it does not turn
the same 58 studies into independent confirmation. It authorizes a paired
image-model ablation only; it is not a forecast of AUC or leaderboard gain.

## Exact fix to implement

For each Family A outer fold and each target:

1. Start from the Steven report soft values already used by the auxiliary arm.
2. Exclude report-silent `0.5` cells from fitting and leave them masked from the
   loss after calibration.
3. On expert studies belonging to the outer fold's training partition only,
   transform addressed probabilities with `logit(clip(p, 1e-4, 1-1e-4))`.
4. Fit a target-specific logistic regression with intercept, fixed L2
   regularization `C=1.0`, `lbfgs`, and no hyperparameter search.
5. Apply that fitted map only to addressed auxiliary probabilities used to
   train the current outer fold. Never fit or tune it on the outer validation
   expert rows.
6. If the fitting subset has only one expert class or the solver fails, retain
   the identity map for that target and record the fallback in the receipt.
7. Keep the existing transfer policy selected from the **raw** source audit.
   Do not use in-sample calibrated Brier to unlock extra targets or inflate
   auxiliary weights.
8. Preserve all current safeguards: expert cells override report cells,
   validation-fold rows contribute to no loss, target weights remain capped,
   and baseline/raw/calibrated arms share initialization and sample order.

The critical implementation boundary is step 7. Calibration can improve the
soft target values, but using its in-sample fit statistics to widen the policy
would combine calibration fitting and target selection on the same small fold.

## Required GPU comparison

Train three arms on the same 2 seeds x 5 folds:

- `baseline`: expert labels only;
- `raw_aux`: current transfer-gated Steven labels;
- `platt_aux`: identical to `raw_aux` except for fold-local calibration above.

Evaluate paired OOF predictions, not best-epoch gold selection. Promote
`platt_aux` only if:

1. it beats `raw_aux` by at least `+0.001` macro AUC in both seeds;
2. it beats `baseline` in both seeds;
3. no target loses more than `0.03` AUC versus `raw_aux` in both seeds;
4. every receipt proves the calibrator used training-fold expert IDs only; and
5. the resulting model adds positive paired OOF gain when blended with the
   deployed ensemble.

Only a scored competition submission can establish the public-leaderboard
effect. The Brier reduction must never be added numerically to AUC.

## Score error propagation

For the eventual paired blend comparison, report

`SE_combined = sqrt(SE_paired_bootstrap^2 + (SD_seed / sqrt(n_seeds))^2)`.

This accounts for finite validation studies and two-seed training variation. It
does **not** account for validation-to-public-leaderboard transport error or the
public/private split. A displayed move from `0.935` to `0.950` requires a latent
gain between approximately `+0.0140` and `+0.0160` under nearest-thousandth
rounding. To prove a latent score of at least `0.950`, target a displayed
`0.951`, because displayed `0.950` can represent a latent value as low as
`0.9495`.
