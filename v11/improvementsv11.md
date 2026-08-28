# improvementsv11.md — V11 Accuracy / Model-Quality Audit

## Scope

This file is intentionally **accuracy-only**. It does not optimize wall-clock time, memory, or Kaggle efficiency-track score except where a reliability bug can directly destroy the prediction file. Runtime-only changes belong in `efficencyv11.md`.

The audit was performed against the repository's V10 notebook, V9 notebook, repository README, OOF harness notes, timing notes, and current Kaggle competition documentation/discussions. The current public reference is V9 at approximately **0.935 public leaderboard AUC**. V10 was still running at the time of this audit.

The key fact for V11 planning is that **V10 has only one declared modeling change over V9: the new content-centroid crop recentering**. The other V10 changes are robustness/correctness fixes. Therefore, if V10 moves materially relative to V9, the crop change is the first place to investigate, not the backbone ensemble.

---

# Executive recommendation

## Safest V11 starting point

Build V11 from:

1. **V9's original fixed-center crop behavior** as the accuracy baseline.
2. **Keep V10's genuine correctness fixes**, especially dropping a Stage-4 arm that is cut off before completing all studies.
3. Fix the additional silent-score hazards identified below before adding another model family.
4. Re-evaluate all hand-tuned pooling/weighting on the 4,349-study OOF harness; do not tune them on the 58-study gold gate or public leaderboard.
5. Add a genuinely decorrelated family only after its inference is reproduced exactly. The repo's OOF work identifies `surasan092` DINOv3-S as unusually decorrelated, but its head has **not** yet been reconstructed exactly; do not blend the current approximate reconstruction.
6. For future retraining, address weak report-label noise, slot-boundary windowing, and physical/DICOM preprocessing before simply scaling the backbone.

### Highest-priority V11 accuracy changes

| Priority | Change | Why it matters | Deployment rule |
|---|---|---|---|
| P0 | Revert V10 content-centroid recrop to V9 fixed center by default | Sole declared V10 modeling delta; frozen checkpoints were trained on fixed center | Re-enable only if paired OOF proves benefit |
| P0 | Never recenter each slice independently | Stage 2 and Stage 4 can introduce artificial inter-slice translation | If recentering is retained, estimate one center per series |
| P0 | Preserve the last valid Stage-1 ensemble after exceptions | Current top-level handler can overwrite a banked valid ensemble with all 0.5 | Never replace valid predictions with fallback |
| P0 | Replace `first matching series` selection | Stage 2 and Stage 4 can choose arbitrary duplicate series | Deterministic quality selector, identical train/inference |
| P0 | Fix slice ordering when some DICOM geometry is missing | Stage 1 can fall back to raw filesystem order | Mixed robust geometry/InstanceNumber fallback |
| P0 | Stop banking partially evaluated Stage-1 members | Runtime can change the mathematical model | A member is full-definition or absent |
| P0 | Remove/re-prove 15× target weights | Repo OOF says weight tuning did not generalize | Equal/capped weights unless nested OOF proves otherwise |
| P1 | Make Stage-4 windows slot-boundary aware and retrain | Current global 3-slice windows can cross series/plane boundaries | Retraining change, not frozen-checkpoint inference patch |
| P1 | Replace report-label BCE with noise-aware supervision | Image labels and reports can disagree | Cross-fit confidence/masking + gold-aware fine-tuning |
| P1 | Freeze Stage-3 blend/stack parameters from OOF | Test-set adaptive weighting can chase test-distribution quirks | No test-derived blend tuning without OOF proof |
| P1 | Add a truly decorrelated family | OOF evidence says diversity paid; weight tuning did not | Exact reconstruction + large-OOF ablation first |

---

# 1. V10's new content-centroid crop should not be the default V11 crop

## What V10 changed

V9 crops the selected volume around `(height // 2, width // 2)` using physical crop size. V10 added `_content_crop_origin`: it thresholds a reference image at **8% of that image's maximum**, computes the centroid of pixels above threshold, and nudges the crop toward that centroid, capped at **25% of the crop half-width**.

The V10 notebook explicitly says this change was **not validated with retraining or an OOF re-score** and also acknowledges that all frozen checkpoints were trained with plain center crops.

This is the most important V10-vs-V9 difference because the input distribution changed while the weights did not.

## Why the current implementation is risky

### 1.1 `0.08 * max` is not an anatomy detector

A max-relative threshold can move substantially with:

- bright fluid,
- fat signal,
- coil shading,
- a single bright artifact,
- sequence type,
- clipping,
- intensity rescaling,
- different field of view.

The centroid of `image > 0.08 * max(image)` is therefore a **foreground-intensity centroid**, not necessarily a knee-joint centroid.

### 1.2 Frozen checkpoints were trained on a different crop distribution

Even a medically more sensible crop can reduce AUC when introduced only at inference. A frozen network can encode stable background/FOV/context patterns from its training preprocessing. The correct comparison is not "does the new crop look better?" but "does the frozen model rank positives and negatives better under the new crop?"

### 1.3 Stage 2 and Stage 4 recenter each slice independently

This is more serious than a one-time volume shift.

Stage 2's `read_crop(path)` computes the threshold/centroid separately for every DICOM slice. Stage 4's `mm_crop_resize(a, ps)` also computes a separate centroid for each slice. Three neighboring slices are later stacked as channels. If their crop origins differ, static anatomy can appear to translate between channels.

That creates synthetic inter-slice motion that was not present in the MRI and was not part of the fixed-center training distribution.

### 1.4 The implementation assumes the common case receives approximately zero shift, but does not test that assertion

The code has no distributional QC such as:

- shift histogram by series/plane,
- percentage of slices shifted,
- median shift in millimeters,
- inter-slice shift variance,
- crop-border anatomy contact before/after,
- prediction-rank delta vs V9.

Without those measurements, the blast radius is unknown.

## V11 solution

### V11-A: safest deployment

Restore **exact V9 fixed-center cropping** for all frozen V9/V10 checkpoints. Keep every non-modeling V10 correctness fix.

This makes V11 a controlled baseline: if V10 underperforms 0.935, the most likely modeling regression is removed immediately.

### V11-B: if off-center handling is still desired

Do not center independently per slice. Compute a single transform per series:

1. Select 5-9 representative slices distributed through the usable series.
2. Normalize them with robust per-series statistics.
3. Form an aggregate foreground map, e.g. median of normalized slices.
4. Threshold with a robust percentile/body-mask rule rather than `0.08 * max`.
5. Keep the largest plausible connected component near the image center or largest body component.
6. Compute one robust median centroid.
7. Convert the shift to millimeters using row/column PixelSpacing.
8. Apply **the exact same `(x0, y0)` to every selected slice in that series**.
9. Gate the correction: only move when the original physical crop demonstrably clips foreground/anatomy near a border.
10. Cap the correction and log it.

### V11-C: train for the crop you deploy

For a new trained model, use the series-level crop during training and add bounded translation augmentation so the model is intentionally robust to acquisition centering. Do not train fixed-center and infer adaptive-center.

## Required validation

A paired experiment must compare, on exactly the same OOF studies/checkpoints:

- V9 fixed center,
- V10 per-slice content centroid,
- series-level centroid,
- gated series-level centroid.

Report macro AUC, all 12 target AUCs, prediction Spearman correlation, and paired bootstrap CI for the delta. Do not accept the crop because a few visual examples look better.

---

# 2. Stage 4 generates semantically invalid cross-slot 3-slice windows

## Confirmed design behavior

Stage 4 builds one 64-slice array by concatenating fixed slots from different series:

- sagittal fluid-sensitive,
- sagittal non-fluid-sensitive,
- coronal fluid-sensitive,
- coronal non-fluid-sensitive,
- axial.

`_eval_centers(mask, D, k)` then treats the global valid range as one continuous stack. `eval_windows` constructs `[c-1, c, c+1]` for global centers.

There is no slot-boundary check. Therefore, windows centered at a boundary can combine, for example, the last sagittal slice with the first coronal slices, or two different contrast series.

The notebook says the windowing mirrors the training code, so this is **not necessarily an inference mismatch**. It is a modeling flaw in the trained Stage-4 representation: some windows are physically meaningless 2.5D neighborhoods.

## Why this can cap accuracy

A 2.5D model gets its advantage from adjacent slices representing adjacent anatomy. Cross-plane or cross-sequence triplets violate that assumption and can:

- waste attention mass,
- create artificial edge/texture transitions,
- make target attention learn to suppress garbage rather than recognize anatomy,
- inject unstable signals when slot availability differs,
- weaken findings that rely on true local continuity.

## V11 solution for retraining

Do **not** change the frozen Stage-4 checkpoint's inference window definition alone. Retrain the Stage-4 family with slot-aware windows:

```text
for each slot:
    valid_centers = centers where c-1,c,c+1 are all inside this slot
    make 3-slice windows only from those centers
    attach slot_id / plane_id / contrast_id metadata
concatenate window embeddings only after encoding
use target-specific attention across valid window embeddings
```

Add learned embeddings for:

- plane,
- fluid/fat-suppression category,
- normalized slice coordinate within the source series,
- optional physical slice position/spacing.

The target-specific attention head can still attend across series after each local 3-slice window has been built correctly.

## Validation

Compare same backbone/training seed where possible:

- current cross-boundary windows,
- boundary-safe windows,
- boundary-safe + slot embeddings.

This is a high-value retraining experiment because it fixes a violation of the fundamental 2.5D assumption.

---

# 3. Series selection is inconsistent and sometimes arbitrary

## Stage 1

`pick_slots` selects the candidate with the largest `n_slices` after plane/fluid/fat-suppression filtering.

Largest slice count is better than arbitrary first, but it is not a full quality criterion. A series with many slices can still have poor coverage, low resolution, repeated/localizer images, motion, or unusual geometry.

## Stage 2 — stronger flaw

For each requested slot, Stage 2 uses:

```python
sub.iloc[0].SeriesInstanceUID
```

There is no explicit quality sort before this selection. If multiple matching sequences exist, the selected series depends on CSV ordering.

## Stage 4 — same class of flaw

`_pick_series_for_slot` creates a candidate list and returns `pref[0]` or `cands[0]`.

Again, duplicate series are not ranked by image quality/coverage.

## Why this matters

Ensembling only helps if each family receives an input consistent with what it learned. Arbitrary duplicate-series choice adds an uncontrolled study-level perturbation before inference and can be especially damaging for meniscus, ligament, and subtle OA findings.

## V11 solution: one deterministic series quality scorer

Create one shared selector used by training and inference. Rank candidate series by a score built from **non-label metadata and geometry**:

1. required plane match,
2. required/preferred fluid-sensitive/fat-suppression match,
3. minimum usable slice coverage,
4. number/fraction of slices with valid IPP/IOP,
5. monotonic physical slice positions,
6. low duplicate-position rate,
7. plausible slice spacing and low spacing irregularity,
8. in-plane pixel spacing / matrix size,
9. physical FOV coverage,
10. slice count within plausible range,
11. preference against obvious scout/localizer descriptions,
12. optional motion/quality score if derived without labels.

Do not simply maximize slice count.

### Important compatibility rule

For existing frozen checkpoints, first reproduce the **training-time selector**. If the checkpoint was trained on `first match`, changing inference to `best series` can itself create a distribution shift. For new V11 training, use exactly the same deterministic selector at train and test.

---

# 4. Stage-1 slice ordering can collapse to raw filesystem order

## Confirmed behavior

In native `order_slices`, each file gets a physical position from IPP/IOP or an InstanceNumber fallback. However, if **any** file still has no key, the function returns the original `files` list and marks ordering unsuccessful.

Filesystem/listing order is not a medical slice-order guarantee.

One malformed slice can therefore invalidate otherwise usable geometry for an entire series.

Stage 2 also relies on InstanceNumber-only ordering and simply skips slices whose header cannot be read. Stage 4 assigns failed-header slices a position of `0.0`, which can cluster them at the beginning.

## V11 solution: robust mixed-key ordering

Use a hierarchy that keeps good geometry instead of discarding it:

1. Compute a robust series normal from valid IOP vectors.
2. Project valid IPP onto that normal.
3. Estimate orientation direction and median spacing from valid positions.
4. Place slices with InstanceNumber but missing IPP using the fitted position-vs-instance relationship when possible.
5. Use InstanceNumber directly when geometry is insufficient.
6. Use natural filename only as the final tie/failure fallback.
7. Remove exact duplicate physical positions or select one deterministically.
8. Log series with low ordering confidence.

For a frozen checkpoint, preserve ascending/descending orientation convention used during training.

---

# 5. The weak report supervision is not the same target as expert image labels

## Competition reality

Only 58 studies carry the official twelve image-derived labels; the remaining 4,349 training studies supply radiology reports that can be converted into weak labels. The hidden test set does **not** provide reports.

A community audit of report statements vs provided labels found substantial disagreement. The central lesson is that report-derived supervision should be treated as **noisy weak supervision**, not ground truth.

This particularly affects Stage 4, which was trained on report-derived soft labels.

## Why report labels differ from image labels

Possible causes include:

- report omission of incidental findings,
- differing severity thresholds,
- uncertain/hedged language,
- translation/negation errors,
- report focusing on clinically salient findings rather than every competition target,
- labels created by independent image review,
- ambiguous mapping from terms to the exact competition definition.

A model can become excellent at reproducing report-writing behavior while leaving leaderboard AUC on the table.

## V11 supervision strategy

### 5.1 Confidence-masked weak labels

Store, per study and target:

- soft probability,
- confidence,
- evidence type: explicit positive / explicit negative / uncertain / absent,
- language/parser confidence.

Use lower or zero loss weight for uncertain report-derived labels instead of forcing them into hard 0/1 targets.

### 5.2 Cross-fit report teachers

If an LLM/text model generates weak labels, produce them out-of-fold or with frozen prompts/rules that never see target images. Avoid tuning the label generator against the same 58 gold cases used to evaluate the image model.

### 5.3 Gold-aware fine-tuning without overfitting 58 cases

The 58 gold studies are too small for aggressive per-target fitting. Use them for:

- low-LR head/backbone fine-tuning with strong regularization,
- leave-one-group-out/repeated CV diagnostics,
- global weak-label trust calibration,
- detecting systematic report/image mismatch.

Avoid twelve independently tuned correction parameters learned directly on 58.

### 5.4 Robust loss options to test

Compare, with identical splits:

- BCE on all weak soft labels,
- confidence-weighted BCE,
- masked BCE for uncertain/unmentioned cases,
- bootstrapped BCE / soft target interpolation,
- generalized cross entropy or another noise-robust loss,
- cross-fitted image-teacher consistency for low-confidence weak examples.

### 5.5 Image/report disagreement mining

Use OOF image predictions to find high-confidence disagreements with report labels. Manually or rule-audit those categories. Do **not** automatically replace weak labels with in-sample image predictions; that creates confirmation bias. Any pseudo-label correction must be cross-fitted.

---

# 6. The 58-study gate is too noisy for blend optimization

The repository's OOF harness already demonstrates this problem:

- weak-label OOF: n = 4,349,
- gold gate: n = 58,
- adding a decorrelated family measured about +0.0010 on large OOF with a positive paired CI,
- the same comparison on 58 gold had a CI too wide to resolve the change.

This means a large fraction of target-specific blend/pooling decisions in the inherited pipeline cannot be trusted merely because they improved a tiny gate or public LB by 0.001-0.003.

## V11 rule

**No new blend weight, target-specific pooling rule, alpha, calibration gate, or model subset should be promoted based only on the 58 gold studies or public leaderboard.**

Use the 4,349-study paired OOF harness for variance reduction, then use the 58 gold cases only as a domain check.

---

# 7. Extreme Stage-1 target-specific legacy weights are a major overfitting risk

V10 retains:

```python
LEGACY_MEMBER_WEIGHT_BY_TARGET = {
    'Lateral Meniscus': 15.0,
    'Medial OA': 2.5,
    'Lateral OA': 15.0,
    'Contusion': 5.0
}
```

A 15× per-member weight is not a small correction; it can dominate a target's ranking.

This conflicts with the repository's own large-OOF conclusion that **per-target blend-weight tuning did not improve over equal weighting**, while adding genuinely decorrelated model families did.

## V11 solution

1. Run an ablation with all legacy target multipliers removed.
2. Compare equal member weights, family-normalized equal weights, and the current 15× scheme.
3. Evaluate per-target prediction correlations to identify duplicate members.
4. If weighting is needed, constrain it heavily, e.g. family weights rather than member-target weights, and fit only in nested CV.
5. Never let one weakly validated member receive 15× simply because a tiny gate favored it.

### Recommended default

Use **equal rank averaging within a model family**, then blend at the **family level**. This prevents a family with 20 highly correlated checkpoints from winning merely by checkpoint count.

---

# 8. Hard-coded target-specific TTA pooling may be leaderboard-tuned noise

Current Stage 1 contains target rules such as:

- max across windows for fracture, contusion, menisci, Baker's,
- top-2 for ACL/MCL,
- original mean for synovitis.

The medical intuition is plausible, but max pooling is statistically aggressive: the more windows evaluated, the higher the chance that one false-positive window becomes the study score. It also changes behavior if runtime changes the number of windows.

## V11 solution

OOF-test the following for each target **inside nested folds**:

- mean probability,
- mean logit,
- smooth log-sum-exp,
- top-k mean with fixed *fraction* rather than fixed count,
- learned target-query attention,
- noisy-OR only if probability calibration is credible.

Prefer one smooth pooling mechanism with target queries over a dictionary of hand-authored target exceptions.

Do not change pooling independently from training if the model was explicitly fitted to a certain aggregation convention.

---

# 9. Partially evaluated Stage-1 members should never enter the same ensemble at full identity

## Confirmed behavior

The Stage-1 runtime scheduler estimates remaining time. If a full set of window starts does not fit, it can take a **central subset** of the windows and still bank that member's predictions into the normal ensemble.

This means the mathematical definition of a member varies with runtime/mount speed. A checkpoint that normally votes from all windows can suddenly vote from only its central windows but keep its normal model weight.

## Why this hurts AUC

- Peripheral pathology can be omitted.
- The member's error distribution changes.
- Rank correlations with other members change.
- Two runs on the same hidden set can produce different ensembles based on transient runtime.

## V11 accuracy rule

A member is either:

- run with its validated complete inference recipe, or
- absent from the blend.

When budget is tight, drop the lowest-value remaining member **before starting it**. Do not silently mutate it into a central-window variant.

If a short-window member is genuinely useful, define it as a separate explicit model and OOF-score it as such.

---

# 10. Stage-1 exception handling can erase a valid ensemble with all-0.5 predictions

Stage 1 atomically banks a valid submission after each successful member. However, the outer `except Exception` then creates a fresh all-0.5 `submission.csv`.

So a late exception after many successful members can erase the exact recovery artifact the banked submission was designed to preserve.

## V11 fix

On exception:

1. If `submission.csv` exists, validate schema, row IDs, finiteness, and non-degenerate variance.
2. If valid, keep it unchanged and record the error.
3. Only write all 0.5 if no valid submission exists at all.
4. Write fallback atomically.

This is a submission-integrity fix with potentially enormous score impact.

---

# 11. Stage-3 adaptive second-pass weighting is test-distribution dependent

Stage 3 computes per-target alpha using:

- fold agreement on the current test predictions,
- diversity relative to another current-test branch,
- hand-set coefficients/priors,
- clipping ranges.

No labels are consulted, so this is not leakage in the ordinary sense. But it is **transductive ensemble weighting**: the model recipe changes according to statistics of the hidden test set rather than using parameters fixed by OOF validation.

That can help, but it can also react to scanner/site/protocol shifts in ways that have never been tied to target AUC.

## V11 solution

Compare:

1. fixed published alpha,
2. fixed OOF-fitted global alpha,
3. fixed OOF-fitted per-target alpha with strong shrinkage,
4. current test-adaptive alpha.

The adaptive variant should be kept only if the exact adaptive formula is applied to OOF fold predictions and produces a repeatable gain across folds/sites.

---

# 12. Stage-3 protocol-metadata calibration may learn site/protocol shortcuts

The Rad branch calibration uses features including counts of:

- total series,
- series per plane,
- fat-suppressed/fluid-sensitive series,
- plane-by-flag counts.

These can genuinely proxy input completeness. They can also proxy institution/scanner/protocol.

A rank-changing calibration that learns site prevalence can score well on a public split and fail if the private split has a different site mix.

## V11 validation/fix

1. Cluster studies by protocol signature and, if possible, institution/scanner identifiers available legally in metadata.
2. Perform grouped OOF where protocol/site groups are held out.
3. Compare image-only branch vs protocol-calibrated branch.
4. Regularize protocol coefficients strongly.
5. Remove protocol features that only help random CV but not grouped CV.
6. Prefer metadata for **input-validity gating** over direct disease prediction unless group-CV proves the latter.

---

# 13. Stage-2 laterality handling is fragile

Stage 2 infers side using `ImagePositionPatient[0]` from the **first ordered file** and flips non-sagittal series when that value is negative. If metadata is missing, it returns 0 and skips the flip.

A single-slice heuristic is unnecessarily brittle.

## V11 solution

Derive laterality once per study/series from robust geometry:

- median/central-slice IPP,
- patient coordinate system,
- explicit laterality tags when trustworthy,
- consistency check across series.

Log disagreement and avoid silently flipping only some sequences. Validate against the training convention before changing frozen-model inference.

---

# 14. Preprocessing is inconsistent across model families

The families do not all apply the same DICOM intensity logic:

- Stage 1 manually applies RescaleSlope/Intercept.
- Stage 2 reads `pixel_array` without an equivalent explicit modality LUT/inversion path.
- Stage 4 applies modality LUT and handles `MONOCHROME1` inversion.

Some diversity is useful, but accidental DICOM interpretation differences are not the kind of diversity to seek.

## V11 strategy

For **newly trained** branches, standardize a canonical decoder:

1. modality LUT / rescale,
2. PhotometricInterpretation handling,
3. robust finite-value cleanup,
4. row and column PixelSpacing separately,
5. per-series intensity normalization matching training,
6. deterministic ordering and laterality.

For frozen public checkpoints, reproduce the exact preprocessing they were trained with; do not normalize everything at inference merely for code cleanliness.

---

# 15. Physical crop size uses one PixelSpacing component

Several crop paths reduce PixelSpacing to one scalar. DICOM PixelSpacing is `(row_spacing, column_spacing)` and should not be assumed isotropic.

## V11 retraining improvement

Crop a physical rectangle using both components:

```text
crop_height_px = round(crop_mm_y / row_spacing)
crop_width_px  = round(crop_mm_x / col_spacing)
```

Then resize to the network's square input.

For frozen models, first quantify how many competition series are meaningfully anisotropic. If effectively none, this is low priority; if not, retrain/OOF before changing inference.

---

# 16. Sample by physical slice position, not only index

Evenly spaced indices are not always evenly spaced anatomy when:

- slice spacing varies,
- duplicate positions exist,
- gaps exist,
- mixed reconstruction spacing occurs.

## V11 retraining improvement

For each series:

1. sort by physical position,
2. deduplicate near-identical positions,
3. estimate usable physical range,
4. sample requested positions uniformly in millimeters,
5. map desired positions to nearest actual slices,
6. record normalized physical coordinate as model metadata.

This can make the same slot represent comparable anatomy across scanners.

---

# 17. Stage-4 model subset was selected on a tiny gate

V10 runs three CoAtNet checkpoints of the same architecture/resolution with different corpus/preprocessing histories. The notebook notes the three-arm selection came from a small gold panel and costs about 3× Stage-4 inference.

For **accuracy**, the problem is not cost itself; it is that same-architecture model selection on ~45/58 studies can overfit the gate.

## V11 solution

Create OOF predictions for all available Stage-4 checkpoints on the 4,349 weak-label set and calculate:

- individual macro AUC,
- per-target AUC,
- pairwise prediction correlation,
- marginal delta when added to the blend,
- paired bootstrap CI,
- gain by target.

Keep a checkpoint only if it contributes independent ranking signal. Different preprocessing is not automatically useful diversity.

---

# 18. The next model addition should maximize residual diversity, not standalone score

The repository's OOF harness found that a fourth decorrelated family was the only tested ensemble change with a resolved positive gain. It also identifies `surasan092` DINOv3-S full fine-tune as the least correlated measured family (roughly 0.644-0.655 correlations with others).

However, the same README says the reconstructed head currently reaches only about `r=0.72` to the published validation predictions and explicitly warns **not to blend that reconstruction**.

## Correct V11 path

1. Reconstruct the exact DINOv3-S/16+ backbone config, including gated MLP.
2. Match the author's **per-series** intensity statistics, not per-slice statistics.
3. Respect `slot_min_coverage` during series selection.
4. Reconstruct the slot/query head exactly.
5. Verify predictions against published validation arrays with near-zero numerical error, not merely correlation.
6. Use the exact 10-window **plain mean** aggregation identified by the repo's reconstruction work.
7. Only then generate OOF predictions and test marginal ensemble gain.

Do not add an approximate "diverse" model. Diversity from a broken reproduction is just error.

---

# 19. Optimize for macro AUC, not probability calibration

The Kaggle metric is ROC AUC. Per target, any strictly monotonic transform of predictions leaves AUC unchanged. Therefore:

- temperature scaling alone cannot improve AUC,
- Platt calibration alone cannot improve AUC,
- probability calibration is not the objective unless the transformation also changes ordering through cross-target/meta features.

## V11 training improvement

Keep stable BCE/soft-label training but test a rank-aware auxiliary objective:

```text
loss = macro_balanced_BCE + lambda * differentiable_pairwise_AUC_surrogate
```

Use equal target weighting at the top level so rare/easy targets do not dominate the shared representation. Evaluate the AUC-surrogate carefully because noisy weak labels can make pairwise ranking loss amplify label errors.

---

# 20. Use target-aware slot fusion instead of hand-written target heuristics

The pipeline already recognizes that findings are visible in different planes/sequences, but some of that knowledge is encoded as manual pooling dictionaries.

A cleaner V11 model is:

1. one shared image encoder,
2. local 2.5D windows that never cross series,
3. learned plane/sequence/position embeddings,
4. twelve target queries,
5. target-specific attention over windows,
6. one output per target.

This allows ACL, menisci, OA, effusion, synovitis, fracture, etc. to learn different evidence distributions without manually coding `max`, `top2`, or target-specific legacy weights.

---

# 21. Add explicit input-quality masks rather than letting zeros act as anatomy

Missing/failed series and failed slices are sometimes filled with zeros or nearest neighbors depending on branch. The model should know the difference between:

- a real very-dark image,
- an imputed duplicate,
- a missing slot,
- a failed decode.

## V11 retraining improvement

Pass:

- slot presence mask,
- slice validity mask,
- imputation flag,
- ordering confidence,
- optional series quality score.

Use these only for attention masking/gating, not as direct disease shortcuts unless group-CV proves they are stable.

---

# 22. Use controlled ensemble family normalization

V10 contains very different numbers of checkpoints per family. If all checkpoints are simply pooled at member level, a family with many nearly identical folds can dominate a family with fewer but more independent models.

## V11 blend hierarchy

Recommended structure:

```text
within each family:
    rank-average folds/seeds -> one family prediction
across families:
    rank-average family predictions
optional:
    one small set of OOF-validated family weights with strong shrinkage toward equal
```

This makes ensemble influence correspond to information diversity rather than checkpoint count.

---

# 23. Add a prediction-drift audit whenever preprocessing changes

Before any V11 preprocessing change reaches Kaggle, create a table per study/target containing:

- old prediction,
- new prediction,
- old rank,
- new rank,
- absolute rank shift,
- crop shift mm,
- selected series UID old/new,
- number of valid slices,
- ordering confidence.

Inspect the largest rank movers. A preprocessing fix that moves hundreds of ordinary centered studies is not conservative, even if a few pathological examples look better.

---

# 24. Required V11 validation protocol

## 24.1 Freeze splits

Create immutable study-level folds. Do not let checkpoints, label generation, or blend tuning see validation labels from their fold.

## 24.2 Two validation domains

### Large weak-label OOF (n=4,349)
Use this to resolve tiny model/ensemble differences with paired comparisons.

### Gold image-label set (n=58)
Use only as an external-domain sanity check. Report wide uncertainty; do not fit many knobs to it.

## 24.3 Add grouped robustness validation

Where metadata permits, construct protocol/site/scanner groups and evaluate held-out groups. This specifically tests whether Stage-3 metadata calibration or series heuristics are learning site shortcuts.

## 24.4 Always report paired deltas

For each candidate:

- macro AUC old/new,
- delta,
- paired bootstrap 95% CI,
- probability candidate is better,
- all 12 target deltas,
- prediction correlation,
- per-protocol/site deltas.

## 24.5 Do not use public LB as a hyperparameter loop

The public leaderboard is approximately 30% of the test data; final standings are based on the other 70%. Treat public LB as a final external check, not an optimizer.

---

# 25. V11 experiment matrix, in recommended order

## Tier 0 — score-integrity / regression isolation

1. **V9 crop + all V10 correctness fixes**.
2. Same model with V10 crop only.
3. V9 crop + fix Stage-1 exception preservation.
4. V9 crop + full-member-only scheduler.
5. V9 crop + equalized legacy weights.

Goal: isolate whether V10's input change is helping or hurting before changing anything else.

## Tier 1 — preprocessing correctness

6. Robust mixed-key slice ordering.
7. Deterministic quality-ranked duplicate-series selection.
8. Series-level/gated recentering.
9. Robust laterality.
10. Physical-position slice sampling.

Run each one independently first.

## Tier 2 — ensemble cleanup

11. Equal family weighting.
12. Remove 15× target multipliers.
13. Re-test target TTA pooling.
14. Freeze Stage-3 adaptive weights from OOF.
15. Audit/remove protocol calibration if group-CV unstable.

## Tier 3 — retraining changes

16. Stage-4 slot-boundary-safe windows.
17. Plane/sequence/position embeddings.
18. Confidence-masked weak-label training.
19. Gold-aware regularized fine-tune.
20. Rank-aware auxiliary loss.

## Tier 4 — new diversity

21. Exact `surasan092` reconstruction.
22. OOF residual-correlation selection.
23. Add only if marginal paired delta is positive and stable.

---

# 26. What V11 should NOT do

Do not:

- tune another dozen per-target blend weights on 58 cases,
- react to a +0.001 public-LB move by hard-coding more target exceptions,
- keep the V10 recrop merely because it fixes an intuitively real edge case,
- change frozen-model preprocessing without paired OOF,
- treat report labels as equivalent to image labels,
- blend the current `surasan092` reconstruction at `r≈0.72`,
- add more same-family checkpoints before measuring residual diversity,
- use a larger backbone as a substitute for fixing series/window/data problems,
- use calibration methods that only change probability scale and expect AUC gain,
- accept runtime-dependent partial models as the same validated member.

---

# 27. Proposed V11 accuracy architecture

For a fresh trainable V11 branch, the strongest direction is a **series-aware multi-view 2.5D model** rather than another monolithic stack:

```text
DICOM series
  -> deterministic quality-ranked slot selection
  -> robust physical ordering / physical position sampling
  -> one crop transform per series
  -> valid within-series 3-slice windows
  -> shared image encoder
  -> add plane + sequence + normalized-position embeddings
  -> twelve target queries attend over all valid window embeddings
  -> 12 logits
```

Training:

```text
weak report labels with confidence masks
+ gold-aware regularized fine-tune
+ macro-balanced objective
+ optional small AUC-ranking auxiliary loss
+ translation/FOV augmentation matching deployed localization
```

Ensembling:

```text
folds -> one prediction per family
families -> rank blend
new family admitted by OOF marginal gain/correlation, not tiny gate score
```

This addresses the largest structural weaknesses without relying on public-LB micro-tuning.

---

# 28. Acceptance checklist before calling a notebook V11

- [ ] Fixed-center V9 baseline can be reproduced.
- [ ] V10 correctness fixes retained.
- [ ] No all-0.5 overwrite of a valid banked submission.
- [ ] No incomplete Stage-4 arm can be blended.
- [ ] No partially evaluated Stage-1 member is treated as the full member.
- [ ] Every series selector is deterministic and logged.
- [ ] Slice ordering never falls to arbitrary filesystem order because of one missing tag.
- [ ] Crop origin is constant within a series.
- [ ] Any adaptive crop was trained/OOF-tested with the same convention.
- [ ] Legacy 15× weights have been ablated.
- [ ] Hand-written target pooling has been re-tested on large OOF.
- [ ] Stage-3 adaptive alpha is OOF-proven or frozen.
- [ ] Protocol calibration survives grouped validation.
- [ ] Weak labels carry confidence/uncertainty.
- [ ] 58 gold cases are not used as a high-dimensional tuning set.
- [ ] New model families reproduce their published/reference predictions exactly.
- [ ] Family-level marginal AUC and correlation are recorded.
- [ ] Public LB is not being used iteratively as a hyperparameter oracle.

---

# Source cross-checks used for this audit

Repository paths and observations were checked against:

- `README.md` — public progression, OOF findings, T4 dtype measurements, checkpoint probe findings, public-LB overfit warning.
- `v10/rsna-knee-ensemble-v10.ipynb` — V10 change log and all four inference stages.
- `v9/rsna-knee-ensemble-v9.ipynb` — fixed-center crop baseline and Stage-4 behavior before V10 fixes.
- `oof_harness/README.md` — n=4,349 paired OOF results, v52 diversity result, `surasan092` reconstruction status.
- `timing/rsna-knee-timing.ipynb` — measured Stage-1 runtime slope and historical stage timing setup.
- Kaggle competition evaluation/rules/leaderboard pages — ROC-AUC metric, efficiency prizes, public/private split.
- Kaggle competition discussions on report-derived labels and hidden-test report availability.

Key repository line regions inspected in V10 include the change log (~lines 40-126), Stage-1 slot/crop/order logic (~892-1048), pooling/legacy weights (~1311-1343), scheduler (~1578-1621), Stage-1 fallback (~1986-1997), Stage-2 preprocessing/series selection (~2135-2227), Stage-2 blend (~2797-2818), Stage-3 adaptive weighting/calibration (~3404-3580), and Stage-4 volume/windowing/inference (~3748-4215).

---

# Bottom line

The most likely V11 win is **not** another tiny blend-weight tweak. The repository's own measurements already say those tweaks are below the noise floor. The highest-value path is:

1. remove V10's unvalidated per-slice input shift as the default,
2. fix deterministic series/order/submission-integrity issues,
3. revalidate all aggressive target-specific ensemble rules on large OOF,
4. train a slot-boundary-aware, noise-robust series model,
5. add only genuinely decorrelated families whose inference has been reproduced exactly.

That gives V11 a much better chance of improving the **private** leaderboard rather than only continuing the public-leaderboard tuning cycle.
