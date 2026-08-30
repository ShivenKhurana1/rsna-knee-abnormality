# RSNA Knee: strategy from 0.935 toward 0.955+

Date: 2026-08-29

Scope correction after user clarification: the training-oriented roadmap below is
an earlier proposal, not the active plan. The user wants existing pretrained models
and continued OrthoDiffusion diagnosis, without training new specialists. The
[reopened reproduction audit](orthodiffusion_diagnosis.md) supersedes this document's
OrthoDiffusion closure, plane-permutation priority, and claim that the receipt's
+0.0088 is a best-case bound for the current ensemble.

## Executive verdict

The present `0.935` public score is strong, but the evidence in this repository does not support a
credible `+0.020` from another public-checkpoint blend. The only statistically resolved ensemble gain
so far is `+0.0010` on the 4,349-study OOF harness. Blend-weight fitting, label-table consensus, and two
approximate checkpoint reconstructions have not produced a deployable gain.

A `+0.020` macro-AUC increase means finding **0.24 total AUC points across the twelve targets**. For
example, four targets must each improve by about `+0.060`, or eight targets by about `+0.030`. That is
an original-model and supervision problem, not a final-blend-weight problem.

The path with a plausible chance is:

1. lock down a trustworthy per-target OOF baseline and remove inference-integrity variance;
2. rebuild weak labels to match the competition's strict image-label definitions;
3. train a series-aware, target-query model with anatomy-appropriate resolution and aggregation;
4. add a small number of target specialists for the weakest residuals;
5. ensemble only families with verified OOF marginal gain;
6. preserve prize eligibility by resolving external-data and model-license questions before training.

The goal should be expressed as a validation gate, not a promise: pursue `+0.020`, but do not treat it
as an expected outcome from the currently available evidence.

## 1. What is actually limiting the current solution

### 1.1 Validation is the first bottleneck

Only 58 studies have expert image labels. A single target has roughly 9-35 positives in that set, so
per-target AUC and tiny blend deltas are extremely unstable. The public leaderboard is only about 30%
of the test set; the remaining 70% determines the final result. A pipeline descended from heavily
forked public notebooks is particularly exposed to public-LB selection bias.

The large OOF harness is useful for comparing image families, but its labels are still report-derived.
It can resolve small deltas against the weak-label task; it cannot prove that a change better matches
the stricter expert image-label task.

Required response:

- keep the 4,349 weak-label OOF for variance-efficient comparisons;
- keep the 58 gold cases as an untouched direction/sanity gate;
- add grouped folds by site/protocol/scanner when metadata permits;
- always report per-target deltas, rank correlation, and paired bootstrap intervals;
- never tune a 12-target blend on the 58 cases or on repeated public-LB submissions.

### 1.2 The supervision target is misaligned

The host says ambiguous findings were labeled negative and gives strict thresholds: high-grade ACL/MCL
tears; meniscal signal reaching a surface on at least two images or definite morphology; greater than
50% cartilage loss over a moderate/large area for OA; moderate/large effusion and Baker's cyst; and an
acute cortical break/line for fracture. The host also confirms that image labels are authoritative when
they disagree with reports.

Current report parsing is broad enough to turn clinically real but competition-negative statements
into positives: low-grade sprain, intrasubstance meniscal degeneration, mild effusion, minimal
synovitis, small Baker's cyst, nonspecific marrow edema, chondropathy, and chronic/subchondral injury.
That is structured label noise, not random noise.

Required response: generate two labels per target from each report:

```text
finding_present       = does the report mention the clinical entity?
competition_positive  = does severity/chronicity/definiteness satisfy the host definition?
confidence            = explicit_positive | explicit_negative | uncertain | unmentioned
```

Train the image model on `competition_positive`; use `finding_present` only as an auxiliary target.
Mask uncertain and unmentioned cases rather than forcing all of them to zero.

### 1.3 Input handling changes across families and sometimes violates MRI geometry

Confirmed repository risks include arbitrary duplicate-series selection, raw filesystem-order fallback,
per-slice recentering, slot-crossing 2.5D windows, one-component PixelSpacing crops, and index rather
than physical-position sampling. These are especially damaging to small structures and findings whose
definition requires continuity across images.

The OrthoDiffusion failure reinforces the same conclusion: target-specific performance can move by
0.05-0.19 even when the backbone and beta schedule are correct, because the input convention is part
of the model.

### 1.4 The current ensemble has more redundancy than independent information

Twenty same-lineage DINOv2 members, several RadImageNet heads, and three same-architecture CoAtNet arms
are not twenty-plus independent votes. Member-level averaging gives large families more influence simply
because they contain more checkpoints. The OOF harness shows that a decorrelated family can help, while
fine weight fitting does not.

Use a hierarchical ensemble:

```text
folds/seeds -> one prediction per family
families    -> equal or strongly-shrunk rank blend
specialist  -> admitted only for its target after OOF residual testing
```

## 2. OrthoDiffusion: final disposition

The reconstruction is **not reproduced and not suitable for blending**. The schedule, model identity,
and primary series selection have been ruled out. Large gaps remain for Effusion, Baker's, PF OA, and
Lateral OA, while several other targets reproduce closely. Current Effusion AUC (`0.8832`) is worse than
the deployed blend (`0.907`).

Only three bounded diagnostics remain defensible:

1. Test all six permutations of axial/coronal/sagittal checkpoint assignment. A mislabeled re-upload
   should produce a coherent improvement pattern, not a one-off target bump.
2. Repeat with several predeclared four-seed panels and measure prediction-rank/AUC variance. If the
   observed `0.075` Effusion gap is far outside that variance, stop blaming seeds.
3. Audit whether `RescaleSlope` or `RescaleIntercept` varies within a series. Apply both per slice before
   any volume-level percentile/min-max operation and handle `MONOCHROME1`/padding consistently.

Stop after those tests. Without reference per-study predictions or source preprocessing, a wider search
is an underdetermined attempt to fit 58 labels and the published receipt.

## 3. Released external models: what is and is not actionable

| Candidate | Released asset | Useful role | Main problem | Decision |
|---|---|---|---|---|
| OrthoFoundation-L | Knee MRI/X-ray pretrained DINOv3-L checkpoint and pretraining code | New MRI-specific encoder for all targets, especially ligament/meniscus/cartilage | Only a backbone; large/slow; corpus includes OAI and a private multicenter cohort, creating an unresolved external-data eligibility issue | Highest-value technical experiment **only after a written host ruling** |
| SKM-TEA model zoo | Released V-Net/U-Net tissue-segmentation checkpoints | Meniscus/cartilage masks, ROI crops, visibility and coverage features | qDESS/domain shift; segmentation is not abnormality classification; external-data provenance must be cleared | Small ROI feasibility test, not a submission arm by itself |
| `aagatti/nnunet_knee` | Released 3D nnU-Net weights for medial/lateral meniscus, cartilage, and bone | Anatomical crops for meniscus and OA specialists | Self-reported cross-domain performance; about 80 s/volume; expects T2-like NIfTI input | Test on 20-30 diverse studies before any training dependency |
| SAMRI | Released MRI segmentation checkpoint/code | Possible generic anatomy mask | Prompt-dependent, not knee-pathology classification, substantial integration/runtime cost | Lower priority than knee-specific segmenters |
| MRNet derivatives | Code and assorted third-party files for ACL/meniscus/abnormality | Relevant task taxonomy | No clean official deployable checkpoint lineage; MRNet data require an agreement; domain and label-definition mismatch | Do not use as a competition arm without provenance and host clearance |
| RadImageNet | Released radiology pretraining | General radiology features | Already represented strongly in the current ensemble | No new family-level diversity unless retrained with a materially different head/input |
| OrthoDiffusion | Three released plane checkpoints | Direct multi-target predictions | Reconstruction is not exact and underperforms the current blend on Effusion | Reject current reconstruction |

OrthoFoundation is the genuinely new lead. Its repository releases a checkpoint pretrained on roughly
1.2 million knee images and reports strong MRI transfer results, but the competition rules require
external resources to be reasonably accessible to all. The paper/repository explicitly includes private
clinical data, and a current host question about OAI/MRNet-style gated data is unanswered. Do not spend
training compute until the host confirms that public checkpoint availability is sufficient.

## 4. Proposed trainable architecture

### 4.1 Canonical study representation

Select up to six slots:

```text
sagittal fluid-sensitive/fat-suppressed
sagittal non-fluid-sensitive
coronal fluid-sensitive/fat-suppressed
coronal non-fluid-sensitive
axial fluid-sensitive/fat-suppressed
axial non-fluid-sensitive
```

For every series:

- apply per-slice rescale slope/intercept before any statistics;
- normalize photometric interpretation and remove padding/outliers;
- order using IOP/IPP projection with a robust InstanceNumber fallback;
- rank duplicate series with a deterministic geometry/coverage quality score;
- use one fixed physical crop transform for the entire series;
- sample slices by physical position, never by directory order;
- construct 2.5D windows only within their source series;
- emit plane, sequence, physical position, validity, and quality metadata.

Cache this representation once. Every experiment must consume the identical cache unless preprocessing
is the controlled variable.

### 4.2 Shared encoder plus target queries

Use one MRI-capable 2D encoder over within-series 3-slice windows, then:

1. project window features to a common embedding;
2. add plane/sequence/physical-position embeddings;
3. use twelve learned target queries to attend over all valid windows;
4. pass the twelve label tokens through a small interaction transformer;
5. predict one logit per target.

Start with DINOv2-S or the existing reproducible DINOv3-S as the control. If allowed, replace the
encoder with OrthoFoundation-L using frozen features, then LoRA/last-block tuning, before attempting a
full fine-tune. The architecture must be validated on the same cache and folds so backbone value is not
confounded with preprocessing.

### 4.3 Loss and weak-label handling

Use a masked, target-balanced objective:

```text
L = masked_soft_BCE(competition_target, confidence)
  + 0.2 * masked_BCE(report_presence_aux)
  + lambda * pairwise_AUC_surrogate(high_confidence_pairs_only)
```

- Gold labels override report labels.
- Explicit high-confidence report positives/negatives receive full weak-label weight.
- Severity-ambiguous or unmentioned targets are masked or receive a very small weight.
- Do not use ordinary positive-class weighting merely to match the enriched 58-study prevalence.
- Cross-fit every learned label teacher; a study must never receive a pseudo-label from a teacher that
  was fitted using that study.

## 5. Target specialists

Do not train twelve independent full models. Add specialists only where the base model has a reproducible
OOF residual and the anatomy supports a different input policy.

| Specialist | Inputs and ROI | Aggregation/auxiliary task | Why it can add new signal |
|---|---|---|---|
| Medial/lateral meniscus | High-resolution sagittal + coronal fluid-sensitive windows; medial/lateral meniscus ROI if segmentation transfers | MIL top-k plus a two-adjacent-image consistency auxiliary target | Matches the host's two-image/surface-contact definition and preserves small tear detail |
| PF OA | Axial plus sagittal cartilage crops at high resolution | Patellar/femoral cartilage visibility and loss-severity auxiliary heads | Current crop/plane errors plausibly hit PF OA hardest |
| Medial/lateral OA | Coronal weight-bearing-compartment ROI plus sagittal confirmation | Compartment-specific cartilage-loss severity | Separates cartilage loss from broad report terms such as chondropathy |
| Effusion/synovitis/Baker's | Axial + sagittal fluid-sensitive images; suprapatellar and posteromedial ROIs | Target-specific attention, not one shared max rule; severity auxiliary labels | Distinguishes fluid amount, synovial thickening, and popliteal cyst location |
| ACL/MCL | Sagittal ACL and coronal MCL emphasis with wider peripheral coverage for MCL | Continuity/visibility auxiliary heads | Prevents central crops from discarding collateral-ligament evidence |
| Contusion/fracture | All fluid-sensitive planes with bone ROI | Sparse-lesion MIL; separate marrow-edema and fracture-line auxiliaries | Prevents nonspecific edema from becoming an acute-fracture label |

A specialist enters the final blend only when it improves its target on cross-fitted OOF and does not
materially degrade robustness groups. Its coefficient should be fixed or heavily shrunk, never selected
from repeated public submissions.

## 6. Experiment order and stop/go gates

### Phase A — establish the real baseline

1. Freeze V9 fixed-center inference as the control.
2. Run V10's content recrop as a paired A/B; do not assume it improved `0.935`.
3. Save one keyed prediction matrix per family for all OOF studies.
4. Produce the missing deployed-blend per-target AUC table on both weak and gold labels.
5. Add site/protocol/scanner group summaries and failure/QC logs.

Gate: no architecture work is admitted until repeated evaluation of the same artifact is deterministic.

### Phase B — supervision ablations

1. Existing labels.
2. Severity-aware competition labels.
3. Confidence masking.
4. Cross-fitted multi-teacher consensus.
5. Gold-aware fine-tuning with strong regularization.

Gate: require positive macro delta on large OOF, the same direction on gold, and no single language/site
accounting for the gain.

### Phase C — input/architecture ablations

1. Boundary-safe windows.
2. Physical-position sampling.
3. Deterministic duplicate-series selection.
4. Target queries with plane/sequence/position embeddings.
5. Higher-resolution weak-target heads.
6. OrthoFoundation encoder only if eligible.

Change one family-level variable at a time. Do not combine a new cache, encoder, head, and label table in
one experiment.

### Phase D — specialists and ensemble

1. Train meniscus specialist first.
2. Train fluid-complex specialist (Effusion/Synovitis/Baker's).
3. Train OA specialist if PF/lateral/medial OA remain weak.
4. Compute residual rank correlations against every existing family.
5. Admit only target/family combinations with positive cross-fitted marginal AUC.
6. Form a family-normalized rank ensemble.

## 7. Evidence budget for a `+0.020` claim

Track the required total target-AUC gain explicitly:

| Source | Required evidence | Contribution counted toward +0.240 target-AUC total |
|---|---|---:|
| Label rebuild | Cross-fitted, paired OOF; direction confirmed on gold | measured only |
| Base architecture | Same cache/folds/labels; bootstrap CI | measured only |
| Meniscus specialist | Two target deltas, residual correlation | measured only |
| Fluid specialist | Three target deltas, residual correlation | measured only |
| OA specialist | Three target deltas, residual correlation | measured only |
| Ensemble | Nested/cross-fitted family selection | incremental delta only |

Do not add standalone improvements; sum only each component's **marginal delta in the final ensemble**.
Until those measured marginal deltas sum to roughly `0.240` across targets, a `+0.020` macro claim is not
supported.

## 8. Immediate next actions

1. Log OrthoDiffusion as inconclusive and stop broad reconstruction work.
2. Run the three bounded OrthoDiffusion checks only if they are cheaper than the next training ablation.
3. Ask the host for a written ruling on pretrained checkpoints derived from OAI and private clinical data,
   specifically OrthoFoundation-L and knee segmentation weights.
4. Generate the deployed blend's full per-target OOF table; this decides which specialist comes first.
5. Build a severity-aware label audit table against all 58 gold studies, recording false positives caused
   by mild/degenerative/chronic wording.
6. Create one canonical six-slot cache on local SSD or a persistent Kaggle dataset.
7. Train the boundary-safe target-query control using the current reproducible encoder.
8. Train meniscus and fluid-complex specialists only after the control clears the validation gate.

## Sources checked

- [Competition overview and strict label definitions](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)
- [Host confirmation that image-derived labels are authoritative](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826)
- [Competition rules: external data, reproducibility, and winner obligations](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
- [Public leaderboard uses about 30% of test data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)
- [OrthoFoundation paper](https://arxiv.org/abs/2601.18250)
- [OrthoFoundation released code/checkpoint](https://github.com/ytrsk/OrthoFoundation)
- [SKM-TEA model zoo](https://github.com/StanfordMIMI/skm-tea/blob/main/MODEL_ZOO.md)
- [Released knee nnU-Net weights](https://huggingface.co/aagatti/nnunet_knee)
- [SAMRI released code/checkpoint](https://github.com/wangzhaomxy/SAMRI)
- [Stanford MRNet task/model description](https://stanfordmlgroup.github.io/projects/mrnet/)

## Bottom line

Close OrthoDiffusion as an unreproduced negative result. The next credible leap is not another guessed
inference convention or another public blend weight. It is a competition-definition-aware label system
paired with anatomically faithful, within-series multi-view modeling, followed by high-resolution
meniscus/fluid/OA specialists selected on residual OOF gain. OrthoFoundation is the strongest newly
released encoder candidate, but it must clear the host's external-data ruling before use.
