# RSNA Knee: execution brief for a 0.95–0.96 target

**Prepared against this workspace and the four supplied public notebooks, September 14, 2026.**

## 1. Decision and target

**Use the reported 0.941 ensemble as a reproduction control, then improve supervision and add one genuinely different, trainable image family. Build dynamic ingestion around saved model contracts.** A larger collection of forks or more public-leaderboard weight probes is a weak route to the next tier.

| Starting score | To 0.950 | To 0.960 |
|---|---:|---:|
| Your recorded 0.935 | +0.015 | +0.025 |
| Reported 0.941 control | +0.009 | +0.019 |

These are **absolute macro-AUC gains**, not accuracy percentages. From 0.941, reaching 0.95 or 0.96 requires reducing the remaining gap to perfect AUC by approximately 15.3% or 32.2%. This is substantial. No retrieved evidence establishes that any proposed recipe will deliver it; the experiment gates below establish whether it works.

The challenge predicts multiple simultaneous binary findings. The metric is the mean of target-wise ROC-AUC. Optimize ranking per finding; do not optimize classification thresholds or use a softmax across findings. [Official evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation).

**Two promises to replace with testable requirements:**

- Dynamic code can make predictions invariant to directory names, file order, row order, and batch composition. It cannot make AUC invariant to a different population, missing diagnostic sequences, or image quality.
- Grouped cross-validation can prevent known leakage and estimate uncertainty. It cannot align perfectly with an unseen private test distribution.

The official data description explicitly says prevalence can differ between train/public/final datasets; reports are unavailable at test time. The design must accommodate both facts. [Official data specification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data).

## 2. What the supplied notebooks actually add

Public source was downloaded through Kaggle's public kernel API and inspected without execution. Full source versions, attachment lists, and SHA-256 fingerprints are saved in the two source-inventory JSON files accompanying this report. **The user-reported 0.941 is not independently reproduced here.** The search index still showed an older v11/0.936 result for RSNA Baseline while the API returned v15; a latest-source download is not a scored-version receipt.

| Supplied source | Inspected version | Finding and practical use |
|---|---:|---|
| [RSNA Baseline](https://www.kaggle.com/code/evgendvorkin/rsna-baseline) | 15 | Large inherited DINO/RadImageNet/Raptor/CoAt ensemble with final outer routing. Reproduction candidate; not a simple fine-tuning script. |
| [Bend the knee to the dinosaurs](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs) | 38 | Shares the same twelve attached datasets and two kernel sources as the other three. Includes residual-gated CoAt checkpoints and target-specific outer blending. |
| [RSNA Knee 0941 restructured](https://www.kaggle.com/code/maverickss26/rsna-knee-0941-restructured) | 5 | Makes configuration/routing easier to inspect; exposes parent/halfway/probe choices and agreement/tie diagnostics. Useful engineering scaffold, not evidence of independent model diversity. |
| [Hybrid Raptor meniscus specialist](https://www.kaggle.com/code/renta0426/rsna-knee-hybrid-raptor-meniscus-specialist) | 1 | Final inspected change is outer routing: default Raptor weight 0.60; ACL 0.75; medial meniscus 0.80; lateral meniscus 1.00; lateral OA 0.75; fracture 0.75. Receipt sets `bag_present=False`, `overlay_applied=False`. The title alone does not demonstrate a separately trained meniscus specialist. |

### Concrete audit findings

1. **All four share substantial ancestry and attachments.** Blending their final CSVs can mostly reweight the same underlying predictions. Require per-arm checkpoint hashes and keyed predictions before treating one as a new family.
2. **The newer embedded Raptor code already distinguishes per-arm cache sizes, spans, and window counts.** Your older V12 preprocessing mismatch is documented, but do not assume the same mismatch remains in this newer control. Evaluate your V13 correction against its actual parent independently.
3. **`rankpct` uses `x.argsort(0).argsort(0)` in the embedded Raptor implementation.** Equal scores receive different ranks based on input ordering. Use average ranks and test row permutation invariance. This matters especially when failed cases share 0.5. It is a correctness fix, not a measured AUC gain.
4. **There are environment-specific assumptions:** the residual CoAt child asserts two GPUs and passes a fixed competition mount. Configure these explicitly or reproduce the required environment; changing GPU topology can require adapting the imported runner, not just deleting an assertion.
5. **The embedded decoder substitutes 0.5 mm for missing spacing and uses only one PixelSpacing component for a square physical crop.** This is a training-contract assumption, not recovered geometry. For new training, support separate row/column spacing and an explicit missing-spacing policy. For old checkpoints, preserve the published transform as a control and test a revised transform separately.
6. **Raptor constructs a concatenated fixed-slot volume and samples triplets over it.** Boundary windows can cross series. A new sequence architecture should keep series boundaries explicit; do not silently change old checkpoint inputs and assume equivalent predictions.

Source fingerprints:

```text
RSNA Baseline v15: eb850464cec6a406d36a225625cc4bf22916720ec0175c8aab1fcf6023eec739
Dinosaurs v38:     5e03fc9f7799b2e46bcf67395d1ad5177d1e9f19b58cc93825c9fc02da6d0155
Restructured v5:   e3d4a594dd5f7b2f5104f05f716a36adc18ed80bb5451d4d94f6266249b93b7c
Hybrid v1:         f7c351b8976a75e7f7227bd78ab5f2cbb00f978221481a266e5f7baf5ec212a4
```

### Evidence from your existing project

These are **recorded project measurements**, not experiments rerun for this report:

- The root README records 0.935 public LB, a stale-checkpoint discovery, and poor training throughput from remote cache I/O. Benchmark the current environment before repeating that older hardware conclusion.
- The OOF harness records **+0.0010, 95% CI [+0.0001, +0.0019]** from adding a fourth family on 4,349 weak-labelled studies. It also records **−0.0008** from per-target weight fitting versus equal weights on the expert evaluation. These are different cohorts/objectives and cannot be summed into an LB forecast.
- The project uses **58 expert-labelled studies** for its clean-label audit. Its measured seed variability and broad confidence intervals make tiny gold-cohort changes unreliable. Larger weak-label evaluation reduces sampling noise but preserves label bias.
- The V15 calibration specification records addressed-cell OOF Brier improving **0.1791 → 0.1347**. It explicitly does not establish an image-model AUC gain. The policy was developed using this cohort, so it remains exploratory.
- Existing `family_a/folds.py` balances study counts and expert-case counts. It does **not** perform multilabel stratification, and it explicitly cannot establish patient separation.

Relevant local evidence: [README](C:/Projects/rsna-knee-abnormality/README.md), [OOF measurements](C:/Projects/rsna-knee-abnormality/oof_harness/README.md), [V13 input contracts](C:/Projects/rsna-knee-abnormality/v13/README.md), [V15 calibration specification](C:/Projects/rsna-knee-abnormality/v15/CALIBRATION_FIX.md), [current folds](C:/Projects/rsna-knee-abnormality/v15/family_a/folds.py).

## 3. Highest-return experiment order

These priorities are engineering judgments informed by the evidence above. **No fabricated per-technique gain ranges are assigned.**

| Priority | Intervention | Why it earns compute | Promotion requirement |
|---|---|---|---|
| P0 | Reproduce one reported 0.941 notebook, freeze artifacts | Establishes the actual control and active arms | Exact source/version/weights; all expected studies; no unexplained arm failure |
| P0 | Dynamic schema, ID alignment, decoder and rank audits | Prevents avoidable hidden-set failures and silent mismatches | Contract tests; unchanged predictions where a refactor should be equivalent |
| P1 | Audit report labels; masked, fold-local calibrated auxiliary supervision | Clean expert supervision is scarce; wrong negatives damage all architectures | Paired baseline/raw-aux/calibrated-aux runs; clean-label and ensemble evaluation |
| P1 | Fine-tune DINOv2-S with target-aware multi-series pooling | Already supported by your Family A code; affordable first trainable model | Adds held-out gain to the complete frozen control |
| P2 | New CNN + within-series BiGRU + target-query pooling | Preserves local slice context and introduces architectural diversity | Beats matched CNN pooling control or adds ensemble value across seeds |
| P2 | High-resolution, anatomy-focused specialist | Targets missed small structures with genuinely new pixels | Better implicated targets without offsetting macro loss or excessive runtime |
| P3 | Conservative BCE/focal/ASL ablation | Can reduce domination by easy negatives | Same folds, samples, epochs and initialization; no rare-target collapse |
| P3 | Small predeclared ensemble/TTA comparison | Harvests measured diversity | Nested/frozen selection and runtime margin |

**First training comparison:** keep the current Family A architecture fixed and compare expert-only, raw report auxiliary, and calibrated report auxiliary. Then change architecture. Simultaneously changing labels, loss, resolution, and model prevents useful attribution.

## 4. Dynamic data pipeline: executable contract

### 4.1 Dynamic discovery with a frozen semantic schema

The runnable [reference_patterns.py](C:/Projects/rsna-knee-abnormality/research/next_tier_095/reference_patterns.py) provides:

- `Schema`: infers target names and output dimension from the sample-submission header; maps checkpoint target order explicitly.
- `find_root`: finds a uniquely matching mount at arbitrary nesting depth.
- `KneeDataset`: reads the current split CSV at initialization and discovers actual series directories per study when loading.
- `adjacent_windows`: chooses centers dynamically while retaining actual adjacent slices as channels.
- `collate_studies`: flattens variable numbers of series, pads variable sequence lengths, and retains study ownership.
- `SequenceMIL`: encodes only real windows, packs sequence lengths into a GRU, and pools by study/target.
- `masked_loss`: handles unknown labels and supplies BCE plus a simplified asymmetric focal variant.
- `stratified_group_folds`: correct SGKF API usage with an explicitly approximate stratification surrogate and all-target audits.
- `write_submission`: joins by study identity and uses live test IDs, never sample-submission row count.

**This is runnable reference code, not a ready-to-submit model.** It includes a synthetic reader for testing; the real reader must implement the audited DICOM/preprocessing contract below. No weights or competition images are bundled.

```python
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from reference_patterns import Schema, KneeDataset, collate_studies

root = Path(config.data_root)  # set externally; optional unique mount discovery
schema = Schema.from_sample(root / 'sample_submission.csv', config.id_col)
schema.save(config.output_dir / 'schema.json')

dataset = KneeDataset(
    root=root, split='test', schema=schema,
    read_series=validated_reader,  # defined by the checkpoint contract below
)
loader = DataLoader(dataset, batch_size=config.study_batch_size,
                    shuffle=False, collate_fn=collate_studies,
                    num_workers=config.workers)
n_targets = len(schema.targets)
# Later: reorder model outputs once using names saved in its checkpoint.
permutation = schema.checkpoint_permutation(checkpoint['targets'])
```

The fixed names `StudyInstanceUID` and `SeriesInstanceUID` are **semantic identifiers**, supplied as configuration. They are not fixed path lengths or class arrays. A named directory layout is an official storage contract; if a different layout is introduced, replace the discovery adapter with header-based grouping by those UIDs. Do not guess identities from filename substrings.

**Never infer targets by “all numeric columns in train.csv.”** That can include identifiers, metadata, and confidence values. Infer targets from the submission schema; retain that ordering and its hash in every checkpoint. Hidden data may change the number of studies or sequences, but a checkpoint cannot acquire a new trained output class by resizing its head at inference. Reject a genuinely changed class set with a clear diagnostic.

### 4.2 Correct handling of multi-label and multi-class targets

For this challenge: independent sigmoid outputs `[batch, len(schema.targets)]`; no categorical encoding of finding names is necessary. Unknown cells stay NaN and have zero loss weight. A silent report is not a confirmed negative.

For another RSNA task with genuinely mutually exclusive severities, derive vocabulary from the task ontology, save it once, and reuse it:

```python
# The ontology defines semantics/order. Do not learn a different mapping per fold.
vocab = {target: tuple(ontology[target]) for target in categorical_targets}
class_to_index = {t: {value: i for i, value in enumerate(values)}
                  for t, values in vocab.items()}
heads = {t: torch.nn.Linear(feature_dim, len(values))
         for t, values in vocab.items()}

def encode_label(target, value):
    if pd.isna(value):
        return -100  # explicit CrossEntropyLoss ignore_index
    if value not in class_to_index[target]:
        raise ValueError(f'Unexpected label for {target}: {value!r}')
    return class_to_index[target][value]
```

If no formal ontology exists, derive observed training vocabularies once and review/save them before splitting. A training fold missing a rare class must retain its output channel. Never fit a label encoder on test data.

### 4.3 DICOM reader: required implementation steps

Use your existing decoder for legacy weights, after numerical reproduction. For a newly trained model, implement this explicit reader contract. The following is **pseudocode for the adapter**, not an assertion that helper functions already exist:

```python
def validated_reader(series_dir, metadata):
    paths = discover_dicom_files(series_dir)  # no assumed slice count
    headers = read_headers_and_deduplicate_sop_uids(paths)
    stacks = partition_by_series_orientation_and_acquisition(headers)
    stack = select_stack_with_frozen_quality_policy(stacks)
    ordered, geometry_flags = order_by_geometry_or_logged_fallback(stack)
    decoded = decode_with_verified_transfer_syntax_plugins(ordered)
    decoded = apply_modality_transform_and_monochrome_policy(decoded)
    decoded = apply_frozen_orientation_and_laterality_policy(decoded)
    decoded = physical_crop_or_logged_fractional_fov_fallback(decoded)
    volume = normalize_per_series_and_resize(decoded, saved_contract)
    return adjacent_windows(volume, saved_contract.offsets,
                            saved_contract.max_centres)
```

**Geometry and pixel rules:**

1. Decode pixels using the DICOM transfer syntax; validate JPEG Lossless and JPEG 2000 as well as uncompressed cases. Bundle compatible offline decoder dependencies. [pydicom compressed-pixel guidance](https://pydicom.github.io/pydicom/stable/guides/user/image_data_handlers.html).
2. Use decoded `array.shape` for actual height/width. Missing CSV Rows/Columns is harmless if pixels decode; missing essential DICOM pixel-description tags may make decoding impossible. Do not invent dimensions by square-rooting byte counts.
3. Within a geometrically consistent stack, compute a **shared** normal `n = cross(IOP[:3], IOP[3:])` and sort by `dot(IPP, n)`. Check orientation consistency before sorting. Prefer complete geometric ordering; use InstanceNumber only as a logged fallback for the entire stack. Never mix millimeter positions and instance numbers in one sort key.
4. Split or reject mixed acquisitions, duplicate locations, and scout stacks using a declared policy. Do not let filename lexicographic order define slice adjacency. A deterministic filename tie-breaker is acceptable only after geometry and duplicate handling.
5. Use row and column spacing separately for physical crops. Validate finite positive spacing. If absent, use valid shared-series metadata when available; otherwise apply a trained fractional-field-of-view fallback and mark spacing unknown. Missing spacing does not justify asserting a measured 0.5 mm value.
6. Apply modality scaling/LUT and a consistent MONOCHROME1 policy. MRI intensities are not CT Hounsfield units; do not copy CT windows. Mask padding/non-image regions when estimating intensities.
7. Normalize per series using a saved policy; handle nonfinite and constant arrays explicitly. Keep the same scope and quantization for training and inference. Per-slice normalization can destroy relative inter-slice intensity information.
8. Maintain physical coverage and orientation. Preserve aspect ratio with padding for a new model unless its trained contract specifies another transform. Resampling to a fixed tensor size is compatible with a dynamic dataset.
9. Treat singleton and short series explicitly: replicate edge slices for context channels, preserve true sequence length, and record reduced coverage. Exclude unusable series with a reason; retain every study ID.

Example aspect-preserving resize for a **new** transform:

```python
def resize_pad(slice_2d, output_hw):
    x = torch.as_tensor(slice_2d, dtype=torch.float32)
    h, w = x.shape                       # inferred from pixels
    out_h, out_w = map(int, output_hw)    # saved model config
    scale = min(out_h / h, out_w / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    x = torch.nn.functional.interpolate(
        x[None, None], size=(rh, rw), mode='bilinear',
        align_corners=False, antialias=True)[0, 0]
    top, left = (out_h-rh)//2, (out_w-rw)//2
    return torch.nn.functional.pad(x, (left, out_w-rw-left,
                                       top, out_h-rh-top))
```

Do not blindly catch every decoding exception and replace the whole study with zeros. Log errors by transfer syntax, site/protocol if available, series, and arm. In training, exclude all-image-missing studies from the image loss. In inference, route to the valid generalist/remaining-series model; use a saved training-prior vector only as a last resort. Persist the fallback reason. If missing studies have no visual evidence, there is no code trick that recovers their diagnostic accuracy.

### 4.4 Variable batch dimensions and missing sequences

The code packs each series separately through the GRU, then attends over all valid contextual windows belonging to a study. This avoids both cross-series recurrence and a hardcoded six-slot input. More series need not require a different model shape.

Shape flow:

```text
study -> variable list of series
series -> T actual neighbor windows [T, C, H, W]
batch -> padded [number_of_series_in_batch, max_T_in_batch, C, H, W]
encoder -> only valid windows
packed BiGRU -> contextual windows within each series
target queries -> [number_of_studies_in_batch, number_of_targets]
```

Lengths must be actual valid lengths, never padded lengths. PyTorch's packed-sequence interface removes padding from recurrent processing. [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html).

Use length/series-count buckets and a configurable total-window budget to bound memory. Encoder microbatching bounds individual forward work, but **does not remove all saved training activations**; use gradient checkpointing, fewer sampled centers, or gradient accumulation when training remains memory-bound.

### 4.5 Metadata and reports without a train/test mismatch

Reports may supervise training but must never be required by the submitted model. Useful candidates for series embeddings include plane, fluid sensitivity, fat suppression, and geometric quality indicators. Their meanings matter more than their column positions.

Discover candidate metadata by name/type, audit availability in both documented schemas, exclude IDs/reports/outcome-bearing fields, then freeze the approved feature schema. Fit numeric imputation/scaling and categorical vocabularies on the training fold only. Use a distinct unknown/missing category and mask at inference. Ignore extra test columns; missing approved fields use saved fallbacks, never a newly fitted test scaler.

```python
# Fit only after splitting; approved_columns is saved configuration built by audit.
from sklearn.feature_extraction import DictVectorizer
encoder = DictVectorizer(sparse=False)
train_meta = train_frame.reindex(columns=approved_columns).fillna('__MISSING__')
encoder.fit(train_meta.to_dict('records'))
test_meta = test_frame.reindex(columns=approved_columns).fillna('__MISSING__')
test_features = encoder.transform(test_meta.to_dict('records'))
# For numeric columns, impute with training-fold medians BEFORE this operation,
# add missing indicators, and scale separately. Do not mix numeric/string types.
```

Unknown categorical values become all-zero contributions with this simple encoder; add a trained explicit unknown bucket for a neural embedding implementation. Missingness dropout during training must preserve at least one usable image series.

### 4.6 Submission and rank semantics

```python
# Correct ties; never double-argsort.
ranked = predictions.rank(method='average', pct=True)

# Reorder checkpoint outputs by saved names, then build a keyed table.
p = torch.sigmoid(logits).cpu().numpy()[:, permutation]
prediction_frame = pd.DataFrame(p, columns=schema.targets)
prediction_frame.insert(0, schema.id_col, batch_ids)
# Concatenate every batch; write_submission validates exact live-test coverage.
```

**Do not rank per batch.** A batch's scores would depend on its neighbors. If preserving a legacy rank ensemble, rank each model over the same complete study set after inference, then blend. Adding unrelated studies can still change empirical-rank ensemble weights/orderings. For a new pipeline requiring per-study subset invariance, prefer a fixed probability/logit blend, or maps fitted on training-side OOF predictions and frozen before test. Evaluate the change; rank and probability blends are not equivalent.

Cache key = hash of source identity/content manifest + decoder version + full preprocessing contract + resolution + crop + spacing/orientation policy + series selection + sampling + normalization. Add encoder checkpoint hash for cached features. Do not cache augmented training images as if they were deterministic or reuse features after changing encoder weights. Rebuild test manifests from live input at each scoring run.

## 5. Architecture: what to train next

### 5.1 DINOv2 Small is fine-tunable, but it is a backbone

The supplied [Kaggle DINOv2 Small model](https://www.kaggle.com/models/metaresearch/dinov2/PyTorch/small) is a plausible initialization. It is not a knee-specific twelve-finding classifier by itself. Meta provides DINOv2 visual backbones and distinct variants with/without registers. Inspect the mounted artifact format and instantiate the exact variant. [Meta's implementation](https://github.com/facebookresearch/dinov2), [DINOv2 paper](https://arxiv.org/abs/2304.07193).

Your existing `family_a/model.py` already contains `Dinov2Encoder` and a target-query head. Reuse that loader where it matches the attached format. A PyTorch state dictionary is not automatically a Hugging Face `from_pretrained` directory.

Fine-tuning configuration to start from, **not an optimized result**:

- Warm up a newly initialized head for one fixed epoch; train last two encoder blocks plus final normalization next.
- Encoder LR `8e-6`; head LR `1e-3` matches your current Family A starting point. Use AdamW, weight decay `0.01`, gradient norm cap `1.0`, short warmup and cosine decay as an initial controlled recipe.
- Choose epochs on an inner training split or predeclare them; keep outer expert labels out of checkpoint selection.
- Start at the existing native preprocessing resolution. For a newly trained patch-14 DINOv2, a possible progression is `224 -> 336 -> 448`, with positional interpolation supported by the loader. Do not blindly substitute a nondivisible size or change the existing checkpoint's crop.
- Use fp16 on a verified T4 configuration; choose autocast from a real arithmetic probe and device capability. Keep losses and probability accumulation in fp32.
- Compare partial fine-tuning against frozen encoder + head. If full fine-tuning is attempted, lower backbone LR and measure overfitting across folds.

DINOv2 is a transformer. The recommended **CNN+GRU** family below is an architectural complement, not a renamed DINO head.

### 5.2 New family: adjacent-slice CNN + BiGRU + label-aware MIL

```text
geometry-ordered MRI series
  -> neighboring slices as channels, e.g. [-1, 0, +1]
  -> pretrained CNN encoder
  -> optional plane/protocol/relative-position embedding
  -> bidirectional GRU within each series
  -> one attention query per finding over contextual windows
  -> independent study-level logits
```

Use a CNN already supported by your environment, e.g. a ConvNeXt or EfficientNet implementation, after verifying checkpoint compatibility. Compare to the **same encoder with mean/max/attention pooling and no GRU**. The GRU earns its cost only if it adds held-out value.

MRNet demonstrates multi-series knee-MRI classification, while attention MIL provides a trainable way to aggregate instances. Neither paper proves this CNN+GRU proposal reaches 0.95 in this competition. [MRNet study](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699), [attention MIL paper](https://arxiv.org/abs/1802.04712).

Implementation details that matter:

- A triplet must contain neighboring slices in the original series, not three widely separated sampled centers. Preserve gaps/physical positions when centers are subsampled.
- Apply the same random spatial transform to all channels/windows in a series. Independent slice rotations invent anatomy.
- Add geometry/relative-position features with a missingness mask; do not connect sagittal and coronal slices through one temporal sequence.
- Target queries allow different findings to attend to different regions. Inspect attention for gross failures, but do not treat an attention map as validated lesion localization.
- Train with conservative series dropout; retain real sequences at inference. Make complete series absence a learned condition where feasible.
- Cache deterministic decoded images locally; cache encoder features only when that encoder is frozen.

### 5.3 Text: privileged training information

**Preferred first experiment:** report-derived soft labels with per-cell masks and fold-local calibration. Distinguish present, explicitly absent, uncertain, unmentioned, historical, and postoperative mentions; retain evidence spans and language/source provenance. Translation and extraction must preserve negation and anatomy.

**Second experiment if budget permits:** paired image/report representation pretraining, followed by an image-only prediction head. A text teacher may supply embedding or probability targets during training; validation-study reports must not train that fold's teacher/student. ConVIRT establishes the general paired-image/report representation-learning approach, not a numerical gain for knee MRI. [ConVIRT paper](https://arxiv.org/abs/2010.00747).

Keep report teacher labels, image pseudo-labels, and expert labels as separate sources. Confidence is not the same as correctness. Never select the extractor prompt, source, target gate, or calibrator using outer-fold expert outcomes. If a foundation teacher has external pretraining, record provenance; any competition-specific adaptation must respect folds.

Your project already documents failure of exact source consensus. Do not promote “two LLMs agree” without an independent transfer audit; correlated errors are common across label sources.

## 6. Weak areas: targeted interventions

### 6.1 Start with an error ledger

For each target, export expert positive/negative/unknown counts, weak-label coverage, AUC, PR-AUC, sensitivity at a predeclared specificity, confidence interval, false-negative study IDs, protocol coverage, and lesion-size annotations where available. PR-AUC and sensitivity diagnose issues; macro ROC-AUC remains the selection metric.

Do not assume the rarest target is the worst-ranked one. With equal target weights:

`macro_gain = sum(per_target_gain) / number_of_targets`.

For twelve targets, improving three by 0.04 each gives only +0.01 macro AUC if every other target is unchanged. This arithmetic explains why a small meniscus routing tweak alone is unlikely to account for a +0.019 jump.

### 6.2 Missing labels and imbalance

Use masked BCE as the control. Compute label frequencies from **observed training-fold cells**, not filled zeros. Keep expert supervision from being diluted by the much larger report-labelled pool; your existing runner uses a declared expert-row sampling proportion. Audit batch counts and unique-study exposure.

Then compare exactly one modification at a time:

| Variant | Initial setting | Best use / failure mode |
|---|---|---|
| Masked BCE | No focusing | Stable baseline, including soft report labels |
| Weighted BCE | Training-fold negative/positive ratio, square-rooted and capped | May help rare positives; large weights overamplify noisy positives |
| Focal | `gamma_pos = gamma_neg = 2`, clip zero in reference ASL function | Downweights easy examples; can emphasize hard mislabeled cases |
| Conservative ASL | `gamma_pos=0`, `gamma_neg=2`, clip zero | Reduces easy-negative domination while retaining positive gradients |
| Stronger ASL ablation | `gamma_pos=0`, `gamma_neg=4`, negative clip `0.05` | Optional comparison after conservative setting; can suppress useful negatives |

Focal and ASL are general classification losses, not uniquely medical losses. Their papers support mechanisms and results on their own tasks; they do not guarantee improvement on this dataset. [Focal Loss](https://arxiv.org/abs/1708.02002), [ASL paper](https://arxiv.org/abs/2009.14119), [official ASL implementation](https://github.com/Alibaba-MIIL/ASL).

```python
expert_mask = expert_mask & images_available[:, None]
aux_mask = aux_mask & images_available[:, None] & ~expert_mask
loss = masked_loss(logits, expert_y, expert_mask, mode='bce')
loss += masked_loss(logits, report_y, aux_mask, mode='bce',
                    target_weights=fold_local_aux_weights)
```

Zero the report-silent mask from the source's documented sentinel **before** calibration. Do not assume every 0.5 in every label source means silence; preserve an explicit addressed flag. Use BCE for soft report probabilities. The supplied ASL pattern deliberately rejects observed soft labels.

If sampling rare classes, sample **studies**, never split slices from the same study across batches/folds as independent cases. Cap oversampling and audit co-occurring positives to avoid repeatedly showing the same few studies. Do not simultaneously apply extreme positive weights and extreme oversampling. Dice/Tversky are appropriate only for a genuine segmentation auxiliary task with spatial supervision; they are not the default study-classification objective.

### 6.3 Tiny lesions and subtle structural abnormalities

**Preserve pixels before buying a larger model.** Upsampling an existing 224/336 cache cannot recover a lesion removed during downsampling.

For a new high-resolution experiment:

1. Rebuild native-source crops at the highest planned resolution, preserving geometry and useful bit depth.
2. Train the whole-study model at low/medium resolution for a fixed schedule.
3. Fine-tune at higher native resolution with lower LR and smaller microbatches.
4. Add a whole-joint branch plus anatomy-focused crops. Retain the whole-joint branch when a crop/localizer fails.
5. Generate validation crops with a localizer trained without that validation group's images/labels. Use the same predicted-crop process at inference.
6. Train a specialist only if residual errors support it, and evaluate its **marginal ensemble gain**.

Potential specialist scopes are menisci/ligaments, cartilage compartments, or marrow findings. These are hypotheses to choose from the error ledger. Localization labels may be newly annotated where permitted; attention-derived crops remain weak supervision and require coverage review.

Avoid large random crops, aggressive erasing, or geometric changes that remove tiny findings. Keep crop margin and a global view. Treat left/right mirroring as a geometry policy requiring visual review; medial/lateral relationships must remain coherent.

### 6.4 Multi-label dependencies

Use shared image features with independent logits as the control. Target-query attention can capture shared anatomy without hard diagnosis rules. A small label-token attention layer is a later ablation; use regularization and missing-label masks.

Do not impose rules such as “effusion implies synovitis” or “ACL injury requires contusion.” Dependencies vary across cohorts and are not deterministic. Fit any classifier-chain/stacking features from training-side OOF predictions; in-sample base logits contaminate a meta-model's validation.

## 7. Validation integrity: grouped, multilabel-aware, and honest

### 7.1 Define the independent unit

Build a study table with one row per study. Connect known same-patient studies, repeat examinations, bilateral studies from a shared patient, and verified duplicate image sets into indivisible groups. Patient IDs must be audited for de-identification/site collisions; a convenient DICOM string is not proof of identity. If no reliable linkage exists, use study/duplicate groups and explicitly report **patient separation unverified**.

Keep all images, reports, crops, feature caches used for supervised training, and pseudo-label adaptations for a group on the same side. An externally pretrained generic backbone is distinct from a competition model trained on validation studies. Full-trained public competition checkpoints cannot generate honest OOF for their own training cohort merely by being loaded inside a fold loop.

### 7.2 Stratified Group K-Fold correctly

Standard `StratifiedGroupKFold` preserves a **one-dimensional class distribution** subject to group constraints. It does not directly accept this challenge's full multilabel matrix. Do not flatten study/label pairs or duplicate studies independently to force it. [scikit-learn SGKF documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

The executable SGKF pattern uses a study's rarest observed positive as a surrogate stratum, distinguishes all-unknown studies, and audits every target afterward. That is a useful minimum implementation, **not exact multilabel stratification**.

```python
fold, audit = stratified_group_folds(binary_or_unknown_y, verified_groups,
                                    n_splits=config.folds, seed=config.split_seed)
for row in audit:
    print(row['fold'], row['positive'], row['negative'], row['scorable'])
```

For the final split, prefer group-level multilabel allocation, with this executable-design pseudocode:

```text
For each indivisible group:
    count studies, expert-observed positives/negatives per target,
    weak-observed positives/negatives separately, unknowns,
    frequent label-pair positives, and available site/protocol categories.
Aggregate counts; compute desired per-fold counts = total / K.
Order groups by rare expert-label support, then group size, seeded tie-break.
Assign each group to the fold minimizing weighted squared normalized
    deviation from desired counts, including study count and expert count.
Do bounded group swaps to reduce imbalance without splitting any group.
Audit train AND validation support for every target and every fold.
Freeze split using only label/group/domain balance, never model scores.
```

Iterative multilabel stratification has a literature basis, including balancing label pairs. Group preservation is an additional constraint you must enforce; an ungrouped iterative splitter is insufficient when patients have multiple studies. [Multilabel stratification research](https://proceedings.mlr.press/v74/szyma%C5%84ski17a.html).

Predeclare five folds only if feasible. If a target has fewer than five independent positive groups, it cannot have positive support in every validation fold. Even with sufficient counts, overlapping group constraints can make a desired allocation impossible. Reduce K or report undefined target-fold AUC explicitly; **never quietly nanmean a different target set per fold**.

With the project's small expert subset, report pooled cross-fitted expert predictions plus per-fold support. Pooled OOF AUC can itself be affected by different fold-model score scales; accompany it with valid per-fold AUCs and a fixed training-only scale-alignment policy if used. Repeating seeds does not create more independent patients.

### 7.3 Three complementary evaluation panels

1. **Expert OOF:** closest label source to the target evaluation, but currently small and repeatedly consulted.
2. **Weak-label OOF:** larger screening panel, useful for rejecting unstable experiments; explicitly label it agreement with report-derived targets. Do not relabel thresholded report probabilities as expert truth.
3. **Domain stress panel:** hold out an institution when reliable site information exists; otherwise examine manufacturer/protocol/geometry groups with appropriate caveats. Also assess missing planes, severe artifacts, spacing fallback, and short/long stacks.

Do not replace patient grouping with scanner grouping; they answer different questions. Test metadata-only and image-only variants to detect scanner/site shortcuts. Standard patient-grouped folds and site-held-out evaluation are complementary.

If allowed and feasible, expand the blinded expert-labelled evaluation cohort before testing many small improvements. Size it using patient-level bootstrap simulations/power from your pilot and target delta; do not invent a universal required N. Labels acquired after examining candidate predictions must be treated carefully to avoid preferential evaluation sampling.

### 7.4 Outer folds must remain outside every learned decision

For each outer fold, training-only decisions include report calibration, pseudo-label generation using adapted teachers, localizer fitting, feature normalization, metadata encoding, target weighting, and hyperparameter/epoch choice. Inner OOF is required for fitted ensemble weights or a learned meta-model. Preserve a final untouched expert set if available; this project's existing 58-case cohort is already development data.

Use saved fold assignments for every arm. Save row-keyed OOF predictions and receipts listing training/validation group IDs and checkpoint hashes. A filename saying OOF is not evidence of held-out training provenance.

### 7.5 Promotion and uncertainty

Predeclare the primary comparison and a practical development threshold, for example **+0.001 macro AUC in each of two seeds** for a new arm. This is a compute-allocation gate, not significance at n=58. Require no systematic damaging target/domain regression and positive marginal gain in the complete blend.

Bootstrap **groups**, using the same sampled groups for baseline and candidate. With verified singleton studies, study bootstrap is equivalent. Preserve within-group rows and paired predictions. Track bootstrap replicates with an unscorable target; reject or qualify an interval dominated by such replicates. Report seed variation separately; two seeds cannot characterize all optimization uncertainty.

Do not present `public_baseline + OOF_delta` or its bootstrap interval as a measured/predictive private score. It assumes perfect transport across label sources and domains. The existing `evaluate_submission_gain.py` exposes explicitly conditional projections; use its paired-development comparison, not those projections, for evidence claims.

Freeze the recipe before a small number of leaderboard checks. Never choose among dozens of target weights on the public split and call the resulting score private-test validation. Correlated forks count as correlated probes.

## 8. Execution plan and required outputs

### Stage A — control and correctness

1. Select one supplied notebook, pin the exact scored source version if available, and save its source/checkpoint/dependency fingerprints.
2. Run that control with its documented attachments and offline environment. Save all arm predictions and runtime/fallback receipts.
3. Add ID/schema/discovery changes as a separate revision. Require probability equality within a declared numerical tolerance before any rank change.
4. Fix tie handling as its own numerical ablation. Test reordered inputs and duplicated scores.
5. Audit the same decoder on a representative sample covering available transfer syntaxes, planes, resolutions and missing metadata. Inspect image grids against originals.

**Output:** `control_receipt.json`, keyed `control_predictions.csv`, `schema.json`, `preprocessing.json`, `decode_audit.csv`, and a locked source manifest. A successful example-test run alone is insufficient.

### Stage B — data and validation

Create `study_manifest.parquet`, `series_manifest.parquet`, `groups.csv`, `folds.csv`, `fold_audit.json`, and a source-aware label table. Preserve explicit expert/report/unknown masks and calibration provenance. Fold assignment is versioned once; downstream jobs consume it.

Benchmark deterministic caching on 100–200 representative studies. Measure warm and cold I/O, decode time, train step time, peak memory, and storage use. Copy working shards to fast local scratch if permitted. Your older remote-memmap bottleneck must be solved before comparing model speed.

### Stage C — the first controlled training experiment

Reuse the existing Family A runner as the starting point. It already implements expert/raw auxiliary arms, two seeds, per-fold policy selection, shard merging and OOF export. Its fixed target/slot contract should be retained for reproduction; a dynamic-series model is a new version requiring new training.

The calibrated auxiliary arm remains a **change specification** in the inspected project; implement and test that arm before claiming to run it. Its exact policy is in the existing calibration specification.

Run order:

1. CPU smoke tests and label/fold audits.
2. One matched-fold smoke run to establish feasible epochs and cache throughput; do not promote a model from it.
3. Full matched expert-only versus raw auxiliary comparison across frozen folds/seeds.
4. Add fold-local calibrated auxiliary as a matched third arm.
5. Promote supervision recipe from the predeclared development gates; then compare a new CNN+GRU family.

Existing Windows smoke command:

```powershell
Push-Location C:\Projects\rsna-knee-abnormality\v15\family_a
python -m unittest discover -p "test_*.py"
Pop-Location
```

Existing training/cache notebook execution needs the data, pretrained models, matching packages, and GPU setup described in [Family A RUN.md](C:/Projects/rsna-knee-abnormality/v15/family_a/RUN.md). That document describes twenty fold-runs for two arms × two seeds × five folds; adding a third arm makes thirty at five folds. Recompute counts if the validated K changes.

**Per-run outputs:** checkpoint with schema/config/fold provenance, expert-only OOF, full-cohort OOF, per-target metrics/support, timing/memory, and sample/auxiliary-weight receipts. Preserve all shards before merging. Do not mix a new fold scheme with old OOF files.

### Stage D — diversity and native resolution

Only after a supervision winner emerges:

- Train matched CNN pooling and CNN+BiGRU models.
- Inspect largest remaining false-negative clusters on development data.
- Train one targeted high-resolution branch; compare global-only, crop-only, and global+crop on the same held-out cases.
- Evaluate a conservative frozen blend coefficient before fitting any target-specific weights. If fitting is necessary, use nested training-side OOF.

**Output:** `ablation_table.csv` with experiment, label policy, encoder, pooling, native resolution, seed, fold hash, expert AUC, weak-label AUC, target deltas, domain results, complete-ensemble delta, inference seconds and peak memory.

### Stage E — private-test engineering rehearsal

Test all of these before final inference:

| Perturbation | Required behavior |
|---|---|
| Arbitrary root nesting and UID lengths | Same keyed raw predictions |
| More test rows than sample submission | Every live test ID scored exactly once |
| Reordered rows/files/batches | Same keyed raw probabilities; tie-aware ranks |
| Batch size 1 versus larger | Same probabilities within declared precision tolerance |
| Different series/slice counts | Valid masked tensors, bounded compute |
| Missing CSV dimensions/optional metadata | Infer from pixels or trained missing-data policy |
| Missing series/all-image-missing study | Valid-series model or explicit saved fallback, logged |
| Unknown category | Frozen unknown encoding, no shape growth |
| Duplicate IDs or checkpoint-label mismatch | Clear failure before writing misleading output |
| Compressed or corrupted DICOM | Supported decoding or localized logged recovery |
| Offline restart | All packages, code and weights available without downloads |
| Different test subset | Raw probability blend invariant; empirical rank blend explicitly cohort-dependent |

Kaggle currently requires offline notebook inference, a file named `submission.csv`, and at most nine hours for CPU/GPU notebooks. [Official code requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/efficiency-prize-evaluation).

Budget from measured study counts/window counts and long-tail runtime, not the example test set. Target a rehearsal with at least 25% time margin as an engineering policy. Complete a generalist pass for all studies first, then optional specialist passes. For a rank ensemble, only completed whole-cohort arms should enter ranking/fusion. Stopping an arm halfway and mixing inconsistent per-study model sets can distort ordering.

Final release checks: exactly one row per live test ID; exact target schema; finite probabilities/scores in [0,1]; no accidental index column; no dropped studies; checkpoint and input hashes; fallback counts; stable completed-arm recipe; atomic final write.

## 9. What was delivered and verified

Delivered this report, four pinned public notebook sources with inventories, readable copies of embedded Raptor code for inspection, and executable reference patterns. **No existing model/notebook was modified, trained, uploaded, or submitted.**

Run the new standalone patterns from any working directory:

```powershell
python C:\Projects\rsna-knee-abnormality\research\next_tier_095\reference_patterns.py
```

The synthetic CPU checks passed: live hidden-ID replacement, arbitrary UID/root lengths, dynamic class count and checkpoint order, variable series/slice counts, missing series metadata, all-image-missing masking, CNN/GRU forward and gradients, padding invariance, all-unknown loss, keyed submission reordering/missing-ID rejection, and grouped split separation with per-target audits.

**Not established:** real-DICOM numerical correctness of a new reader, GPU throughput of new models, trained AUC, reproduction of the reported 0.941, or attainment of 0.95–0.96. Those require the concrete runs above. The path to the requested score is to improve measured image evidence and supervision while using these dynamic contracts to prevent avoidable loss at inference.
