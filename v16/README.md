# V16 — dynamic knee MRI training and inference

**Implemented accuracy candidate. Not yet trained on the competition data or scored.**

V16 implements the new model proposed in the [engineering report](C:/Projects/rsna-knee-abnormality/research/next_tier_095/REPORT.md). It is a new trainable branch, not a relabelled 0.941 checkpoint. Keep the supplied 0.941 notebook as the paired control. V16 cannot use that ensemble's weights as if they were weights for this architecture.

## What runs

**2026-09-16 stability update:** the first Kaggle pilot failed before completing
epoch one because gradient clipping raised on an AMP overflow before the loss
scaler could recover. The sequence/attention head now runs in FP32. The optimizer
rejects overflowing gradients, lowers the scale, and retries the same batch up to
12 times; persistent failures still stop with the study IDs. Progress includes
batch counts and retry counts. The Kaggle launcher tests 20 synthetic training
steps on its allocated GPU before starting image preparation. This addresses the
observed failure path; successful full-data training must still be verified.

- Real single-frame DICOM decoding, geometry ordering, identity checks, per-series normalization, rectangular physical crops, and aspect-preserving resizing.
- Missing CSV dimensions are irrelevant: spatial shape comes from decoded pixels. Missing spacing uses valid series spacing or a logged full-FOV fallback. Corrupt/unusable series are logged; every test study keeps its ID.
- Dynamic output schema from the sample-submission header, live test IDs, variable series/slice counts, masked padding, checkpoint-label reordering.
- DINOv2 or a CNN encoder, within-series bidirectional GRU, geometry/protocol embeddings, and finding-specific attention across a study's valid windows.
- Group-preserving approximate multilabel fold balancing with per-target support audits. Without a groups CSV, patient separation is explicitly unverified.
- Expert-only (`none`), raw report auxiliary (`raw`), and fold-local calibrated report auxiliary (`platt`) arms.
- Fixed-epoch training, gradient checkpointing, conservative encoder LR, capped auxiliary weights, deterministic epoch sampling, checksum-bound resume and fold merging.
- Streaming offline inference, missing-study priors fitted from training labels, keyed output, complete-arm banking, and paired group-bootstrap evaluation.

## Files

| File | Purpose |
|---|---|
| `config.json` | Starting configuration; edit model paths before preparation |
| `rsna-knee-v16-prepare.ipynb` | Self-contained Kaggle cache/fold preparation |
| `rsna-knee-v16-train.ipynb` | Self-contained training for one fold/seed/arm |
| `rsna-knee-v16-infer.ipynb` | Self-contained offline inference from trained V16 checkpoints |
| `rsna-knee-v16.zip` | Notebook and source package |
| `data.py` | DICOM reader, compact train cache, streaming inference |
| `model.py` | Encoders, sequence model, masked BCE/ASL |
| `validation.py` | Grouped folds and fold-local supervision/calibration |
| `runner.py` | Preparation, training, resume, OOF merging, inference |
| `evaluation.py` | Paired expert-label comparison and group bootstrap |

## 1. Install and attach inputs

Use a separate environment with a working CUDA-compatible PyTorch build. Install the dependencies in `requirements.txt` there. The offline Kaggle run must have these packages available already or install them from attached compatible wheels with `--no-index`; these notebooks never download dependencies or models.

```powershell
Set-Location C:\Projects\rsna-knee-abnormality
python -m pip install -r v16/requirements.txt
```

Required training inputs:

1. Competition root with current `train.csv`, `train_series.csv`, `train_series/`, and `sample_submission.csv`.
2. An attached DINOv2 encoder, in either format below.
3. For auxiliary training: a CSV containing the study identity field plus finding columns named exactly as in the submission schema. It must cover every training study, including expert studies needed to fit the fold-local policy. Values may be probabilities or missing. Default `silent_value=0.5` follows this project's report-source convention; change it if your source uses another convention. Do not reinterpret ordinary uncertain 0.5 probabilities as silence without checking the source.
4. Recommended: a CSV with the study identity field and `group_id`, covering every training study. Supply verified patient/duplicate groups. Do not assume DICOM PatientID is reliable without auditing it.

### DINOv2 encoder format

**Hugging Face directory:** set `encoder_path` to a directory containing `config.json` and local weights loadable by `Dinov2Model.from_pretrained`. V16 saves the architecture configuration and all encoder weights in its trained checkpoint, so the original pretrained directory is unnecessary for inference.

**Original Meta Small `.pth`:** set `encoder_path` to the exact plain backbone state dictionary and `dinov2_repo` to an attached local `facebookresearch/dinov2` code checkout containing `hubconf.py`. It loads `dinov2_vits14` with `source='local'`, `pretrained=False`, then strict state loading. The local code checkout is needed at inference; override its new mount with `--encoder-code` or `V16_DINOV2_REPO`. Registers/base/large checkpoints are not interchangeable with this Small variant. This native path requires reproduction with your actual artifact before training.

**CNN alternative:** set `encoder=cnn`, `cnn_arch` to a compatible timm architecture, and `encoder_path` to an attached backbone-only state dictionary for `num_classes=0`. Strict loading intentionally refuses unmatched classifier keys. Default candidate architecture is `convnext_tiny`.

`encoder=tiny` is reserved for synthetic tests; it has no pretrained medical-image capability.

## 2. Set the starting configuration

Edit [config.json](C:/Projects/rsna-knee-abnormality/v16/config.json) before preparing data.

The shipped starting recipe is 336 pixels, three adjacent context slices, up to 24 centers per series, up to 8 series per study, two unfrozen DINO blocks, a BiGRU with hidden size 128, eight fixed epochs, and study batch size one. These are explicit model/compute settings, not assumptions about the dataset's row count or class count.

Series/slice selection, resizing, and intensity policies are frozen into each checkpoint. Change resolution or sampling by making a new preparation directory and training a new candidate. Changing the hidden study count or directory name does not require changing the model.

**Disk planning:** training caches are native-source normalized float16 tiles, deduplicated across overlapping context windows. This preserves more intensity precision than uint8 but can still require substantial local storage. Do not assume all training caches fit Kaggle's working disk. Measure representative cache sizes and place the full preparation directory on adequate local scratch. Preparation stops before exhausting disk and can resume identical cached studies. Hidden-set inference streams directly and saves only diagnostics/predictions.

## 3. Prepare once: labels, groups, folds, and DICOM cache

The examples below are PowerShell. Replace the three input paths with actual data locations.

```powershell
Set-Location C:\Projects\rsna-knee-abnormality
$dataRoot = 'D:\rsna-data'
$workRoot = 'D:\rsna-v16-work'
$reportLabels = 'D:\rsna-labels\report_labels.csv'
$patientGroups = 'D:\rsna-labels\groups.csv'

python -m v16 prepare --data-root $dataRoot --work $workRoot --config v16/config.json --report-labels $reportLabels --groups $patientGroups
```

Omit `--groups` only when verified linkage is unavailable; the audit records the weaker grouping. For an expert-only preparation without a report table, set `auxiliary` to `none` and omit `--report-labels`. Such a preparation cannot meaningfully test auxiliary supervision later unless rebuilt with the report source.

Outputs:

```text
v16-work/
  config.json                 frozen starting configuration
  schema.json                 ordered finding names
  expert_labels.csv           expert values / missing cells
  report_labels.csv           raw report values / missing cells
  folds.csv                   one row per study with group and fold
  fold_audit.json             target support, grouping caveat, split hash
  preparation.json            input/config/code fingerprints
  cache/manifest.json         every study, availability, checksums, decode events
  cache/<content-hash>.pt     compact per-study tile/index tensors
```

**Read `fold_audit.json` before GPU training.** The allocator balances positive and negative expert-label counts at the group level; it cannot manufacture rare positive groups. Reduce K and prepare a new work directory if support is inadequate. Metrics return `macro_auc=null` when any target lacks both classes; they never silently drop that target. No fold allocation guarantees the private leaderboard distribution.

## 4. Run matched training arms

Start with one fold to establish memory/throughput. It is a smoke run, not a promotion decision:

```powershell
python -m v16 train --work $workRoot --fold 0 --seed 1400 --arm none
python -m v16 train --work $workRoot --fold 0 --seed 1400 --arm raw
python -m v16 train --work $workRoot --fold 0 --seed 1400 --arm platt
```

Then use the actual saved folds, not a hardcoded fold count:

```powershell
$foldNumbers = Import-Csv (Join-Path $workRoot 'folds.csv') | Select-Object -ExpandProperty fold -Unique
foreach ($seed in @(1400, 1401)) {
    foreach ($arm in @('none', 'raw', 'platt')) {
        foreach ($fold in $foldNumbers) {
            python -m v16 train --work $workRoot --fold $fold --seed $seed --arm $arm
            if ($LASTEXITCODE -ne 0) { throw "Training failed: $seed / $arm / $fold" }
        }
        python -m v16 merge --work $workRoot --seed $seed --arm $arm
        if ($LASTEXITCODE -ne 0) { throw "OOF merge failed: $seed / $arm" }
    }
}
```

Independent fold jobs can be run on separate GPU sessions. Copy every completed fold directory back under its matching work root before merging. Preserve the same prepared labels, folds, cache manifest, config, and source package. The merger refuses inconsistent shard configurations and checksums.

Training resumes an identical checkpoint automatically at the next epoch boundary. An interrupted partial epoch is rerun from the preceding saved state. Changing code, labels, folds, sampling settings, or configuration invalidates resume; use a new work directory for a changed experiment. There is no outer-fold early stopping.

Outputs per fold:

```text
runs/seed1400/platt/fold0/
  checkpoint.pt      model + encoder configuration + optimizer/scheduler/RNG
  oof.csv            predictions only for that fold's held-out studies
  receipt.json       policies, fitting IDs, training IDs, hashes, loss, AUC/support
```

The raw gate is fitted from training-fold expert cases only. Platt calibration changes addressed auxiliary probabilities without using its in-sample fit to unlock new targets. Expert cells override report cells; auxiliary weights are capped; validation-fold studies contribute to no training loss. Repeating seeds does not make the existing expert cohort an independent test set.

## 5. Evaluate held-out improvement

After all folds are merged:

```powershell
python -m v16 evaluate --work $workRoot --baseline-oof "$workRoot/runs/seed1400/raw/oof.csv" --candidate-oof "$workRoot/runs/seed1400/platt/oof.csv" --output "$workRoot/platt_vs_raw_seed1400.json"
```

This compares expert-label AUC with paired group-bootstrap uncertainty. Repeat for the second seed and the expert-only control. A practical +0.001 development threshold can determine whether to spend more compute; it is not a significance claim on 58 studies.

For marginal ensemble value, replace the baseline with independently audited OOF from your complete control and pass an explicitly predeclared candidate weight, e.g. `--weight 0.10`. The number is an experiment setting, not an optimized coefficient. The evaluator refuses incomplete row coverage and does not produce projected leaderboard scores. It cannot establish that an arbitrary uploaded baseline CSV was genuinely held out; inspect its training provenance.

## 6. Generate the live-test submission

Attach only the trained V16 checkpoints you selected using validation:

```powershell
$models = Get-ChildItem "$workRoot/runs/seed1400/platt/fold*/checkpoint.pt" | Select-Object -ExpandProperty FullName
python -m v16 infer --data-root $dataRoot --checkpoints $models --output "$workRoot/submission.csv" --cache "$workRoot/test_diagnostics" --device cuda
```

`--cache` is the inference diagnostics directory; V16 streams test DICOMs and does not build a full hidden-image disk cache. Multiple checkpoints currently perform separate streaming passes. Runtime must be measured on representative data before final selection.

The inference entry point:

1. Reads current `test.csv` IDs and sample-submission header.
2. Verifies checkpoint training completion, runtime code, preprocessing and target schema.
3. Loads the saved architecture/weights without downloading pretrained models.
4. Streams variable-length studies and averages model probabilities using fixed coefficients.
5. Uses the saved training-fold prior only for an all-image-missing study and lists every fallback.
6. Banks an output only after a checkpoint covers the entire cohort.
7. Uses measured completed-pass time to avoid starting another arm unlikely to fit `--max-seconds` (default 28,800 seconds); skipped arms are listed in the receipt. This does not preempt the first pass or guarantee completion under a hard platform deadline.
8. Writes `submission.csv`, `submission_model.csv`, `submission.receipt.json`, and per-checkpoint decoder audits.

To test a frozen blend with an existing baseline submission, supply both `--baseline-csv` and an explicit `--blend-weight`. Baseline and candidate must cover the exact live study set. A baseline built from empirical cohort ranks retains its original cohort dependence; V16 does not remove it by averaging.

Inference supports a changed output-column order by matching names, but refuses a changed class set. The model cannot learn an unseen finding at inference by adding an output column.

## 7. Kaggle notebook workflow

Use the notebooks included in [rsna-knee-v16.zip](C:/Projects/rsna-knee-abnormality/v16/rsna-knee-v16.zip), or regenerate them:

```powershell
python -m v16.build_notebooks
```

Each notebook embeds the exact runtime modules. In its settings cell, edit paths directly or provide:

| Setting | Meaning |
|---|---|
| `RSNA_DATA_ROOT` | Competition directory; unique automatic discovery is available |
| `V16_WORK` | Writable preparation/training directory |
| `V16_ENCODER_PATH` | HF directory or native DINOv2 Small .pth |
| `V16_DINOV2_REPO` | Local Meta code checkout, only for native format |
| `V16_REPORT_LABELS` | Full aligned report-label CSV |
| `V16_GROUPS_CSV` | Verified group CSV, if available |
| `V16_FOLD`, `V16_SEED`, `V16_ARM` | One training job's identity |
| `V16_CHECKPOINTS` | JSON array of attached trained V16 checkpoint paths |

Prepare → preserve the preparation artifacts → copy them to writable scratch for a training session → run fold jobs → merge/evaluate → attach selected checkpoints to the inference notebook. Only the inference notebook produces a submission. Training caches may exceed Kaggle disk limits at the default high-coverage recipe; use an adequate external training host or a deliberately smaller new training configuration.

No notebook has been pushed to your Kaggle account. No competition submission was made.

## Verification and limits

```powershell
python -m unittest v16.test_pipeline v16.test_encoders -v
```

Tests cover generated real DICOM files, geometric ordering, missing spacing, corrupt input handling, grouping, fold-local calibration leakage, preparation, training, exact completed-run resume, shard merging, missing-study inference, reordered output schemas/IDs, cache tampering, padding invariance, masked loss, local HF model loading/rebuilding, and CNN microbatch-one training.

HF/CNN loader tests use random model fixtures, not trained competition models. The original native Meta `.pth` path and compressed clinical transfer syntaxes require validation against your attached artifacts. Enhanced multiframe MRI is explicitly logged as unsupported by this adapter; the competition describes single-slice DICOM files.

**No 0.95–0.96 claim is made.** This version supplies the engineering and controlled experiments needed to measure whether the new branch contributes that improvement. Known limits include the small expert cohort, unverifiable patient linkage without supplied groups, missing-sequence information loss, potentially large training caches, unmeasured full-cohort GPU runtime, and domain shift.
