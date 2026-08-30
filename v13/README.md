# V13: pretrained input-contract accuracy candidate

V13 makes actual prediction changes relative to V12. It does **not** have a
measured AUC or a guaranteed improvement. No new models are trained, no arbitrary
blend weights are introduced, and OrthoDiffusion remains excluded.

## Confirmed input mismatches corrected

| CoAtNet checkpoint | V12 input | V13 input, following its author |
|---|---|---|
| MaxSpan v5 SWA | 336 cache; 2–98% span; 62 windows | Unchanged |
| Native384 v10 | 336 cache, then 384 interpolation | Direct 384 cache; same 2–98% span and 62 windows |
| WideDense v4 | 2–98% span; 62 windows | 6–94% span; 42 windows; same 336 cache |

All use the same 140 mm fixed-center crop and 384-pixel network. The distinction
between cache and network resolution matters: loss of detail during the 336-pixel
resize/uint8 quantization cannot be undone by upsampling.

Sources checked on 2026-08-29:

- [MaxSpan model card](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-maxspan).
- [Native384Dense model card](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-native384dense).
- [WideDense model card](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense).
- [Author's WideDense inference, version 6](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model): `IMG=336`, `K_EVAL=42`, `int(n*0.06)` through `int(n*0.94)-1`. Downloaded source SHA-256: `e1f3aa1fe0f76c01d80fc0aa0718bd7ac5573dbcaca465943db0e0666fb6f2f4`.
- [DINOsaur V4](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4) also explicitly uses a legacy-arm 6–94%/42-window contract. This corroborates the input fix; its hand-tuned target blend is not copied.

The original model cards' standalone/gold scores are not V13 ensemble scores.
Correcting a mismatch is plausible improvement, not evidence that every target
or the final ensemble will improve. The 42-window policy also reduces v4 inference
work versus V12, but real GPU runtime has not been measured.

## Run

Import `rsna-knee-ensemble-v13.ipynb`, attach the **same inputs as V12** (enumerated
in `kernel-metadata.json`), select GPU T4 x2, keep internet off, and Save & Run All.
The notebook is self-contained. The ZIP includes notebook, metadata, manifest,
and this guide; import the notebook inside it, not the ZIP itself. Owner remains
`seanzhang2445`; change that metadata if using a different Kaggle account.

Keep `submission.csv`, `v13_submission_ready.json`, `v13_run_receipt.json`, and
`v13_diagnostics/<run_id>/`. The ready marker checks output integrity only.

Completed CoAtNet arms now save keyed raw predictions and their hashes before
fusion. Their per-arm input contracts are logged. Cache keys include the complete
contract so incompatible volumes cannot be reused. Disk caching defaults to zero
because each of these distinct contracts is consumed only once per arm; streaming
does not change the model input. Stages 1–3, ensemble coefficients, and inherited
failure recovery are unchanged. Checkpoint target order/resolution is checked.

V12 is preserved as the control. No notebook has been uploaded, run on Kaggle, or
submitted from this workspace. Local tests do not exercise torch, real DICOM
decoding, OpenCV interpolation, hidden-test runtime, or AUC.

## Verification

```powershell
python v13/build_v13.py
python v13/test_v13.py
python v13/test_comparison.py
```

To measure paired held-out predictions once available:

```powershell
python v13/compare_predictions.py --labels gold_labels.csv --baseline v12_gold.csv --candidate v13_gold.csv --output paired_v13_report.json
```

All three CSVs must cover exactly the same studies and contain the twelve target
columns plus `StudyInstanceUID`; labels must be binary or missing. The tool
reports per-target AUC, macro delta, paired 95% bootstrap intervals and whether
the +0.01/+0.02 thresholds are met on that cohort. It does not train or tune
anything and cannot verify that predictions were held out during model training.
Do not score full-trained CoAtNets on their own 4,349 training studies and call
that OOF. The actual gold/prediction files are not present in this repository.

Public score/source audit: `../research/public_score_audit.md`. Original model
and community notebook attributions are retained in the notebook and root README.
