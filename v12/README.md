# V12 Kaggle submission package

Import **rsna-knee-ensemble-v12.ipynb** into Kaggle. It is self-contained; no helper
Python files need to be attached. The ZIP bundles the notebook, Kaggle input
configuration, build manifest and this guide. It does not contain model weights
or competition data.

## Run and submit

1. Import the `.ipynb` (not the ZIP).
2. Attach **RSNA Knee Abnormality Detection** and the inputs below. If using the
   Kaggle CLI, `kernel-metadata.json` supplies these attachments. Its owner is
   inherited as `seanzhang2445`; change the owner if uploading under another account.
3. Select **GPU T4 x2**; keep internet disabled for the scored run.
4. **Save Version → Save & Run All**. Wait for successful completion.
5. Check `v12_submission_ready.json` for `READY_FOR_KAGGLE_SUBMISSION` and check
   that `submission.csv` is present. Submit the completed notebook version through
   the competition interface, selecting `submission.csv` if prompted.

The ready marker checks schema, study IDs/order, finite scores in [0,1], nonconstant
predictions on multi-study inputs, a completed current-run member, and the final
prediction hash against this run's receipt. A failed preflight resets the marker
to `NOT_READY`. Model loaders still check weights and actual GPU execution.

V12 preserves V11's pretrained DINOv2, DINOv3, RadImageNet and CoAtNet recipe. No
training, new weights, or OrthoDiffusion predictions have been added. It adds
submission preflight/final gates, valid notebook cell IDs, and V12-named receipts.
The bounded temporary Stage-4 cache is cleaned after use; prediction snapshots
remain under `v12_diagnostics/<run_id>/`.

**Not yet measured:** Kaggle execution, hidden-test runtime, AUC, or leaderboard
improvement. The reported 0.935 is a user-provided baseline, not a V12 result.
Local packaging/regression tests cannot establish leaderboard performance.

## Required attachments

Datasets:

- `antoinegg1/rsna-knee-e11-diverse-heads-v20`
- `antoinegg1/rsna-knee-e9-radimagenet-heads-v15`
- `dreaddevelopment/raptor-knee-maxspan`
- `dreaddevelopment/raptor-knee-native384dense`
- `dreaddevelopment/raptor-knee-widedense`
- `marwanmath/resnet-50-radimagenet-marwan`
- `mattiaangeli/knee-mri-fold-weights`
- `mattiaangeli/rsna-knee-rad-calibration`
- `mattiaangeli/rsna-knee-radimagenet-foldsv1-heads`
- `pilkwang/rsna-knee-llm-labels`
- `pilkwang/rsna-knee-weights`
- `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812`
- `stevenleehans/rsna-knee-llm-report-labels`
- `tonylica/rsna2026-models`

Notebook outputs (already-fitted weights; these are not training runs in V12):

- `sofiaanjenje/rsna-knee-e11-train`
- `sofiaanjenje/rsna-knee-e13-train`

Model:

- `metaresearch/dinov2/PyTorch/small/1`

These are inherited input references, not freshly downloaded or access-verified
in this release. Attach the corresponding resources in your Kaggle account and
retain their original attributions/licenses.

## Rebuild/test locally

```powershell
python v12/build_v12.py
python v12/test_v12.py
```

The builder verifies the V11 source against its manifest before packaging.
Regenerate and test V11 first if you intentionally change its recipe. V12's ZIP
and notebook are deterministic outputs. The CPU tests need NumPy and pandas;
no GPU, competition data, or training is used locally.
