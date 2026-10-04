# RSNA Knee Abnormality Detection: pipeline, harnesses and measurements

Work for the Kaggle competition [`rsna-knee-abnormality-detection`](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
(macro-AUC over twelve knee-MRI findings, plus an efficiency track).


| Directory | What it is |
|---|---|
| `v16` | [Dynamic DICOM training/inference package](v16/README.md): variable-series DINOv2/CNN + BiGRU, grouped folds, fold-local report calibration, resumable training and offline notebooks; competition AUC unmeasured |
| `v5` … `v9`, `v9-flash`, `v10` | Successive submission notebooks (see below) |
| `v11` | [Runnable pretrained V11 candidate](v11/README.md), regression tests, and accuracy/efficiency audits; AUC not yet measured |
| `v12` | [Kaggle submission package](v12/README.md): self-contained notebook, input metadata, release ZIP and final-output checks |
| `v13` | [Accuracy candidate](v13/README.md): checkpoint-specific CoAtNet input corrections, unchanged weights/blend, paired scoring tool; AUC unmeasured |
| `research` | [Public-score/source audit](research/public_score_audit.md) and public notebook source downloader |
| `variant_native` | A/B: stage-1 native pool vs public frontier |
| `ckpt_probe` | **CPU-only** kernel that ranks every published checkpoint by reading `gold_auc` out of the `.pt` headers |
| `oof_harness` | CPU reconstruction and evaluation scripts at n=4,349 instead of the 58-study gate; data not redistributed |
| `timing` | Per-stage runtime calibration using training studies as a pseudo test set |
| `train_p0`, `train_p1` | Training feasibility measurement on Kaggle T4 |
| `probe_arch`, `recon` | Attempt to reconstruct an unpublished model arm from its checkpoint |

