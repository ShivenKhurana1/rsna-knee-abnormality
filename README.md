# RSNA Knee Abnormality Detection — pipeline, harnesses and measurements

Work for the Kaggle competition [`rsna-knee-abnormality-detection`](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
(macro-AUC over twelve knee-MRI findings, plus an efficiency track).

Public LB progression: **0.910 → 0.915 → 0.930 → 0.935** (rank ~206 / 2406, top 8.6%).

## What is actually here

This repo holds **notebooks, harnesses and measurements** — no competition data and no
third-party weights. Everything mounts its inputs from Kaggle at run time; see each
`kernel-metadata.json` for the exact dataset list.

| Directory | What it is |
|---|---|
| `v5` … `v9`, `v9-flash`, `v10` | Successive submission notebooks (see below) |
| `variant_native` | A/B: stage-1 native pool vs public frontier |
| `ckpt_probe` | **CPU-only** kernel that ranks every published checkpoint by reading `gold_auc` out of the `.pt` headers |
| `oof_harness` | Out-of-fold evaluation at n=4,349 instead of the 58-study gate (README only; data not redistributed) |
| `timing` | Per-stage runtime calibration using training studies as a pseudo test set |
| `train_p0`, `train_p1` | Training feasibility measurement on Kaggle T4 |
| `probe_arch`, `recon` | Attempt to reconstruct an unpublished model arm from its checkpoint |

## Findings worth reusing

Most of the value here is negative results that cost real GPU-hours to establish.

**Kaggle's P100 is unusable with the current torch build.** It fails with
`cudaErrorNoKernelImageForDevice` on a plain tensor copy — not just on bf16 math. Since
`torch.cuda.is_available()` returns `True` for it, code must *probe* with a real op before
trusting a device. Set `"machine_shape": "NvidiaTeslaT4"` in `kernel-metadata.json`; the
server silently accepts invalid values and falls back to P100.

**On T4, use fp16 — never bf16.** Measured at 384px/bs4: bf16 796.6 ms/step, fp16 254.4 ms/step,
fp32 607.8 ms/step. sm_75 has no native bf16, so bf16 is 3.1× slower than fp16 *and* slower than
fp32.

**Training on Kaggle is not viable.** 5.41 s/study = **6.54 h per epoch**, ~35× the published
"16 epochs in ~3 h on one 4090" reference. Only one epoch fits a 12 h session. Analysis of the
gap: ~0.8 s/study is compute, so the rest is reading 5 MB/study out of a memmapped 22 GB corpus
on network storage (~1 MB/s). Local disk would be the fix.

**Blend-weight tuning does nothing; adding a diverse family does.** Measured on out-of-fold
predictions at n=4,349 (the 58-study gate has an sd of ~0.05 across models differing only by
seed, so it cannot resolve these effects): per-target weights fitted by 5-fold CV scored
**−0.0008** vs equal weights; label-table consensus **−0.0054**; the deployed E13/E11 ratio was
not beaten by equal weighting. Adding a fourth, genuinely decorrelated family was the only thing
that paid: **+0.0010, 95% CI [+0.0001, +0.0019]**.

**Rank published checkpoints before doing anything else.** `ckpt_probe` reads `gold_auc` straight
out of `.pt` headers — no inference, no GPU. It found the deployed pipeline running a stale arm
(0.9054) when a better one (0.9214) was already published, which was the single largest gain of
the project and took five CPU minutes.

**A notebook cell carrying `from __future__ import annotations` cannot be indented** under an
`if`/`try`, and `ast.parse` does **not** catch it — only `compile()` does. Validate notebooks
with `compile()`.

## Provenance and credit

The submission notebooks are forks of community work and are **not original**. The lineage is
`pilkwang/rsna-knee-baseline-v1` → `prvsiyan` (Apache 2.0) → `mattiaangeli/bend-the-knee-to-dinov3-ensembled`,
with a CoAtNet blend from `dreaddevelopment`. Weights, label tables and RadImageNet heads are the
work of `pilkwang`, `mattiaangeli`, `marwanmath`, `antoinegg1`, `prvsiyan`, `sofiaanjenje`,
`tonylica`, `stevenleehans`, `cf696666`, `romantamrazov`, `surasan092` and `dreaddevelopment`.

Original contributions in this repo are the harnesses and the measurements above: the
checkpoint probe, the OOF evaluation method, the timing calibration, the training-feasibility
measurement, and robustness fixes (GPU probe gating, NaN-safe blending, per-stage wall-clock
cutoffs) applied on top of the inherited pipeline.

Upstream's own warning is worth repeating: the base notebook states it is *"likely overfit to the
public leaderboard"* after a fork-and-republish race chasing 0.001–0.003 movements.
