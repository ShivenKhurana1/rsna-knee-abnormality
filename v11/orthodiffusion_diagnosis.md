# OrthoDiffusion: reopened reproduction investigation

Audit date: 2026-08-29. Scope: diagnose the existing pretrained pipeline; no new
specialist training, production inference changes, or submissions.

## Conclusion

The reported Effusion AUC of 0.8832 is still not reproduced against the 0.9578
receipt. This is an **open reproduction issue**, not a closed negative result.
The actual reconstruction notebook, its descriptors, and its 58-study predictions
are not present in this checkout. Consequently no specific line in that pipeline
has yet been established as the cause, and no corrected AUC has been measured.

We recovered official source, the exact public fitted head/receipt, and historical
author preprocessing code. Checkpoint plane swapping is no longer a productive
leading hypothesis for the audited public re-upload. Compare preprocessing and
the fitted feature/head interface next, before searching seeds or training models.

## 1. What the advertised number actually measures

The exact Effusion number, 0.9577639751552796, is in
`rsna_teacher_ridge_probe_v1/receipt.json` in the
[public two-teacher bundle](https://www.kaggle.com/datasets/prvsiyan/rsna-knee-two-teacher-bundle-v1).
It is a community ridge classifier on frozen OrthoDiffusion descriptors, not an
official twelve-target OrthoDiffusion benchmark or Kaggle leaderboard score.

The receipt reports:

| Quantity | Value |
|---|---:|
| Gold audit studies | 58 |
| Report-derived fitting rows | 4,346 |
| Firewalled rows | 61 |
| Ridge standalone macro AUC | 0.6980483 |
| Baseline macro AUC | 0.8419935 |
| Selected blend macro AUC | 0.8507967 |
| Selected blend delta | +0.0088033 |
| Promotion gate | false |
| Baseline Effusion AUC | 0.9068323 |
| Ridge Effusion AUC | 0.9577640 |
| Selected blend Effusion AUC | 0.9465839 |

The +0.0088 is an observed local receipt delta against that baseline, not an upper
bound or promised gain over the user's 0.935 leaderboard ensemble. The quoted
current-blend Effusion 0.907 also rounds to the receipt's baseline number; verify
its provenance against the deployed blend's actual predictions before comparing.

A larger pretrained encoder does not guarantee better downstream classification.
This package compresses each plane's 1,792 descriptors to 128 PCA coordinates and
uses a fitted linear head. Its task, labels, feature extraction, and validation
cohort differ from the [official paper](https://arxiv.org/abs/2602.20752).
That explains why advertising is not a performance guarantee; it does NOT by itself
explain failure to reproduce the same fitted head on the same 58 studies.

## 2. Verified checkpoint and schedule findings

Compared [public EMA re-upload](https://www.kaggle.com/datasets/cemdusenkalkan/orthodiffusion-ema-weights)
with the EMA state in the [author's original weights](https://huggingface.co/lanstat0123/orthodiffusion).

- All 257 tensor names, layouts, storage sizes and ZIP storage CRCs match for
  each correspondingly named plane.
- Six distributed weight tensors additionally match SHA-256 byte fingerprints
  per plane; none match either of the other two planes.
- The beta, square-root cumulative alpha, and square-root noise buffers match
  element by element between the original and the re-upload (maximum error zero).
- Original checkpoints report step 21,499 and include model, EMA, and optimizer
  state; the smaller re-upload retains EMA. Smaller file size is not evidence of
  a smaller neural network.
- The reconstructed official cosine formula, cast to float32, matches the beta
  buffer elementwise. Comparing only beta sums would not have proved this.
- At index 100: beta = 0.0005318991607055068; signal coefficient =
  0.985685408115387; noise coefficient = 0.16859513521194458.

This validates re-upload naming relative to the official naming, not the anatomical
provenance of the author's original training data. It also does not prove the user's
runtime loads the matching file or uses those buffers correctly.

Official [implementation source](https://github.com/lt-0123/OrthoDiffusion) is public.
The inspected re-upload's UNet and diffusion trainer have no code differences from
the checked official source after ignoring line-ending changes.

## 3. Recovered preprocessing: a concrete comparison target

The author's historical [notebook version 4](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes?scriptVersionId=341937872)
contains the complete `orthodiff_full_bridge_v4` uint8 cache construction. Its
contract matches the bundle's advertised input shape and plane order. A cache hash
or extraction receipt is still needed to prove that this exact cache generated the
released PCA/head, rather than another compatible cache revision.

The recovered construction is:

1. Plane order AXIAL, CORONAL, SAGITTAL.
2. Choose the lexicographically highest tuple `(4*fluid, 3*fatsat,
   2*(slice_count >= 16), min(slice_count,128))`; equal-score candidates retain
   their first metadata-row occurrence. This is not a summed score or UID tiebreak.
3. Sort slices by projection of ImagePositionPatient onto the cross-product of
   ImageOrientationPatient vectors, with InstanceNumber/filename fallbacks.
4. Select `files[D//2-8:D//2+8]`. For odd D, this starts one slice later than
   `(D-16)//2`. Short series use symmetric edge repetition.
5. Apply each slice's slope/intercept and handle MONOCHROME1.
6. Resize selected slices to 256x256 with OpenCV INTER_LINEAR.
7. Compute min/max over **only the selected, resized 16-slice volume**.
8. Round with `np.rint(255 * normalized)` and cast to uint8.
9. The descriptor bundle expects `uint8.astype(float) / 127.5 - 1`.

Normalizing the full original stack before center selection is not equivalent to
step 7. Likewise, extrema may change on resizing. This is a concrete potential
mismatch with the user's phrase "whole volume", not proof their code does it wrong.
Checking only fluid/fatsat flags and slice counts does not establish identical
SeriesInstanceUID selection, slice indices, or tensor values.

## 4. Fitted descriptor/head contract recovered from arrays

The v1 head has coefficients (12,387), target-specific means/scales (12,387), and
plane priors (12,3). Each PCA matrix is (128,1792), with a 1792-element mean.
Do not mix these with the separate routed-v3 head in the same archive.

Expected algebra inferred from the fitted arrays, for target t and plane p:

```text
z_p = (descriptor_p - pca_mean_p) @ pca_components_p.T
x_t = concat(prior[t,p] * z_p for p in [AXIAL,CORONAL,SAGITTAL], presence_flags)
score_t = sum(((x_t - scaler_mean[t]) / scaler_scale[t]) * coef[t]) + intercept[t]
```

Evidence: dividing each target's PCA-coordinate scaler means/scales by its plane
priors recovers the same base means/scales across all 12 targets. Maximum relative
scale discrepancy is 1.04e-7, and maximum mean discrepancy is 2.76e-9.
Thus the priors were applied before standardization, and their multiplicative
effect cancels there. They are not an independent post-scaling attention weight.
Omitting priors while keeping the target-specific scaler, multiplying after scaling,
or using one target's scaler for every target can alter rankings target-selectively.
The three appended presence coefficients are exactly zero in this v1 head.

A synthetic 20-row NumPy test comparing that head evaluation with its collapsed
affine equivalent agrees within 2.83e-8. This checks algebra only, not real AUC.

The descriptors concatenate 256 means, 256 maxima, 256 standard deviations, and
1024 channel-major 1x2x2 pooled values. The PCA means provide an internal check:
the first 256 means equal the average of each channel's four pooled quadrant means
within 5.43e-6. A spatial-major flattening instead gives errors up to 1.0671.

Historical [version 1 source](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes?scriptVersionId=341910200)
also uses population standard deviation (`unbiased=False`), channel-major pooling,
and averaging descriptors after forward/reverse extraction. It is not the exact
four-draw ridge extractor: its plane order and head differ. Do not transplant its
RNG behavior without checking the released bundle's actual extraction provenance.

## 5. Tests that can establish the cause without training

1. Obtain the notebook/script that produced 0.8832, with exact input dataset
   versions, UID-aligned predictions, and preferably saved per-plane descriptors.
2. Re-score its existing descriptors through the v1 algebra above. If AUC recovers,
   the mismatch is downstream of extraction; no GPU run is needed for that test.
3. On fixed studies, compare series UIDs, ordered slice indices, and uint8 tensor
   hashes against the historical cache convention. Check normalization timing,
   interpolation, rounding, and rescale parameters before rerunning all studies.
4. If inputs/head match, trace q_sample -> mid_2 -> descriptors -> PCA -> standardized
   features. Assert tensor layout, timestep index, eval mode, population std, and
   channel-major pooling. Pooling max/std before versus after averaging maps is
   not interchangeable.
5. Only then resolve four-draw RNG details: CPU/CUDA generator, seed scope per
   study/plane/batch, reverse-pass reuse, and batch-order invariance. If no reference
   extractor exists, characterize variance with a fixed preregistered seed panel;
   do not choose seeds to maximize the 58-study AUC.
6. Admit predictions only after UID-aligned reproduction and measured marginal
   ensemble benefit. Lower standalone AUC does not logically rule out blend value,
   but unreproduced predictions have not earned that admission.

Matching a rounded MCL AUC does not establish matching MCL predictions, nor rule
out a global feature permutation. Same AUC can arise from different rankings.

The browser-assisted historical-source check also found that the current version
47 uses different, partially fine-tuned plane models and a five-target head; its
input panel includes private datasets. That is not the frozen v1 ridge model and
is not yet a verified publicly downloadable replacement. No new model training
was performed or recommended as part of this reproduction investigation.
