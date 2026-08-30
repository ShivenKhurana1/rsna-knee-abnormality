# OOF evaluation harness (CPU only)

The 58 gold studies cannot resolve differences smaller than ~0.05 (measured: that is
the sd across models differing only by random seed). This harness replaces that gate
with **out-of-fold predictions over all 4,407 studies**, which is the only way to
measure a blending change honestly without a GPU.

## What's here

| File | Contents |
|---|---|
| `F_dino.npy`  | (4407,12) OOF from surasan092's DINOv3-S focal folds |
| `F_rad11.npy` | (4407,12) OOF from the RadImageNet E11 heads |
| `F_rad13.npy` | (4407,12) OOF from the RadImageNet E13 heads |
| `gold_mask.npy` | (4407,) bool — the 58 expert-labelled studies |
| `oof_rad_e1*.csv` | the same RadImageNet OOF, keyed by StudyInstanceUID |
| `v52_oof.csv` | (4407,12) OOF from prvsiyan's `read-the-report-then-the-knee` v52 |

Row order for the `.npy` files is **train.csv row order**. That mapping was verified,
not assumed: applying it reproduces a gold macro-AUC of 0.8200 for the DINOv3 family,
consistent with the 0.796-0.852 per-fold figures its author published.

`surasan092`'s indices come from `focal_fold{k}_val_indices.npy`; the RadImageNet OOF
comes from `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` under
`kernel-sources/rsna-knee-e{11,13}-train/rsna_rad_e11/v52_e11_oof.csv`. Note both e11
and e13 ship a file with the *same* name — download them to separate paths or one
silently overwrites the other.

## What it established (2026-08-25)

- Family gold-gate AUCs: dino 0.8200, e11 0.8375, e13 0.8576
- Pairwise rank corr 0.64-0.80 — genuinely diverse, unlike the label tables (0.86-0.995)
- **Equal-weight blending beats the best single family by +0.005**
- **Optimising the weights adds nothing**: per-target weights fitted by 5-fold CV on
  4,349 weak-labelled studies scored -0.0008 vs equal weights on the held-out gold
- The deployed E13/E11 ratio (~0.77/0.23) is not beaten by equal weighting
  (-0.0025, 95% CI [-0.0134, +0.0083])

Conclusion: blend *more diverse families*, do not tune the weights between them.

## Adding a family

Any team publishing OOF predictions over the training set can be dropped in. Score it,
check its rank correlation against the existing three, and only bother if the
correlation is low — that is where the +0.005 came from.

## Update 2026-08-25: a fourth family, and the first resolved gain

Added `prvsiyan/rsna-knee-v52-oof-receipt-20260810` as a fourth family. Family gold AUCs are
now dino 0.8200 / e11 0.8375 / e13 0.8576 / v52 0.8420, with v52 correlating 0.655-0.797
against the others.

Evaluated at **n=4,349** against the best weak-label table (these are OOF, so every study was
held out by its own model):

| blend | weak n=4349 | gold n=58 |
|---|---|---|
| dino+e11+e13+v52 | **0.8624** | 0.8651 |
| dino+e11+e13 | 0.8614 | 0.8630 |
| dino+e13+v52 | 0.8607 | 0.8608 |
| e13 alone | 0.8389 | 0.8576 |

Adding v52 to the three-family blend: **+0.0010, 95% CI [+0.0001, +0.0019], P(better) 98%.**
On the 58-study gate the same comparison was +0.0021 with CI [-0.0080, +0.0128] — unresolvable.
That contrast is the whole argument for this harness: the large set resolves what gold cannot.

### The concrete next addition

`surasan092`'s DINOv3-S full-FT model (`rsna-knee-dinov3-s-v13-1-gpu-util-full-ft-model`) is
the **least correlated** family measured (0.644-0.655 against everything else) and is **not in
the v7 pipeline** — v7's stage 2 uses `mattiaangeli/knee-mri-fold-weights`, a different DINOv3.
It is the weakest family alone (0.8200) but diversity is what paid here, not individual strength.

Wiring it in needs inference code for its architecture (DINOv3-S/16+, slot attention, d_model
256, focal head; see its `config.json`) and a GPU run to validate. The CPU evidence says it is
the right thing to add; it cannot say the wiring is correct.

## Reconstructing the surasan092 arm — progress (2026-08-25, all CPU)

No inference code was published for `surasan092/rsna-knee-dinov3-s-v13-1-gpu-util-full-ft-model`,
so the arm must be rebuilt from `focal_fold*_final.pt`. Two of three unknowns are now solved
*exactly*, with no GPU and (for the second) no images at all.

### SOLVED — backbone, exactly
`model_state` holds a HuggingFace DINOv3 ViT-S/16+ under `backbone.*`. The HF repos are gated,
so build the config directly:

    DINOv3ViTConfig(hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
                    patch_size=16, num_register_tokens=4, image_size=336,
                    intermediate_size=1536, use_gated_mlp=True)

`use_gated_mlp=True` is essential and is the "+" in ViT-S/16+. Without it you get
`missing 0, unexpected 24` (`mlp.gate_proj.*`) and `strict=False` **silently drops those
weights**, producing plausible-but-wrong predictions. With it: `missing 0, unexpected 0`.

At 336px the backbone emits 5 prefix tokens (CLS + 4 registers) and 441 patch tokens.
CPU cost 301 ms/image -> ~21 min per fold to score the 58 gold studies.

### SOLVED — window pooling, exactly
`focal_fold0_val_window_predictions.npy` is `(10, 882, 12)`; `val_predictions.npy` is
`(882, 12)`. Testing poolings against each other gives **plain mean over the 10 windows,
max|err| = 0.0** (top-10 mean is the same thing). It is NOT top-k: `focal_top_fraction: 0.125`
refers to patch-token pooling inside the model, not to windows. Solved without touching an image.

### FAILED — the head forward (attempted 2026-08-25, r=0.72)
12 tensors: `feature_proj` (1152->256), `slot_embed` (6,256), `slot_queries` (12,256),
`attn_gate` (12,), `slot_prior` (12,6), `classifier_w/b`, `slot_norm`, `study_norm`.
1152 = 3 x 384 and the shapes are consistent with `concat(CLS, mean-patch, mean of top-12.5%
patches)`, but how `attn_gate` and `slot_prior` modulate the 12-query-over-6-slot attention is
not determined. Verifying needs real images: build the 6-slot/336px/12-slice cache (v7 stage 1
already produces exactly this format — same slot names, crop_mm 130, window 3) for a handful of
fold-0 validation studies, then grid the remaining choices until predictions reproduce
`val_predictions.npy`. That is a bounded search with an exact target.

### Outcome of the head reconstruction: NOT REPRODUCED

Built the full path on CPU (`blackpearls1/rsna-knee-recon`): preprocessing from the config,
backbone features cached once per (study, slot, window), then 18 head variants graded against
`val_predictions.npy` on 12 fold-0 studies. Best variant `none/add/scale=sqrt` reached
**mean|err| 0.100, r = 0.7192** — correlated but not the model. All 18 variants sat at r 0.64-0.72.

**Do not blend an r=0.72 reconstruction.** It is a different, worse model using their weights.

The result is confounded: one signal, two unknowns (head wiring AND preprocessing). Preprocessing
is the likelier fault — `intensity_stat_workers` in the config implies **per-series** intensity
statistics, whereas the attempt normalised **per slice** (percentile 1/99 on each cropped image),
a systematic difference on every input. `slot_min_coverage` was also ignored during series
selection. Fix those before touching the head search again, and find a way to decouple the two
unknowns (e.g. verify preprocessing alone against something with a known output) rather than
grading them jointly.

Effort/payoff note: even a perfect reconstruction was worth about **+0.001** by the measured
four-family analog. It was never a route to prize-level score.

## Retry (2026-08-29): preprocessing fixes ruled out, not confirmed

Three fixes suggested by an independently-published fork of the same lineage
(`anhadmahajan06`'s notebook) were applied to `recon/rsna-knee-recon.ipynb` and re-run on Kaggle
(`seanzhang2445/rsna-knee-recon`, kernel version 1): per-slot (not per-slice) intensity
normalisation, `slot_min_coverage`-gated series selection, and geometric laterality
canonicalisation (median DICOM-geometry centroid, +/-20mm threshold, mirror Right to the Left
canonical layout).

All three fixes were confirmed to actually engage (log: 5/6 slots filled on average, laterality
resolved on all 12 studies -- 5 Right, 7 Left, 0 unresolved, each correctly mirrored). Result:
**r=0.7258, mean|err|=0.0990** (best variant `none/add/scale=sqrt`) -- statistically
indistinguishable from the original attempt's r=0.7192 given n=12 studies.

**Conclusion: per-slice intensity normalisation was NOT the dominant cause of the r=0.72
ceiling**, contrary to this document's earlier hypothesis. The remaining gap is more likely in
the head-forward wiring itself (how `attn_gate` and `slot_prior` actually modulate the
12-query-over-6-slot attention) rather than in preprocessing -- the original framing of this as
"confounded: one signal, two unknowns" still holds, but this result shifts weight toward the
wiring unknown. Given the already-established low payoff ceiling (~+0.001 even if perfectly
reconstructed), further reconstruction attempts are not recommended as a priority; the
`OrthoDiffusion` lead documented in `v12/PLAN.md` has a larger plausible payoff for the same or
less effort.
