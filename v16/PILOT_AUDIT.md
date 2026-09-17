# V16 first GPU pilot: audit and next steps

**2026-09-16.** Covers the first real end-to-end V16 training run and the
follow-up checks it prompted. Written from the actual Kaggle kernel outputs
(`receipt.json`, run log, `train.csv`/`train_series.csv`), not from the run's
own summary text.

## 1. What actually ran

Kernel `seanzhang2445/rsna-knee-v16-gpu-pilot` (script: `v16/build_kaggle_gpu.py`,
notebook: `v16/kaggle_gpu/train.ipynb`), status `COMPLETE`. Trained fold 0, seed
1400, `platt` arm (fold-local calibrated report auxiliary), 8 epochs, on a
Kaggle T4. Wall clock ~4h56m (~3.3h data preparation + ~1.6h training). Loss
0.1763 → 0.0792. 8 AMP overflow events logged; the per-study `overflow_retries`
counts in the receipt sum to 9, all recovered before the 12-retry cap — this
matches the earlier stability fix described in `v16/README.md` (FP32 sequence
head, bounded same-batch retry).

**Important: this pilot used a deliberately shrunk config, not the repo
default.** `build_kaggle_gpu.py` hardcodes `image_size=224, max_series=3,
max_centres=8, encoder_chunk=4` for a fast first stability check. The repo's
checked-in `v16/config.json` (and the code baked into the same pushed kernel
at `runtime/v16/config.json`) already carries the full recipe — `image_size=336,
max_series=8, max_centres=24, encoder_chunk=8` — confirmed byte-identical to
what's committed. No code fix was needed; the pilot just never used it.

## 2. Real per-target ledger (fold 0, 12 expert-labelled studies)

Pulled directly from `receipt.json['metrics']['targets']` — not the run's own
prose summary.

| Target | AUC | positive/negative (of 12) |
|---|---:|---|
| Effusion | 1.000 | 8/4 |
| Medial OA | 0.963 | 3/9 |
| Baker's | 0.950 | 2/10 |
| Lateral OA | 0.900 | 2/10 |
| ACL | 0.889 | 6/6 |
| Lateral Meniscus | 0.800 | 5/7 |
| Synovitis | 0.778 | 6/6 |
| Fracture | 0.778 | 3/9 |
| PF OA | 0.688 | 4/8 |
| Contusion | 0.656 | 4/8 |
| Medial Meniscus | 0.629 | 5/7 |
| MCL | 0.600 | 2/10 |

Macro AUC 0.8025 = plain unweighted mean of the 12 rows above (verified by
recomputation), each target counted equally regardless of support.

**Caveat before treating the bottom four as confirmed weak points:** Lateral
OA and Baker's also have only 2 positives and scored 0.90/0.95 — low support
alone doesn't explain MCL's 0.60. At n=12, one flipped ranking pair moves
MCL's AUC by 0.05, Medial Meniscus's by ~0.03, Contusion's by ~0.03. The
ordering among the bottom four is not statistically distinguishable at this
sample size. This is 1 of 5 folds × 1 of 2 seeds × 1 of 3 arms; no `none`/`raw`
arm has been run on these same studies for a paired comparison yet.

A join against `train_series.csv` confirmed all 12 of these studies actually
have 4–7 series available (full protocol coverage: axial/coronal/sagittal,
fluid-sensitive and fat-suppressed series present) — more than the pilot's
`max_series=3` cap used. Since that cap doesn't exist in the repo's default
config (§1), this is not a bug to fix, just a reason this pilot's specific
weak-target read may partly reflect the shrunk config rather than the
architecture or supervision policy.

## 3. Competition rules and model provenance (checked directly)

- **External data/tools**: allowed. Competition rules §2.6: *"You may use
  data other than the Competition Data... provided [it is] publicly available
  and equally accessible to... all Participants... at no cost, or satisfies
  the Reasonableness criteria."* §3.6.b additionally deems any Competition
  Code shared publicly on Kaggle "licensed under an Open Source
  Initiative-approved license" by virtue of being shared that way. Building
  on other competitors' public kernels/datasets is explicitly fine.
- **DINOv2 license has a real discrepancy, not yet resolved**: the Kaggle
  `metaresearch/dinov2` model-card badge says `Apache 2.0` (linking to Meta's
  GitHub repo); the same page's embedded Hugging-Face-sourced README says
  `license: cc-by-nc-4.0`. Doesn't block use either way — rules §2.5.a
  exempts "pretrained models with an incompatible license" from the winner
  open-source obligation — but don't assert one license as settled fact.
- **The reported 0.941 is a ~12-dataset, multi-user ensemble**, not a single
  trainable model: DINOv2 (small + base), a RadImageNet ResNet-50, and
  several CoAtNet ("Raptor") checkpoints from different Kaggle users, per
  `research/next_tier_095/source_audit/source_inventory.json`. It was never
  independently reproduced in this project
  (`research/next_tier_095/verification.json` lists it under `not_run`), and
  the source audit already flags shared ancestry across the four inspected
  public notebooks (§2 finding 1: "blending their final CSVs can mostly
  reweight the same underlying predictions"). V16 already uses the same
  DINOv2-small backbone family (`metaresearch/dinov2/pytorch/small`), so the
  ensemble's real remaining value-add is the RadImageNet/CoAtNet diversity —
  separate architecture-diversity work, not a shortcut to adopt wholesale.

## 4. Full-recipe calibration (in progress)

Rough math from the pilot's own cache-size formula (`n × max_series ×
max_centres × len(offsets) × image_size² × 2 bytes`) puts the full recipe
(§1) at roughly 18× the shrunk pilot's decoded-pixel volume — extrapolating
the pilot's ~3.3h prepare time, a full ~4,349-study pool prepare at full
recipe could plausibly take ~20-30h, likely exceeding one Kaggle session.
`build_kaggle_gpu.py` also stages its cache on `/tmp` and only copies to
persisted output after `prepare()` fully returns, so a mid-run timeout there
would lose all progress rather than resume.

Pushed `seanzhang2445/rsna-knee-v16-full-recipe-calibration`
(`v16/build_kaggle_gpu_calib.py` → `v16/kaggle_gpu_calib/`) instead of
guessing: same full recipe, but on a 208-study subset (150 random pool + all
58 expert studies, reusing this project's existing 150+58 dev-cohort
convention) so real per-study prepare/train cost at full resolution can be
measured cheaply first. No DICOM bytes are copied — `train_series/` is
symlinked whole and only the CSVs are truncated to the subset, since
`v16/common.py`'s `child()` path-escape guard rejects a symlinked *per-study*
subfolder (it resolves through to a target outside a non-symlinked root) but
accepts a symlinked *parent* directory (root and path both resolve through
the same link).

**Status as of this report: kernel `RUNNING`, not yet complete.** Fold 0 in
this subset still contains the same ~12 expert studies, so this run's own
per-target ledger will be a (heavily caveated — only 150 pool studies behind
the `platt` auxiliary policy here, vs. ~4,200 in a real run) directional check
of whether full series/resolution coverage changes the §2 weak-target
picture, plus the throughput numbers needed to size the real full-pool run.
This section should be updated once it finishes.

## 5. Open items

- [ ] Check `rsna-knee-v16-full-recipe-calibration` to completion; record its
      per-study prepare/epoch timing and extrapolated full-pool estimate.
- [ ] Decide full-run sharding strategy from that timing (one session vs.
      several, chained via persisted work-directory datasets).
- [ ] Run the `none` (expert-only) arm on the same fold 0 / seed 1400 cohort
      for a real paired supervision comparison — still not done at any config.
- [ ] Only after a merged, full-cohort OOF (all 5 folds) should any
      target-specific intervention for MCL/Medial Meniscus/Contusion be
      chosen — the current bottom-four ranking is not yet distinguishable
      from noise (§2).
