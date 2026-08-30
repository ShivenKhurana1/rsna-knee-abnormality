# V12 plan — actually moving the score

## 0. Why this document exists

`v11/PLAN.md` was, by its own design, a safety pass: every item that could plausibly
change the score was tagged **DEBATE — needs paired OOF** and explicitly *not* shipped
(`v11/PLAN.md` §3, §10). v9-flash, v10, and v11 held the public LB flat at 0.935 as a
direct consequence of that discipline, not as an unexplained regression — see the
analysis in this session's earlier turn. v12 currently ships zero changes over v11
(its own changelog cell says so).

This document's job is different: **pick the highest-expected-value items off that
backlog and actually spend what's needed (CPU time now, Kaggle GPU-hours later) to
clear their OOF gate**, so v12 can ship a change that has evidence behind it, not just
a "looks more correct" argument. The discipline from `v11/PLAN.md` §7 still applies in
full — nothing here ships without a paired bootstrap CI on the n=4,349 harness. This
document only reprioritizes *which* gated items to actually resolve first, and how.

---

## 1. What's actually testable right now, and what isn't

Checked before writing anything else below, because it changes what's actionable this
week versus what needs a new Kaggle job:

- `oof_harness/` holds exactly four **study-level** (4407, 12) prediction arrays —
  `dino` (surasan092 DINOv3-S), `e11`, `e13` (RadImageNet), `v52` (prvsiyan) — plus
  `gold_mask`. That's enough to test blend-level questions **among those four
  families**, and nothing else.
- **Checked, not assumed:** `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812` is
  already present in `dataset_sources` as far back as `v8/kernel-metadata.json`, and
  Stage 3's `_rad_load_public_heads` loads a file named exactly
  `v52_radimagenet_heads.pt`. So the oof_harness "fourth family" result documents a
  choice **already shipped** in the live pipeline (part of the "v15+E13 fused family"
  the README table describes) — it is not an unshipped lever sitting on the table.
  Don't re-propose adding v52; it's already in.
- The oof_harness arrays **cannot** test: Stage 1's internal
  `LEGACY_MEMBER_WEIGHT_BY_TARGET` (needs per-member OOF — Stage 1 has ~20 DINOv2
  members, not 4 families), `PUBLIC_FRONTIER_TARGET_POOL` TTA pooling (needs
  window-level, pre-aggregation predictions — both constants confirmed live in Stage
  1's cell defining `TTA_TARGET_POOL`/`LEGACY_FOLD_SOFTPOOL_*`), or anything
  preprocessing-level (crop rule, intensity rule, series selection, slice ordering,
  laterality) — those are upstream of any cached prediction and require re-running
  actual DICOM decode + frozen-checkpoint inference, which needs Kaggle GPU + the
  competition images this repo does not hold.
- Stage-3's adaptive alpha (`_rad_adaptive_alpha`, §3.8) **is** testable today: it's a
  pure function of the four families' predictions, exactly what's already sitting in
  `oof_harness/*.npy`.

This sets the three tiers below.

---

## 2. Tier 1 — ship this round, zero new GPU cost

### 2.1 Stage-3 adaptive alpha vs. fixed alternatives (resolves v11/PLAN.md §3.8)
No new data needed. Using `F_dino.npy`, `F_rad11.npy`, `F_rad13.npy`, `v52_oof.csv`,
`gold_mask.npy` as-is, compute in this order and report macro-AUC + paired bootstrap
95% CI + P(better) for each vs. the current adaptive alpha, on n=4,349:
1. Fixed published alpha (whatever `_rad_adaptive_alpha`'s hand-set prior currently
   defaults to),
2. Fixed OOF-fitted global alpha (one scalar, fit on the full 4,349),
3. Fixed OOF-fitted per-target alpha with strong shrinkage toward the global value,
4. Current test-adaptive alpha (the transductive, prediction-correlation-driven one).

Per `v11/PLAN.md` §3.8: keep the current adaptive alpha only if it beats every fixed
alternative *repeatably* — don't accept a one-shot win. This is a same-day, CPU-only
script; there's no reason it's still open.

### 2.2 Re-check the deployed blend ratio with all four families named explicitly
The existing "-0.0025, CI [-0.0134, +0.0083]" result (equal weighting doesn't beat the
deployed ratio) was run as a 3-family comparison per `v11/PLAN.md` §3.6's own
callout. Since v52 is confirmed already deployed, redo that specific check across all
four families as actually blended, to close the gap between what was measured and
what's shipped. Expect no change (this reprioritization doesn't claim otherwise) —
this is closing a loose end, not a new hypothesis.

---

## 3. Tier 2 — one Kaggle job, no preprocessing change, moderate cost

### 3.1 Stage-1 legacy per-target weights (resolves v11/PLAN.md §3.6)
Blocked only on missing data: no member-level OOF for Stage 1's DINOv2 members exists
anywhere in this repo. Action: run Stage 1's members (frozen weights, code unchanged)
over the 4,349 weak-labelled studies on Kaggle, save **per-member per-target**
predictions before they're collapsed into the weighted rank mean. Then ablate
`LEGACY_MEMBER_WEIGHT_BY_TARGET`: current (15x-capped) vs. all-1.0 vs.
exclude-legacy-family vs. a single small global family weight, exactly as prescribed.
Prior from the Stage 2/3 experiment says this is likely overfit — confirm on its own
ablation, don't delete on the prior alone (`v11/PLAN.md` §3.6 is explicit about this).

### 3.2 TTA pooling policy (resolves v11/PLAN.md §3.7)
Same job as 3.1 can produce this too: also dump **per-window**, pre-pooling
predictions for the targets in `PUBLIC_FRONTIER_TARGET_POOL`
(Fracture/Contusion/Medial Meniscus/Lateral Meniscus/ACL/MCL/Baker's) and
`TTA_TARGET_POOL`'s `Synovitis` entry. Compare mean-probability / mean-logit /
smooth-log-sum-exp / fixed-fraction top-k / learned target-query attention against
the current hard-coded per-target max/top2/original_mean dictionary, nested within
folds.

**Batch 3.1 and 3.2 into one Stage-1-only Kaggle run** — they need overlapping
intermediate outputs (member- and window-level predictions before final pooling), so
running Stage 1 twice for these would waste GPU-hours for no reason.

---

## 4. Tier 3 — real preprocessing re-inference, expensive, touches frozen-checkpoint inputs

Each item below needs actual DICOM re-decode + a forward pass through the relevant
frozen checkpoint(s) under each variant, over some or all of the 4,349 weak-labelled
studies. This is real, multi-hour-per-variant Kaggle GPU cost — sequence them by
expected payoff, not by document order:

1. **Crop rule (§3.1)** — V9 fixed-center vs. V10 per-slice centroid vs. series-level
   centroid vs. gated series-level centroid. Do this first: it changes every pixel a
   frozen checkpoint sees on every study, not just a rare fallback path, so it has the
   largest plausible effect size of anything in this tier.
2. **Intensity rule (§3.2)** — the 2x2 grid (crop x intensity) specified in
   `v11/PLAN.md` §3.2. Run after 1, since its result narrows which crop setting is
   worth pairing with `INTENSITY='series'`.
3. **Series-level vs. per-slice crop consistency (§3.3)** — whether adopting Stage
   1/3's already-existing series-level crop design in Stage 2/4 changes anything,
   independent of which crop rule wins in item 1.
4. **Slice-ordering fallback parity (§3.5)** — lowest priority in this tier because
   `v11/PLAN.md` §3.5 itself scopes it to a **targeted subset** (only studies where the
   IOP/IPP fallback actually triggers), not a blanket re-score — cheapest of the four,
   but also the smallest plausible effect since it only touches the fallback-rate
   subset.

---

## 5. Blocked — needs information not present in this repo

### 5.1 Series selection selector (§3.4) and laterality convention (§3.10)
Both explicitly require reconstructing the **training-time** convention before it's
safe to change frozen-checkpoint inference — a "more correct" selector or laterality
rule is exactly the same "looks better, shifts the input distribution" risk flagged
throughout `v11/PLAN.md`. Action before any experiment here: find and read the
original training scripts (not published in this repo — check with `pilkwang`,
`mattiaangeli`, `dreaddevelopment`'s published training kernels first). Don't spend
Kaggle GPU-hours on this until that convention is confirmed; an ablation without it
can't distinguish "more correct" from "off-distribution."

### 5.2 Protocol-metadata calibration (§3.9)
Needs a grouped-CV split by protocol/site/scanner cluster, which needs per-study
protocol metadata not currently in `oof_harness/`. Lower priority than §5.1 since it's
not blocked on external information, just on pulling one more column of DICOM header
data into the harness — worth doing opportunistically alongside the Tier 3 Kaggle
jobs (they already re-read headers) rather than as its own trip.

---

## 6. Retrain-only backlog (V11-B, unchanged from v11/PLAN.md §4, §9)

Not on the critical path for v12, but the largest single-item upside in the whole
audit remains here, not in any frozen-checkpoint patch:

- **Slot-boundary-aware Stage-4 windows** (§4.1) — the plan's own words: "one of the
  higher-value retraining experiments in this whole plan," since it fixes an actual
  violation of the 2.5D locality assumption rather than a cosmetic issue.
- Series-consistent, translation-augmented cropping trained in from scratch (§4.2).
- One validated, geometry-based laterality convention trained against (§4.3).
- A deterministic, quality-ranked series selector, used identically at train and test
  time (§3.4/V11-B).
- Confidence-masked weak-label supervision (explicit-positive / explicit-negative /
  uncertain / absent), given the confirmed disagreement between report-derived weak
  labels and the 58 expert image labels.
- `surasan092` full-FT reconstruction retry, only after fixing per-series (not
  per-slice) intensity statistics and respecting `slot_min_coverage` during series
  selection (`oof_harness/README.md`'s reconstruction section) — flagged low priority
  again here: even a perfect reconstruction was only ever worth ~+0.001 by the
  measured four-family analog, and the current best attempt (r=0.72) is explicitly
  not blendable.

---

## 7. Recommended order for v12

1. **Now, no GPU:** §2.1 (adaptive alpha) and §2.2 (4-family ratio re-check). Both are
   scripts against data already in this repo.
2. **One combined Kaggle job:** §3.1 + §3.2 (Stage 1 legacy weights + TTA pooling),
   sharing one inference run over the 4,349 studies.
3. **Kaggle, in this order:** §4 item 1 (crop rule) → item 2 (intensity) → item 3
   (crop consistency) → item 4 (ordering parity). Stop and ship after any item that
   clears its OOF gate rather than batching all four before shipping anything.
4. **Do not start** §5.1 (series selection, laterality) until the training-time
   convention is confirmed from an external source. Opportunistically fold §5.2
   (protocol metadata) into whichever Tier 3 job already re-reads DICOM headers.
5. Log the retrain-only backlog (§6) as the next planning horizon once the
   frozen-checkpoint backlog above is exhausted — it's where the largest remaining
   upside actually lives.

Every item above reports, per `v11/PLAN.md` §7: macro AUC old/new, paired bootstrap
95% CI, P(better), all 12 per-target deltas, and prediction rank correlation vs.
current behavior, on the n=4,349 harness. The 58-gold gate stays a sanity check only.
Nothing ships on a public-LB nudge, a visual example, or the gold gate alone — same
rule as v11, now actually being used to *ship* something instead of only to defer.
