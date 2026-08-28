# V11 plan — RSNA knee abnormality detection

## Provenance

This consolidates a multi-round audit: two independent audits (model-quality and
efficiency) of `v10/rsna-knee-ensemble-v10.ipynb`, each separately verified
line-by-line against the actual v9/v10 code and against `oof_harness/README.md`
and the root `README.md`, followed by three rounds of adversarial debate between
the audit author and the verifiers. All of that process content (the original
`improvementsv11.md`/`efficencyv11.md` audits, the two verification passes, and
the debate rounds) is superseded by this document and has been removed from
`v11/` to save space. Everything load-bearing from that process — every
confirmed bug, every piece of code evidence, every OOF number, and every
remaining disagreement — is captured below. Nothing in this plan is asserted
without a code location or a measured number behind it.

---

## 1. How to use this document

Every action item below is tagged with a status:

- **GO** — verified safe, ship it now, no OOF gate required (it either doesn't
  change predictions, or it only fixes something already broken).
- **GO (investigate)** — confirmed to be true of the code, but the right next
  step is diagnosis, not a change.
- **DEBATE — needs paired OOF** — plausible and probably correct, but changes
  what a frozen checkpoint sees or how it's blended, so it must clear the
  large-OOF harness (n=4,349, see §7) before it can replace current behavior.
  Do not ship on the strength of "it looks more correct."
- **Retrain-only** — the current behavior is baked into what a frozen checkpoint
  learned. Do not patch frozen inference. Only valid as a training-time change
  for a new checkpoint.
- **Benchmark-only** — an efficiency question with no accuracy risk by itself,
  but no confirmed win either; needs a timing/memory measurement, not a review.
- **Open — one measurement needed** — a live disagreement in this process that
  text-based argument can't resolve; the exact measurement that would settle it
  is specified.

Two branches are assumed throughout:

- **V11-A** — the branch that ships. Frozen checkpoints, V9-era preprocessing,
  only fixes that are GO. This is what should be scored next.
- **V11-B** — a retraining branch for the structural fixes that can only be
  validated by training a new checkpoint. Not on the critical path for the next
  submission.

---

## 2. Tier 0 — unconditional GO (ship in V11-A immediately)

### 2.1 Preserve the last valid Stage-1 submission instead of overwriting with all-0.5 — **GO, highest priority of everything in this document**

**The bug, exactly:** Stage 1's `bank()` writes a valid, incrementally-improving
`submission.csv` after every successfully-scored ensemble member, inside a
`try/except` that only drops the one failing member
(`worker()` → `_run_member()` wrapped locally). But each pixel-*group*'s setup
code — `adopt_config_globals(cfg)` (raises `WeightsError` on a members/slots
mismatch), `build_cache(...)`, `_combine(...)` — runs once per group, **outside**
any per-member protection, in between groups. If any of that group-setup code
raises on group 2+ (after group 1 has already banked real predictions), the
exception propagates all the way out of `main()` and is caught by the
notebook's outer handler:

```python
try:
    main()
except LabelSourceError:
    raise
except Exception:
    traceback.print_exc()
    t = pd.read_csv(find_root() / 'test.csv')
    for c in TARGETS: t[c] = 0.5
    t.to_csv('submission.csv', index=False)
```

There is no check of whether `submission.csv` already holds a valid, banked
ensemble before this fires. This is a deterministic, fully-identified trigger
(not a vague "could happen") that converts a near-complete real ensemble into a
uniform-0.5 file — macro AUC ≈ chance for every target.

**Fix:** before writing the 0.5 fallback, validate any existing `submission.csv`
(schema, row IDs, finiteness, non-degenerate variance — the notebook already has
a validator, `_v37_validate_submission`, that does exactly this and can be
reused). Only write 0.5 if nothing valid exists. Keep the fallback write atomic
(see 2.3).

### 2.2 Make Stage-1's 0.5-fallback write atomic — **GO**
The fallback path writes `submission.csv` directly (`t.to_csv('submission.csv',
index=False)`), unlike `write_submission()` two cells earlier in the same stage,
which already does `tmp = str(path) + '.tmp'; sub.to_csv(tmp); os.replace(tmp,
path)`. Apply the same pattern to the fallback write. (This is secondary to 2.1
— fixing 2.1 makes this fallback fire far less often, but it should still be
atomic on the rare path where it's needed.)

### 2.3 Make Stage-2's final submission write atomic — **GO**
Stage 1, Stage 3, and Stage 4 all commit their final submission via
tmp-file + `os.replace`. Stage 2's final commit
(`_a5_sub.to_csv('/kaggle/working/submission.csv', index=False)`) is the one
stage that writes directly. Bring it in line with the other three — trivial,
zero prediction impact.

### 2.4 Keep T4 fp16 (already implemented — no code change, just don't regress it) — **GO**
Measured: T4 384px/bs4 — bf16 796.6 ms/step, fp16 254.4 ms/step, fp32
607.8 ms/step (sm_75 has no native bf16; bf16 is 3.1× slower than fp16 *and*
slower than fp32). v9 already fixed a hardcoded `AMP_PREF='bf16'` bug; v10's
capability-based choice (`if AMP_PREF == 'bf16' and cc < (8,0): use fp16
instead`) is correct as shipped. **Action for V11: add a regression test that
asserts fp16 is selected on a T4 capability probe** — this is a "don't break it"
item, not new work.

### 2.5 Fix the stale `K_EVAL=24` comment — **GO**
Live value: `K_EVAL = 62  # every window position the volume holds, not an
evenly spaced subset`. A stale comment two cells later, inside `load_model`'s
DataParallel-removal note, still says "a single T4 handles K_EVAL=24 windows
fine." Documentation-only fix, but a real trap for anyone sizing batches off the
comment instead of the live constant.

---

## 3. Changes that need paired OOF before shipping to V11-A

For every item below: run it through the 4,349-study weak-label OOF harness
(§7), report macro AUC delta + paired bootstrap 95% CI + per-target deltas +
prediction rank correlation vs. the current behavior. **Do not accept any of
these on the strength of a few visual examples, a public-LB nudge, or the
58-study gold gate alone** — the gold gate's cross-model sd is ~0.05, which
cannot resolve effects this size (see §7).

### 3.1 V10's content-centroid crop vs. V9's fixed-center crop
V9 crops at `(h//2, w//2)`, no content awareness. V10's `_content_crop_origin`
thresholds a reference slice at 8% of its max, computes the centroid of
above-threshold pixels, and nudges the crop toward it, capped at 25% of the
crop half-width — confirmed unchanged from what the original audit described,
and the notebook's own changelog says this was never validated by retraining or
an OOF re-score.

**Both sides of this are legitimate** and the OOF test needs to cover the real
question, not just "old vs new": V10's motivation is real (some acquisitions are
genuinely off-center, and the shift is capped, not unbounded), but frozen
checkpoints were trained on the fixed-center distribution, so a "more sensible"
crop can still cost AUC purely from distribution shift.

Run **all** of:
- V9 fixed center (control/default for V11-A while pending),
- V10 per-slice content centroid (current),
- series-level centroid (one transform per series, not per slice — see 3.3),
- gated series-level centroid (only correct when the fixed crop demonstrably
  clips foreground near a border).

**Default while pending: V9 fixed center.** It is the input distribution the
frozen checkpoints actually trained on; every other option is not proven either
way.

### 3.2 Stage 2's `INTENSITY='series'` vs. the deployed `INTENSITY='slice'`
Stage 2 hardcodes `INTENSITY = 'slice'` — every slice gets its own independent
1st/99th-percentile contrast stretch (`render()` → `read_crop()` →
`np.percentile(crop[::4,::4], [1,99])` per slice). A second, unused branch
(`INTENSITY == 'series'`) already exists in the same function: it pools
percentiles across all slices in a slot before stretching each one. This is
**dead code** — nothing in v10's ~124-line changelog mentions it, tests it, or
explains why it's there unused.

This belongs in the exact same risk class as 3.1, not below it: flipping it
changes the actual pixel values fed to a frozen network, and there is zero
evidence the deployed checkpoint was ever trained with or evaluated against
per-series normalization. Do not ship this as a "one-line, near-zero-risk" fix.
Run a 2×2 paired OOF grid instead:

| | crop = fixed | crop = V10 centroid |
|---|---|---|
| **intensity = slice (current)** | A | C |
| **intensity = series** | B | D |

No GO on either axis until measured. (There are real, competing physical
arguments for per-slice normalization too — coil sensitivity and non-standardized
MRI intensity scales are legitimate reasons per-slice stretching could already be
the right call, not just an artifact to fix.)

### 3.3 Never recenter slices independently — series-level transform only
Stage 2's `read_crop()` and Stage 4's `mm_crop_resize()` each recompute the crop
centroid **per slice** (confirmed: called once per slice inside their
respective per-slice loops). Three neighboring slices are later stacked as
2.5D channels — if their crop origins differ, static anatomy can appear to
translate between channels, which is not present in the source MRI and not
part of either frozen model's training distribution.

Stage 1/3's shared `read_slot`, by contrast, already computes one crop origin
per series (from the single middle slice of the stacked volume,
`vol[len(vol)//2]`) and applies it to the whole stack — this is the
series-level design Stage 2/4 should adopt, not something that needs inventing.

**Caveat found in this same review that should be folded into the fix:** Stage
1/3's reference slice is itself a single-point estimate — whatever slice lands
at the middle index. If that slice has an artifact or atypical anatomy, the one
`(y0,x0)` it produces gets applied to the entire series. If a more robust
version is built for Stage 2/4 (median-of-normalized-slices aggregate
centroid, per improvementsv11.md's original V11-B proposal), apply the same
robust reference-slice logic to Stage 1/3 too rather than treating it as
already correct.

**Frozen checkpoints: do not change without reproducing training preprocessing.**
**V11-B (retrain): strongly test series-consistent, translation-augmented
cropping** — this is one of the higher-value retraining experiments in this
whole plan.

### 3.4 Replace `sub.iloc[0]` / `pref[0]` arbitrary series selection
Confirmed verbatim in three places:
- Stage 1's `pick_slots`: `cand.sort_values('n_slices', ascending=False).iloc[0]`
  (sorts by slice count only, not real quality).
- Stage 2: `ordered_files(SERIES_ROOT / study / sub.iloc[0].SeriesInstanceUID)`
  — no quality sort on `sub` at all.
- Stage 4's `_pick_series_for_slot`: `pref[0]` / `cands[0]` — same, no ranking.

**The catch that matters more than the fix:** we do not know what selector the
training corpus used. If training always happened to pick (say) the first
eligible fat-suppressed sagittal sequence in whatever order the corpus builder
used, and V11 changes inference to pick the largest/highest-resolution series
instead, that can feed a systematically different sequence family into a frozen
checkpoint — a regression that would look like an improvement in code review
and be one in practice.

**Frozen checkpoints: GO only after the training-time selector is reconstructed
and matched** (not in this repo — check the training script before touching
this for v9/v10 checkpoints).
**New V11-B training: build one deterministic quality selector** ranking by
plane/sequence match, usable slice coverage, fraction of slices with valid
IPP/IOP, monotonic physical positions, spacing regularity, in-plane pixel
spacing, physical FOV, plausible slice-count range, and a localizer/scout
penalty — use identically at train and test time.

### 3.5 Robust mixed-key slice ordering
Confirmed: Stage 1's `order_slices` builds a per-file position from IOP/IPP
projection or an InstanceNumber fallback — but **if any single file** in the
series has neither, the whole function bails to raw filesystem/glob order for
the entire series. Stage 4's independently-written `order_and_meta` is a
different function with different (also imperfect — unresolvable slices default
to `pos=0.0`, clustering them at the start) fallback behavior. These are two
separate bugs in two separate functions, not one bug repeated.

**Fix:** a shared, robust ordering hierarchy — series normal from valid IOP,
project valid IPP onto it, fit position-vs-InstanceNumber for slices missing
IPP, InstanceNumber-only as next fallback, filename order as last resort, log
every fallback and its confidence.

**This is close to GO but still needs a targeted parity check**, since it
changes actual slice sequence for frozen networks: run old vs. new specifically
on the subset of studies where the fallback triggers, before/after prediction
comparison, not a blanket OOF run.

### 3.6 Remove/re-prove the 15× legacy target weights
Confirmed verbatim: `LEGACY_MEMBER_WEIGHT_BY_TARGET = {'Lateral Meniscus': 15.0,
'Medial OA': 2.5, 'Lateral OA': 15.0, 'Contusion': 5.0}`. A 15× per-member
weight can dominate a target's ranking outright.

**The evidentiary chain to get right:** `oof_harness/README.md`'s measured
result — per-target weights fitted by 5-fold CV scored **-0.0008** vs. equal
weighting on 4,349 studies, and the deployed E13/E11 ratio wasn't beaten by
equal weighting either (-0.0025, CI [-0.0134, +0.0083]) — is real and solid, but
it was measured on **Stage 2/3's cross-family blend** (dino / e11 / e13 / v52),
a completely different weighting mechanism than Stage 1's
`LEGACY_MEMBER_WEIGHT_BY_TARGET`, which weights individual CoAtNet/DINOv2
members within Stage 1's own ensemble. Nothing in `oof_harness/README.md`
mentions Stage 1's native ensemble or this dict. Citing the Stage 2/3 result as
direct proof this dict is overfit overstates what was measured — it's a strong
prior, not the same experiment.

**Action: run the dedicated ablation** — current legacy weights vs. all-1.0 vs.
excluding the legacy family vs. a small global family-level weight — on the
4,349-study OOF harness. A 15× value is exactly the shape of thing that tends to
be overfit, and the prior from the Stage 2/3 experiment is a reasonable bet that
it will lose; but confirm it on its own ablation before deleting it, and don't
write up the deletion as "the repo's own measurement already proved this wrong."

### 3.7 Re-test hard-coded target-specific TTA pooling
Confirmed verbatim: `PUBLIC_FRONTIER_TARGET_POOL = {'Fracture': 'max',
'Contusion': 'max', 'Medial Meniscus': 'max', 'Lateral Meniscus': 'max', 'ACL':
'top2', 'MCL': 'top2', "Baker's": 'max'}`, plus `'Synovitis': 'original_mean'`.
Max-pooling is statistically aggressive — the more windows evaluated, the higher
the chance one false-positive window sets the study score, and it changes
behavior if the number of windows changes at runtime.

**Action:** OOF-test mean probability / mean logit / smooth log-sum-exp /
fixed-*fraction* top-k / learned target-query attention against the current
per-target dictionary, inside nested folds. "Looks heavily tuned" is a correct
observation, not a measurement — simplify only if the current policy shows no
stable OOF advantage.

### 3.8 Stage-3 adaptive alpha (transductive per-target blend weight)
Confirmed verbatim: `_rad_adaptive_alpha` computes `diversity` from
`corrcoef(_rad_pass2_consensus[:,t], e10_rank[:,t])` (both are functions of
current test predictions, not labels) combined with fold-agreement stability and
a hand-set per-target prior, then clipped twice. No labels are read — not
leakage in the ordinary sense — but it is genuinely transductive: the ensemble
recipe reacts to statistics of the hidden test set itself.

**Getting the framing right matters here:** transductive is not automatically
bad — if the hidden set genuinely has a different model-correlation structure,
dynamically down-weighting a redundant family can help, and the risk is
instability/distribution-dependence, not label leakage. The right test isn't
"remove it," it's: **does adaptive weighting beat a fixed alpha consistently
across many pseudo-test subsets drawn from OOF predictions?** Compare, in this
order: fixed published alpha → fixed OOF-fitted global alpha → fixed
OOF-fitted per-target alpha with strong shrinkage → current test-adaptive
alpha. **Keep the current adaptive alpha only if it wins that comparison
repeatably across folds/sites** — until tested, keep it as-is rather than
removing it on the transductive-ness argument alone.

### 3.9 Protocol-metadata calibration (Stage 3 Rad branch)
Confirmed verbatim: features are exact counts of total series, series per plane
(Sag/Cor/Axi), and fat-suppressed/fluid-sensitive series both overall and per
plane. These can proxy genuine input completeness (protocol composition
determines what pathology is even visible) or scanner/site/institution — both
readings are plausible and the feature list alone can't tell them apart.

**Action: grouped OOF, holding out protocol/site/scanner clusters, comparing
image-only vs. protocol-calibrated branches.** Keep the features by default;
**remove only the ones that help random CV but not grouped CV** — don't strip
them preemptively just because they *could* leak site, and don't keep them
blindly either.

### 3.10 Stage-2 laterality heuristic
Confirmed verbatim: `series_side(path)` reads
`ImagePositionPatient[0]` from the **first ordered file only**, flips
non-sagittal series when negative, and silently returns 0.0 (no flip) if the
tag is missing or unreadable. A single-slice, single-coordinate heuristic.

**Frozen checkpoints: do not redesign without confirming the training-time
convention first** — a "more robust" laterality rule is exactly the same
"looks better but shifts the input distribution" risk as 3.1-3.3.
**New V11-B training: establish one validated, geometry-based laterality
convention** (median/central-slice IPP, patient coordinate system, explicit tags
where trustworthy, consistency check across series) and use it consistently.

### 3.11 Stop banking partially-evaluated Stage-1 members as full members
Confirmed verbatim: the scheduler's `pop_next()` computes `n_win` from remaining
wall-clock budget, and if it's less than the full window count, takes a
**central subset** (`mid = (len(starts_full) - n_win) // 2; starts =
starts_full[mid:mid+n_win]`). That truncated `starts` gets passed to
`_run_member` and banked via `bank()` at the member's **normal, full weight** —
no distinction between "ran on all windows" and "ran on a runtime-dependent
central subset." This means the mathematical definition of a member changes
with wall-clock timing measured mid-run — not reproducible run-to-run.

**GO: never silently treat "incomplete" as "complete" — a member's identity
must be fixed regardless of runtime.**
**Open design choice on the how (not a correctness question):**
- Policy A — an incomplete member is dropped from the blend entirely, or
- Policy B — a reduced-window member is defined in advance (before the run,
  not adaptively) with its own independently-validated weight.

Under severe time pressure, B could genuinely outperform A. Either is
acceptable; what's not acceptable is the current "silently mutate the member
mid-run, keep its normal vote" behavior.

---

## 4. Retrain-only items — do not patch frozen inference for these

### 4.1 Stage-4 cross-slot 3-slice windows
Confirmed verbatim: `_eval_centers` treats the entire 64-slot concatenated stack
(sagittal fluid/non-fluid, coronal fluid/non-fluid, axial, concatenated
back-to-back with a running offset) as one continuous index range with **no
slot-boundary check**. `[c-1, c, c+1]` windows near a seam can mix, e.g., the
last sagittal slice with the first coronal slices.

**Critical scoping point, confirmed by the notebook's own comment**
("verbatim from finetune_raptor.py StudyWindows (train=False)"): this exact
cross-boundary behavior **mirrors training**. It is not an inference-time bug —
it's a property of what the frozen CoAtNet checkpoints actually learned. A
"buggy-looking" pattern in frozen inference code is not sufficient justification
to change it when the training code produced the same pattern on purpose (or at
least, consistently).

**Frozen CoAtNet arms: NO CHANGE.**
**V11-B retraining: GO to slot-aware windows** — build 3-slice windows only from
centers whose full window sits inside one slot, attach plane/sequence/position
embeddings, concatenate window embeddings after encoding, use target-specific
attention across valid windows. This is one of the higher-value retraining
experiments in this plan because it fixes an actual violation of the 2.5D
locality assumption, not a cosmetic issue.

### 4.2 Series-level aggregate cropping (see §3.3) as a trained-in convention
Same status as 4.1: worth strongly testing in V11-B with translation/FOV
augmentation so a new model is trained to be robust to acquisition centering,
rather than trained fixed-center and served adaptive-center.

### 4.3 One canonical laterality convention (see §3.10)
Same status: establish and train against one validated geometry-based rule in
V11-B rather than patching the heuristic used by frozen checkpoints.

---

## 5. Preprocessing/caching architecture (mostly efficiency, touches accuracy risk)

**The core principle for this whole section:** canonicalize only what cannot
change model input — file discovery, DICOM header metadata, ordered-path
manifests, and (only where exact numerical equivalence can be demonstrated) raw
decoded pixels. **Do not** converge Stage 1/2/4's per-family preprocessing
(intensity rule, crop rule, laterality rule) into one shared final tensor —
different checkpoint families may have been trained on genuinely different LUT/
inversion/normalization/crop conventions, and forcing one canonical
representation can erase that as easily as it removes redundant IO.

Concretely, build a layered cache:

```
Layer 0: study index          — series UID -> metadata, ordered files, geometry
Layer 1: decoded canonical slices — file -> modality-corrected image + spacing
Layer 2: family physical crop  — (series, crop_mm, crop_rule) -> cropped stack
Layer 3: model tensor          — (family, resolution, slice picks, intensity
                                   rule, laterality) -> uint8/float tensor
```

Share Layers 0-1 freely across stages (same DICOM read/decode/order underneath
all four stages today, confirmed as three separately-written decoders with
different intensity/crop conventions — Stage 1 manually applies
RescaleSlope/Intercept with no PhotometricInterpretation handling; Stage 2's
`read_crop` does neither; Stage 4's `read_px` applies the modality LUT and
handles MONOCHROME1 inversion). **Never share Layer 3 across families unless
every element of the cache key matches** — `(SeriesInstanceUID,
ordered_slice_hash, crop_mm, crop_rule_version, slice_band, slice_count,
intensity_rule, laterality_rule, output_resolution)`.

This is a real, substantial rewrite risk (three independently-authored decoder
pipelines, two of which are "verbatim from" upstream training scripts per their
own comments) — schedule it as deliberate engineering work, not a drop-in
optimization, and validate every migrated call site against the old
pixel-for-pixel output before trusting it.

---

## 6. Efficiency items — status and rationale

| Item | Status | Note |
|---|---|---|
| Bounded Stage-4 futures (currently *all* studies submitted to the pool at once) | **GO to bound it; benchmark the depth** | Confirmed: `pool.submit(build_study, sid, ...) for sid in test_ids` submits the entire test set in one shot to an 8-worker pool; the budget guard only cancels not-yet-started futures. A bounded window (e.g. `2*workers`) is right in direction; don't assume `2x` vs `4x` without measuring — too shallow stalls the GPU, too deep raises RAM and makes cancellation less effective. |
| Bounded/streaming Stage-4 volume cache (currently caches every study's volume in RAM) | **GO to bound it** | Confirmed math, recomputed twice: `64*336*336 = 7,225,344` bytes = 6.891 MiB/study; ×1,300 studies = 8.749 GiB, before Python/dict/mask/future overhead. The existing `MemoryError` fallback was previously dead code in v9 (per-future exception handling swallowed it before it could fire) and is now fixed in v10 — but "cache everything or fall back to full recompute" is still coarse. Move to a bounded producer/consumer or disk-backed cache. |
| Stage-4 multi-model residency (load 2-3 CoAtNet arms at once instead of sequentially) | **Open — one measurement needed** | See §8, this is the one item this process did not converge on. |
| Stage-4 study-level microbatching (currently one study, one forward, per arm) | **Benchmark-only, not a guaranteed win** | `K_EVAL=62` windows/study already; a batch of B studies pushes the real backbone batch to `B×62` before any internal chunking — B=2 already means 124 images per forward call. Confirm with an OOM-backoff probe (`try B; on OOM: empty_cache(); B //= 2`), and check prediction parity after batching (float batch-order can shift results at the floating-point-noise level). |
| Reuse Stage-4 resized windows across arms when actual checkpoint resolution matches | **Near-GO** | All three declared arms specify `res: 384`, but `load_model` can override from checkpoint metadata (`ck.get('res', res_default)`), and V10 conservatively recomputes per arm because it doesn't check this in advance — a deliberate, explained tradeoff, not an oversight. Fix: load each arm, read its actual `ck_res`, group arms by actual resolution, build/resize windows once per resolution group, and assert the shared tensor is bit-identical to the old per-arm tensor on a test study before trusting it. This is memoization, not modeling — closest thing to a free efficiency win in this whole plan. |
| `torch.inference_mode()` on remaining `no_grad()` sites | **Partial GO** | Confirmed: `inference_mode()` already used at 3 sites (one is a tiny CUDA capability probe, not real inference load; two are in Stage 3's actual head-inference functions). `no_grad()` remains at 5 sites across Stage 1/2/4's real prediction loops — convert those specifically. Expect a small win, not transformational. |
| Pinned CPU memory for the one `non_blocking=True` transfer | **Benchmark, no unconditional GO** | Confirmed: exactly one `non_blocking=True` call exists (Stage 2), and zero `.pin_memory()` calls exist anywhere — so the nonblocking transfer currently has nothing to overlap with. Still, pinned memory consumes page-locked host RAM and is not free; benchmark before/after rather than assuming it helps. |
| Remove Stage-2's subprocess/pip-install `timm` probe | **Keep it — do not remove** | Confirmed: runs once per notebook execution (not per-study/per-fold), costs a few seconds in the common case where the environment is already correct. Removing it trades a few seconds for reduced robustness against Kaggle environment drift. Not worth it in a multi-hour pipeline. |
| Tune `HDR_THREADS=16` / `PIX_THREADS=12` / `ORDER_THREADS=32` | **Benchmark-only** | Values confirmed exactly as configured. The IO-bound finding that motivates concern about thread contention (`~0.8 s/study compute, rest is 5 MB/study over ~1 MB/s network storage`) is from the *training* notebooks, not measured against these specific *inference*-time pools. Don't replace the current numbers with an intuitively nicer guess — run the matrix (4/8/12/16 header threads × 4/8/12 pixel workers × 8/16/24/32 order workers) against real throughput first. |
| `torch.cuda.empty_cache()` call frequency | **Leave alone — confirmed correctly placed** | All 13 call sites sit at genuine model/member/arm/fold boundaries (immediately after `del model`/`gc.collect()`, or before an OOM-retry), never inside a per-study/per-window inner loop. No action needed. |
| `torch.compile` / INT8 quantization | **Low priority, not for the next V11 submission** | Neither appears anywhere in v9/v10/v9-flash today. Correctly scoped in the original audit as forward-looking, not corrections to existing code. Exhaust fp16/batching/caching/Pareto-pruning wins first; if `torch.compile` is tried, measure break-even point per backbone and keep an eager fallback; if INT8 is tried, validate every target's rank/AUC on OOF before trusting it. |
| AUC-per-second Pareto ablation of the three same-architecture Stage-4 CoAtNet arms | **Strong GO as an experiment** | Confirmed: all three arms share architecture string and resolution (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, 384px), and the notebook's own comment already flags this as closer to the "tuning does nothing" case than the "genuinely decorrelated family" case the OOF harness measured — the repo's authors made this an informed, provisional choice, not a blind spot. Run the ablation (v5_swa alone / v5_swa + best complementary / all three) with paired marginal-AUC-vs-seconds on the large OOF set before pruning anything. |

---

## 7. Validation protocol (applies to every DEBATE item above)

- **Large weak-label OOF, n=4,349** (`oof_harness/`) is the only instrument that
  can resolve effects this small. Established family gold-gate AUCs: dino
  0.8200, e11 0.8375, e13 0.8576, v52 0.8420. Pairwise rank correlation 0.64-0.80
  across these four families — genuinely diverse, unlike the label tables
  (0.86-0.995).
- **The 58-study gold gate cannot resolve differences smaller than ~0.05**
  (measured: that's the sd across models differing only by random seed). Use it
  only as an external-domain sanity check, never as a tuning target.
- Concrete precedent for why this matters: adding a 4th, genuinely decorrelated
  family (`prvsiyan`'s v52) measured **+0.0010, 95% CI [+0.0001, +0.0019],
  P(better) 98%** on the n=4,349 harness; the identical comparison on the
  58-gold gate was +0.0021 with CI [-0.0080, +0.0128] — unresolvable. That
  contrast is the whole argument for using the large harness.
- **Report for every candidate change:** macro AUC old/new, paired bootstrap
  95% CI, probability the candidate is better, all 12 per-target deltas,
  prediction rank correlation vs. current behavior, and (where protocol/site
  metadata permits) grouped/held-out-protocol deltas.
- **Do not use the public leaderboard as a hyperparameter loop.** It's ~30% of
  the test data; final standing is the other 70%. Treat it as a final external
  check only.
- **The `surasan092` DINOv3-S family remains a do-not-blend.** Backbone
  (DINOv3 ViT-S/16+, `use_gated_mlp=True`) and window pooling (plain mean over
  10 windows) are solved exactly; the attention/query head is not — best
  reconstruction reaches r=0.7192 against the published validation array, likely
  because the reconstruction normalized per-slice while the original config
  implies per-series intensity statistics (`intensity_stat_workers`) and ignored
  `slot_min_coverage` during series selection. Fix preprocessing and decouple it
  from the head-wiring search before attempting reconstruction again. Even a
  perfect reconstruction was only ever worth ~+0.001 by the measured 4-family
  analog — it's a nice-to-have, not a route to a materially different score.

---

## 8. The one open item — Stage-4 multi-model residency

**Claim in dispute:** whether running 2-3 CoAtNet arms with all their models
resident in memory simultaneously (instead of the current strictly-sequential
one-model-at-a-time design) is a closed question or an open one.

**What's confirmed:** the notebook's own comment says plainly —

```
SEQUENTIAL ARMS (the OOM fix): only ONE model is resident at a time, so peak
system RAM == one model == the single-arm champion's footprint... Holding both
arms simultaneously OOM'd system RAM on the full hidden test.
```

**What's confounded, and why "closed" is premature:** this comment sits
immediately below the code that builds `_vol_cache`, the full-test volume
cache measured at ~8.75 GiB for 1,300 studies (§6). The measured OOM happened
with **both** two resident models **and** the entire test set's decoded volumes
cached in RAM at once. Nothing in the notebook isolates which factor caused the
OOM or by how much each contributes to peak RSS. Since this same plan already
calls for bounding that cache (§6), the memory profile at the moment a second
model would load is not fixed — it changes by however many GiB the bounded
cache frees, plausibly most of the 8.75 GiB, against a second CoAtNet model
whose weights + K=62-window activation footprint is very likely a few hundred
MB to low GiB.

**How to actually resolve this** (further argument won't move it further):
when the bounded/streaming Stage-4 cache ships (§6), log peak system RSS
immediately before attempting a second model load, once, as a side effect of
that work. That single number — headroom after the cache fix vs. a second
model's measured footprint — settles this directly.

**Until that measurement exists:** treat this as **closed for the current
(full-cache) memory profile, open pending re-test once the bounded cache
ships** — not permanently closed. Don't spend engineering time re-testing it
under today's unbounded-cache conditions; that specific retry really would be
wasted effort and would reproduce a known result.

---

## 9. Recommended implementation order

**Immediate (V11-A, this round):**
1. Stage-1 last-good-submission preservation (§2.1) — highest priority in this
   entire document; deterministic, catastrophic-if-hit, trivial to fix.
2. Atomic Stage-1 fallback write (§2.2) and Stage-2 final write (§2.3).
3. fp16-on-T4 regression test (§2.4); fix the stale `K_EVAL` comment (§2.5).
4. Freeze the V9-fixed-center + V10-correctness-fixes baseline as V11-A's
   default (§3.1) — the crop, intensity, and cropping-locality questions (§3.1,
   3.2, 3.3) all default to "keep V9/current behavior" until their OOF
   ablations land, not to "ship the new idea because it looks better."
5. Run the OOF ablations for §3.4-§3.11 (series selection, slice ordering
   parity, legacy weights, TTA pooling, Stage-3 alpha, protocol calibration,
   laterality) — none of these block the V11-A freeze; they gate what replaces
   current behavior next.
6. Efficiency: bound the Stage-4 futures queue and volume cache (§6); reuse
   resized windows across same-resolution arms after verifying actual
   checkpoint resolutions (§6); convert the remaining `no_grad()` sites (§6).
   Benchmark thread counts, pinned memory, and study microbatching rather than
   assuming wins.
7. Once the bounded cache ships: take the one RSS measurement in §8 and close
   out the multi-model-residency question either way.

**V11-B (separate, not on the critical path for the next scored submission):**
8. Slot-boundary-aware Stage-4 windows with plane/sequence/position embeddings
   (§4.1).
9. Series-consistent, translation-augmented cropping trained in from scratch
   (§4.2/§3.3).
10. One validated geometry-based laterality convention, trained against (§4.3/
    §3.10).
11. A deterministic, quality-ranked series selector used identically at train
    and test time (§3.4).
12. Confidence-masked weak-label supervision distinguishing explicit-positive/
    explicit-negative/uncertain/absent report evidence, given the confirmed
    substantial disagreement between report-derived weak labels and the 58
    expert image labels.
13. Only after the above: attempt the `surasan092` reconstruction again, with
    per-series (not per-slice) intensity statistics and `slot_min_coverage`
    respected during series selection, and only blend it if OOF marginal gain
    is positive and stable — never blend an approximate reconstruction on
    correlation alone.

---

## 10. What V11 should not do (unchanged from the original audit, still holds)

- Tune another round of per-target blend weights on the 58 gold studies or the
  public leaderboard.
- React to a small public-LB move by hard-coding more target-specific
  exceptions.
- Keep any preprocessing change "because it fixes an intuitively real edge
  case" without a paired OOF win.
- Change frozen-checkpoint preprocessing (crop, intensity, ordering,
  laterality, series selection) without reproducing what that checkpoint was
  actually trained on first.
- Treat report-derived weak labels as equivalent to the 58 expert image labels.
- Blend an approximate model reconstruction on correlation alone (r≈0.72 is not
  a green light).
- Add more same-family checkpoints before measuring residual diversity via OOF.
- Use a larger backbone as a substitute for fixing series/window/label
  problems.
- Expect probability calibration methods to move AUC on their own — AUC is
  invariant to monotonic per-target transforms; only a transform that changes
  cross-target/cross-example ordering can move it.
- Treat a runtime-truncated Stage-1 member as equivalent to its full-window
  definition.
- Assert an efficiency optimization is "closed" or "proven safe" from a single
  historical comment without checking whether the conditions that produced it
  are still in force (see §8).

---

## 11. Acceptance checklist before calling a notebook "V11"

- [ ] Fixed-center V9 baseline reproduces exactly.
- [ ] Stage-1 last-good-submission preservation ships and is tested against a
      deliberately-injected group-2 exception.
- [ ] No incomplete Stage-4 arm and no partially-evaluated Stage-1 member can
      enter the blend at full weight.
- [ ] Every series selector is deterministic and logged; frozen-checkpoint
      selectors match their (reconstructed) training-time convention.
- [ ] Slice ordering never falls back to arbitrary filesystem order because of
      one missing tag; fallback rate and confidence are logged per series.
- [ ] Crop origin is constant within a series wherever centroid cropping is
      used at all; any adaptive crop shipped to a frozen checkpoint has cleared
      a paired-OOF gate first.
- [ ] Legacy 15× weights and hand-written target TTA pooling have each been
      re-ablated on the 4,349-study harness, not assumed disproven by a
      different experiment.
- [ ] Stage-3 adaptive alpha is either OOF-proven to beat a fixed alpha across
      simulated test distributions, or replaced with the winning fixed
      alternative.
- [ ] Protocol calibration features are grouped-CV validated, not removed or
      kept on intuition alone.
- [ ] Weak labels carry confidence/uncertainty rather than being forced to
      hard 0/1 (V11-B).
- [ ] 58 gold studies are used only as a sanity check, never as a tuning set.
- [ ] Every efficiency change has a before/after prediction-parity check
      (exact study IDs/order, 12 columns, no NaN/Inf, no unexpected 0.5
      placeholders, per-stage correlation vs. baseline) — speed changes must
      not silently become accuracy changes.
- [ ] Public leaderboard was not used as an iterative optimizer at any point in
      this process.
