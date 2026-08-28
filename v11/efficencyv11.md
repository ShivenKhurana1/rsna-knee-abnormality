# efficencyv11.md — V11 Efficiency Audit Without Sacrificing Detection Quality

> Filename follows the requested spelling: `efficencyv11.md`.

## Scope

This document is strictly about making V10 **faster, more memory-efficient, more deterministic, and less likely to miss the Kaggle execution budget** while preserving abnormality-detection quality. Any optimization that materially changes predictions must be treated as a model experiment and pass the accuracy-preservation gates below.

The efficiency goal is not "make it fast at any cost." The competition includes a dedicated efficiency track, but the primary requirement here is to preserve the high-AUC ensemble behavior.

---

# Executive efficiency plan

## Highest-value changes

1. **Keep T4 fp16 everywhere it is supported.** The repo measured 384px/bs4 at about 254.4 ms/step in fp16 vs 796.6 ms bf16 and 607.8 ms fp32. bf16 on T4 is a severe regression.
2. **Stop decoding/preprocessing the same DICOM content independently in multiple stages.** Build one canonical study/series cache and derive family-specific tensors from it.
3. **Replace Stage-4 full-test RAM cache with bounded streaming/preload.** One Stage-4 uint8 volume is about 6.89 MiB; around 1,300 studies is roughly 8.75 GiB before Python/future/model overhead.
4. **Batch Stage-4 studies/windows across GPU calls.** Current code performs one study at a time and transfers one 62-window tensor per forward.
5. **Pre-read Stage-4 checkpoint metadata and group arms by actual resolution.** All declared V10 arms are 384px; if checkpoint metadata confirms this, resize/normalize each study's 62 windows once and reuse them across all three arms.
6. **Use bounded producer/consumer preprocessing rather than submitting every Stage-4 study future at once.** Avoid RAM spikes and wasted queued work when the budget cutoff fires.
7. **Remove runtime-dependent rework and repeated process startup in Stage 2.** Prepackage the required `timm` build and avoid subprocess import probes/pip installation during the scored notebook.
8. **Tune CPU concurrency from measurements.** V10 simultaneously uses high thread counts (`HDR_THREADS=16`, `PIX_THREADS=12`, `ORDER_THREADS=32`) plus other pools. More threads can make Kaggle network-backed DICOM IO slower.
9. **Use `torch.inference_mode()` and pinned/nonblocking transfers where safe.** Low-risk inference overhead reduction.
10. **Build an AUC-per-second Pareto table before pruning models.** Never remove an arm merely because it is expensive; remove it only if OOF shows negligible marginal AUC.

---

# 1. Establish an efficiency-preservation contract

Before optimizing anything, freeze a reference run and save:

- final prediction matrix before rank conversion,
- final ranked submission,
- per-stage outputs,
- selected series UIDs,
- ordered slice lists,
- crop origins,
- slot masks,
- model/fold list,
- wall-clock timings,
- peak GPU memory,
- peak process RSS,
- DICOM decode counts.

## Optimization classes

### Class A — mathematically equivalent

Examples:

- caching,
- batching identical tensors,
- pinned memory,
- `inference_mode`,
- avoiding duplicate resize,
- bounded prefetch,
- atomic file writes.

Acceptance target: output should be identical or differ only at normal fp16 floating-point noise with essentially unchanged per-target ranks.

### Class B — numerically equivalent but execution reordered

Examples:

- larger GPU microbatch,
- concurrent CPU preprocessing,
- grouped arm execution.

Acceptance: prediction/rank difference must be negligible; macro AUC must be unchanged within measurement noise.

### Class C — model pruning / fewer windows / lower resolution / quantization

These can change AUC and **must not be called pure efficiency optimizations**. Deploy only after OOF demonstrates an acceptable quality-speed trade.

---

# 2. Preserve the T4 fp16 fix everywhere

The repository already measured a critical hardware fact:

```text
T4 384 px, batch 4
bf16: ~796.6 ms/step
fp16: ~254.4 ms/step
fp32: ~607.8 ms/step
```

T4/Turing has no native bf16 acceleration for this workload. V9 fixed a Stage-2 hardcoded bf16 preference; V10 keeps the capability-based choice.

## V11 rules

- T4: use fp16 autocast by default.
- Ampere+: bf16 may be tested, not assumed faster.
- P100: do not trust `torch.cuda.is_available()` alone; retain the repo's real CUDA-op probe.
- Keep fp32 retry only for the specific study/operation that fails, not as a global mode switch.
- Log the dtype actually used by every stage.

---

# 3. Build one canonical DICOM cache across stages

## Current inefficiency

Stages 1, 2, 3, and 4 each contain substantial DICOM discovery, header reading, ordering, pixel decode, crop, normalization, and resize logic. V9/V10 improved reuse **inside** Stage 4, but the notebook still repeats work across stages.

The same study can therefore be:

- walked for headers multiple times,
- have slice ordering recomputed multiple times,
- decode overlapping DICOM slices multiple times,
- be cropped/resized independently for different branches.

DICOM/network IO is expensive enough that the repo's training feasibility analysis found IO dominating compute.

## V11 architecture

Split preprocessing into layers:

```text
Layer 0: Study index
  series UID -> metadata, ordered file list, geometry, quality flags

Layer 1: Decoded canonical slices
  file -> modality-corrected float/uint16/float16 image + spacing metadata

Layer 2: Family physical crop
  (series UID, crop_mm, crop_origin_rule) -> cropped canonical stack

Layer 3: Model tensor
  (family, resolution, slice picks, intensity rule, laterality) -> uint8/float tensor
```

Use explicit cache keys so reuse never accidentally mixes incompatible preprocessing.

### Safe cache key example

```text
(
  SeriesInstanceUID,
  ordered_slice_hash,
  crop_mm,
  crop_rule_version,
  slice_band,
  slice_count,
  intensity_rule,
  laterality_rule,
  output_resolution
)
```

Do not share a tensor across families unless the entire key matches.

---

# 4. Stage-4 full-test RAM caching is too coarse

## Current V10 design

Stage 4 precomputes every study volume in a thread pool and stores it in `_vol_cache` so the three arms do not repeat DICOM preprocessing.

Each volume is:

```text
64 × 336 × 336 uint8
```

Raw payload:

```text
~6.89 MiB/study
~8.75 GiB for 1,300 studies
```

That excludes:

- NumPy object overhead,
- dictionary overhead,
- masks,
- all pending futures and their references,
- temporary float arrays during decode/crop,
- three arm prediction arrays,
- model state,
- Python runtime,
- other stage leftovers.

V10 correctly added a `MemoryError` fallback, but "cache the whole hidden test or fall back to recomputing" is still a coarse memory strategy.

## V11 solution: bounded streaming cache

Use a producer/consumer ring buffer:

```text
CPU workers preprocess next B studies
        |
        v
bounded queue of B volumes
        |
        v
GPU inference consumes current batch for arm(s)
        |
        v
release volumes immediately when no longer needed
```

### Two ways to preserve three-arm reuse

#### Option A — arm-interleaved per study batch

For a batch of studies:

1. preprocess volume once,
2. build windows once,
3. run arm 1,
4. run arm 2,
5. run arm 3,
6. free study batch.

This requires loading all three models or swapping them frequently. Three simultaneous models may exceed RAM/VRAM, so do not use this unless measured safe.

#### Option B — disk-backed preprocessing cache

Preprocess volumes once into a compact local `/kaggle/working` memmap/zarr-like binary cache, then run arms sequentially. This preserves V10's one-model-at-a-time RAM behavior while avoiding three DICOM decode passes.

The cache should use contiguous uint8 and an index table. Local ephemeral disk is preferable to network-mounted input for repeated reads.

#### Option C — bounded RAM chunks across sequential arms

Process e.g. 64-128 studies at a time:

1. preprocess chunk once,
2. load arm 1 and infer chunk,
3. free arm 1; arm 2; arm 3,
4. release chunk,
5. next chunk.

Model reload cost must be benchmarked. If reload is cheap relative to DICOM decode, this can be a strong compromise.

---

# 5. Do not submit all Stage-4 preprocessing futures at once

Current code creates a future for every test study:

```python
_vol_futs = {
    pool.submit(build_study, sid, ...): sid
    for sid in test_ids
}
```

Even with only eight worker threads, every task/future exists immediately. When the wall-clock budget gets tight, cancellation can stop queued futures, but the notebook has already created the whole task graph and any running tasks continue.

## V11 solution

Maintain a bounded number of outstanding futures, e.g. `2 * workers` or `4 * workers`:

```text
submit 16
as one completes -> consume/store -> submit one more
stop submitting immediately when budget guard trips
```

Benefits:

- lower Python object overhead,
- lower temporary-memory pressure,
- faster response to budget cutoff,
- no large queue of work that may never be used.

---

# 6. Stage 4 should batch studies, not forward one study at a time

## Current path

For every study and arm:

1. construct 62 three-slice windows,
2. convert to float,
3. optionally resize,
4. normalize,
5. add a batch dimension,
6. transfer to GPU,
7. one model forward,
8. copy 12 probabilities to CPU.

The model's backbone then sees `B*K` images internally, but `B=1` at study level.

## Why this leaves throughput on the table

- many small Python calls,
- frequent H2D transfers,
- CPU/GPU synchronization after every study,
- poor opportunity to overlap next-study preprocessing with current GPU work,
- repeated allocation.

## V11 solution: microbatch multiple studies

Batch studies with the same number of windows/resolution:

```text
input: [B, K, 3, H, W]
backbone sees [B*K, 3, H, W]
head returns [B, 12]
```

Choose `B` adaptively from available VRAM. Start at 2 and probe upward; T4 memory, CoAtNet activation size, and `K=62` will determine the safe batch.

### Important

Do not simply set a huge batch. Use a measured binary/backoff probe:

```text
try B
if OOM -> empty cache -> B //= 2
cache safe B per (model,res,K,dtype)
```

Unlike V10's Stage-1 fallback, the selected batch size should not change which windows are evaluated; only grouping changes.

---

# 7. Reuse Stage-4 resized windows across all three arms

All three declared V10 Stage-4 arms use the same architecture string and fallback resolution of **384**. `load_model` can override resolution from checkpoint metadata, so V10 conservatively recomputes `eval_windows` per arm.

## Better V11 implementation

Before loading full models, inspect only lightweight checkpoint metadata:

```text
arm -> actual `res`
group arms by `res`
```

For each study/chunk and each unique resolution:

1. build 62 uint8/float windows once,
2. perform the 336→resolution resize once,
3. normalize once,
4. reuse the CPU tensor for every arm in that resolution group.

If all three checkpoints confirm `res=384`, this removes two of three resize/normalization passes without changing the model input.

### Memory note

Do not cache float32 62×3×384×384 tensors for the entire dataset. Reuse them within a bounded study chunk.

---

# 8. Reuse window index tensors and normalization constants

Small but safe reductions:

- Precompute valid window center indices for each slot-mask pattern when possible.
- Keep `_MEAN`/`_STD` on GPU once per stage if normalization is performed there.
- Avoid repeated construction of identical slot IDs, study IDs, and masks in Stage 2.
- Reuse output buffers rather than allocating new arrays inside every study loop.

These do not change predictions.

---

# 9. Prefer `torch.inference_mode()` over `torch.no_grad()` for pure inference

V10 uses `@torch.no_grad()` in several inference functions.

For functions that never require autograd/version-counter semantics, `torch.inference_mode()` can reduce overhead further.

## V11 rule

Convert pure inference functions one at a time and regression-test outputs. Do not wrap code that mutates tensors in a way incompatible with inference tensors.

Candidate functions:

- Stage-1 member prediction,
- Stage-2 fold inference,
- Stage-3 feature/head inference,
- Stage-4 `infer_probs`.

---

# 10. Use pinned CPU memory and asynchronous H2D transfer

Some V10 `.to(dev, non_blocking=True)` calls are present, but nonblocking H2D is most useful when the source is pinned.

## V11 pipeline

For bounded tensor batches:

```python
cpu_tensor = cpu_tensor.pin_memory()
gpu_tensor = cpu_tensor.to(device, non_blocking=True)
```

Use a CUDA stream only if profiling shows CPU preprocessing can overlap meaningfully with GPU compute. Avoid adding stream complexity without a trace proving benefit.

---

# 11. Remove Stage-2 subprocess and pip-install startup from the scored path

Stage 2 launches a subprocess to test whether `timm` registers the DINOv3 model. If not, it scans for a wheel and invokes offline pip, then probes again.

This is robust notebook behavior but avoidable production overhead.

## V11 solution

Before submission:

1. attach the known-good wheel as an input,
2. pin/version it,
3. import it directly in-process,
4. assert the required model exists,
5. fail Stage 2 cleanly if the known package is missing.

If Kaggle's base environment already has the exact version, skip installation entirely.

Keep a development notebook with fallback installation; remove it from the final efficiency notebook.

---

# 12. Tune CPU thread counts instead of assuming more threads are faster

V10 configuration includes roughly:

```text
HDR_THREADS   = 16
PIX_THREADS   = 12
ORDER_THREADS = 32
Stage-4 preprocess workers <= 8
```

The notebook can also have process pools and library-internal threads. On network-backed Kaggle storage, too many concurrent small DICOM reads can create contention and lower total throughput.

## V11 benchmark matrix

For 100 representative studies, test:

```text
header threads: 4, 8, 12, 16
pixel workers:  4, 8, 12
order workers:  8, 16, 24, 32
```

Measure:

- files/s,
- studies/s,
- median and p95 DICOM decode latency,
- CPU utilization,
- process RSS,
- GPU idle percentage while preprocessing.

Pick the smallest worker count near peak throughput. Do not optimize each pool independently if they overlap in time.

Also set OpenMP/MKL/PyTorch CPU thread counts intentionally to avoid hidden oversubscription.

---

# 13. Parse DICOM headers once and reuse geometry everywhere

Stage 1's walk/probe and later order pass can both read headers. Stage 2 and Stage 4 independently repeat header reads.

## V11 study index

Build one metadata record per DICOM file containing the needed fields:

```text
StudyInstanceUID
SeriesInstanceUID
filename
InstanceNumber
IPP
IOP
PixelSpacing
Rows/Columns
RescaleSlope/Intercept
PhotometricInterpretation
selected series tags
```

Persist the compact index in RAM or local disk. Ordering, laterality, physical slice sampling, and series quality selection should consume this index rather than re-opening headers.

This can substantially reduce small-file IO while also making preprocessing more deterministic.

---

# 14. Decode only slices that are actually used

The pipeline often needs a subset of slices per series. Do not decode full pixel arrays merely to decide ordering or slice selection.

Correct order:

1. header-only scan,
2. series selection,
3. physical slice selection,
4. pixel decode selected slices only.

Stage 4 already samples fixed slot counts, which is good. The shared V11 cache should preserve this laziness.

---

# 15. Cache decoded pixels at the right granularity

Caching entire full-resolution studies wastes RAM; caching nothing repeats expensive decompression.

Best compromise:

- cache only selected source slices,
- store as compact uint16/float16/uint8 where mathematically safe,
- evict by LRU/chunk boundary,
- derive multiple model resolutions from the same selected crop when preprocessing rules match.

Do not cache normalized float32 images unless they are immediately reused; float32 multiplies memory by ~4 over uint8.

---

# 16. Move rank operations out of hot loops

Stage 1 writes a new rank-mean submission after every banked member. The CSV is small, so this is not the main bottleneck, but repeated Pandas ranking and disk writes are unnecessary for every checkpoint if the notebook is stable.

## V11 compromise

Maintain the accumulator in memory and checkpoint:

- after every N members (e.g. 2-4),
- at group boundaries,
- before risky model loads,
- near budget cutoffs,
- at stage completion.

Because crash resilience matters, do not eliminate intermediate commits entirely.

For rank calculation itself, use preallocated NumPy ranking if profiling shows Pandas overhead is significant.

---

# 17. Make Stage-2 submission commit atomic

Stage 1 and Stage 4 use temp-file + `os.replace`; Stage 2 writes `submission.csv` directly.

A kernel kill during that write can leave a truncated file. The write is small, so this is primarily robustness rather than speed, but robustness is part of effective efficiency: a fast run that produces a corrupt artifact scores nothing.

Use:

```text
submission.csv.tmp
fsync/close if desired
os.replace(tmp, submission.csv)
```

---

# 18. Preserve last-good submission instead of recomputing after failures

Stage 1's outer exception currently can overwrite a valid banked submission with all 0.5. This is disastrous both for score and for compute efficiency: hours of completed inference are discarded logically.

V11 should treat each successfully committed stage/member as a durable checkpoint. Any later failure keeps the last valid artifact.

---

# 19. Rework wall-clock scheduling around complete model units

Efficiency and quality both improve when the scheduler operates on complete validated units.

Current Stage 1 may reduce windows per member when time is short. That saves time but changes model quality unpredictably.

## V11 scheduler

For every candidate family/member, estimate:

```text
fixed_load_time
seconds_per_study
windows_per_study
expected marginal AUC from OOF
```

Then rank remaining units by something like:

```text
marginal_OOF_AUC / predicted_seconds
```

while requiring a member to run its full validated inference recipe.

When budget is low, skip the next low-value member entirely rather than mutilating its window set.

---

# 20. Build an AUC-per-second Pareto frontier for all ensemble components

The repo already found an important pattern: **diverse-family additions can help; fine blend-weight tuning often does not**.

For the efficiency track, measure each stage/family with:

| Component | OOF macro AUC alone | Marginal AUC when added | Seconds/study | Fixed seconds | Peak VRAM | Peak RAM | Marginal AUC / 1k sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage-1 DINOv2 family | | | | | | | |
| Stage-2 DINOv3 | | | | | | | |
| Stage-3 Rad | | | | | | | |
| Stage-4 CoAtNet v5 | | | | | | | |
| Stage-4 CoAtNet v10 | | | | | | | |
| Stage-4 CoAtNet v4 | | | | | | | |

Use **marginal**, not standalone AUC. An individually strong model can be redundant; a weaker model can be efficient if it adds decorrelated signal.

---

# 21. The three Stage-4 arms need an efficiency-specific ablation

V10 uses three same-architecture CoAtNet arms at equal weight. The notebook itself notes roughly 3× Stage-4 model runtime.

For the main accuracy notebook, keep all arms until OOF says otherwise. For the efficiency version, test:

1. v5_swa only,
2. v5_swa + best complementary arm,
3. all three.

Measure **paired marginal macro AUC vs seconds** on the large OOF set.

### Safe pruning criterion

Do not define a fixed tolerance arbitrarily. Use paired bootstrap confidence:

- if an arm's removal has no resolvable negative AUC effect and materially saves time, remove it in the efficiency variant;
- if it has a stable positive marginal AUC, retain it.

This avoids sacrificing detection quality based on a tiny 58-study gate.

---

# 22. Stage 3 is likely a high-value efficiency component because heads share features

Stage 3 encodes images once and evaluates multiple lightweight heads/features. That pattern is generally superior in AUC-per-FLOP to running several full independent backbones.

V11 should preserve and expand this principle:

```text
one expensive encoder
many cheap diverse heads / target queries
```

rather than adding another full backbone for every small ensemble gain.

Before making a claim about exact speed, measure it in the V10 timing harness because current stage-specific rates have changed from older notebooks.

---

# 23. Distill expensive same-family ensembles for a true efficiency V11

The best long-term efficiency improvement requires training, not notebook micro-optimization.

## Student distillation plan

Teacher:

- final OOF ensemble/family rank logits or probabilities,
- optionally per-family targets.

Student:

- one DINO/ConvNeXt/CoAtNet-sized encoder,
- same multi-view slots,
- target-query head.

Loss:

```text
supervised weak/gold-aware loss
+ teacher soft-logit distillation
+ optional feature distillation
```

Train student out-of-fold so evaluation is honest.

A student that reproduces most ensemble ranking can replace dozens of full forward passes in an efficiency-track notebook.

Do not train the student solely on in-sample teacher predictions.

---

# 24. Avoid risky quantization before exhausting fp16/batching/cache wins

T4 fp16 is already highly effective. INT8 post-training quantization can alter subtle medical-image ranking and may have unsupported/slow kernels for some attention/CoAtNet operations.

## V11 priority

1. fp16,
2. batching,
3. cache/reuse,
4. streaming/prefetch,
5. model Pareto pruning,
6. distillation,
7. only then investigate INT8.

If INT8 is tested, validate every target's rank/AUC on OOF. Do not assume smaller precision means faster on the actual Kaggle stack.

---

# 25. `torch.compile` should be benchmarked, not assumed beneficial

Compilation can help repeated fixed-shape inference but has startup cost and can create graph breaks/dynamic-shape issues.

For each major backbone:

- measure eager warm runtime,
- compile time,
- post-compile throughput,
- break-even number of studies,
- numerical prediction drift,
- memory increase.

Only keep compile if hidden-test scale exceeds break-even comfortably. Never let compile failure take down the stage; preserve eager fallback.

---

# 26. Keep cuDNN benchmark only for stable shapes

V10 enables `torch.backends.cudnn.benchmark=True` in Stage 4. This is sensible when input shapes are stable. If V11 introduces many dynamic microbatch/window shapes, algorithm search can become overhead.

Bucket work into a small number of fixed shapes and benchmark per bucket.

---

# 27. Do not call `torch.cuda.empty_cache()` after every small unit unless profiling justifies it

V10 frequently calls garbage collection and `empty_cache()` after model work. This is useful between large model loads but can create synchronization/allocator overhead if done too often.

## V11 rule

- keep `empty_cache()` after unloading an entire model/arm,
- avoid it inside normal per-study/per-batch loops,
- use peak-memory profiling to decide whether it is needed at stage boundaries.

---

# 28. Reduce Python/Pandas overhead in Stage 3/ensemble blending

The test set is not huge, so this is lower priority than DICOM/GPU work, but final blending repeatedly constructs DataFrames and ranks columns.

Implement a tested NumPy rank function once and use contiguous arrays through hot blend paths. Convert to DataFrame only for final schema merge/write.

Ensure tie behavior matches the reference method before swapping implementations.

---

# 29. Clean up stale comments/config because they cause bad future performance decisions

V10 contains stale Stage-4 comments referring to `K_EVAL=24` or 42 windows while actual configuration uses `K_EVAL=62`. Stale performance assumptions can cause someone to pick an unsafe batch size or memory budget.

V11 should generate config logs from live variables rather than prose:

```text
MAXS=64
K_EVAL=62
arm actual res=[384,384,384]
input dtype=fp16
expected bytes/study=...
```

Treat comments as documentation, not runtime truth.

---

# 30. Add stage-level performance telemetry

Every submission run should print/save:

```text
stage
fixed setup seconds
studies processed
seconds/study
DICOM files decoded
header reads
pixel decodes
cache hit rate
CPU preprocess seconds
GPU inference seconds
GPU utilization if available
peak VRAM
peak RSS
number of retries/OOMs
number of model loads
number of skipped models
```

Without this, future optimization becomes guesswork.

Save to a small `timing_v11.json` artifact.

---

# 31. Use representative timing samples, not 3-study estimates

The older timing notebook measured Stage 1 properly over a larger pseudo-test set but noted Stage 2/3 per-study rates were not yet measured in the same way.

V11 should run timing regression at multiple sizes, e.g.:

```text
N = 20, 80, 160
```

Fit:

```text
wall_seconds = fixed + seconds_per_study * N
```

Do this for every stage, including Stage 4. Use the upper confidence estimate for the wall-clock scheduler.

---

# 32. Avoid network-backed temporary caches for repeated heavy reads

The repo's training study found a large gap between compute time and network memmap IO. For inference, any cache that will be read repeatedly should live on Kaggle's local working disk when possible, not be repeatedly traversed from the mounted input tree.

Suggested approach:

- input DICOM remains read-only on mounted storage,
- build compact selected-slice/volume cache once into `/kaggle/working`,
- run multiple model arms from that local contiguous cache.

Benchmark local disk capacity before committing to full-test cache; otherwise use chunked cache.

---

# 33. Compress cache structure, not image precision blindly

Safe storage choices:

- selected normalized slices as uint8 when that exactly matches the model's current preprocessed representation,
- masks as bit/uint8,
- metadata arrays as compact numeric dtypes,
- contiguous memmap rather than one file per study.

Avoid lossy JPEG or aggressive quantization of MRI pixels merely to save IO; that changes model input.

---

# 34. Make cache format append-only and crash-resumable within the run

A local preprocessing cache can include:

```text
header
study offsets
valid flag
volume bytes
mask bytes
```

Commit a study only after its bytes are complete. If preprocessing fails later, previously completed cache entries remain valid for later arms/stages in the same kernel.

This reduces duplicate work after recoverable branch failures.

---

# 35. Overlap CPU preprocessing and GPU inference

Current stages often alternate between CPU-heavy cache building and GPU-heavy inference.

A better pipeline:

```text
CPU workers: preprocess batch n+1
GPU: infer batch n
writer: checkpoint prior completed results
```

Use a small bounded queue to avoid RAM growth. Measure GPU idle periods with timestamps; the goal is not maximum CPU usage but minimum end-to-end wall time.

---

# 36. Stage-1 group caching should reuse source decoding across resolutions where possible

Stage 1 can have model groups at different image resolutions/configs. Existing `pixel_cache` keys include crop/resolution/slice settings, so a different output resolution causes another decode path.

V11 can separate:

```text
source selected/cropped slice cache
from
resized model tensor cache
```

If two groups select the same source DICOMs and physical crop but differ only in output size, decode/crop once and resize twice.

Keep the cache key strict enough that intensity/crop conventions never cross accidentally.

---

# 37. Stage-2 ProcessPool preprocessing should be compared with a thread-based decoder

Stage 2 uses `ProcessPoolExecutor`. Processes can help CPU-heavy codecs but have startup/pickling/memory costs, especially when moving large NumPy arrays back to the parent.

Benchmark:

- 1 process,
- 2-4 processes,
- thread pool with codecs that release the GIL,
- chunked process tasks returning several studies per result.

Record parent/child RSS and bytes transferred. Keep whichever gives best end-to-end throughput, not maximum per-core CPU.

---

# 38. Avoid copying large arrays unnecessarily

Examples to inspect with a profiler:

- `astype(np.float32)` before operations that could use float16/uint8,
- `np.stack` creating full copies of 62 windows,
- `torch.from_numpy(...).float()` producing another buffer,
- `torch.cat` repeatedly allocating combined tensors.

Possible safe improvements:

- preallocate window tensor and fill it,
- normalize on GPU after uint8 transfer when H2D bandwidth/memory permits,
- use `torch.from_numpy` on contiguous buffers without intermediate copies,
- reuse a pinned staging buffer.

Validate numerical equivalence.

---

# 39. Consider GPU-side resize/normalization for Stage 4

Current Stage 4 forms float CPU windows and uses `F.interpolate` before transfer. An alternate path is:

1. keep compact uint8 windows on CPU,
2. transfer a batched uint8/half staging tensor,
3. convert/resize/normalize on GPU,
4. immediately run backbone.

This trades GPU compute for lower CPU memory/copy overhead. Whether it is faster depends on GPU utilization, so profile both paths.

Do not change resize interpolation/alignment settings.

---

# 40. Failure fallback should be scoped, not global

The good V10 pattern is "one arm/stage fails -> retain previous valid predictor." Preserve that.

V11 should never:

- re-run an entire stage because one study failed,
- discard completed arm predictions,
- replace a valid ensemble with all 0.5,
- retry every study in fp32 because one fp16 study failed.

Fallback granularity should be the smallest failed unit consistent with a valid submission.

---

# 41. Build two V11 notebook variants

## `v11-accuracy`

- runs every OOF-proven family,
- full window definitions,
- no quality-sacrificing pruning,
- optimized caching/batching only.

## `v11-efficiency`

- starts from the exact same preprocessing/model code,
- prunes only components proven redundant on OOF,
- may use distilled student if validated,
- tuned for efficiency-track score.

Maintaining one codebase with feature flags reduces drift.

---

# 42. Proposed V11 execution architecture

```text
START
  |
  +-- real CUDA probe -> choose device/dtype
  |
  +-- build one compact metadata/index table
  |
  +-- choose series + physical order ONCE
  |
  +-- bounded selected-slice decoder / local cache
  |
  +-- Stage 1
  |     family-group tensor derivation
  |     adaptive GPU microbatch
  |     durable checkpoint
  |
  +-- Stage 2
  |     reuse index/cache where preprocessing-compatible
  |     no subprocess package probe
  |     batch folds/studies
  |     atomic checkpoint
  |
  +-- Stage 3
  |     reuse encoder features aggressively
  |     cheap heads
  |     atomic checkpoint
  |
  +-- Stage 4
        read compact local/chunk cache
        group arms by actual resolution
        build/resize windows once per resolution/chunk
        batched study inference
        one model resident at a time unless measured safe
        durable per-arm results
        final atomic blend
```

---

# 43. Benchmark plan

## Phase A — baseline V10

Measure 20/80/160 pseudo-test studies:

- Stage 1 fixed + slope,
- Stage 2 fixed + slope,
- Stage 3 fixed + slope,
- Stage 4 preprocess fixed + slope,
- Stage 4 per-arm inference slope.

## Phase B — equivalent optimizations

Test individually:

1. bounded future queue,
2. metadata/header reuse,
3. chunked local volume cache,
4. `inference_mode`,
5. pinned transfer,
6. Stage-4 B=2 study batching,
7. reused 384 resize across arms,
8. tuned CPU worker counts,
9. Stage-2 no-subprocess startup.

Keep only wins that reproduce reference ranks.

## Phase C — combined optimized run

Re-fit timing slope after combining changes; speedups are not always additive.

## Phase D — OOF component Pareto analysis

Only after the equivalent optimizations are done should you test dropping expensive models/windows.

---

# 44. Regression tests for every efficiency patch

## Prediction tests

- exact study IDs/order,
- exact 12 columns,
- no NaN/Inf,
- no unexpected 0.5 placeholders,
- per-stage prediction correlation vs baseline,
- max/mean absolute logit/probability delta,
- per-target rank correlation.

## Preprocessing tests

- identical selected series,
- identical ordered files,
- identical slice picks,
- identical crop boxes,
- identical laterality flip,
- identical resize interpolation,
- identical intensity statistics.

## Performance tests

- wall time,
- peak RSS,
- peak VRAM,
- number of DICOM header reads,
- number of pixel decodes,
- cache hit ratio,
- GPU batches/second,
- GPU idle time.

---

# 45. Things that look efficient but should NOT be done without OOF proof

Do not automatically:

- lower 384→224 resolution,
- reduce Stage-4 `K_EVAL=62`,
- remove peripheral slices,
- switch max/top-k pooling to mean only for speed,
- quantize backbone to INT8,
- drop Stage 3 because it looks complicated,
- remove a weak standalone model that is strongly decorrelated,
- run only center slices,
- skip a plane/sequence,
- use lossy image compression,
- replace three-slice windows with single slices.

All of those can reduce medical signal and are Class-C model changes.

---

# 46. Things that should be safe first targets

High-confidence, low-quality-risk efficiency wins:

- keep fp16 on T4,
- real CUDA probe,
- one header index,
- one ordered-file list,
- bounded queues,
- local contiguous preprocessing cache,
- resize reuse for same-resolution arms,
- batched study forwards,
- pinned memory/nonblocking copies,
- `inference_mode`,
- no repeated subprocess package probing,
- tuned worker counts,
- atomic writes,
- last-good artifact preservation,
- fewer needless `gc`/`empty_cache` calls,
- NumPy blend path after equivalence testing.

---

# 47. Recommended implementation order

## P0 — before the next scored efficiency run

1. Add timing/RSS/VRAM/decode counters.
2. Make Stage-2 final write atomic.
3. Preserve last-good Stage-1 output on exception.
4. Pre-read all Stage-4 arm metadata and confirm actual resolutions.
5. If all 384, reuse resized windows per bounded chunk.
6. Replace all-study Stage-4 future submission with bounded queue.
7. Benchmark Stage-4 multi-study microbatch.
8. Pin DINOv3/timm package and remove subprocess/pip startup.

## P1 — larger structural speedups

9. Shared metadata/header index across stages.
10. Shared selected-slice decode cache.
11. Local chunked/memmap Stage-4 volume cache.
12. CPU/GPU pipeline overlap.
13. Thread/process tuning.
14. `inference_mode` conversion.

## P2 — efficiency-track model optimization

15. OOF AUC-per-second table.
16. Stage-4 1/2/3-arm ablation.
17. Same-family fold/member compression.
18. Ensemble distillation.
19. Compile/INT8 only after the above.

---

# 48. V11 efficiency acceptance checklist

- [ ] T4 uses fp16, never accidental bf16.
- [ ] Real CUDA op probe still gates GPU use.
- [ ] Every stage has measured fixed/slope timing.
- [ ] Peak RSS and VRAM are recorded.
- [ ] Header reads are not repeated needlessly across stages.
- [ ] DICOM pixel decodes are cached/reused where preprocessing matches.
- [ ] Stage-4 cache is bounded or disk-backed, not all-or-nothing RAM.
- [ ] Stage-4 outstanding futures are bounded.
- [ ] Stage-4 actual checkpoint resolutions are inspected before reuse.
- [ ] Same-resolution windows are resized once per study/chunk.
- [ ] Stage-4 can process multiple studies per GPU call if memory permits.
- [ ] CPU tensors are pinned where nonblocking transfer is used.
- [ ] Pure inference functions use `inference_mode` if regression-tested.
- [ ] No scored-path pip install/subprocess model-registry probe is needed.
- [ ] Thread counts were benchmarked rather than guessed.
- [ ] `empty_cache()` is not called inside hot loops.
- [ ] Stage-2 submission write is atomic.
- [ ] A valid previous submission is never overwritten by global fallback.
- [ ] Runtime scheduler skips full models rather than changing their window definition.
- [ ] Any model pruning is backed by paired OOF marginal-AUC analysis.
- [ ] Final optimized predictions match the reference ranking before submission.

---

# Source cross-checks used for this efficiency audit

Repository paths checked:

- `README.md` — measured fp16/bf16/fp32 T4 timings, IO bottleneck findings, OOF ensemble findings.
- `v10/rsna-knee-ensemble-v10.ipynb` — thread counts, cache keys, Stage-2 subprocess setup, Stage-4 volume cache, 64-slice stack, `K_EVAL=62`, three 384px CoAtNet arms, per-study inference loop.
- `v9/rsna-knee-ensemble-v9.ipynb` — predecessor Stage-4 budget/cache behavior.
- `timing/rsna-knee-timing.ipynb` — historical measured Stage-1 fixed/slope timing and 9-hour budget design.
- `oof_harness/README.md` — evidence that diversity, rather than blend-weight tuning, produced measurable gain.
- Kaggle competition rules/evaluation pages — ROC-AUC evaluation and dedicated efficiency prizes.

Important V10 line regions inspected include configuration (~595-619), DICOM walking/order/cache (~856-1140), Stage-1 scheduler/banking (~1578-1621), Stage-2 dependency bootstrap (~2026 onward), Stage-2 preprocessing/batching (~2135-2762), Stage-3 shared feature/head path (~2830-3580), and Stage-4 config/window/cache/arm loops (~3748-4215).

---

# Bottom line

The largest V10 efficiency opportunity is **not** to reduce medical input yet. It is to remove repeated work:

- decode/order once,
- cache at the correct granularity,
- stream instead of retaining the whole test set,
- batch multiple studies,
- reuse same-resolution windows across the three 384px Stage-4 arms,
- keep T4 fp16,
- and choose ensemble components by **marginal OOF AUC per second**.

Those changes can materially cut runtime and memory while leaving the model's abnormality-ranking behavior essentially unchanged. Only after those are exhausted should V11 consider pruning windows/models, quantization, or resolution changes.
