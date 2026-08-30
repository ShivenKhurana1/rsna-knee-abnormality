# V11: implemented pretrained-ensemble candidate

`rsna-knee-ensemble-v11.ipynb` is the runnable, self-contained candidate. It is
generated from the saved V10 notebook, not from an assumed missing OrthoDiffusion
run. The user's reported 0.935 is treated as the reference score, not as a locally
verified measurement. V11 has **not** been run on Kaggle or measured for AUC.

## Implemented changes

| Change | Purpose | Prediction implications |
|---|---|---|
| Exact V9 fixed-center crop functions restored in Stages 1/3, 2, and 4 | Remove V10's unvalidated per-slice centroid shift | Deliberately changes V10 inputs back to the saved V9 convention |
| Current-run recovery checkpoints and atomic writes | A late failure cannot replace valid scores with an all-0.5 fallback | Successful normal outputs unchanged; error handling changes |
| Transactional Stage-1 banking, retry deduplication | A failed save cannot leave a duplicated ensemble vote | Normal successful vote math unchanged |
| Full-window, fixed no-jitter Stage-1 admission | Never silently replace a model with a cheaper central-window variant | May skip members under tight budgets; explicitly recorded |
| Per-study vote normalization | Missing member coverage cannot dilute a study's scores | Identical when all members cover the same studies |
| Stage-4 tie-preserving ranking | Equal predictions cannot get arbitrary ranks based on row order | Identical to the inherited formula when there are no ties |
| Incomplete/non-finite Stage-4 arms excluded | Failed rows cannot enter the blend as untouched placeholders | Strictly drops the affected arm and retains completed arms/base |
| Current-run public-frontier and CoAtNet readiness guards | Prevent an old output file being mistaken for a new successful branch | A skipped/failed branch cannot promote stale output |
| Lazy bounded temporary disk cache | Avoid holding the full cohort and every preprocessing future in RAM | Cached volume and mask values are exact; runtime unmeasured |
| Build hashes, member events, per-stage CSV snapshots | Make the next result attributable to a concrete recipe | Diagnostic only |
| Training fallback disabled and working GPU required | Missing weights/GPU cannot silently trigger training or a CPU marathon | Stops with a clear error |

The model families, input datasets, checkpoint list, and blend coefficients are
retained from V10. No new injury-specific networks were trained. Deliberate model
changes such as new pooling rules, target-weight tuning, or cross-slot window
retraining were not implemented without validation. Existing family-specific
normalization and interpolation remain separate; they are not replaced by a
single shared preprocessing recipe.

OrthoDiffusion is **not** newly mixed into the final submission. The pasted 0.8832
result does not identify a reproducible local extractor, and its AUC has not been
corrected. See `orthodiffusion_diagnosis.md` for the recovered contracts and remaining
uncertainty. This candidate addresses concrete issues in the available ensemble.

## Run on Kaggle

1. Import `rsna-knee-ensemble-v11.ipynb` as a notebook.
2. Attach the competition and the same inputs listed in `kernel-metadata.json`.
   The owner in the supplied metadata is inherited from V10; adjust it if using a
   different account. No upload or Kaggle submission has been performed here.
3. Select GPU T4 x2. The inherited notebook runs offline using the attached weights.
4. Save and run all cells. Keep `submission.csv`, `v11_run_receipt.json`, and the
   `v11_diagnostics/<run_id>/` stage snapshots.
5. Compare against the baseline on the same studies. A successful three-study
   visible-test run establishes execution, not AUC or hidden-test runtime.

The Stage-4 cache admits up to 12 GiB by default and leaves 2 GiB disk reserve.
It is created in a fresh temporary directory and removed after Stage 4. It holds
only preprocessed volumes/masks; no downloaded weights or user files are removed.
If caching cannot fit or write, inference recomputes the volume instead. Setting
`V11_VOLUME_CACHE_GIB = 0` disables cache storage.

The run receipt records output status (`changed_valid`, `unchanged`, or `restored`),
not a claim that an unchanged stage successfully executed a particular model.
Use member/arm events and notebook logs to inspect actual completion. A missing
valid current-run baseline stops execution rather than manufacturing predictions.

## Local verification and regeneration

Requires Python, NumPy, and pandas; tests need neither torch nor competition data:

```powershell
python v11/build_v11.py
python v11/test_v11.py
```

The builder copies exact crop functions from V9, embeds `runtime_helpers.py`,
clears notebook outputs, and compiles every code cell. It also emits
`build_manifest.json` and Kaggle metadata. Edit the builder/helpers and regenerate
instead of editing the generated notebook independently.

Tests cover compilation, deterministic generation, crop-source equivalence,
prediction contracts, current-run-only byte-preserving recovery, bank retry
deduplication, complete-pass admission, rank fusion and ties, cache failure modes,
and the actual generated Stage-4 orchestration with stubbed model inference.

No numerical GPU equivalence, real DICOM extraction, end-to-end hidden-test run,
or AUC gain has been verified. The code improvements are implemented; +0.020 and
a top-three finish remain goals, not measured or promised results.
