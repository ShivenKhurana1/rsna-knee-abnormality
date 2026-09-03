# Family A / Run 4 — run instructions

Scope: baseline (no auxiliary loss) vs. auxiliary (transfer-gated, capped, masked
report-label auxiliary loss) training of the six-slot 2.5D target-query model,
across two seeds, gated into a Family C go/no-go decision. See `RUN.md`'s
docstrings in each module for what's actually being tested and why; this file is
just the command sequence.

## 0. Before you touch a GPU: CPU smoke test (no data, no GPU, ~30s)

```
cd v15/family_a
python3 -m unittest discover -p "test_*.py"
```
All 42 tests should pass. This exercises the entire pipeline shape (masking,
checkpoint resume, fold coverage, OOF export, comparison, gate) on synthetic
data. If this fails, nothing below will work either — fix it here first.

## 1. Get the competition data reachable

Either attach it the Kaggle way (`/kaggle/input/rsna-knee-abnormality-detection/`
with `train.csv`, `train_series.csv`, `train_series/<uid>/`), or point at a local
copy:

```
export RSNA_DATA_ROOT=/path/to/rsna-knee-abnormality-detection
```

If running locally on your own GPU (not an actual Kaggle kernel), `/kaggle/working`
and `/kaggle/temp` are also referenced as plain absolute paths by the reused V13/V14
code — on Linux or WSL2, just `mkdir -p /kaggle/working /kaggle/temp` once; you
don't need to be running inside a real Kaggle notebook for these to work.

## 2. Build the two notebooks (already done once; rerun only if you change policy/config)

```
python3 build_notebook.py
```
Produces `rsna-knee-v15-family-a-cache-pool.ipynb`, `...-cache-gold.ipynb`, and
`...-train.ipynb`. The training notebook bakes in the auxiliary-loss policy
computed from `../transfer_audit_report.json` at build time — rerun this script
if you regenerate that report.

## 3. Run the cache notebooks (CPU-only, no GPU needed, real DICOM decode)

Push to Kaggle (`kaggle kernels push -p .` after copying the relevant
`kernel-metadata-cache-{pool,gold}.json` to `kernel-metadata.json`), or execute
locally:

```
jupyter nbconvert --to notebook --execute --inplace rsna-knee-v15-family-a-cache-pool.ipynb
jupyter nbconvert --to notebook --execute --inplace rsna-knee-v15-family-a-cache-gold.ipynb
```

This is the slow, I/O-bound, non-GPU step — benchmark it on ~100-200 studies
before committing to the full ~4,349-study pool run (the plan's own instruction;
don't extrapolate a total from guesswork). Output: `/kaggle/working/family_a_pool_cache/`
and `family_a_gold_cache/`, each with a `manifest.json` and `cached_ids.csv`.

## 4. Run the training notebook (GPU)

```
jupyter nbconvert --to notebook --execute --inplace rsna-knee-v15-family-a-train.ipynb
```

Trains baseline and auxiliary arms at two seeds (5-fold each, so 20 fold-runs
total). Checkpoints resume automatically if interrupted — rerunning the same
notebook cell picks up from the last completed epoch per fold, it does not
restart. Output: `/kaggle/working/family_a_run4/seed<N>/{baseline,auxiliary}/`,
each with `*_oof.csv` and `*_receipt.json`. Check each receipt's
`best_epoch_summary` — if every fold's best epoch sits well below what you
requested, more epochs are not the lever to pull.

## 5. Compare and gate

```
for seed in 1400 1401; do
  python3 compare_oof.py \
    --labels /kaggle/working/family_a_gold_prep/gold_labels.csv \
    --baseline-oof /kaggle/working/family_a_run4/seed${seed}/baseline/baseline_oof.csv \
    --auxiliary-oof /kaggle/working/family_a_run4/seed${seed}/auxiliary/auxiliary_oof.csv \
    --out /kaggle/working/family_a_run4/compare_seed${seed}.json
done

python3 family_c_gate.py \
  --seed-reports /kaggle/working/family_a_run4/compare_seed1400.json \
                 /kaggle/working/family_a_run4/compare_seed1401.json \
  --out /kaggle/working/family_a_run4/family_c_gate.json
```

Read the gate's `decision` field. `PASS_PROCEED_TO_FAMILY_C` means both seeds
agreed in sign and cleared the minimum gain — a project-management signal to
start Family C, not a claim of statistical significance at n=58 (the gate JSON
says this explicitly). Either way, remember the non-independence caveat: the
auxiliary-loss policy was chosen using this same 58-study cohort, so this
comparison is exploratory, not confirmatory, regardless of which way it comes out.

## What this package deliberately does not do yet

- No Family C code (gated on this package's own output, per the plan).
- No blend with the current frozen ensemble — that's Run 9 in the plan, after
  Family A's own value is established.
- No real DINOv2 weights are bundled; `Dinov2Encoder` needs `transformers` and
  the `metaresearch/dinov2/PyTorch/small/1` model attached (see
  `kernel-metadata-train.json`).
