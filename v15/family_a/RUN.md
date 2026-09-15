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
All tests should pass. This exercises the entire pipeline shape (masking,
checkpoint resume, fold coverage, OOF export, comparison, gate) on synthetic
data. If this fails, nothing below will work either — fix it here first.

The optional three-source label experiment is specified separately in
`../LABEL_CONSENSUS_AUDIT.md`. Run its label-only gate before training any
consensus-label image model; a public notebook title is not sufficient evidence.

Exact consensus has now failed that gate. If adding a third arm after the
current baseline/raw-aux run, follow the leakage-safe fold-local calibration
specification and promotion criteria in `../CALIBRATION_FIX.md`.

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

The generated Kaggle notebook reads the two cache notebook outputs from their
mounted `/kaggle/input/rsna-knee-v15-family-a-{pool,gold}-cache/` paths. It
requires a working CUDA device, verifies it with a real arithmetic operation,
uses fp16 autocast on T4, and transfers images in batches rather than attempting
to place the roughly 8.3 GiB uint8 cache on the GPU.

It trains baseline and auxiliary arms at two seeds (5-fold each, so 20 fold-runs
total). The auxiliary target policy is re-derived inside every outer fold using
only that fold's training-side expert cases; held-out expert cases cannot choose
the auxiliary targets or their weights. The scratch head uses LR `1e-3`, while
the partially unfrozen pretrained encoder uses LR `8e-6`. In the auxiliary arm,
every eligible report-only row is visited once per epoch and expert rows are
sampled to a declared 10% of row instances; without that anchor the roughly
46:3,500 training-fold imbalance would dilute expert supervision to about 1.3%.
Checkpoints resume
automatically if interrupted. The epoch count is fixed before training:
outer-fold expert labels are scored once afterward and never select an epoch or
checkpoint. Output is
`/kaggle/working/family_a_run4/seed<N>/{baseline,auxiliary}/`, each with
gold-only `*_oof.csv`, full-cohort `*_all_oof.csv`, and `*_receipt.json`.

The shipped notebook defaults to `FAMILY_A_SHARD=0`, a single safe T4 job. On a
time-limited T4, run ten independent jobs with `FAMILY_A_SHARD=0` through `9`.
On an unrestricted GPU host, explicitly set `FAMILY_A_SHARD=all`. Shards `0..4`
are seed 1400/folds `0..4`; shards `5..9` are seed
1401/folds `0..4`. Each job runs the matching baseline and auxiliary fold and
writes a partial OOF plus a hash-bound receipt. Example:

```
FAMILY_A_SHARD=0 jupyter nbconvert --to notebook --execute --inplace \
  rsna-knee-v15-family-a-train.ipynb
```

Preserve each job's `family_a_run4/shardNN` directory. Once all ten are copied
under one root, merge each seed/arm independently; the merger rejects missing,
overlapping, misconfigured, re-ordered, or hash-mismatched shards:

```
for seed in 1400 1401; do
  for arm in baseline auxiliary; do
    python3 merge_shards.py \
      --receipts /path/to/family_a_run4/shard0*/seed${seed}/${arm}/${arm}_partial_receipt.json \
      --ids /path/to/family_a_run4/all_study_ids.csv \
      --gold-ids /path/to/family_a_run4/gold_study_ids.csv \
      --out-dir /path/to/family_a_run4/merged/seed${seed}/${arm}
  done
done
```

For sharded runs, substitute `family_a_run4/merged/seed<N>/...` in the commands
below. The shard index changes scheduling only; it does not change folds,
policies, learning rates, seeds, or epoch count.

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

Read the gate's `decision` field. Its default minimum is `+0.001` macro AUC;
`PASS_PROCEED_TO_FAMILY_C` means both seeds agreed in sign and cleared that
minimum — a project-management signal to
start Family C, not a claim of statistical significance at n=58 (the gate JSON
says this explicitly). Either way, remember the non-independence caveat: the
original global auxiliary-loss policy was discovered using this same 58-study
cohort. Fold-local policy selection now prevents direct held-out-label leakage,
but the broader target-policy design was still developed on this cohort, so the
comparison remains developmental rather than fully independent confirmation.

Do not translate Family A's standalone OOF delta directly into a leaderboard
forecast. A score claim requires a frozen blend with the current ensemble and a
paired same-study comparison of that complete blend. A public score of at least
0.950 is proved only by a scored submission, not by this development gate.

After both seeds finish, use `evaluate_submission_gain.py` to test a predeclared
10% rank blend against a frozen, keyed OOF prediction file for the existing
ensemble. Use the `*_all_oof.csv` files with `--source-kind report-soft` and the
exact audited report-label CSV for the large development check. This is the
variance-efficient promotion gate, but it is still not a leaderboard score:

```
python3 evaluate_submission_gain.py \
  --labels /path/to/llm_labels_v2.csv \
  --source-kind report-soft \
  --baseline-oof /path/to/frozen_existing_ensemble_oof.csv \
  --family-oof /kaggle/working/family_a_run4/seed1400/auxiliary/auxiliary_all_oof.csv \
               /kaggle/working/family_a_run4/seed1401/auxiliary/auxiliary_all_oof.csv \
  --weights 0.10 0.05 0.15 \
  --out /kaggle/working/family_a_run4/submission_gain.json
```

## What this package deliberately does not do yet

- No Family C code (gated on this package's own output, per the plan).
- No blend with the current frozen ensemble — that's Run 9 in the plan, after
  Family A's own value is established.
- No real DINOv2 weights are bundled; `Dinov2Encoder` needs `transformers` and
  the `metaresearch/dinov2/PyTorch/small/1` model attached (see
  `kernel-metadata-train.json`).
