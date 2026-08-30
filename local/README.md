# Running the training locally

**You cannot produce a competition submission locally.** This is a code competition: the
notebook must execute on Kaggle against the hidden test set, and a local checkout only ever
sees the 3 public studies. What is worth doing locally is **training** — it needs no
competition images at all, only the preprocessed corpus, and it sidesteps Kaggle's 30 h/week
GPU quota and 12 h session cap.

## Requirements

- an **NVIDIA GPU** (the script autocasts on `cuda`; ~15 GB is enough at `bs 4`, 24 GB lets you
  use `bs 8`). Apple Silicon/MPS is not supported by this script as written.
- ~25 GB free disk
- `torch`, `timm`, `numpy`, `pandas`, `scikit-learn`, `pyarrow`
- Kaggle CLI authenticated, competition rules accepted

## 1. Fetch the data (~22 GB, once)

    ./fetch_data.sh ./data

Downloads both corpus parts, the LLM soft labels, and `train.csv`, then merges the two corpus
parts into a single `all_vols.npy` of shape `(4407, 44, 336, 336)` uint8.

## 2. Train

    python train_knee.py \
      --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
      --bs 4 --k 12 --k_eval 16 --grad_ckpt --ckpt timm \
      --labels ./data/labels_llm_soft.parquet \
      --epochs 16 --tag local

Run it from the directory holding the corpus, or symlink `all_vols.npy`, `all_masks.npy` and
`all_ids.npy` next to the script — it prefers a local single-file corpus.

**On a 24 GB card** use `--bs 8`; `bs 4` was sized for Kaggle's 15 GB T4.

It **resumes itself**. If it is interrupted, re-run the identical command: it restores
model/optimiser/scheduler/scaler/epoch from `resume_local.pt` and continues.

## What it produces

- `raptor_ft_local.pt` — best epoch by the 58-study gold gate
- `raptor_ft_local.json` — per-epoch trajectory and per-target AUCs
- `raptor_gold_local.npz` — gold-set predictions, for measuring ensemble diversity

Reference points on that same gate: **0.9036** (our Kaggle run, 16 epochs), **0.8991** (best
public label extractor), **0.9214** (best public checkpoint).

## Notes carried from the Kaggle runs

- `--ckpt timm` is required: the script's default points at an SSL backbone that was never
  published. This means ImageNet init, which is a real handicap versus the published weights.
- AMP dtype is chosen by GPU capability — bf16 on Ampere and newer, fp16 otherwise. On Kaggle's
  T4 bf16 was **3.13x slower** than fp16 and slower than fp32.
- Timing: 1.08 s/study on a T4 with the corpus on local disk (0.96 h/epoch). A 4090 should be
  several times faster. Off a network mount it was 5.41 s/study — keep the corpus on local disk.

## Using the result

To get it into a submission it still has to reach Kaggle: upload the checkpoint as a dataset,
then add it to the `ARMS` list in `../v9/`. See `../p3_weights/` for the metadata format.
