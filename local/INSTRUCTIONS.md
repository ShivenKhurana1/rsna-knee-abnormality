# Running the training locally

Repo: <https://github.com/ShivenKhurana1/rsna-knee-abnormality/tree/v9-own-trained-model/local>

## Read this first

**You cannot produce a competition submission locally.** RSNA Knee is a code competition — the
notebook must execute on Kaggle against the hidden test set, and a local checkout only ever sees
the 3 public studies. What moves off Kaggle usefully is **training**: it needs no competition
images at all, and it sidesteps both the 30 h/week GPU quota and the 12 h session cap.

## Requirements

- **NVIDIA GPU.** The script autocasts on `cuda`. Apple Silicon / MPS will not work as written.
- ~15 GB VRAM at `--bs 4`; 24 GB lets you use `--bs 8`
- ~25 GB free disk
- `torch`, `timm`, `numpy`, `pandas`, `scikit-learn`, `pyarrow`
- Kaggle CLI authenticated, with the competition rules accepted on that account

## 1. Clone

```bash
git clone -b v9-own-trained-model https://github.com/ShivenKhurana1/rsna-knee-abnormality.git
cd rsna-knee-abnormality/local
```

## 2. Fetch data — ~22 GB, once

```bash
./fetch_data.sh ./data
```

Pulls both corpus parts, the LLM soft labels and `train.csv`, then merges the corpus into a
single `(4407, 44, 336, 336)` uint8 array.

## 3. Train

```bash
python train_knee.py \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --bs 4 --k 12 --k_eval 16 --grad_ckpt --ckpt timm \
  --labels ./data/labels_llm_soft.parquet \
  --epochs 16 --tag local
```

Run it from the directory holding the corpus, or symlink `all_vols.npy`, `all_masks.npy` and
`all_ids.npy` next to the script — it prefers a local single-file corpus.

**If interrupted, re-run the identical command.** It restores model / optimiser / scheduler /
scaler / epoch from `resume_local.pt` and continues.

## Four things that will bite otherwise

1. **`--ckpt timm` is mandatory.** The default points at an SSL backbone that was never
   published; omit it and the run dies immediately with `FileNotFoundError`. It also means
   ImageNet init, which is a real handicap versus the published weights.
2. **Keep the corpus on local disk.** Off a network mount it ran at 5.41 s/study; locally 1.08.
   That single difference is what made training feasible at all.
3. **Use `--bs 8` on a 24 GB card.** `bs 4` was sized for Kaggle's 15 GB T4; the upstream default
   `bs 8 / k 12` assumes 24 GB and OOMs below that.
4. **Do not force bf16.** AMP dtype is selected by GPU capability — bf16 on Ampere and newer,
   fp16 below. On a T4, bf16 was **3.13x slower than fp16** and slower than fp32.

## Outputs, and how to read them

| file | contents |
|---|---|
| `raptor_ft_local.pt` | best epoch by the 58-study gold gate |
| `raptor_ft_local.json` | per-epoch trajectory and per-target AUCs |
| `raptor_gold_local.npz` | gold-set predictions, for measuring ensemble diversity |

Benchmarks on that same gate:

| | gold AUC |
|---|---|
| our Kaggle run, 16 epochs | **0.9036** |
| best public label extractor | 0.8991 |
| best public checkpoint | 0.9214 |

The gate is 58 studies with sd ~0.05 across models differing only by seed — treat small
differences as noise.

## Getting a local result into a submission

The checkpoint still has to reach Kaggle: upload it as a dataset (see
`../p3_weights/dataset-metadata.json` for the format), then add it to the `ARMS` list in `../v9/`.

**Highest-value thing to do with a fast local GPU:** train a *different preprocessing variant*
rather than another seed. Eight public checkpoints of the identical architecture at identical
resolution, differing only in corpus geometry, span **0.8997 to 0.9214** on the gold gate. That
0.022 from slice-span / spacing / FOV is the largest measured lever in this competition, and it
is the one thing Kaggle's quota made impractical to explore.
