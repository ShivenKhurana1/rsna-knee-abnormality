# Your laptop training run

The official Kaggle download and first-fold pipeline have been launched. Do not
start another download/training task while they are running.

## What happens automatically

1. Download the competition ZIP from Kaggle into `local/downloads` on C:.
2. Validate the ZIP directory and extract to `D:/RSNA-Knee/data`, checking ZIP
   checksums while reading each file and retaining 20 GiB of disk headroom.
3. Build the normalized image cache and folds in `local/work-laptop` on C:.
4. Train seed 1400, calibrated auxiliary labels, fold 0 for eight epochs.
5. Save the checkpoint, held-out predictions and receipt under
   `local/work-laptop/runs/seed1400/platt/fold0`.

The source archive is retained. Do not move/delete it while extraction is running.
Keep the laptop plugged in, connected to the internet, and awake. Closing VS Code
does not stop these background processes, but shutdown/restart interrupts them.
No Windows power settings have been changed.

## Check progress

- `local/status.json`: current stage or failure reason.
- `local/logs/download.err.log`: downloaded bytes, speed and estimated time.
- `local/logs/pipeline.out.log`: extraction/preparation/training progress.
- `local/logs/pipeline.err.log`: errors and library warnings.

In VS Code, **Terminal → Run Task** offers the full pipeline and a prepared-fold
resume task. Only restart the full pipeline after the existing process has ended.
An incomplete download is rejected before extraction; the Kaggle CLI determines
whether a repeated download can reuse partial data. Do not assume byte-range resume.
Training resumes from the last complete epoch with the same data/config/code.

## Installed and checked

- `.venv-v16` uses the existing CUDA-enabled PyTorch installation.
- Kaggle DINOv2 Small weights: `local/models/dinov2-kaggle`.
- Kaggle report labels: `local/labels/llm_labels_v2.csv`.
- Laptop config: `local/config.laptop.json`, 336 pixels, batch size 1,
  encoder microbatch 2, mixed precision and gradient checkpointing.
- Actual pretrained encoder passed a CUDA forward/backward/optimizer step using
  synthetic images at the configured maximum study size.
- Report IDs match all 4,407 training IDs and the submission target schema.
- All five provisional folds contain positive and negative expert examples for
  every target. Some classes have very few examples, so estimates remain noisy.

This first run uses study-level groups; cross-study patient separation is
unverified. It is a pilot to check throughput and learning, not proof of a
0.95–0.96 leaderboard score. Review its receipt and preparation diagnostics before
launching the remaining folds or comparing architectures.
