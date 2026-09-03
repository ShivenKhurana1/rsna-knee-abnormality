"""Training loop: deterministic seeding, grouped folds, checkpoint resume,
per-arm (baseline / auxiliary) supervision, OOF export, and a receipt for every
run. Loss is computed on the training split only; validation-fold rows are
scored with a bare forward pass (no grad) purely to produce the OOF prediction
for that row -- this is also why no special-casing is needed to keep a fold's
held-out expert label from leaking through the auxiliary term: the auxiliary
loss is only ever summed over training-split rows in the first place.
"""
import contextlib
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compare_oof import aucs
from contract import TARGETS, UID
from folds import assign_folds
from labels import build_supervision
from losses import combined_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent, check=True).stdout.strip()
    except Exception:
        return None


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def write_receipt(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
        f.write('\n')


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


def _batches(idx, batch_size, rng):
    idx = idx.copy()
    rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield idx[start:start + batch_size]


def _predict(model, imgs_t, masks_t, idx):
    model.eval()
    with torch.no_grad():
        idx_arr = np.array(idx)
        logits, attention = model(imgs_t[idx_arr], masks_t[idx_arr])
        return torch.sigmoid(logits).numpy(), attention.numpy()


def _val_macro_auc(probs, y_expert, expert_mask, val_idx):
    """Macro AUC over whichever val-fold rows actually carry an expert label
    (report-only pool studies in the validation split contribute nothing here,
    same as they contribute nothing to the final scored OOF)."""
    val_idx = np.array(val_idx)
    gold_rows = expert_mask[val_idx].any(axis=1)
    if gold_rows.sum() < 4:
        return None  # too few gold rows in this fold's validation split to trust a per-epoch AUC
    y = np.where(expert_mask[val_idx][gold_rows], y_expert[val_idx][gold_rows], np.nan)
    per_target = aucs(y, probs[gold_rows])
    valid = per_target[np.isfinite(per_target)]
    return float(valid.mean()) if len(valid) else None


def train_one_fold(model_fn, imgs, masks, y_expert, expert_mask, y_aux, aux_mask, aux_weight,
                    train_idx, val_idx, epochs, lr, seed, checkpoint_path, batch_size=8, resume=True,
                    patience=None):
    """Checkpoints every epoch (for resume) and separately tracks the best-so-far
    epoch by validation macro AUC on this fold's gold rows, so a long run doesn't
    have to guess the right epoch count in advance: OOF predictions and the final
    receipt both come from the best epoch, not whichever epoch happened to be
    last when training stopped. `patience`: stop early after this many epochs
    with no improvement over the best (None = always run the full `epochs`)."""
    device = torch.device('cpu')
    set_seed(seed)
    model = model_fn().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    start_epoch = 0
    stopping_reason = 'completed_all_epochs'
    checkpoint_path = Path(checkpoint_path)
    best_path = checkpoint_path.with_suffix('.best.pt')
    best_epoch, best_val_auc, epochs_since_improvement = None, -float('inf'), 0
    if resume and checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        restore_rng_state(state['rng_state'])
        start_epoch = state['epoch'] + 1
        best_epoch = state.get('best_epoch')
        best_val_auc = state.get('best_val_auc', -float('inf'))
        epochs_since_improvement = state.get('epochs_since_improvement', 0)

    imgs_t = torch.from_numpy(imgs)
    masks_t = torch.from_numpy(masks).float()
    y_expert_t = torch.from_numpy(np.nan_to_num(y_expert, nan=0.0)).float()
    expert_mask_t = torch.from_numpy(expert_mask)
    y_aux_t = torch.from_numpy(np.nan_to_num(y_aux, nan=0.0)).float()
    aux_mask_t = torch.from_numpy(aux_mask)
    aux_weight_t = torch.from_numpy(aux_weight).float()

    batch_rng = np.random.default_rng(seed + 1)
    history = []
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses = []
        for batch in _batches(np.array(train_idx), batch_size, batch_rng):
            has_expert = bool(expert_mask_t[batch].any())
            has_aux = bool(aux_mask_t[batch].any()) and float(aux_weight_t.abs().sum()) > 0
            if not (has_expert or has_aux):
                # A batch that is entirely report-only pool studies under the
                # baseline (zero-weight) policy has no supervised cell at all --
                # skip it rather than backward() through a graph-disconnected
                # zero loss, which raises rather than silently no-op-ing.
                epoch_losses.append({'expert': 0.0, 'aux': 0.0})
                continue
            optimizer.zero_grad()
            logits, _ = model(imgs_t[batch], masks_t[batch])
            loss, parts = combined_loss(logits, y_expert_t[batch], expert_mask_t[batch],
                                        y_aux_t[batch], aux_mask_t[batch], aux_weight_t)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at epoch {epoch}: {parts}')
            loss.backward()
            optimizer.step()
            epoch_losses.append(parts)

        val_probs, _ = _predict(model, imgs_t, masks_t, val_idx)
        val_auc = _val_macro_auc(val_probs, y_expert, expert_mask, val_idx)
        improved = val_auc is not None and val_auc > best_val_auc
        if improved:
            best_val_auc, best_epoch, epochs_since_improvement = val_auc, epoch, 0
            atomic_save(best_path, {'model': model.state_dict(), 'epoch': epoch, 'val_macro_auc': val_auc})
        else:
            epochs_since_improvement += 1
        history.append({'epoch': epoch,
                        'mean_expert_loss': float(np.mean([p['expert'] for p in epoch_losses])),
                        'mean_aux_loss': float(np.mean([p['aux'] for p in epoch_losses])),
                        'val_macro_auc': val_auc, 'improved': improved})
        atomic_save(checkpoint_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                      'epoch': epoch, 'rng_state': rng_state(), 'seed': seed,
                                      'best_epoch': best_epoch, 'best_val_auc': best_val_auc,
                                      'epochs_since_improvement': epochs_since_improvement})
        if patience is not None and epochs_since_improvement >= patience:
            stopping_reason = f'early_stop: no improvement for {patience} epochs'
            break

    if best_epoch is None:
        # never had >=4 gold rows to score in this fold's validation split, or no
        # epoch ever beat -inf (e.g. epochs=0): fall back to the last trained epoch.
        stopping_reason = 'no_valid_epoch_auc_available; used final epoch weights'
        probs, attention = _predict(model, imgs_t, masks_t, val_idx)
    else:
        best_state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_state['model'])
        probs, attention = _predict(model, imgs_t, masks_t, val_idx)
    return probs, history, stopping_reason, attention, best_epoch, best_val_auc


def run_arm(arm_name, cache_dir, ids, gold_ids, gold_labels, report_source, policy, out_dir,
            model_fn, k=5, epochs=3, lr=1e-3, seed=1400, batch_size=8, resume=True, fold_seed=1400,
            patience=None):
    """Trains one arm ('baseline' has an all-zero policy; 'auxiliary' uses a
    transfer-gated policy) across k grouped folds and writes an OOF prediction
    file plus a receipt. Only studies in `gold_ids` ever contribute an expert-BCE
    term or an OOF row scored against ground truth; report-only pool studies
    still participate in training (auxiliary term only) but are not evaluated."""
    from cache import load_batch

    start = time.time()
    out_dir = Path(out_dir)
    imgs, masks = load_batch(cache_dir, ids)
    y_expert, expert_mask, y_aux, aux_mask, aux_weight = build_supervision(
        ids, gold_labels, report_source, policy)

    fold_info = assign_folds(ids, k=k, seed=fold_seed)
    fold_of = np.array([fold_info['fold_assignment'][u] for u in ids])

    oof = np.full((len(ids), len(TARGETS)), np.nan)
    oof_covered = np.zeros(len(ids), dtype=bool)
    fold_histories = {}
    checkpoints = {}
    for fold in range(k):
        val_idx = np.flatnonzero(fold_of == fold)
        train_idx = np.flatnonzero(fold_of != fold)
        if len(val_idx) == 0 or len(train_idx) == 0:
            raise ValueError(f'fold {fold} has an empty split; reduce k or add studies')
        ckpt = out_dir / 'checkpoints' / f'{arm_name}_fold{fold}.pt'
        probs, history, reason, _, best_epoch, best_val_auc = train_one_fold(
            model_fn, imgs, masks, y_expert, expert_mask, y_aux, aux_mask, aux_weight,
            train_idx, val_idx, epochs, lr, seed, ckpt, batch_size, resume, patience)
        oof[val_idx] = probs
        oof_covered[val_idx] = True
        fold_histories[fold] = history
        checkpoints[str(fold)] = {
            'path': str(ckpt), 'sha256': digest_file(ckpt), 'stopping_reason': reason,
            'best_epoch': best_epoch, 'best_val_macro_auc': best_val_auc,
            'epochs_run': len(history), 'epochs_requested': epochs,
        }

    if not oof_covered.all():
        raise RuntimeError('Incomplete OOF coverage; refusing to write a partial prediction file')

    gold_only = [i for i, u in enumerate(ids) if u in set(gold_ids)]
    frame = pd.DataFrame(oof[gold_only], columns=TARGETS)
    frame.insert(0, UID, [ids[i] for i in gold_only])
    oof_path = out_dir / f'{arm_name}_oof.csv'
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    with oof_path.open('x', encoding='utf-8') as f:
        frame.to_csv(f, index=False)

    best_epochs = [c['best_epoch'] for c in checkpoints.values() if c['best_epoch'] is not None]
    receipt = {
        'arm': arm_name, 'git_commit': git_commit(), 'seed': seed, 'fold_seed': fold_seed,
        'k': k, 'epochs_requested': epochs, 'patience': patience, 'lr': lr, 'batch_size': batch_size,
        'n_studies_total': len(ids), 'n_studies_scored': len(gold_only),
        'policy': policy, 'fold_grouping_caveat': fold_info['grouping_caveat'],
        'checkpoints': checkpoints, 'fold_histories': fold_histories,
        'best_epoch_by_fold': {k_: c['best_epoch'] for k_, c in checkpoints.items()},
        'best_epoch_summary': (
            f'range {min(best_epochs)}-{max(best_epochs)} across {len(best_epochs)}/{k} folds; '
            'if this sits well below epochs_requested, more epochs were not being used -- raising '
            'epochs_requested further is unlikely to help without also addressing what capped it '
            '(the 58-study gold BCE term is small and easy to overfit)'
        ) if best_epochs else 'no fold produced a valid best-epoch AUC (too few gold rows per fold)',
        'runtime_seconds': time.time() - start, 'device': 'cpu',
        'oof_sha256': digest_file(oof_path),
    }
    write_receipt(out_dir / f'{arm_name}_receipt.json', receipt)
    return frame, receipt
