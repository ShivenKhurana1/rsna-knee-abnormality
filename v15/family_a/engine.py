"""Training loop: deterministic seeding, grouped folds, checkpoint resume,
per-arm (baseline / auxiliary) supervision, OOF export, and a receipt for every
run. Loss is computed on the training split only; validation-fold rows are
scored with a bare forward pass (no grad) purely to produce the OOF prediction
for that row -- this is also why no special-casing is needed to keep a fold's
held-out expert label from leaking through the auxiliary term: the auxiliary
loss is only ever summed over training-split rows in the first place.
"""
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
from labels import build_supervision, crossfit_transfer_policy
from losses import combined_loss


def resolve_device(requested='auto'):
    """Return a usable device after a real CUDA arithmetic probe."""
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == 'cpu':
        return torch.device('cpu')
    try:
        if torch.cuda.is_available():
            probe = torch.ones(1, device='cuda') * 2
            if float(probe.cpu()) != 2.0:
                raise RuntimeError('CUDA arithmetic probe returned the wrong value')
            torch.cuda.synchronize()
            return torch.device('cuda')
    except Exception as exc:
        if requested == 'cuda':
            raise RuntimeError(f'CUDA failed a real arithmetic probe: {exc}') from exc
    if requested == 'cuda':
        raise RuntimeError('CUDA was required but is unavailable')
    return torch.device('cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False


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


def rng_state(batch_rng=None):
    state = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state()}
    try:
        if torch.cuda.is_initialized():
            state['torch_cuda'] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    if batch_rng is not None:
        state['batch_rng'] = batch_rng.bit_generator.state
    return state


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def _batches(idx, batch_size, rng):
    idx = idx.copy()
    rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield idx[start:start + batch_size]


def _epoch_training_indices(train_idx, expert_mask, aux_mask, aux_weight,
                            expert_fraction, rng):
    """Visit all auxiliary-only rows and sample enough expert rows to anchor them.

    With 46 training-side gold studies among roughly 3,500 report-only studies,
    a plain shuffle exposes the expert objective in only ~1.3% of rows. The
    declared fraction prevents that accidental dilution without discarding any
    eligible auxiliary row.
    """
    train_idx = np.asarray(train_idx, dtype=int)
    weighted_aux = aux_mask & (aux_weight[None, :] > 0)
    expert_rows = train_idx[expert_mask[train_idx].any(axis=1)]
    aux_only_rows = train_idx[(~expert_mask[train_idx].any(axis=1)) &
                              weighted_aux[train_idx].any(axis=1)]
    if not len(expert_rows):
        return aux_only_rows
    if not len(aux_only_rows):
        return expert_rows
    if not 0 < expert_fraction < 1:
        raise ValueError('expert_fraction must be strictly between 0 and 1')
    requested = max(len(expert_rows), int(np.ceil(
        expert_fraction * len(aux_only_rows) / (1 - expert_fraction))))
    sampled_expert = rng.choice(expert_rows, size=requested,
                                replace=requested > len(expert_rows))
    return np.concatenate([aux_only_rows, sampled_expert])


def _predict(model, imgs_t, masks_t, idx, device, batch_size=8, use_amp=False):
    model.eval()
    probabilities, attentions = [], []
    idx = np.asarray(idx)
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            batch = idx[start:start + batch_size]
            x = imgs_t[batch].to(device, non_blocking=True)
            m = masks_t[batch].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, attention = model(x, m)
            probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
            attentions.append(attention.float().cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(attentions)


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


def build_optimizer(model, head_lr=1e-3, backbone_lr=8e-6, weight_decay=0.02):
    """AdamW with a conservative rate for the partially unfrozen encoder.

    The target-query head is initialized from scratch, whereas the encoder is
    pretrained and only its final blocks are trainable. Updating both at the
    head rate destroys the representation Family A is intended to transfer.
    """
    encoder_ids = {id(p) for p in getattr(model, 'encoder', ()).parameters()} \
        if hasattr(model, 'encoder') else set()
    encoder = [p for p in model.parameters() if p.requires_grad and id(p) in encoder_ids]
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    groups = []
    if head:
        groups.append({'params': head, 'lr': head_lr, 'group_name': 'head'})
    if encoder:
        groups.append({'params': encoder, 'lr': backbone_lr, 'group_name': 'encoder'})
    if not groups:
        raise ValueError('Model has no trainable parameters')
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_one_fold(model_fn, imgs, masks, y_expert, expert_mask, y_aux, aux_mask, aux_weight,
                    train_idx, val_idx, epochs, lr, seed, checkpoint_path, batch_size=8, resume=True,
                    patience=None, device='auto', amp=None, prediction_batch_size=None,
                    backbone_lr=8e-6, weight_decay=0.02, expert_fraction=0.10):
    """Train a fixed, predeclared epoch count and score the outer fold once.

    Outer-fold expert labels are not inspected during training or used for
    checkpoint selection. `patience` is rejected because early stopping on the
    outer fold would make the resulting OOF estimate optimistic.
    """
    if epochs < 1:
        raise ValueError('epochs must be at least 1')
    if patience is not None:
        raise ValueError('Outer-fold early stopping is prohibited; use fixed epochs')
    device = resolve_device(device)
    use_amp = device.type == 'cuda' if amp is None else bool(amp and device.type == 'cuda')
    prediction_batch_size = prediction_batch_size or batch_size
    set_seed(seed)
    model = model_fn().to(device)
    optimizer = build_optimizer(model, head_lr=lr, backbone_lr=backbone_lr,
                                weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    training_config = {'head_lr': lr, 'backbone_lr': backbone_lr,
                       'weight_decay': weight_decay, 'batch_size': batch_size,
                       'expert_fraction': expert_fraction}
    start_epoch = 0
    stopping_reason = 'completed_predeclared_fixed_epochs'
    checkpoint_path = Path(checkpoint_path)
    history = []
    batch_rng = np.random.default_rng(seed + 1)
    if resume and checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state.get('training_config') != training_config:
            raise ValueError('Checkpoint training configuration differs from this run; '
                             'use a fresh output directory')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if 'scaler' in state:
            scaler.load_state_dict(state['scaler'])
        restore_rng_state(state['rng_state'])
        start_epoch = state['epoch'] + 1
        history = state.get('history', [])
        if 'batch_rng' in state['rng_state']:
            batch_rng.bit_generator.state = state['rng_state']['batch_rng']

    imgs_t = torch.from_numpy(imgs)
    masks_t = torch.from_numpy(masks).float()
    y_expert_t = torch.from_numpy(np.nan_to_num(y_expert, nan=0.0)).float()
    expert_mask_t = torch.from_numpy(expert_mask)
    y_aux_t = torch.from_numpy(np.nan_to_num(y_aux, nan=0.0)).float()
    aux_mask_t = torch.from_numpy(aux_mask)
    aux_weight_t = torch.from_numpy(aux_weight).float()

    train_idx = np.asarray(train_idx, dtype=int)
    if not (expert_mask[train_idx].any() or
            (aux_mask[train_idx] & (aux_weight[None, :] > 0)).any()):
        raise ValueError('No supervised rows remain in this training fold')

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses = []
        epoch_idx = _epoch_training_indices(
            train_idx, expert_mask, aux_mask, aux_weight, expert_fraction, batch_rng)
        expert_instances = int(expert_mask[epoch_idx].any(axis=1).sum())
        for batch in _batches(epoch_idx, batch_size, batch_rng):
            optimizer.zero_grad()
            x = imgs_t[batch].to(device, non_blocking=True)
            m = masks_t[batch].to(device, non_blocking=True)
            ye = y_expert_t[batch].to(device, non_blocking=True)
            em = expert_mask_t[batch].to(device, non_blocking=True)
            ya = y_aux_t[batch].to(device, non_blocking=True)
            am = aux_mask_t[batch].to(device, non_blocking=True)
            aw = aux_weight_t.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, _ = model(x, m)
                loss, parts = combined_loss(logits, ye, em, ya, am, aw)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at epoch {epoch}: {parts}')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(parts)

        history.append({'epoch': epoch,
                        'mean_expert_loss': float(np.mean([p['expert'] for p in epoch_losses])),
                        'mean_aux_loss': float(np.mean([p['aux'] for p in epoch_losses])),
                        'training_row_instances': len(epoch_idx),
                        'expert_row_instances': expert_instances,
                        'expert_row_fraction': expert_instances / len(epoch_idx),
                        'outer_val_macro_auc': None,
                        'selection_note': 'outer labels not inspected during training'})
        atomic_save(checkpoint_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                      'scaler': scaler.state_dict(), 'epoch': epoch,
                                      'rng_state': rng_state(batch_rng), 'seed': seed,
                                      'history': history, 'selection': 'fixed_final_epoch',
                                      'training_config': training_config})

    probs, attention = _predict(model, imgs_t, masks_t, val_idx, device,
                                prediction_batch_size, use_amp)
    selected_epoch = epochs - 1
    outer_val_auc = _val_macro_auc(probs, y_expert, expert_mask, val_idx)
    return probs, history, stopping_reason, attention, selected_epoch, outer_val_auc


def run_arm(arm_name, cache_dir, ids, gold_ids, gold_labels, report_source, policy, out_dir,
            model_fn, k=5, epochs=3, lr=1e-3, seed=1400, batch_size=8, resume=True, fold_seed=1400,
            patience=None, device='auto', amp=None, prediction_batch_size=None,
            backbone_lr=8e-6, weight_decay=0.02, crossfit_policy=False,
            policy_bootstrap=1000, policy_seed=1400, folds=None,
            expert_fraction=0.10):
    """Trains one arm ('baseline' has an all-zero policy; 'auxiliary' uses a
    transfer-gated policy) across k grouped folds and writes an OOF prediction
    file plus a receipt. Only studies in `gold_ids` ever contribute an expert-BCE
    term or an OOF row scored against ground truth; report-only pool studies
    still participate in training (auxiliary term only) but are not evaluated."""
    from cache import load_batch

    start = time.time()
    out_dir = Path(out_dir)
    resolved_device = resolve_device(device)
    use_amp = resolved_device.type == 'cuda' if amp is None else bool(amp and resolved_device.type == 'cuda')
    imgs, masks = load_batch(cache_dir, ids)

    fold_info = assign_folds(ids, k=k, seed=fold_seed, priority_ids=gold_ids)
    fold_of = np.array([fold_info['fold_assignment'][u] for u in ids])
    selected_folds = list(range(k)) if folds is None else sorted(set(map(int, folds)))
    if not selected_folds or any(fold < 0 or fold >= k for fold in selected_folds):
        raise ValueError(f'folds must be a non-empty subset of 0..{k - 1}')

    oof = np.full((len(ids), len(TARGETS)), np.nan)
    oof_covered = np.zeros(len(ids), dtype=bool)
    fold_histories = {}
    fold_policies = {}
    fold_policy_audits = {}
    checkpoints = {}
    for fold in selected_folds:
        val_idx = np.flatnonzero(fold_of == fold)
        train_idx = np.flatnonzero(fold_of != fold)
        if len(val_idx) == 0 or len(train_idx) == 0:
            raise ValueError(f'fold {fold} has an empty split; reduce k or add studies')
        fold_policy = policy
        if crossfit_policy:
            fold_policy, fold_audit = crossfit_transfer_policy(
                ids, train_idx, gold_labels, report_source,
                bootstrap=policy_bootstrap, seed=policy_seed + fold * 100)
            fold_policy_audits[str(fold)] = fold_audit
        fold_policies[str(fold)] = fold_policy
        y_expert, expert_mask, y_aux, aux_mask, aux_weight = build_supervision(
            ids, gold_labels, report_source, fold_policy)
        ckpt = out_dir / 'checkpoints' / f'{arm_name}_fold{fold}.pt'
        probs, history, reason, _, selected_epoch, outer_val_auc = train_one_fold(
            model_fn, imgs, masks, y_expert, expert_mask, y_aux, aux_mask, aux_weight,
            train_idx, val_idx, epochs, lr, seed, ckpt, batch_size, resume, patience,
            resolved_device.type, use_amp, prediction_batch_size, backbone_lr, weight_decay,
            expert_fraction)
        oof[val_idx] = probs
        oof_covered[val_idx] = True
        fold_histories[fold] = history
        checkpoints[str(fold)] = {
            'path': str(ckpt), 'sha256': digest_file(ckpt), 'stopping_reason': reason,
            'selected_epoch': selected_epoch, 'outer_val_macro_auc_after_selection': outer_val_auc,
            'selection': 'fixed final epoch; outer labels were not used for selection',
            'epochs_run': len(history), 'epochs_requested': epochs,
        }
        if resolved_device.type == 'cuda':
            torch.cuda.empty_cache()

    common_receipt = {
        'arm': arm_name, 'git_commit': git_commit(), 'seed': seed, 'fold_seed': fold_seed,
        'k': k, 'selected_folds': selected_folds,
        'fold_assignment_sha256': hashlib.sha256(json.dumps(
            fold_info['fold_assignment'], sort_keys=True).encode()).hexdigest(),
        'epochs_requested': epochs, 'patience': patience,
        'head_lr': lr, 'backbone_lr': backbone_lr, 'weight_decay': weight_decay,
        'batch_size': batch_size, 'expert_fraction': expert_fraction,
        'n_studies_total': len(ids), 'n_studies_scored': len(gold_ids),
        'policy': policy, 'policy_selection': ('cross-fitted on training-fold expert cases'
                                              if crossfit_policy else 'fixed predeclared policy'),
        'policy_bootstrap': policy_bootstrap if crossfit_policy else None,
        'policy_seed': policy_seed if crossfit_policy else None,
        'fold_policies': fold_policies, 'fold_policy_audits': fold_policy_audits,
        'fold_grouping_caveat': fold_info['grouping_caveat'],
        'checkpoints': checkpoints, 'fold_histories': fold_histories,
        'epoch_selection': ('fixed final epoch declared before training; outer-fold expert labels '
                            'were scored once afterward and never selected a checkpoint'),
        'runtime_seconds': time.time() - start, 'device': str(resolved_device),
        'amp_fp16': use_amp,
    }

    if not oof_covered.all():
        partial_frame = pd.DataFrame(oof[oof_covered], columns=TARGETS)
        partial_frame.insert(0, UID, np.asarray(ids)[oof_covered])
        partial_path = out_dir / f'{arm_name}_partial_oof.csv'
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        with partial_path.open('x', encoding='utf-8') as f:
            partial_frame.to_csv(f, index=False)
        receipt = {
            **common_receipt,
            'status': 'COMPLETE_FOLD_SHARD_NOT_COMPLETE_ARM',
            'partial_oof_path': str(partial_path),
            'partial_oof_sha256': digest_file(partial_path),
            'partial_oof_rows': len(partial_frame),
        }
        write_receipt(out_dir / f'{arm_name}_partial_receipt.json', receipt)
        return partial_frame, receipt

    all_frame = pd.DataFrame(oof, columns=TARGETS)
    all_frame.insert(0, UID, ids)
    all_oof_path = out_dir / f'{arm_name}_all_oof.csv'
    all_oof_path.parent.mkdir(parents=True, exist_ok=True)
    with all_oof_path.open('x', encoding='utf-8') as f:
        all_frame.to_csv(f, index=False)

    gold_only = [i for i, u in enumerate(ids) if u in set(gold_ids)]
    frame = pd.DataFrame(oof[gold_only], columns=TARGETS)
    frame.insert(0, UID, [ids[i] for i in gold_only])
    oof_path = out_dir / f'{arm_name}_oof.csv'
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    with oof_path.open('x', encoding='utf-8') as f:
        frame.to_csv(f, index=False)

    receipt = {
        **common_receipt,
        'status': 'COMPLETE_ARM',
        'n_studies_scored': len(gold_only),
        'gold_oof_sha256': digest_file(oof_path),
        'all_oof_sha256': digest_file(all_oof_path),
        'all_oof_scope': ('Includes cross-fold predictions for report-only pool studies so the '
                          'incremental frozen-ensemble blend can be evaluated at n≈4,349. Weak '
                          'labels remain development evidence, not expert truth.'),
    }
    write_receipt(out_dir / f'{arm_name}_receipt.json', receipt)
    return frame, receipt
