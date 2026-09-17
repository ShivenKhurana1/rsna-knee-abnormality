import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .common import (aligned, atomic_csv, atomic_json, check_ids, fingerprint,
                     read_studies, schema, sha, write_predictions)
from .data import CachedStudies, DicomStudies, build_cache, collate, to_device, transform_contract
from .model import KneeModel, masked_loss
from .validation import load_groups, make_folds, metrics, supervision


def code_hash():
    names = ('__init__.py', '__main__.py', 'common.py', 'data.py', 'model.py',
             'validation.py', 'runner.py', 'evaluation.py')
    return fingerprint({name: (Path(__file__).parent/name).read_text(encoding='utf-8') for name in names})


def device_for(request):
    if request not in ('cpu', 'cuda', 'auto'):
        raise ValueError('device must be cpu, cuda, or auto')
    if request == 'cpu':
        return torch.device('cpu')
    try:
        probe = (torch.ones(1, device='cuda')*2).cpu().item()
        if probe != 2:
            raise RuntimeError('CUDA arithmetic failed')
        return torch.device('cuda')
    except Exception as exc:
        if request == 'cuda':
            raise RuntimeError('Required CUDA device failed its arithmetic probe') from exc
        return torch.device('cpu')


def prepare(root, work, cfg, report_path=None, groups_path=None):
    root, work = Path(root), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if cfg['encoder'] != 'tiny':
        assets = Path(cfg['encoder_path'])
        if not cfg['encoder_path'] or not assets.exists():
            raise ValueError('Set encoder_path to your attached pretrained encoder before preparing the cache')
        if cfg['encoder'] == 'dinov2' and not (assets/'config.json').is_file():
            if not assets.is_file() or not (Path(cfg['dinov2_repo'])/'hubconf.py').is_file():
                raise ValueError('Native DINOv2 .pth requires dinov2_repo with local hubconf.py')
    contract = schema(root, cfg['id_col'])
    frame = read_studies(root, 'train', contract)
    ids = check_ids(frame, contract['id_col'])
    y = frame[contract['targets']].to_numpy(float)
    if not (np.isnan(y) | (y == 0) | (y == 1)).all():
        raise ValueError('train.csv expert targets must be binary or missing')
    if report_path is not None:
        report_frame = pd.read_csv(report_path, dtype={cfg['id_col']: str})
        report = aligned(report_frame, ids, contract)
    else:
        report = np.full_like(y, np.nan)
        if cfg['auxiliary'] != 'none':
            raise ValueError('Provide --report-labels for auxiliary training or set auxiliary=none')
    groups, caveat = load_groups(groups_path, ids, cfg['id_col'])
    fold, audit = make_folds(ids, y, groups, cfg['folds'], cfg['fold_seed'])
    signature = {'config': cfg, 'schema': contract, 'train_csv': sha(root/'train.csv'),
                 'reports': sha(report_path) if report_path else None,
                 'groups': sha(groups_path) if groups_path else None, 'code': code_hash()}
    if (work/'preparation.json').exists():
        old = json.loads((work/'preparation.json').read_text())
        if old['signature'] != signature:
            raise ValueError('Preparation differs from existing work directory; use a new work directory')
    atomic_json(work/'config.json', cfg)
    atomic_json(work/'schema.json', contract)
    atomic_csv(work/'expert_labels.csv', frame[[cfg['id_col']]+contract['targets']])
    reports = pd.DataFrame(report, columns=contract['targets'])
    reports.insert(0, cfg['id_col'], ids)
    atomic_csv(work/'report_labels.csv', reports)
    atomic_csv(work/'folds.csv', pd.DataFrame({cfg['id_col']: ids, 'group_id': groups, 'fold': fold}))
    audit['grouping_caveat'] = caveat
    atomic_json(work/'fold_audit.json', audit)
    cache = build_cache(root, 'train', ids, cfg, work/'cache')
    receipt = {'signature': signature, 'fold_audit': audit, 'studies': len(ids),
               'missing_images': cache['missing_studies'],
               'cache_manifest_sha256': sha(work/'cache/manifest.json')}
    atomic_json(work/'preparation.json', receipt)
    return receipt


def _atomic_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    os.replace(temp, path)


def finite_optimizer_step(model, optimizer, scaler, max_norm):
    """Let GradScaler reject overflow before clipping; never update invalid grads."""
    scaler.unscale_(optimizer)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite = all(torch.isfinite(g).all().item() for g in grads)
    if not finite:
        if not scaler.is_enabled():
            raise FloatingPointError('Nonfinite gradients in full precision')
        # unscale_ recorded the overflow. step skips the optimizer, update lowers
        # the scale. The caller retries the SAME batch, preserving its exposure.
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return False
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return True


def _predict(model, dataset, indices, device, batch_size, prior):
    predictions, available = [], []
    model.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, collate_fn=collate)
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                logits, has_images = model(batch)
            p = logits.float().sigmoid().cpu().numpy()
            present = has_images.cpu().numpy()
            p[~present] = prior
            predictions.append(p)
            available.extend(present.tolist())
    return np.concatenate(predictions), available


def train(work, fold, seed=None, arm=None):
    work = Path(work)
    start = time.monotonic()
    cfg = json.loads((work/'config.json').read_text())
    if seed is not None:
        cfg['seed'] = seed
    if arm is not None:
        cfg['auxiliary'] = arm
    if cfg['auxiliary'] not in ('none', 'raw', 'platt'):
        raise ValueError('Unknown supervision arm')
    contract = json.loads((work/'schema.json').read_text())
    split = pd.read_csv(work/'folds.csv', dtype={cfg['id_col']: str, 'group_id': str})
    ids, groups = check_ids(split, cfg['id_col']), split.group_id.to_numpy()
    if fold not in set(split.fold):
        raise ValueError('Requested fold is absent')
    y = aligned(pd.read_csv(work/'expert_labels.csv', dtype={cfg['id_col']: str}), ids, contract)
    report = aligned(pd.read_csv(work/'report_labels.csv', dtype={cfg['id_col']: str}), ids, contract)
    tr, va = np.flatnonzero(split.fold.to_numpy() != fold), np.flatnonzero(split.fold.to_numpy() == fold)
    if set(groups[tr]) & set(groups[va]):
        raise ValueError('Group leakage in supplied folds')
    values, aux_mask, weights, policies = supervision(y, report, groups, tr, cfg, contract['targets'], ids)
    dataset = CachedStudies(work/'cache', ids, training=True, dropout=cfg['series_dropout'])
    if dataset.manifest['contract'] != transform_contract(cfg):
        raise ValueError('Cache preprocessing/runtime contract mismatch')
    present = np.array([dataset.manifest['entries'][u]['series'] > 0 for u in ids])
    expert_mask = np.isfinite(y) & present[:, None]
    aux_mask &= present[:, None]
    expert_rows = tr[expert_mask[tr].any(1)]
    aux_rows = tr[(~expert_mask[tr].any(1)) & aux_mask[tr].any(1)]
    if not len(expert_rows) and not len(aux_rows):
        raise ValueError('No usable supervised training studies')
    prior = (np.nansum(y[tr], axis=0)+1)/(np.isfinite(y[tr]).sum(0)+2)
    out = work/'runs'/f'seed{cfg["seed"]}'/cfg['auxiliary']/f'fold{fold}'
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out/'checkpoint.pt'
    binding = {'config': cfg, 'schema': contract, 'fold': int(fold), 'code': code_hash(),
               'folds': sha(work/'folds.csv'), 'expert': sha(work/'expert_labels.csv'),
               'report': sha(work/'report_labels.csv'), 'cache': sha(work/'cache/manifest.json')}
    signature = fingerprint(binding)
    device = device_for(cfg['device'])
    torch.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True) if checkpoint_path.exists() else None
    if saved and saved['signature'] != signature:
        raise ValueError('Checkpoint is bound to different data/configuration/code')
    model = KneeModel(cfg, len(contract['targets']), initialize=saved is None,
                      encoder_spec=saved['encoder_spec'] if saved else None).to(device)
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if p.requires_grad and id(p) in encoder_ids], 'lr': cfg['encoder_lr']},
        {'params': [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids], 'lr': cfg['head_lr']}],
        weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    scaler = torch.amp.GradScaler(device.type, init_scale=1024., enabled=device.type == 'cuda')
    history, start_epoch = [], 0
    if saved:
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        torch.set_rng_state(saved['rng'])
        if device.type == 'cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_epoch, history = saved['epoch'], saved['history']
    y_t = torch.tensor(y, dtype=torch.float32)
    aux_t = torch.tensor(values)
    em_t, am_t = torch.tensor(expert_mask), torch.tensor(aux_mask)
    weight_t = torch.tensor(weights, device=device)
    id_index = {uid: i for i, uid in enumerate(ids)}
    for epoch in range(start_epoch, cfg['epochs']):
        rng = np.random.default_rng(cfg['seed']+epoch)
        # Keep every expert study and every eligible report-only study; supplement expert exposure.
        extra = 0 if not len(expert_rows) else max(0, int(np.ceil(
            len(aux_rows)*cfg['expert_fraction']/(1-cfg['expert_fraction'])))-len(expert_rows))
        indices = np.concatenate([expert_rows, aux_rows, rng.choice(expert_rows, extra, replace=True)
                                  if extra else np.array([], int)])
        rng.shuffle(indices)
        loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=cfg['batch_size'],
                            shuffle=False, collate_fn=collate)
        model.train()
        running, steps, overflow_retries = 0., 0, 0
        for batch in loader:
            index = torch.tensor([id_index[u] for u in batch['ids']])
            batch = to_device(batch, device)
            for attempt in range(12):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    logits, has_images = model(batch)
                expert_loss = masked_loss(logits, y_t[index].to(device),
                                          em_t[index].to(device) & has_images[:, None], cfg)
                auxiliary_loss = masked_loss(logits, aux_t[index].to(device),
                                             am_t[index].to(device) & has_images[:, None], cfg,
                                             auxiliary=True, weights=weight_t)
                loss = expert_loss+auxiliary_loss
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite training loss for {batch["ids"]}')
                scaler.scale(loss).backward()
                if finite_optimizer_step(model, optimizer, scaler, cfg['grad_clip']):
                    break
                overflow_retries += 1
                print(f'AMP overflow: retry {attempt+1}/12, scale={scaler.get_scale()}, '
                      f'studies={batch["ids"]}', flush=True)
            else:
                raise FloatingPointError(f'Persistent gradient overflow for {batch["ids"]}')
            running += loss.item()
            steps += 1
            if steps % 100 == 0:
                print(f'Fold {fold}, epoch {epoch+1}/{cfg["epochs"]}, batch {steps}/{len(loader)}, '
                      f'loss={running/steps:.5f}, AMP retries={overflow_retries}', flush=True)
        scheduler.step()
        history.append({'epoch': epoch+1, 'loss': running/max(steps, 1), 'steps': steps,
                        'row_instances': len(indices), 'overflow_retries': overflow_retries})
        state = {'model': model.state_dict(), 'encoder_spec': model.encoder.spec, 'config': cfg,
                 'schema': contract, 'preprocessing': dataset.manifest['contract'],
                 'prior': prior.tolist(), 'signature': signature, 'binding': binding,
                 'train_ids': [ids[i] for i in tr], 'val_ids': [ids[i] for i in va],
                 'train_groups': sorted(set(groups[tr])), 'val_groups': sorted(set(groups[va])),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'scaler': scaler.state_dict(), 'epoch': epoch+1, 'history': history,
                 'rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                 'policies': policies}
        _atomic_checkpoint(checkpoint_path, state)
        print(f'Fold {fold}, {cfg["auxiliary"]}, epoch {epoch+1}/{cfg["epochs"]}: loss={history[-1]["loss"]:.5f}', flush=True)
    dataset.training = False
    p, has_images = _predict(model, dataset, va.tolist(), device, cfg['batch_size'], prior)
    write_predictions(out/'oof.csv', [ids[i] for i in va], p, contract)
    receipt = {'status': 'FOLD_COMPLETE', 'seed': cfg['seed'], 'arm': cfg['auxiliary'], 'fold': int(fold),
               'checkpoint_sha256': sha(checkpoint_path), 'oof_sha256': sha(out/'oof.csv'),
               'signature': signature, 'binding': binding, 'metrics': metrics(y[va], p, contract['targets']),
               'train_ids': [ids[i] for i in tr], 'val_ids': [ids[i] for i in va],
               'fallback_ids': [ids[i] for i, ok in zip(va, has_images) if not ok],
               'epoch_selection': 'fixed final epoch; outer labels used only after training',
               'policies': policies, 'history': history, 'runtime_seconds': time.monotonic()-start}
    atomic_json(out/'receipt.json', receipt)
    return receipt


def merge(work, seed, arm):
    work = Path(work)
    contract = json.loads((work/'schema.json').read_text())
    split = pd.read_csv(work/'folds.csv', dtype={contract['id_col']: str})
    ids = check_ids(split, contract['id_col'])
    parts, checkpoint_hashes, common_binding = [], [], None
    root = work/'runs'/f'seed{seed}'/arm
    for fold in sorted(split.fold.unique()):
        directory = root/f'fold{fold}'
        receipt = json.loads((directory/'receipt.json').read_text())
        if receipt['seed'] != seed or receipt['arm'] != arm or receipt['fold'] != fold:
            raise ValueError('Shard identity mismatch')
        if receipt['binding']['folds'] != sha(work/'folds.csv') or receipt['oof_sha256'] != sha(directory/'oof.csv') or receipt['checkpoint_sha256'] != sha(directory/'checkpoint.pt'):
            raise ValueError('Shard provenance/checksum mismatch')
        binding = {k: v for k, v in receipt['binding'].items() if k != 'fold'}
        if common_binding is not None and binding != common_binding:
            raise ValueError('Shards have different training configurations/data')
        common_binding = binding
        part = pd.read_csv(directory/'oof.csv', dtype={contract['id_col']: str})
        expected = split.loc[split.fold == fold, contract['id_col']].tolist()
        aligned(part, expected, contract, probabilities=True)
        if set(receipt['train_ids']) != set(ids)-set(expected) or set(receipt['val_ids']) != set(expected):
            raise ValueError('Shard held-out provenance mismatch')
        parts.append(part)
        checkpoint_hashes.append(receipt['checkpoint_sha256'])
    p = aligned(pd.concat(parts), ids, contract, probabilities=True)
    write_predictions(root/'oof.csv', ids, p, contract)
    y = aligned(pd.read_csv(work/'expert_labels.csv', dtype={contract['id_col']: str}), ids, contract)
    result = {'status': 'ALL_FOLDS_COMPLETE', 'metrics': metrics(y, p, contract['targets']),
              'oof_sha256': sha(root/'oof.csv'), 'checkpoint_hashes': checkpoint_hashes,
              'cohort': 'expert-label evaluation; unknown cells excluded; no target silently dropped'}
    atomic_json(root/'oof_receipt.json', result)
    return result


def infer(root, checkpoints, output, cache_dir, device='auto', baseline=None, blend_weight=None,
          encoder_code=None, max_seconds=28800):
    output, root = Path(output), Path(root)
    start = time.monotonic()
    if max_seconds <= 0:
        raise ValueError('max_seconds must be positive')
    if baseline is not None and (blend_weight is None or not 0 < blend_weight <= 1):
        raise ValueError('Baseline blending requires explicit --blend-weight in (0,1]')
    if baseline is not None and Path(baseline).resolve() in (
        output.resolve(), output.with_name(output.stem+'_model.csv').resolve()):
        raise ValueError('Baseline input must be separate from output files')
    paths = list(dict.fromkeys(str(Path(p).resolve()) for p in checkpoints))
    if len(paths) != len(checkpoints):
        raise ValueError('Repeated checkpoints would silently change ensemble weights')
    total, manifest, rows, contract, receipts = None, None, None, None, []
    baseline_values = None
    actual_device = device_for(device)
    skipped = []
    for path in paths:
        if receipts and time.monotonic()-start + max(r['runtime_seconds'] for r in receipts)*1.25 > max_seconds:
            skipped = paths[len(receipts):]
            break
        arm_start = time.monotonic()
        state = torch.load(path, map_location='cpu', weights_only=True)
        cfg, stored = dict(state['config']), state['schema']
        if state['epoch'] != cfg['epochs']:
            raise ValueError('Checkpoint has not completed its declared training schedule')
        if state['signature'] != fingerprint(state['binding']) or state['binding']['code'] != code_hash():
            raise ValueError('Checkpoint provenance/runtime code differs; use its matching V16 package')
        if encoder_code:
            cfg['dinov2_repo'] = encoder_code
        live = schema(root, stored['id_col'])
        if set(live['targets']) != set(stored['targets']):
            raise ValueError('Checkpoint target set differs from live schema')
        permutation = [stored['targets'].index(t) for t in live['targets']]
        frame = read_studies(root, 'test', live)
        ids = check_ids(frame, live['id_col'])
        if state['preprocessing'] != transform_contract(cfg):
            raise ValueError('Inference decoder/runtime differs from saved training contract')
        if manifest is None:
            manifest = {'contract': transform_contract(cfg)}
            rows, contract = ids, live
            if baseline is not None:
                baseline_values = aligned(pd.read_csv(baseline, dtype={live['id_col']: str}), ids, live, probabilities=True)
        elif manifest['contract'] != state['preprocessing'] or rows != ids or contract != live:
            raise ValueError('This ensemble requires matching preprocessing/schema/study contracts')
        model = KneeModel(cfg, len(stored['targets']), initialize=False, encoder_spec=state['encoder_spec']).to(actual_device)
        model.load_state_dict(state['model'], strict=True)
        dataset = DicomStudies(root, ids, cfg)
        p, present = _predict(model, dataset, list(range(len(ids))), actual_device, cfg['batch_size'], np.array(state['prior']))
        p = p[:, permutation]
        total = p.astype(np.float64) if total is None else total+p
        receipts.append({'checkpoint': path, 'sha256': sha(path),
                         'fallback_ids': [u for u, ok in zip(ids, present) if not ok], 'encoder': cfg['encoder'],
                         'runtime_seconds': time.monotonic()-arm_start})
        atomic_json(Path(cache_dir)/f'decode_audit_{len(receipts)}.json', dataset.events)
        # Bank only whole-cohort arms. Never mix partial per-study model sets.
        banked = total/len(receipts)
        if baseline is not None:
            banked = (1-blend_weight)*baseline_values + blend_weight*banked
        write_predictions(output, rows, banked, contract)
        del model, state
        if actual_device.type == 'cuda':
            torch.cuda.empty_cache()
    if total is None:
        raise ValueError('No checkpoints supplied')
    p = total / len(receipts)
    raw_path = output.with_name(output.stem+'_model.csv')
    write_predictions(raw_path, rows, p, contract)
    if baseline is not None:
        p = (1-blend_weight)*baseline_values + blend_weight*p
    write_predictions(output, rows, p, contract)
    receipt = {'status': 'INFERENCE_COMPLETE', 'studies': len(rows), 'schema': contract,
               'checkpoints': receipts, 'preprocessing': manifest['contract'],
               'blend': 'fixed arithmetic mean; no inference-batch ranking',
               'baseline': {'sha256': sha(baseline), 'candidate_weight': blend_weight} if baseline else None,
               'submission_sha256': sha(output), 'model_predictions_sha256': sha(raw_path),
               'skipped_checkpoints_for_budget': skipped,
               'runtime_seconds': time.monotonic()-start, 'max_seconds': max_seconds,
               'measured_auc': None, 'code_sha256': code_hash()}
    atomic_json(output.with_suffix('.receipt.json'), receipt)
    return receipt
