"""Clean-reference nested grouped CV on frozen image features.

No V13 outputs/weights are inputs. Protocol is fixed before this pilot runs.
PatientID grouping is provisional, and the historical gold set is development.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
FOCUS = [3, 5, 6, 8]
PROTOCOL = {
    'name': 'clean-dinov2-small-regional-pilot-v1',
    'encoder': 'facebook/dinov2-small',
    'encoder_revision': 'ed25f3a31f01632728cabb09d1542f84ab7b0056',
    'encoder_frozen': True, 'expert_labels_only': True,
    'outer_folds': 5, 'inner_folds': 3, 'seed': 1401,
    'regularization_grid': [1.0, 0.1, 0.01],
    'blend_grid': [0.0, 0.25, 0.5, 1.0],
    'inner_auc_min_gain': 0.01, 'inner_bce_max_regression': 0.01,
    'fixed_blend_weight': 0.25, 'focus_targets': [TARGETS[j] for j in FOCUS],
    'image_size': 336, 'slices_per_series': 12, 'max_series_per_plane': 2,
    'slice_quantiles': [0.1, 0.9], 'planes': ['Sagittal', 'Coronal', 'Axial'],
    'regional_features': 'All four image-coordinate patch quadrants, per plane; not anatomical segmentation',
    'primary': 'inner_selected_specialist', 'secondary': ['fixed_25pct_specialist', 'regional_only_focus'],
    'patient_identity_semantics_verified': False, 'exact_v13_comparator': False,
    'untouched_confirmation': False, 'auto_promotion': False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def splits(groups, folds, seed):
    groups = np.asarray(groups, str)
    if groups.ndim != 1 or any(not s.strip() or s.lower() in {'nan', 'none'} for s in groups):
        raise ValueError('Nonmissing group IDs required')
    unique = sorted(set(groups), key=lambda g: hashlib.sha256(f'{seed}|{g}'.encode()).hexdigest())
    if folds < 2 or len(unique) < folds:
        raise ValueError('Insufficient distinct groups')
    # Label-independent balanced group allocation; repeated examinations stay together.
    buckets, counts = {}, np.zeros(folds, int)
    for g in unique:
        k = int(np.argmin(counts))
        buckets[g] = k
        counts[k] += int(np.sum(groups == g))
    assignment = np.array([buckets[g] for g in groups])
    return [(np.flatnonzero(assignment != k), np.flatnonzero(assignment == k)) for k in range(folds)]


def bce(y, pred):
    p = np.clip(pred, 1e-7, 1 - 1e-7)
    return float(np.mean(-y * np.log(p) - (1-y) * np.log1p(-p)))


def fit_head(x, y, regularization):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x) or not x.shape[1]:
        raise ValueError('Invalid feature/label shapes')
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Finite features and expert binary labels required')
    lam = np.broadcast_to(np.asarray(regularization, float), (y.shape[1],))
    if not np.isfinite(lam).all() or (lam <= 0).any():
        raise ValueError('Positive finite regularization required')
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-6, scale, 1.)
    z = (x - mean) / scale / np.sqrt(x.shape[1])
    # Exact train-row-space representation, NOT truncated PCA. Any optimal
    # L2-regularized linear head lies in this span. Scaling is training-only.
    eig, u = np.linalg.eigh(z @ z.T)
    keep = eig > max(float(eig.max()), 1.) * 1e-10
    transform = z.T @ (u[:, keep] / np.sqrt(eig[keep]))
    reduced = z @ transform
    weight, intercept = np.zeros((x.shape[1], y.shape[1])), np.zeros(y.shape[1])
    supported = np.zeros(y.shape[1], bool)
    for j in range(y.shape[1]):
        target = y[:, j]
        prior = (target.sum() + 1) / (len(target) + 2)
        intercept[j] = np.log(prior / (1-prior))
        if min(target.sum(), len(target)-target.sum()) < 2:
            continue
        def objective(theta):
            logits = reduced @ theta[:-1] + theta[-1]
            error = (expit(logits) - target) / len(target)
            loss = np.mean(np.logaddexp(0, logits) - target * logits) + lam[j] * (theta[:-1] @ theta[:-1]) / 2
            grad = np.r_[reduced.T @ error + lam[j] * theta[:-1], error.sum()]
            return loss, grad
        initial = np.r_[np.zeros(reduced.shape[1]), intercept[j]]
        result = minimize(objective, initial, method='L-BFGS-B', jac=True,
                          options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-7})
        if not result.success:
            raise RuntimeError(f'Head optimization failed: {result.message}')
        weight[:, j], intercept[j], supported[j] = transform @ result.x[:-1], result.x[-1], True
    return {'mean': mean, 'scale': scale, 'weight': weight, 'intercept': intercept,
            'supported': supported, 'regularization': lam.copy()}


def predict_head(model, x):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != len(model['mean']) or not np.isfinite(x).all():
        raise ValueError('Feature contract mismatch')
    return expit(((x-model['mean']) / model['scale'] / np.sqrt(x.shape[1])) @ model['weight'] + model['intercept'])


def tune_heads(x, y, groups, seed, protocol=PROTOCOL):
    grid = protocol['regularization_grid']
    pred = np.empty((len(grid), len(y), y.shape[1]))
    fold_log = []
    for k, (tr, va) in enumerate(splits(groups, protocol['inner_folds'], seed)):
        assert not set(groups[tr]) & set(groups[va])
        fold_log.append({'fold': k, 'train_groups': sorted(set(groups[tr])), 'valid_groups': sorted(set(groups[va]))})
        for i, lam in enumerate(grid):
            model = fit_head(x[tr], y[tr], lam)
            pred[i, va] = predict_head(model, x[va])
    # AUC for model ranking, log loss only for tie break; ties favor stronger L2.
    choices, selected, scores = [], np.empty_like(y, dtype=float), []
    for j in range(y.shape[1]):
        values = []
        for i, lam in enumerate(grid):
            auc = float(roc_auc_score(y[:, j], pred[i, :, j])) if len(np.unique(y[:, j])) == 2 else 0.5
            values.append({'regularization': lam, 'auc': auc, 'bce': bce(y[:, j], pred[i, :, j])})
        best = max(range(len(grid)), key=lambda i: (values[i]['auc'], -values[i]['bce'], grid[i]))
        choices.append(grid[best])
        selected[:, j] = pred[best, :, j]
        scores.append(values)
    return np.array(choices), selected, {'inner_folds': fold_log, 'grid_scores': scores}


def train_outer(xbase, xregion, y, groups, train, valid, seed, protocol=PROTOCOL):
    # No y[valid] access in this function: even alpha and scale fitting stay inside train.
    if set(groups[train]) & set(groups[valid]):
        raise ValueError('Group overlap')
    base_lam, inner_base, base_log = tune_heads(xbase[train], y[train], groups[train], seed, protocol)
    reg_lam, inner_reg, reg_log = tune_heads(xregion[train], y[train][:, FOCUS], groups[train], seed, protocol)
    base_model = fit_head(xbase[train], y[train], base_lam)
    reg_model = fit_head(xregion[train], y[train][:, FOCUS], reg_lam)
    base, regional = predict_head(base_model, xbase[valid]), predict_head(reg_model, xregion[valid])
    selected, fixed, raw = base.copy(), base.copy(), base.copy()
    alpha, decisions = [], []
    for jj, j in enumerate(FOCUS):
        t, b, r = y[train, j], inner_base[:, j], inner_reg[:, jj]
        base_auc = float(roc_auc_score(t, b)) if len(np.unique(t)) == 2 else 0.5
        candidates = []
        for a in protocol['blend_grid']:
            mixed = (1-a)*b + a*r
            score = float(roc_auc_score(t, mixed)) if len(np.unique(t)) == 2 else 0.5
            eligible = a == 0 or (score-base_auc >= protocol['inner_auc_min_gain'] and
                                  bce(t, mixed)-bce(t, b) <= protocol['inner_bce_max_regression'])
            candidates.append({'alpha': a, 'auc': score, 'bce': bce(t, mixed), 'eligible': eligible})
        choice = max((c for c in candidates if c['eligible']), key=lambda c: (c['auc'], -c['alpha']))['alpha']
        # Unsupported regional heads cannot replace a supported baseline.
        if not reg_model['supported'][jj]:
            choice = 0.
        alpha.append(choice)
        selected[:, j] = (1-choice)*base[:, j] + choice*regional[:, jj]
        fixed[:, j] = (1-protocol['fixed_blend_weight'])*base[:, j] + protocol['fixed_blend_weight']*regional[:, jj]
        raw[:, j] = regional[:, jj]
        decisions.append({'target': TARGETS[j], 'selected_alpha': choice, 'candidates': candidates})
    log = {'train_groups': sorted(set(groups[train])), 'valid_groups': sorted(set(groups[valid])),
           'base_regularization': base_lam.tolist(), 'regional_regularization': reg_lam.tolist(),
           'base_inner': base_log, 'regional_inner': reg_log, 'blend_decisions': decisions}
    return {'reference': base, 'inner_selected_specialist': selected,
            'fixed_25pct_specialist': fixed, 'regional_only_focus': raw}, base_model, reg_model, np.array(alpha), log


def evaluate_features(features, output, protocol=PROTOCOL):
    from diagnostics import compare
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(features, allow_pickle=False) as data:
        ids, groups = data['StudyInstanceUID'].astype(str), data['GroupID'].astype(str)
        xb, xr, y = data['base'].astype(float), data['regional'].astype(float), data['labels'].astype(float)
    if len(set(ids)) != len(ids) or len(ids) != len(groups) or y.shape != (len(ids), 12):
        raise ValueError('Invalid feature archive alignment')
    if len(xb) != len(y) or len(xr) != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Feature/label alignment or expert label contract failed')
    names = ['reference', 'inner_selected_specialist', 'fixed_25pct_specialist', 'regional_only_focus']
    oof = {name: np.full_like(y, np.nan) for name in names}
    assignment, logs = np.full(len(y), -1), []
    for k, (tr, va) in enumerate(splits(groups, protocol['outer_folds'], protocol['seed'])):
        pred, bm, rm, alpha, log = train_outer(xb, xr, y, groups, tr, va, protocol['seed']+k+1, protocol)
        for name in names:
            oof[name][va] = pred[name]
        assignment[va] = k
        np.savez_compressed(output/f'fold_{k}_reference.npz', **bm)
        np.savez_compressed(output/f'fold_{k}_regional.npz', **rm, alpha=alpha, target_indices=np.array(FOCUS))
        log.update(fold=k, train_ids=ids[tr].tolist(), valid_ids=ids[va].tolist())
        logs.append(log)
        print(f'CLEAN HEAD TRAINING outer fold {k+1}/{protocol["outer_folds"]} complete; train={len(tr)} valid={len(va)} alpha={alpha.tolist()}', flush=True)
    if any(not np.isfinite(p).all() for p in oof.values()) or (assignment < 0).any():
        raise ValueError('Incomplete OOF predictions')
    def frame(values):
        f = pd.DataFrame(values, columns=TARGETS)
        f.insert(0, 'StudyInstanceUID', ids)
        return f
    label_frame = frame(y)
    label_frame.to_csv(output/'gold_labels.csv', index=False)
    pd.DataFrame({'StudyInstanceUID': ids, 'GroupID': groups, 'fold': assignment}).to_csv(output/'folds.csv', index=False)
    for name, pred in oof.items():
        frame(pred).to_csv(output/f'{name}_oof.csv', index=False)
    results = {}
    for name in names[1:]:
        report = compare(label_frame, frame(oof['reference']), frame(oof[name]), groups=groups, bootstrap=2000, seed=1401)
        report['calibration'] = {kind: {'bce': bce(y, oof[n]), 'brier': float(np.mean((oof[n]-y)**2))}
                                 for kind, n in [('reference', 'reference'), ('candidate', name)]}
        report['ci_caveat'] += ' Fixed OOF prediction resampling does not refit models; training/fold uncertainty is omitted. Historical gold and architecture selection make this exploratory.'
        report['unchanged_other_eight_targets'] = bool(np.array_equal(oof[name][:, [j for j in range(12) if j not in FOCUS]], oof['reference'][:, [j for j in range(12) if j not in FOCUS]]))
        report['changed_values'] = int(np.count_nonzero(oof[name] != oof['reference']))
        fold_auc = []
        for k in range(protocol['outer_folds']):
            idx = assignment == k
            valid_targets = [j for j in range(12) if len(np.unique(y[idx, j])) == 2]
            fold_auc.append({'fold': k, 'studies': int(idx.sum()), 'scorable_targets': [TARGETS[j] for j in valid_targets],
                             'target_deltas': {TARGETS[j]: float(roc_auc_score(y[idx, j], oof[name][idx, j])-roc_auc_score(y[idx, j], oof['reference'][idx, j])) for j in valid_targets},
                             'not_a_complete_macro_auc': len(valid_targets) != 12})
        report['within_fold_diagnostics'] = fold_auc
        results[name] = report
    receipt = {'status': 'COMPLETED_PROVISIONAL_GROUPED_RESEARCH', 'protocol': protocol,
               'studies': len(ids), 'groups': len(set(groups)), 'new_head_training_performed': True,
               'saved_reference_heads': protocol['outer_folds'], 'saved_regional_heads': protocol['outer_folds'],
               'outer_label_isolation': 'Outer validation labels were not passed to fitting/selection; feature encoder fixed and frozen',
               'feature_sha256': sha(features), 'folds': logs, 'results': results,
               'patient_independence_confirmed': False, 'v13_improvement_verified': False,
               'leaderboard_prediction': None, 'leaderboard_submission': False,
               'limitations': ['58 historical expert-labeled studies only; no weak report labels or V13 models used',
                               'PatientID tags are anonymized singletons; repeat-patient linkage unverified',
                               'Source encoder is public external pretraining; exact image-level membership is not independently audited',
                               'Pooled OOF AUC may include cross-fold calibration differences; within-fold diagnostics also provided',
                               'The three fixed candidate definitions must all be reported; no post-hoc winner deployment']}
    receipt['artifact_sha256'] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    (output/'training_results.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({name: {k: r[k] for k in ['baseline_macro_auc', 'candidate_macro_auc', 'macro_delta', 'macro_delta_ci95', 'changed_values']} for name, r in results.items()}, indent=2), flush=True)
    return receipt
