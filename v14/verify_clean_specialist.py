"""Independently replay saved clean-head weights and score original artifacts."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from clean_specialist import PROTOCOL, FOCUS, TARGETS, predict_head, splits, sha, train_outer
from diagnostics import compare


def verify(root, refit=False):
    started = time.perf_counter()
    root = Path(root)
    ev, feat = root/'evaluation', root/'features'
    report = json.loads((ev/'training_results.json').read_text())
    run = json.loads((root/'run_receipt.json').read_text())
    protocol = json.loads((root/'predeclared_protocol.json').read_text())
    if protocol != PROTOCOL or report['protocol'] != protocol:
        raise ValueError('Protocol mismatch')
    if run['recipe_sha256'] != sha(root/'predeclared_protocol.json'):
        raise ValueError('Recipe hash mismatch')
    if report['feature_sha256'] != sha(feat/'clean_features.npz'):
        raise ValueError('Feature hash mismatch')
    for name, digest in report['artifact_sha256'].items():
        if Path(name).name != name or sha(ev/name) != digest:
            raise ValueError('Artifact/hash mismatch')
    with np.load(feat/'clean_features.npz', allow_pickle=False) as f:
        ids, groups, y, xb, xr = [f[n] for n in ['StudyInstanceUID','GroupID','labels','base','regional']]
    folds = pd.read_csv(ev/'folds.csv', dtype={'StudyInstanceUID': str, 'GroupID': str})
    pre = pd.read_csv(root/'predeclared_folds.csv', dtype={'StudyInstanceUID': str, 'GroupID': str})
    pd.testing.assert_frame_equal(pre, folds)
    if not np.array_equal(ids, folds.StudyInstanceUID) or not np.array_equal(groups, folds.GroupID):
        raise ValueError('Feature/fold ID alignment mismatch')
    names = ['reference', 'inner_selected_specialist', 'fixed_25pct_specialist', 'regional_only_focus']
    saved = {name: pd.read_csv(ev/f'{name}_oof.csv', dtype={'StudyInstanceUID': str}) for name in names}
    values = {n: f[TARGETS].to_numpy() for n, f in saved.items()}
    if any(not np.array_equal(f.StudyInstanceUID, ids) for f in saved.values()):
        raise ValueError('Saved prediction ID mismatch')
    errors, refit_errors = [], []
    for k, (tr, va) in enumerate(splits(groups, protocol['outer_folds'], protocol['seed'])):
        log = report['folds'][k]
        if set(log['train_ids']) != set(ids[tr]) or set(log['valid_ids']) != set(ids[va]):
            raise ValueError('Training receipt ID mismatch')
        if set(groups[tr]) & set(groups[va]) or not np.all(folds.fold.to_numpy()[va] == k):
            raise ValueError('Outer fold leakage')
        for block in ['base_inner', 'regional_inner']:
            held = []
            for inner in log[block]['inner_folds']:
                it, iv = set(inner['train_groups']), set(inner['valid_groups'])
                if it & iv or it | iv != set(groups[tr]) or (it | iv) & set(groups[va]):
                    raise ValueError('Inner fold leakage')
                held.extend(iv)
            if sorted(held) != sorted(set(groups[tr])):
                raise ValueError('Inner fold coverage mismatch')
        with np.load(ev/f'fold_{k}_reference.npz', allow_pickle=False) as data:
            bm = dict(data)
        with np.load(ev/f'fold_{k}_regional.npz', allow_pickle=False) as data:
            rm = dict(data)
        if not np.array_equal(rm['target_indices'], FOCUS):
            raise ValueError('Regional target contract mismatch')
        b, r = predict_head(bm, xb[va]), predict_head(rm, xr[va])
        replay = {n: b.copy() for n in names}
        for jj, j in enumerate(FOCUS):
            a = rm['alpha'][jj]
            replay['inner_selected_specialist'][:,j] = (1-a)*b[:,j]+a*r[:,jj]
            replay['fixed_25pct_specialist'][:,j] = .75*b[:,j]+.25*r[:,jj]
            replay['regional_only_focus'][:,j] = r[:,jj]
        for n in names:
            error = float(np.max(np.abs(replay[n]-values[n][va])))
            errors.append(error)
            if error > 1e-10:
                raise ValueError(f'Saved weights fail replay: {n}, fold {k}, error {error}')
        if refit:
            pred, *_ = train_outer(xb, xr, y, groups, tr, va, protocol['seed']+k+1, protocol)
            error = max(float(np.max(np.abs(pred[n]-values[n][va]))) for n in names)
            refit_errors.append(error)
            if error > 1e-5:
                raise ValueError(f'Independent CPU refit mismatch in fold {k}: {error}')
        print(f'INDEPENDENT REPLAY fold {k+1}/5 passed', flush=True)
    labels = pd.read_csv(ev/'gold_labels.csv', dtype={'StudyInstanceUID': str})
    if not np.array_equal(labels.StudyInstanceUID, ids) or not np.array_equal(labels[TARGETS], y):
        raise ValueError('Label archive mismatch')
    recomputed = {}
    for name in names[1:]:
        metrics = compare(labels, saved['reference'], saved[name], groups=groups, bootstrap=2000, seed=1401)
        for key in ['baseline_macro_auc','candidate_macro_auc','macro_delta','macro_delta_ci95']:
            if not np.allclose(metrics[key], report['results'][name][key], atol=1e-12, rtol=0):
                raise ValueError('Metric or interval mismatch')
        auc_sklearn = float(roc_auc_score(y, values[name], average='macro'))
        if abs(auc_sklearn-metrics['candidate_macro_auc']) > 1e-12:
            raise ValueError('Independent sklearn AUC mismatch')
        recomputed[name] = {k:metrics[k] for k in ['baseline_macro_auc','candidate_macro_auc','macro_delta','macro_delta_ci95']}
        recomputed[name]['changed_values'] = int(np.count_nonzero(values[name] != values['reference']))
    return {'status': 'INDEPENDENT_ARTIFACT_REPLAY_PASSED', 'studies': len(ids),
            'patient_tag_groups': len(set(groups)), 'artifact_hashes_checked': len(report['artifact_sha256']),
            'max_weight_replay_error': max(errors), 'local_refit_performed': refit,
            'max_local_refit_prediction_error': max(refit_errors) if refit_errors else None,
            'metrics': recomputed, 'elapsed_seconds': time.perf_counter()-started,
            'patient_independence_confirmed': False, 'exact_v13_improvement_verified': False,
            'leaderboard_forecast': None,
            'scope': 'New-head training and provisional group-fold outputs verified; not independent clinical/leaderboard confirmation'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--refit', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        result = verify(args.root, args.refit)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))
