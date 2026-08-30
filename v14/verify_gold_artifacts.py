"""Independent local replay/scoring of private Kaggle gold CSV-table captures.

Expected files in --captures: labels.json, baseline.json, stage3.json,
arm0.json, arm1.json, arm2.json. Each holds {source, heading, rows}; source
must identify the same exact script version and each table must be complete.
This reports exploratory evidence only, never promotes a recipe automatically.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compare_saved_outputs import load_table
from diagnostics import UID, TARGETS, align, aucs, compare, prediction_delta, require_changed_rankings, write_new
from fusion import v14_fuse


def verify(folder, script_version):
    frames, hashes = {}, {}
    for name in ['labels', 'baseline', 'stage3', 'arm0', 'arm1', 'arm2']:
        path = Path(folder) / (name + '.json')
        frame, source = load_table(path)
        if f'scriptVersionId={script_version}' not in source:
            raise ValueError('Mixed/stale script version in ' + name)
        frames[name] = frame
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    ids = frames['labels'][UID].tolist()
    if len(ids) != 58:
        raise ValueError('This specific run must contain all 58 gold studies; refuse partial UI tables')
    frames = {k: align(v, ids, labels=(k == 'labels')) for k, v in frames.items()}
    labels, baseline = frames['labels'], frames['baseline']
    y, b, tr = (frames[k][TARGETS].to_numpy(float) for k in ['labels', 'baseline', 'stage3'])
    arm_names = ['raptor_ft_coatnet_v5_full_swa.pt', 'raptor_ft_coatnet_v10_full.pt', 'raptor_ft_coatnet_v4_full.pt']
    arms = {name: frames[f'arm{i}'][TARGETS].to_numpy(float) for i, name in enumerate(arm_names)}
    control = v14_fuse(tr, arms)
    error = float(np.max(np.abs(control - b)))
    if error > 1e-12:
        raise ValueError(f'Control does not replay exactly: {error}; inspect serialized ties before claiming differences')
    metrics = aucs(y, b)
    sklearn = np.array([roc_auc_score(y[:, j], b[:, j]) for j in range(12)])
    result = {'status': 'EXPLORATORY_GOLD_SELECTED_NOT_CONFIRMATION', 'script_version': script_version,
              'studies': len(ids), 'capture_sha256': hashes, 'baseline_replay_max_error': error,
              'auc_sklearn_max_error': float(np.max(np.abs(metrics - sklearn))),
              'baseline_macro_auc': float(metrics.mean()),
              'plus_0_02_verified': False, 'leaderboard_gain_measured': False,
              'baseline_target_auc': dict(zip(TARGETS, metrics.tolist())),
              'arm_index': dict(enumerate(sorted(arms))), 'candidates': {}}
    for i, name in enumerate(sorted(arms)):
        for tag, selected in [('only', [name]), ('without', [k for k in sorted(arms) if k != name])]:
            candidate = pd.DataFrame(v14_fuse(tr, arms, selected), columns=TARGETS)
            candidate.insert(0, UID, ids)
            detail = compare(labels, baseline, candidate, bootstrap=2000, seed=1400)
            try:
                require_changed_rankings(baseline, candidate)
                detail['output_difference_gate'] = 'PASS_CHANGED_RANKINGS'
            except ValueError as exc:
                detail['output_difference_gate'] = 'FAIL: ' + str(exc)
            detail['changed_numeric_cells'] = int(np.count_nonzero(candidate[TARGETS].to_numpy() != b))
            detail['confirmation_passed'] = False
            result['candidates'][f'{tag}_arm_{i}'] = detail
    result['all_six_candidates_have_changed_rankings'] = all(
        v['output_difference_gate'] == 'PASS_CHANGED_RANKINGS' for v in result['candidates'].values())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--script-version', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.captures, args.script_version)
    write_new(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['candidates', 'capture_sha256']}, indent=2))
    for name, detail in result['candidates'].items():
        print(name, detail['changed_numeric_cells'], detail['macro_delta'], detail['macro_delta_ci95'])
