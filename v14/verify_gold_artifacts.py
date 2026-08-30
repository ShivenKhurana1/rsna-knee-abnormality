"""Independent local replay/scoring of private Kaggle gold CSV-table captures.

Default files in --captures: labels.json, baseline.json, stage3.json,
arm0.json, arm1.json, arm2.json. Each holds {source, heading, rows}; source
must identify the same exact script version and each table must be complete.
With --downloaded, read the original CSV files, download_manifest.json and the
saved six candidate CSVs/gold_results.json; verify bytes and replay saved outputs.
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


def load_downloads(folder, script_version):
    """Load original CSV bytes after verifying the UI-download provenance manifest."""
    folder = Path(folder)
    manifest = json.loads((folder / 'download_manifest.json').read_text(encoding='utf-8'))
    if str(manifest['script_version']) != str(script_version):
        raise ValueError('Mixed/stale download script version')
    for filename, record in manifest['files'].items():
        if Path(filename).name != filename or '/' in filename or '\\' in filename:
            raise ValueError('Unsafe download filename')
        if f'/kaggle-script-versions/{script_version}/output/' not in record['artifact_path']:
            raise ValueError('Mixed/stale artifact path')
        if not record['artifact_path'].endswith('/' + filename):
            raise ValueError('Artifact filename mismatch')
        if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Downloaded artifact hash mismatch: ' + filename)
    names = {'labels': 'gold_labels.csv', 'baseline': 'v13_gold_predictions.csv',
             'stage3': 'stage3.csv', **{f'arm{i}': f'coatnet_arm_{i}_raw.csv' for i in range(3)}}
    frames, hashes = {}, {}
    for name, filename in names.items():
        hashes[name] = manifest['files'][filename]['sha256']
        frames[name] = pd.read_csv(folder / filename, dtype={UID: str})
    return frames, hashes, manifest


def verify(folder, script_version, downloaded=False):
    frames, hashes = {}, {}
    if downloaded:
        frames, hashes, manifest = load_downloads(folder, script_version)
    else:
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
    if downloaded:
        result['artifact_source'] = manifest['source']
        result['original_csv_bytes_verified'] = True
        result['download_manifest'] = manifest
    for i, name in enumerate(sorted(arms)):
        for tag, selected in [('only', [name]), ('without', [k for k in sorted(arms) if k != name])]:
            candidate = pd.DataFrame(v14_fuse(tr, arms, selected), columns=TARGETS)
            candidate.insert(0, UID, ids)
            if downloaded:
                saved = align(pd.read_csv(Path(folder) / f'{tag}_arm_{i}.csv', dtype={UID: str}), ids)
                replay_error = float(np.max(np.abs(saved[TARGETS].to_numpy(float) - candidate[TARGETS].to_numpy(float))))
                if replay_error > 1e-12:
                    raise ValueError(f'Saved candidate replay differs: {tag}_arm_{i}: {replay_error}')
                # Compare actual saved outputs, not just reconstructed candidates.
                candidate = saved
            detail = compare(labels, baseline, candidate, bootstrap=2000, seed=1400)
            if downloaded:
                detail['saved_candidate_replay_max_error'] = replay_error
                detail['saved_csv_sha256'] = manifest['files'][f'{tag}_arm_{i}.csv']['sha256']
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
    if downloaded:
        remote = json.loads((Path(folder) / 'gold_results.json').read_text(encoding='utf-8'))
        errors = [abs(remote['macro_auc'] - result['baseline_macro_auc'])]
        for name, detail in result['candidates'].items():
            errors.extend(abs(remote['candidates'][name][key] - detail[key])
                          for key in ['macro_delta', 'candidate_macro_auc'])
            errors.extend(np.abs(np.array(remote['candidates'][name]['macro_delta_ci95']) - detail['macro_delta_ci95']).tolist())
        result['kaggle_report_max_numeric_error'] = max(errors)
        if max(errors) > 1e-12:
            raise ValueError('Local scoring disagrees with saved Kaggle report')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--script-version', required=True)
    parser.add_argument('--downloaded', action='store_true', help='Read original CSVs with download_manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.captures, args.script_version, downloaded=args.downloaded)
    write_new(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['candidates', 'capture_sha256']}, indent=2))
    for name, detail in result['candidates'].items():
        print(name, detail['changed_numeric_cells'], detail['macro_delta'], detail['macro_delta_ci95'])
