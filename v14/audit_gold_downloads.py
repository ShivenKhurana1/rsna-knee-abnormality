"""Verify downloaded run receipts and summarize all stages without promoting any."""
import argparse
import json
from pathlib import Path

import pandas as pd

from diagnostics import UID, TARGETS, align, aucs, write_new
from verify_gold_artifacts import load_downloads


def audit(folder, version):
    folder = Path(folder)
    frames, _, manifest = load_downloads(folder, version)
    records = json.loads((folder / 'v13_run_receipt.json').read_text())
    ablation = json.loads((folder / 'ablation_receipt.json').read_text())
    cohort = json.loads((folder / 'cohort_audit.json').read_text())
    metadata = json.loads((folder / 'loaded_checkpoint_audit.json').read_text())
    complete = json.loads((folder / 'validation_complete.json').read_text())
    hashes = {k: v['sha256'] for k, v in manifest['files'].items()}
    checked = []

    def check(filename, expected):
        if hashes[filename] != expected:
            raise ValueError('Receipt hash mismatch: ' + filename)
        checked.append(filename)

    check('gold_labels.csv', cohort['labels_sha256'])
    check('v13_gold_predictions.csv', ablation['baseline_sha256'])
    for name, spec in ablation['candidates'].items():
        check(name + '.csv', spec['sha256'])
    for event in records['events']:
        if event['event'] == 'stage4_arm_predictions':
            check(Path(event['path']).name, event['sha256'])
    for snap in records['snapshots']:
        if snap['stage'] in ('stage1', 'stage2', 'stage3'):
            check(snap['stage'] + '.csv', snap['sha256'])
        elif snap['stage'] in ('stage4', 'final'):
            check('v13_gold_predictions.csv', snap['sha256'])
    members = {e['member'] for e in records['events'] if e['event'] == 'stage1_member_committed'}
    arms = {e['arm'] for e in records['events'] if e['event'] == 'stage4_arm_completed' and e['studies'] == 58}
    if len(members) != 24 or len(arms) != 3 or complete['status'] != 'EXPLORATORY_COMPLETE_DO_NOT_SUBMIT':
        raise ValueError('Incomplete benchmark')
    labels = align(frames['labels'], labels=True)
    ids = labels[UID]
    y = labels[TARGETS].to_numpy(float)
    stages = {}
    for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
        frame = frames['baseline'] if stage == 'stage4' else pd.read_csv(folder / (stage + '.csv'), dtype={UID: str})
        scores = aucs(y, align(frame, ids)[TARGETS].to_numpy(float))
        stages[stage] = {'macro_auc': float(scores.mean()), 'target_auc': dict(zip(TARGETS, scores.tolist()))}
    observed = metadata['records']
    return {'status': 'EXPLORATORY_GOLD_SELECTED_NOT_CONFIRMATION', 'script_version': version,
            'source': manifest['source'], 'studies': len(ids),
            'receipt_hashes_verified': sorted(set(checked)),
            'stage1_members_completed': len(members), 'coatnet_arms_completed': len(arms),
            'checkpoint_load_observations': len(observed),
            'unique_checkpoint_paths': len({r['path'] for r in observed}),
            'explicit_id_sets_found': sum(len(r['explicit_id_sets']) for r in observed),
            'training_exclusion_verified': False,
            'stage_results': stages, 'target_counts': cohort['counts'],
            'stage1_gain_over_final': stages['stage1']['macro_auc'] - stages['stage4']['macro_auc'],
            'stage1_warning': 'Near-perfect gold AUC with unknown training exposure is NOT proof of generalization; do not promote stage1-only.',
            'plus_0_02_verified': False, 'promoted': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads', type=Path, required=True)
    parser.add_argument('--script-version', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.downloads, args.script_version)
    write_new(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['stage_results', 'target_counts']}, indent=2))
    print(json.dumps({k: v['macro_auc'] for k, v in result['stage_results'].items()}, indent=2))
