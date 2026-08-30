"""Confirm ONE frozen candidate against V13; no fitting, automatic upload or promotion.

The evidence manifest is an audit declaration, not cryptographic proof that
training/selection manifests are complete or that a freeze preceded label access.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from diagnostics import UID, align, compare, digest, write_new


def verify_evidence(evidence, root, labels, groups):
    required = {
        'label_quality': 'expert_image', 'candidate_frozen_before_confirmation': True,
        'confirmation_labels_used_for_selection': False,
        'all_component_training_and_selection_ids_included': True,
    }
    for key, value in required.items():
        if evidence.get(key) != value:
            raise ValueError(f'Evidence requirement not met: {key}={value!r}')
    if not evidence.get('provenance_notes', '').strip():
        raise ValueError('Document all encoders/checkpoints, training exclusions and candidate freeze')
    if not evidence.get('exclusion_manifests'):
        raise ValueError('List training AND model-selection manifests for every component')
    held_ids, held_groups = set(labels[UID]), set(groups)
    hashes = {}
    for item in evidence['exclusion_manifests']:
        path = (root / item['path']).resolve()
        if digest(path) != item['sha256']:
            raise ValueError('Exclusion manifest hash mismatch')
        frame = pd.read_csv(path, dtype=str)
        if not {UID, 'GroupID'}.issubset(frame) or frame[[UID, 'GroupID']].isna().any().any():
            raise ValueError('Exclusion manifest needs nonmissing StudyInstanceUID and GroupID')
        if set(frame[UID]) & held_ids or set(frame['GroupID']) & held_groups:
            raise ValueError('Training/selection overlap with confirmation studies or groups')
        hashes[str(path)] = digest(path)
    artifact = evidence['frozen_candidate_artifact']
    path = (root / artifact['path']).resolve()
    if digest(path) != artifact['sha256']:
        raise ValueError('Frozen candidate artifact changed')
    hashes[str(path)] = digest(path)
    return hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('labels', 'baseline', 'candidate', 'groups', 'evidence', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args()
    labels = align(pd.read_csv(args.labels, dtype={UID: str}), labels=True)
    g = pd.read_csv(args.groups, dtype=str)
    if g[UID].duplicated().any() or set(g[UID]) != set(labels[UID]):
        raise ValueError('Confirmation group coverage mismatch')
    groups = g.set_index(UID).loc[labels[UID], 'GroupID'].to_numpy()
    evidence = json.loads(args.evidence.read_text(encoding='utf-8'))
    hashes = verify_evidence(evidence, args.evidence.parent, labels, groups)
    frames = [pd.read_csv(path, dtype={UID: str}) for path in (args.baseline, args.candidate)]
    result = compare(labels, *frames, bootstrap=args.bootstrap, groups=groups)
    result['evidence_sha256'] = hashes
    result['input_sha256'] = {k: digest(getattr(args, k)) for k in ('labels', 'baseline', 'candidate', 'groups', 'evidence')}
    result['decision'] = ('CONFIRMATION_COHORT_GAIN_GE_0_02_SUPPORTED'
                          if result['ci_lower_bound_at_least_0_02'] else 'DO_NOT_CLAIM_PLUS_0_02')
    result['leaderboard_gain_guaranteed'] = False
    result['evidence_limit'] = 'Manifest completeness, dates and prediction/model association are declarations requiring independent audit.'
    write_new(args.output, result)
    print(result['decision'])
    print(f"Delta {result['macro_delta']:+.6f}, paired CI {result['macro_delta_ci95']}")


if __name__ == '__main__':
    main()
