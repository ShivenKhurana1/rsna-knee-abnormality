"""Cheap, label-blind ablations using the exact V13 float32/rank fusion order.

This module is embedded in the Kaggle notebook. No per-target recipe is chosen
on hidden-test rows. Ablations remain diagnostic outputs, not automatic promotions.
"""
import numpy as np
import pandas as pd

V14_TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
               'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
V14_UID = 'StudyInstanceUID'


def v14_rank(values):
    return pd.DataFrame(values).rank(method='average', pct=True).to_numpy()


def v14_fuse(transformer, arms, selected=None):
    """Reproduce V13: rank each arm, average, cast float32, rank, 50/50, rank."""
    selected = list(arms) if selected is None else list(selected)
    if not selected:
        raise ValueError('CoAtNet selection must not be empty')
    tr = np.asarray(transformer, dtype=float)
    if tr.ndim != 2 or tr.shape[1] != 12 or not len(tr):
        raise ValueError('Expected nonempty N x 12 transformer predictions')
    for a in [tr] + [np.asarray(arms[k]) for k in selected]:
        if a.shape != tr.shape or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
            raise ValueError('Invalid or incompatible arm predictions')
    weights = np.ones(len(selected), dtype=np.float64) / len(selected)
    # CoAtNet uses (rank-1)/(N-1), whereas outer fusion uses rank/N.
    # The distinction matters at ties after float32 serialization.
    arm_ranks = [(pd.DataFrame(arms[k]).rank(method='average').to_numpy() - 1)
                 / max(1, len(tr) - 1) for k in selected]
    cr = np.tensordot(weights, np.stack(arm_ranks), axes=(0, 0))
    return v14_rank(.5 * v14_rank(tr) + .5 * v14_rank(cr.astype(np.float32)))


def v14_export_ablations(run):
    """Export completed-arm ablations; preserve the current submission byte-for-byte."""
    from pathlib import Path
    import hashlib
    import json
    primary_before = run.primary.read_bytes()
    folder = run.folder
    stage3 = pd.read_csv(folder / 'stage3.csv', dtype={V14_UID: str})
    ids = list(run.ids)

    def values(frame):
        if frame[V14_UID].duplicated().any() or set(frame[V14_UID]) != set(ids):
            raise ValueError('Ablation input study coverage mismatch')
        return frame.set_index(V14_UID).loc[ids, V14_TARGETS].to_numpy(float)

    arms = {}
    for event in run.receipt['events']:
        if event['event'] != 'stage4_arm_predictions':
            continue
        path = Path(event['path'])
        if path.parent.resolve() != folder.resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != event['sha256']:
            raise ValueError('Arm predictions are stale or modified')
        arms[event['arm']] = values(pd.read_csv(path, dtype={V14_UID: str}))
    output = folder / 'ablations'
    output.mkdir(exist_ok=False)
    receipt = {'status': 'DIAGNOSTIC_ONLY', 'promoted': False, 'measured_auc': None,
               'complete_arms': sorted(arms), 'baseline_sha256': hashlib.sha256(primary_before).hexdigest(),
               'candidates': {}, 'v13_reproduced': None}
    tr = values(stage3)
    candidates = {}
    if arms:
        control = v14_fuse(tr, arms)
        current = values(pd.read_csv(run.primary, dtype={V14_UID: str}))
        receipt['v13_reproduced'] = bool(np.allclose(control, current, atol=1e-12, rtol=0))
        if not receipt['v13_reproduced']:
            raise ValueError('Recomputed V13 does not match current output; refuse confounded ablations')
        candidates['v13_control'] = control
        for i, name in enumerate(sorted(arms)):
            candidates[f'only_arm_{i}'] = v14_fuse(tr, arms, [name])
            if len(arms) > 1:
                candidates[f'without_arm_{i}'] = v14_fuse(tr, arms, [k for k in sorted(arms) if k != name])
        receipt['arm_index'] = dict(enumerate(sorted(arms)))
    for name, pred in candidates.items():
        frame = pd.DataFrame(pred, columns=V14_TARGETS)
        frame.insert(0, V14_UID, ids)
        path = output / (name + '.csv')
        frame.to_csv(path, index=False)
        receipt['candidates'][name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if run.primary.read_bytes() != primary_before:
        raise RuntimeError('Diagnostics must never modify the submission')
    (output / 'ablation_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    run.note('v14_ablations_exported', **receipt)
    return receipt
