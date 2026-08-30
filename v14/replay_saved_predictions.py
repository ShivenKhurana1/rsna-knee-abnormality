"""Replay V14 ablations using REAL V13 saved visible-run predictions (no GPU).

No labels exist for these rows. Different predictions are not better predictions.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from compare_saved_outputs import load_table
from diagnostics import TARGETS, UID, align, digest, prediction_delta, write_new
from fusion import v14_fuse


def replay(folder):
    baseline, url = load_table(folder / 'v13_submission_dom.json')
    tr = align(load_table(folder / 'v13_stage3_dom.json')[0], baseline[UID])
    arms = {str(i): align(load_table(folder / f'v13_coatnet_arm_{i}_raw_dom.json')[0], baseline[UID])[TARGETS].to_numpy()
            for i in range(3)}
    control = v14_fuse(tr[TARGETS].to_numpy(), arms)
    error = float(np.max(np.abs(control - baseline[TARGETS].to_numpy())))
    if error != 0:
        raise ValueError('V13 control replay is not exact; do not trust ablations')
    reports = {}
    for i in arms:
        for name, selected in [(f'only_arm_{i}', [i]), (f'without_arm_{i}', [j for j in arms if j != i])]:
            pred = v14_fuse(tr[TARGETS].to_numpy(), arms, selected)
            frame = pd.DataFrame(pred, columns=TARGETS)
            frame.insert(0, UID, baseline[UID].to_numpy())
            delta = prediction_delta(baseline, frame)
            delta['changed_numeric_cells'] = int(np.sum(pred != baseline[TARGETS].to_numpy()))
            reports[name] = delta
    return {'scope': 'REAL cached predictions on 3 visible studies; no new GPU run and no labels.',
            'source': url, 'control_max_absolute_error': error,
            'ablations': reports, 'best_candidate_selected': False,
            'accuracy_improvement_measured': False, 'target_plus_0_02_achieved': False,
            'inputs_sha256': {p.name: digest(p) for p in sorted(folder.glob('v13_*_dom.json'))}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = replay(a.artifacts)
    write_new(a.output, result)
    print('Real V13 control replay maximum absolute error:', result['control_max_absolute_error'])
    for name, r in result['ablations'].items():
        print(f"{name}: {r['changed_numeric_cells']}/36 final cells changed; AUC unknown")
