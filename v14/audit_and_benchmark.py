"""Reproducible source audit + SYNTHETIC mechanism tests, not a knee AUC benchmark."""
import ast
import json
from pathlib import Path
import platform
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score

from diagnostics import TARGETS, UID, aucs, digest, prediction_delta, write_new
from fusion import v14_fuse
from residual_specialist import crossfit

HERE = Path(__file__).resolve().parent


def table(x):
    frame = pd.DataFrame(x, columns=TARGETS)
    frame.insert(0, UID, [f'synthetic-{i:05d}' for i in range(len(x))])
    return frame


def run():
    rng = np.random.default_rng(1400)
    source_audit = {}
    notebooks = {}
    for version in (11, 12, 13):
        path = HERE.parent / f'v{version}/rsna-knee-ensemble-v{version}.ipynb'
        n = json.loads(path.read_text(encoding='utf-8'))
        notebooks[version] = n
        for cell in n['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'v{version}', 'exec')
        stage4 = next(''.join(c['source']) for c in n['cells'] if '    ARMS = [' in ''.join(c['source']))
        assign = next(node for node in ast.walk(ast.parse(stage4)) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'ARMS' for t in node.targets))
        config = next(''.join(c['source']) for c in n['cells'] if 'COATNET_FLAT    =' in ''.join(c['source']))
        source_audit[f'v{version}'] = {'sha256': digest(path), 'coatnet_arms': ast.literal_eval(assign.value),
            'flat_coatnet_weight_half': 'COATNET_FLAT    = True' in config and 'COATNET_W       = 0.50' in config}

    def canon(cell):
        code = re.sub(r'(?i)v1[123]', 'vCONTROL', ''.join(cell['source']))
        return ast.dump(ast.parse(code))

    # V12 inserted the preflight at index 1; V13 keeps that layout.
    same = [i for i in range(5, 36) if notebooks[13]['cells'][i]['cell_type'] == 'code'
            and canon(notebooks[13]['cells'][i]) == canon(notebooks[11]['cells'][i - 1])]
    source_audit['v11_v13_unchanged_stage1_to3_code_cells'] = same
    source_audit['coatnet_checkpoints_weights_identical'] = source_audit['v11']['coatnet_arms'] == source_audit['v13']['coatnet_arms']
    n = 4349
    tr = rng.uniform(.01, .99, (n, 12))
    arms = {str(i): rng.uniform(.01, .99, (n, 12)) for i in range(3)}
    before = v14_fuse(tr, arms)
    changed = {k: v ** 2 for k, v in arms.items()}
    after = v14_fuse(tr, changed)
    mechanism = prediction_delta(table(before), table(after))
    mechanism['scope'] = 'SYNTHETIC monotone input perturbation; NOT V11/V13 MRI inference'
    mechanism['max_raw_arm_probability_change'] = float(max(np.max(np.abs(arms[k] - changed[k])) for k in arms))
    latencies = []
    for _ in range(7):
        start = time.perf_counter()
        v14_fuse(tr, arms)
        latencies.append(time.perf_counter() - start)
    y = rng.integers(0, 2, (n, 12)).astype(float)
    reference_error = float(np.max(np.abs(aucs(y, before) - roc_auc_score(y, before, average=None))))
    # New-feature signal omitted from baseline; only a head implementation check.
    from scipy.special import expit
    x = rng.normal(size=(1200, 8))
    by = rng.normal(size=(1200, 12))
    b = expit(by)
    signal = x @ rng.normal(size=(8, 12)) / 2
    labels = (rng.random((1200, 12)) < expit(by + signal)).astype(float)
    start = time.perf_counter()
    pred, folds = crossfit(x, b, labels, np.repeat(np.arange(400), 3), regularization=.05)
    head_seconds = time.perf_counter() - start
    report = {
        'status': 'NO_REAL_MODEL_ACCURACY_BENCHMARK_AVAILABLE',
        'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
        'scope': 'Current local sources and synthetic tests only; remote scored-version equivalence not established.',
        'source_audit': source_audit,
        'rank_invariance_mechanism': mechanism,
        'cpu_rank_fusion': {'studies': n, 'arms': 3, 'repetitions': 7,
                            'median_seconds': float(np.median(latencies)), 'gpu_inference_measured': False},
        'auc_max_absolute_error_vs_sklearn': reference_error,
        'synthetic_residual_head': {'baseline_auc': float(aucs(labels, b).mean()),
            'candidate_oof_auc': float(aucs(labels, pred).mean()), 'elapsed_seconds': head_seconds,
            'group_folds': int(len(np.unique(folds))),
            'scope': 'Artificial omitted-feature signal; NOT knee data; NOT evidence for a Kaggle gain.'},
        'r2_vs_r_squared_demo': {'r_squared': float(np.corrcoef(tr[:, 0], tr[:, 0] + 2)[0, 1] ** 2),
                                'r2_score': float(r2_score(tr[:, 0], tr[:, 0] + 2))},
        'arithmetic': {'reconstruction_r_squared': .7258 ** 2, 'required_summed_target_auc_gain': 12 * .02,
                       'relative_reduction_in_one_minus_auc': .02 / (1 - .935)},
        'real_v14_auc': None, 'real_v14_minus_v13': None, 'goal_plus_0_02_achieved': False,
    }
    return report


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = run()
    write_new(args.output, report)
    print(json.dumps(report, indent=2))
