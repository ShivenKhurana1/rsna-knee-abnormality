"""Private, exploratory gold-cohort adapter. Never a leaderboard submission."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

GOLD_UID = 'StudyInstanceUID'
GOLD_TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
                'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def gold_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare_gold(real, work, temporary_parent=None):
    """Expose fully labeled studies as test, without reports or labels in test inputs."""
    real, work = Path(real), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if (work / 'cohort_audit.json').exists():
        raise RuntimeError('Existing validation artifacts: use a fresh session/work directory')
    tr = pd.read_csv(real / 'train.csv', dtype={GOLD_UID: str})
    if tr[GOLD_UID].isna().any() or tr[GOLD_UID].duplicated().any():
        raise ValueError('Invalid training study identities')
    gold = tr.loc[tr[GOLD_TARGETS].notna().all(axis=1), [GOLD_UID] + GOLD_TARGETS].copy()
    if len(gold) < 20 or not gold[GOLD_TARGETS].isin([0, 1]).all().all():
        raise ValueError('Expected at least 20 fully expert-labeled binary studies')
    if any(gold[t].nunique() != 2 for t in GOLD_TARGETS):
        raise ValueError('Every target requires both classes')
    ids = gold[GOLD_UID].tolist()
    series = pd.read_csv(real / 'train_series.csv', dtype={GOLD_UID: str, 'SeriesInstanceUID': str})
    selected = series.loc[series[GOLD_UID].isin(ids)].copy()
    if set(selected[GOLD_UID]) != set(ids):
        raise ValueError('Gold study missing series metadata')
    for uid in ids:
        if '/' in uid or '\\' in uid or uid in ('.', '..'):
            raise ValueError('Unsafe study identifier')
        if not (real / 'train_series' / uid).is_dir():
            raise FileNotFoundError('Gold study image folder absent: ' + uid)
    if temporary_parent is not None:
        Path(temporary_parent).mkdir(parents=True, exist_ok=True)
    pseudo_parent = Path(tempfile.mkdtemp(prefix='v14-gold-', dir=temporary_parent))
    pseudo = pseudo_parent / 'rsna-knee-abnormality-detection'
    (pseudo / 'test_series').mkdir(parents=True)
    gold[[GOLD_UID]].to_csv(pseudo / 'test.csv', index=False)
    selected.to_csv(pseudo / 'test_series.csv', index=False)
    sample = gold[[GOLD_UID]].assign(**{t: .5 for t in GOLD_TARGETS})
    sample.to_csv(pseudo / 'sample_submission.csv', index=False)
    # Stage 1 uses training row count to size its cache, not for inference labels.
    tr[[GOLD_UID]].to_csv(pseudo / 'train.csv', index=False)
    for uid in ids:
        os.symlink((real / 'train_series' / uid).resolve(), pseudo / 'test_series' / uid,
                   target_is_directory=True)
    gold.to_csv(work / 'gold_labels.csv', index=False)
    gold[[GOLD_UID]].to_csv(work / 'gold_study_ids.csv', index=False)
    audit = {
        'status': 'EXPLORATORY_NOT_INDEPENDENT_CONFIRMATION', 'studies': len(gold),
        'cohort_definition': 'All train.csv rows with all 12 expert targets present; no outcome filtering.',
        'source_train_csv_sha256': hashlib.sha256((real / 'train.csv').read_bytes()).hexdigest(),
        'labels_sha256': hashlib.sha256((work / 'gold_labels.csv').read_bytes()).hexdigest(),
        'test_inputs_contain_labels_or_reports': False, 'training_enabled': False,
        'patient_grouping_verified': False, 'bootstrap_unit': 'study (patient grouping unavailable)',
        'coatnet_source_audit': {
            'source': 'https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model',
            'gradient_exclusion': 'Author source excludes gold_ids from train_ids.',
            'checkpoint_selection': 'Gold AUC selects epochs/top-k/SWA; GOLD IS NOT UNTOUCHED.',
            'actual_checkpoint_id_manifests': 'Not independently verified by source alone.'},
        'other_families': 'Training/calibration exposure UNKNOWN unless explicit loaded metadata establishes it.',
        'plus_0_02_verified': False, 'leaderboard_submission_allowed': False,
        'counts': {t: {'positive': int(gold[t].sum()), 'negative': int((gold[t] == 0).sum())}
                   for t in GOLD_TARGETS},
    }
    gold_write(work / 'cohort_audit.json', audit)
    print('GOLD AUDIT BEFORE INFERENCE:', json.dumps(audit), flush=True)
    return pseudo, work


def audit_metadata(payload, gold_ids):
    """Read explicit checkpoint metadata only; never infer exclusion from a fold number."""
    result = {'explicit_id_sets': [], 'metadata': {}}
    gold_ids = set(map(str, gold_ids))
    id_keys = {'train_ids', 'training_ids', 'train_uids', 'training_uids', 'val_ids',
               'valid_ids', 'validation_ids', 'val_uids', 'gold_ids', 'test_ids'}
    scalar_keys = {'epoch', 'gold_auc', 'auc', 'best_auc', 'n_train', 'n_gold', 'fold',
                   'version', 'tag', 'seed', 'training_studies', 'validation_studies'}

    def visit(value, prefix='', depth=0):
        if depth > 4 or not isinstance(value, dict):
            return
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            name = prefix + key
            if key.lower() in id_keys and isinstance(item, (list, tuple, set, np.ndarray)):
                overlap = sorted(gold_ids.intersection(map(str, item)))
                result['explicit_id_sets'].append({'key': name, 'count': len(item),
                                                   'gold_overlap_count': len(overlap),
                                                   'gold_overlap_ids': overlap})
            elif key in scalar_keys and isinstance(item, (str, int, float, bool, type(None))):
                if not isinstance(item, float) or np.isfinite(item):
                    result['metadata'][name] = item
            elif key in ('meta', 'metadata', 'config', 'cfg', 'manifest', 'provenance'):
                visit(item, name + '.', depth + 1)
            elif key == 'folds' and isinstance(item, list):
                for i, record in enumerate(item):
                    visit(record, name + f'.{i}.', depth + 1)
    visit(payload)
    result['exclusion_verified'] = False  # Absence of IDs is never evidence of exclusion.
    return result


def install_checkpoint_audit(work, ids):
    """Observe the normal loads; no second GPU/model load and no changed tensors."""
    import torch
    import threading
    original = torch.load
    records, lock = [], threading.Lock()

    def observed(*args, **kwargs):
        payload = original(*args, **kwargs)
        path = str(args[0] if args else kwargs.get('f', 'unknown'))
        record = {'path': path, **audit_metadata(payload, ids)}
        with lock:
            records.append(record)
            gold_write(Path(work) / 'loaded_checkpoint_audit.json', {
                'scope': 'Observed metadata only; no missing-ID exclusion inference.',
                'independent_confirmation_eligible': False, 'records': records})
        return payload
    torch.load = observed
    return records


def evaluate_gold(run, ablations, diagnostics):
    """All results remain descriptive/exploratory, including positive deltas."""
    labels = pd.read_csv(run.work / 'gold_labels.csv', dtype={GOLD_UID: str})
    baseline = pd.read_csv(run.primary, dtype={GOLD_UID: str})
    labels = diagnostics.align(labels, run.ids, labels=True)
    baseline = diagnostics.align(baseline, run.ids)
    y = labels[GOLD_TARGETS].to_numpy(float)
    baseline_aucs = diagnostics.aucs(y, baseline[GOLD_TARGETS].to_numpy(float))
    result = {'status': 'EXPLORATORY_GOLD_SELECTED_NOT_CONFIRMATION',
              'studies': len(labels), 'macro_auc': float(baseline_aucs.mean()),
              'plus_0_02_verified': False, 'leaderboard_improvement_measured': False,
              'ci_caveat': 'Study-level, no verified patient grouping; selected gold set; multiple candidates unadjusted.',
              'baseline': {}, 'candidates': {}, 'stage_macro_auc': {}}
    for j, t in enumerate(GOLD_TARGETS):
        result['baseline'][t] = {'auc': float(baseline_aucs[j]), 'positive': int(y[:, j].sum()),
                                 'negative': int((y[:, j] == 0).sum()),
                                 'perfect_target_max_macro_gain': float((1 - baseline_aucs[j]) / 12)}
    for stage in ('stage1', 'stage2', 'stage3', 'stage4'):
        frame = diagnostics.align(pd.read_csv(run.folder / (stage + '.csv'), dtype={GOLD_UID: str}), run.ids)
        scores = diagnostics.aucs(y, frame[GOLD_TARGETS].to_numpy(float))
        result['stage_macro_auc'][stage] = float(scores.mean())
    # Fixed candidates declared before seeing labels. No auto-selection/promotion.
    for name, spec in ablations['candidates'].items():
        if name == 'v13_control':
            continue
        candidate = pd.read_csv(spec['path'], dtype={GOLD_UID: str})
        try:
            comparison = diagnostics.compare(labels, baseline, candidate, bootstrap=2000, seed=1400)
        except ValueError as error:
            aligned = diagnostics.align(candidate, run.ids)
            scores = diagnostics.aucs(y, aligned[GOLD_TARGETS].to_numpy(float))
            comparison = {'candidate_macro_auc': float(scores.mean()),
                          'macro_delta': float((scores - baseline_aucs).mean()),
                          'bootstrap_unavailable_reason': str(error)}
        comparison['independent_confirmation'] = False
        result['candidates'][name] = comparison
        gold_write(run.work / (name + '_exploratory.json'), comparison)
    gold_write(run.work / 'gold_results.json', result)
    pd.DataFrame(result['baseline']).T.rename_axis('target').reset_index().to_csv(
        run.work / 'per_target_auc.csv', index=False)
    pd.DataFrame([{'candidate': name, 'macro_auc': item['candidate_macro_auc'],
                   'delta': item['macro_delta'], 'status': 'EXPLORATORY_ONLY'}
                  for name, item in result['candidates'].items()]).to_csv(
        run.work / 'candidate_summary.csv', index=False)
    print('EXPLORATORY V13 GOLD AUC:', result['macro_auc'], flush=True)
    print(pd.DataFrame(result['baseline']).T.to_string(), flush=True)
    print('EXPLORATORY CANDIDATE DELTAS:', json.dumps({k: v['macro_delta'] for k, v in result['candidates'].items()}), flush=True)
    print('NO LEADERBOARD GAIN OR +0.02 GUARANTEE IS ESTABLISHED.', flush=True)
    return result
