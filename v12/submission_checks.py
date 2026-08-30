"""Kaggle packaging checks; embedded into V12, with no training or network calls."""

import csv as _v12_check_csv
import hashlib as _v12_check_hashlib
import importlib.util as _v12_check_importlib
import json as _v12_check_json
import os as _v12_check_os
from pathlib import Path as _V12CheckPath


V12_SUBMISSION_TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture',
]


def _v12_write_check_json(path, value):
    path = _V12CheckPath(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(_v12_check_json.dumps(value, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    _v12_check_os.replace(temporary, path)


def v12_preflight(input_root='/kaggle/input', work='/kaggle/working',
                   check_dependencies=True, change_directory=True):
    """Check mounts and schema before expensive model loading; never inspect labels."""
    work = _V12CheckPath(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    # An interrupted rerun must not leave an earlier READY marker looking current.
    _v12_write_check_json(work / 'v12_submission_ready.json', {
        'status': 'NOT_READY', 'reason': 'This run has not completed inference and final validation.'})
    base = _V12CheckPath(input_root)
    candidates = [base / 'competitions/rsna-knee-abnormality-detection',
                  base / 'rsna-knee-abnormality-detection']
    root = next((candidate for candidate in candidates if (candidate / 'test.csv').is_file()), None)
    if root is None:
        raise FileNotFoundError('Attach the RSNA Knee Abnormality Detection competition data.')
    for name in ['test.csv', 'test_series.csv', 'sample_submission.csv']:
        if not (root / name).is_file():
            raise FileNotFoundError(f'Missing competition input: {name}')
    if not (root / 'test_series').is_dir():
        raise FileNotFoundError('Missing competition test_series directory.')
    with (root / 'test.csv').open(newline='', encoding='utf-8-sig') as stream:
        reader = _v12_check_csv.DictReader(stream)
        if 'StudyInstanceUID' not in (reader.fieldnames or []):
            raise ValueError('test.csv is missing StudyInstanceUID')
        ids = [row['StudyInstanceUID'] for row in reader]
    if not ids or any(not uid for uid in ids) or len(ids) != len(set(ids)):
        raise ValueError('test.csv has empty or duplicate study IDs')
    with (root / 'sample_submission.csv').open(newline='', encoding='utf-8-sig') as stream:
        columns = next(_v12_check_csv.reader(stream), [])
    if columns != ['StudyInstanceUID'] + V12_SUBMISSION_TARGETS:
        raise ValueError('Competition submission schema differs from the pretrained heads.')
    with (root / 'test_series.csv').open(newline='', encoding='utf-8-sig') as stream:
        columns = next(_v12_check_csv.reader(stream), [])
    if not {'StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane',
            'Fluid_Sensitive', 'Fat_Suppression'}.issubset(columns):
        raise ValueError('test_series.csv is missing required series metadata')
    dependencies = ['numpy', 'pandas', 'torch', 'torchvision', 'transformers', 'timm',
                    'cv2', 'pydicom', 'scipy', 'sklearn', 'joblib']
    if check_dependencies:
        missing = [name for name in dependencies if _v12_check_importlib.find_spec(name) is None]
        if missing:
            raise RuntimeError('Required runtime packages missing: ' + ', '.join(missing)
                               + '. Prepare the notebook environment before the offline scored run.')
    if change_directory:
        _v12_check_os.chdir(work)
    print(f'[V12 preflight] {len(ids)} studies; competition schema verified. '
          'Pretrained weights and GPU execution are checked by the model loaders.', flush=True)
    return {'competition_root': str(root), 'work': str(work), 'studies': len(ids),
            'dependency_presence_checked': bool(check_dependencies)}


def v12_finalize_submission(run):
    """Emit READY only for the validated final snapshot of this inference run."""
    if list(run.targets) != V12_SUBMISSION_TARGETS:
        raise ValueError('Final targets differ from competition schema')
    if not any(event['event'] == 'stage1_member_committed' for event in run.receipt['events']):
        raise RuntimeError('No current-run pretrained member completed')
    data = run._validated_bytes(run.primary)
    digest = _v12_check_hashlib.sha256(data).hexdigest()
    persisted = _v12_check_json.loads((run.work / 'v12_run_receipt.json').read_text(encoding='utf-8'))
    snapshots = persisted.get('snapshots', [])
    if (persisted.get('run_id') != run.run_id or not snapshots
            or snapshots[-1].get('stage') != 'final' or snapshots[-1].get('sha256') != digest):
        raise RuntimeError('Final submission does not match this run\'s final snapshot')
    if persisted['build'].get('training_enabled') is not False:
        raise RuntimeError('Expected inference-only build')
    ready = {
        'status': 'READY_FOR_KAGGLE_SUBMISSION',
        'validation_scope': 'Current-run output contract only; not leaderboard performance.',
        'run_id': run.run_id, 'submission': str(run.primary), 'submission_sha256': digest,
        'studies': len(run.ids), 'columns': ['StudyInstanceUID'] + list(run.targets),
        'build': persisted['build'], 'measured_auc': None,
    }
    _v12_write_check_json(run.work / 'v12_submission_ready.json', ready)
    print(f'[V12] submission.csv ready: {len(run.ids)} studies, 12 targets, sha256={digest}', flush=True)
    print('[V12] Save & Run All on Kaggle; submit that completed notebook version. AUC is unmeasured.', flush=True)
    return ready
