"""Package the frozen V11 inference recipe as a self-contained V12 Kaggle release."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _source(cell):
    return ''.join(cell['source'])


def _code_cell(source, name):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'id': name, 'source': source.splitlines(keepends=True)}


def build():
    parent_path = ROOT / 'v11/rsna-knee-ensemble-v11.ipynb'
    parent_bytes = parent_path.read_bytes()
    parent_sha = hashlib.sha256(parent_bytes).hexdigest()
    parent_manifest = json.loads((ROOT / 'v11/build_manifest.json').read_text())
    if parent_sha != parent_manifest['notebook_sha256']:
        raise ValueError('V11 notebook changed since its build manifest; regenerate and test V11 first.')
    notebook = copy.deepcopy(json.loads(parent_bytes))
    checks = (HERE / 'submission_checks.py').read_text(encoding='utf-8')
    metadata = json.loads((ROOT / 'v11/kernel-metadata.json').read_text())
    metadata.update(id=metadata['id'].split('/')[0] + '/rsna-knee-ensemble-v12',
                    title='RSNA Knee Ensemble V12', code_file='rsna-knee-ensemble-v12.ipynb')
    build_info = {
        'recipe': 'v12-pretrained-submission-package-v1', 'parent_v11_sha256': parent_sha,
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'submission_checks_sha256': hashlib.sha256(checks.encode()).hexdigest(),
        'training_enabled': False, 'orthodiffusion_in_final_blend': False,
        'reported_baseline_auc': 0.935, 'measured_v12_auc': None,
    }
    for cell in notebook['cells']:
        code = _source(cell).replace('V11', 'V12').replace('v11', 'v12')
        cell['source'] = code.splitlines(keepends=True)
        if cell['cell_type'] == 'code' and code.startswith('V12_BUILD_INFO = '):
            cell['source'] = ('V12_BUILD_INFO = ' + repr(build_info) + '\n'
                              + code.split('\n', 1)[1]).splitlines(keepends=True)
    notebook['cells'][0]['source'] = '''# RSNA Knee Ensemble V12 — submission package

Self-contained pretrained inference notebook. Carries forward V11's fixed-center
crops, validated prediction recovery, complete-member fusion, bounded disk cache,
and current-run-only model outputs. The model families and blend weights are unchanged.

**Run:** attach the inputs in `kernel-metadata.json`, select GPU T4 x2, and Save & Run
All. The code runs offline and writes `/kaggle/working/submission.csv`. No companion
Python files are required on Kaggle. Do not upload the release ZIP as a notebook;
import the `.ipynb` inside it.

At successful completion, `v12_submission_ready.json` must report
`READY_FOR_KAGGLE_SUBMISSION`. The ready receipt validates output integrity, not AUC.
Keep `v12_run_receipt.json` and `v12_diagnostics/` for troubleshooting.

No new training; no unverified OrthoDiffusion blend. User-reported baseline 0.935;
V12 score and hidden-test runtime are unmeasured. Local tests are not a Kaggle run.

Lineage: `mattiaangeli/bend-the-knee-to-dinov3-ensembled` plus the community CoAtNet
branch. Credit: pilkwang, mattiaangeli, marwanmath, antoinegg1, prvsiyan, sofiaanjenje,
tonylica, stevenleehans, cf696666, romantamrazov, dreaddevelopment. Original licenses
and checkpoint terms still apply.
'''.splitlines(keepends=True)
    notebook['cells'].insert(1, _code_cell(checks + '\nV12_PREFLIGHT = v12_preflight()\n', 'v12-preflight'))
    notebook['cells'][-1] = _code_cell('''# Final current-run submission gate, with recoverable stage snapshots.
V12_RUN.snapshot('final')
V12_READY = v12_finalize_submission(V12_RUN)
print(f'Total wall clock: {(time.time() - WALL_T0) / 60:.1f} minutes')
''', 'v12-final-submission-gate')
    # nbformat 4.5 requires unique cell IDs; the inherited notebook omitted most.
    for index, cell in enumerate(notebook['cells']):
        cell['id'] = f'v12-cell-{index:03d}'
        if cell['cell_type'] == 'code':
            compile(_source(cell), f'v12-cell-{index}', 'exec')
            cell['execution_count'] = None
            cell['outputs'] = []
    notebook['metadata'].pop('v11_build', None)
    notebook['metadata']['v12_build'] = build_info
    notebook['nbformat'], notebook['nbformat_minor'] = 4, 5
    notebook_path = HERE / metadata['code_file']
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    (HERE / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    manifest = {**build_info, 'notebook_sha256': hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
                'cells': len(notebook['cells']),
                'code_cells_compiled': sum(c['cell_type'] == 'code' for c in notebook['cells']),
                'kaggle_executed': False, 'kaggle_submitted': False}
    (HERE / 'build_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    # Deterministic release archive: no weights, medical data, or local test files.
    with zipfile.ZipFile(HERE / 'rsna-knee-ensemble-v12.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [metadata['code_file'], 'kernel-metadata.json', 'build_manifest.json', 'README.md']:
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, (HERE / name).read_bytes())
    return notebook


if __name__ == '__main__':
    built = build()
    print(f'Built V12: {len(built["cells"])} cells; notebook, Kaggle metadata and release ZIP ready.')
