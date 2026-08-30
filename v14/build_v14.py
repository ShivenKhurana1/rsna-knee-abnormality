"""Build V14's inference-control/ablation notebook without changing V13's recipe."""
import copy
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def source(cell):
    return ''.join(cell['source'])


def build():
    path = ROOT / 'v13/rsna-knee-ensemble-v13.ipynb'
    parent = path.read_bytes()
    parent_sha = hashlib.sha256(parent).hexdigest()
    if parent_sha != json.loads((ROOT / 'v13/build_manifest.json').read_text())['notebook_sha256']:
        raise ValueError('Frozen V13 notebook no longer matches its build manifest')
    notebook = copy.deepcopy(json.loads(parent))
    fusion = (HERE / 'fusion.py').read_text(encoding='utf-8')
    info = {'recipe': 'v14-v13-control-plus-paired-ablations-v1',
            'parent_v13_sha256': parent_sha,
            'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'fusion_sha256': hashlib.sha256(fusion.encode()).hexdigest(),
            'training_enabled': False, 'orthodiffusion_in_final_blend': False,
            'reported_v13_public_auc': .935, 'measured_v14_auc': None,
            'submission_recipe_changed': False, 'full_precision_csv_guard': True,
            'plus_0_02_verified': False}
    for cell in notebook['cells']:
        code = source(cell).replace('V13', 'V14').replace('v13', 'v14')
        if code.startswith('V14_BUILD_INFO = '):
            code = 'V14_BUILD_INFO = ' + repr(info) + '\n' + code.split('\n', 1)[1]
            if code.count('frame.to_csv(tmp, index=False)') != 1:
                raise ValueError('Atomic CSV serialization anchor drift')
            code = code.replace('frame.to_csv(tmp, index=False)',
                                "frame.to_csv(tmp, index=False, float_format='%.17g')")
        cell['source'] = code.splitlines(keepends=True)
    notebook['cells'][0]['source'] = '''# V14 — V13 control and paired ablations

**This is NOT a measured 0.955 model.** The default submission is the unchanged
V13 control. V14 exports same-run CoAtNet leave-one-out/single-arm candidates;
none is promoted without independent held-out confirmation. No extra GPU passes.

Import this notebook, attach inputs from kernel-metadata.json, select T4 x2,
internet off, Save & Run All. Retain v14_run_receipt.json and v14_diagnostics/.
The final ablation cell checks that it can reproduce the control before export.
Its CSVs are experiments, not guaranteed improvements. Visible test rows have no
accuracy labels. Do not rank candidates by their visible-test probabilities.

The separate residual_specialist.py trains a new target-specific feature head;
it is not embedded in this notebook and has no trained weights yet. README.md
and PLAN.md describe the real-data confirmation needed to deploy it.

Original pretrained/source credit retained: dreaddevelopment, pilkwang,
mattiaangeli, marwanmath, antoinegg1, prvsiyan, sofiaanjenje, tonylica,
stevenleehans, cf696666 and romantamrazov. Their original licenses still apply.
'''.splitlines(keepends=True)
    # Export after the original final gate: no label access and no inference mutation.
    code = fusion + '\nV14_ABLATIONS = v14_export_ablations(V14_RUN)\nprint(V14_ABLATIONS)\n'
    notebook['cells'].append({'cell_type': 'code', 'metadata': {}, 'source': code.splitlines(keepends=True),
                              'execution_count': None, 'outputs': []})
    for i, cell in enumerate(notebook['cells']):
        cell['id'] = f'v14-cell-{i:03d}'
        if cell['cell_type'] == 'code':
            compile(source(cell), cell['id'], 'exec')
            cell['execution_count'], cell['outputs'] = None, []
    notebook['metadata'].pop('v13_build', None)
    notebook['metadata']['v14_build'] = info
    metadata = json.loads((ROOT / 'v13/kernel-metadata.json').read_text())
    metadata.update(id='seanzhang2445/rsna-knee-ensemble-v14-diagnostics',
                    title='RSNA Knee V14 Paired Diagnostics', code_file='rsna-knee-ensemble-v14.ipynb')
    target = HERE / metadata['code_file']
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    (HERE / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    manifest = {**info, 'notebook_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'cells': len(notebook['cells']), 'code_cells_compiled': sum(c['cell_type'] == 'code' for c in notebook['cells']),
                'kaggle_executed': False, 'kaggle_submitted': False}
    (HERE / 'build_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    with zipfile.ZipFile(HERE / 'rsna-knee-ensemble-v14.zip', 'w') as archive:
        for name in [metadata['code_file'], 'kernel-metadata.json', 'build_manifest.json', 'README.md', 'PLAN.md',
                     'diagnostics.py', 'fusion.py', 'residual_specialist.py', 'confirm.py']:
            entry = zipfile.ZipInfo(name, date_time=(2026, 8, 30, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, (HERE / name).read_bytes())
    return notebook


if __name__ == '__main__':
    n = build()
    print(f'Built {len(n["cells"])} cells. Default remains V13; no +0.02 claim.')
