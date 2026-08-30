"""Build the V13 input-contract accuracy candidate from the frozen V12 package."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'v11'))
from build_v11 import replace_once, function_span


def source(cell):
    return ''.join(cell['source'])


def build():
    parent_path = ROOT / 'v12/rsna-knee-ensemble-v12.ipynb'
    parent_bytes = parent_path.read_bytes()
    parent_sha = hashlib.sha256(parent_bytes).hexdigest()
    parent_manifest = json.loads((ROOT / 'v12/build_manifest.json').read_text())
    if parent_sha != parent_manifest['notebook_sha256']:
        raise ValueError('V12 changed; rebuild and test V12 before regenerating V13')
    notebook = copy.deepcopy(json.loads(parent_bytes))
    helpers = (HERE / 'coatnet_contracts.py').read_text(encoding='utf-8')
    info = {
        'recipe': 'v13-per-checkpoint-coatnet-input-contracts-v1',
        'parent_v12_sha256': parent_sha,
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'contracts_sha256': hashlib.sha256(helpers.encode()).hexdigest(),
        'training_enabled': False, 'orthodiffusion_in_final_blend': False,
        'reported_baseline_auc': 0.935, 'measured_v13_auc': None,
        'blend_weights_changed': False,
    }
    for cell in notebook['cells']:
        code = source(cell).replace('V12', 'V13').replace('v12', 'v13')
        if code.startswith('V13_BUILD_INFO = '):
            code = 'V13_BUILD_INFO = ' + repr(info) + '\n' + code.split('\n', 1)[1]
            code += '\n' + helpers
        cell['source'] = code.splitlines(keepends=True)
    stage = notebook['cells'][37]
    code = source(stage)
    if 'def _make_reader():' not in code or 'V13_COATNET_READY' not in code:
        raise ValueError('Parent Stage-4 layout changed')

    # The reader closes over this arm's cache geometry; do not mutate IMG globally.
    code = replace_once(code, 'def _make_reader():', 'def _make_reader(contract):')
    code = replace_once(code, 'cpx = int(round(CROP_MM / max(ps, 1e-3)))',
                        "cpx = int(round(contract['crop_mm'] / max(ps, 1e-3)))")
    code = replace_once(code, 'return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)',
                        "return cv2.resize(a, (contract['cache_img'], contract['cache_img']), interpolation=cv2.INTER_AREA)")
    code = replace_once(code, 'def build_study(sid, ser_records, tsdir, reader):',
                        'def build_study(sid, ser_records, tsdir, reader, contract):')
    code = replace_once(code, 'vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()',
                        "vol = np.zeros((MAXS, contract['cache_img'], contract['cache_img']), np.uint8); idx = 0; used = set()")
    code = replace_once(code, 'n = len(files); lo, hi = int(n * 0.02), int(n * 0.98) - 1; hi = max(hi, lo)',
                        "n = len(files); lo, hi = int(n * contract['span_lo']), int(n * contract['span_hi']) - 1; hi = max(hi, lo)")
    code = replace_once(code, 'ck_res = int(ck.get("res", res_default))',
                        'ck_res = int(ck.get("res", res_default))\n'
                        '        v13_check_coatnet_checkpoint(ck, LAB, v13_coatnet_contract(os.path.basename(pt_path)))')
    code = replace_once(code, '        reader = _make_reader()\n', '')
    code = replace_once(code, 'pd.read_csv(ROOT + "/test.csv")',
                        'pd.read_csv(ROOT + "/test.csv", dtype={"StudyInstanceUID": str})')
    code = replace_once(code, 'pd.read_csv(ROOT + "/test_series.csv")',
                        'pd.read_csv(ROOT + "/test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})')
    code = replace_once(code, 'def _study_volume(sid):\n'
                        '            return _vol_cache.get(sid, lambda uid: build_study(uid, SER, tsdir, reader))',
                        'def _study_volume(sid, contract, reader):\n'
                        '            key = v13_volume_key(sid, contract)\n'
                        '            return _vol_cache.get(key, lambda _: build_study(sid, SER, tsdir, reader, contract))')
    code = replace_once(code, '                    wp = find_weight_file(arm["file"])',
                        '                    contract = v13_coatnet_contract(arm["file"])\n'
                        '                    reader = _make_reader(contract)\n'
                        "                    V13_RUN.note('stage4_arm_contract', arm=arm['file'], **contract)\n"
                        '                    wp = find_weight_file(arm["file"])')
    code = replace_once(code, 'vol, mask = _study_volume(sid)',
                        'vol, mask = _study_volume(sid, contract, reader)')
    code = replace_once(code, 'xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM)',
                        "xw = eval_windows(vol, mask, k=contract['k_eval'], res=res, norm=NORM)")
    code = replace_once(code, '                    if arm_finished:\n                        completed.append(a)',
                        "                    if arm_finished:\n"
                        "                        arm_frame = pd.DataFrame(arm_probs[a], columns=LAB)\n"
                        "                        arm_frame.insert(0, 'StudyInstanceUID', test_ids)\n"
                        "                        v13_validate_submission(arm_frame, test_ids, LAB)\n"
                        "                        arm_path = V13_RUN.folder / f'coatnet_arm_{a}_raw.csv'\n"
                        "                        v13_atomic_csv(arm_frame, arm_path)\n"
                        "                        V13_RUN.note('stage4_arm_predictions', arm=arm['file'], path=str(arm_path),\n"
                        "                                     sha256=_v13_contract_hashlib.sha256(arm_path.read_bytes()).hexdigest())\n"
                        "                        completed.append(a)")
    # A single contract is now consumed once per arm. Caching all three distinct
    # cohorts wastes disk/IO without hits; retain cache implementation for opt-in use.
    config = source(notebook['cells'][2])
    config = replace_once(config, 'V13_VOLUME_CACHE_GIB = 12', 'V13_VOLUME_CACHE_GIB = 0')
    config += '\n# Stage 4 now uses checkpoint-specific geometry and window count.\n'
    notebook['cells'][2]['source'] = config.splitlines(keepends=True)
    # Remove the inherited long, stale Stage-4 preamble claiming one shared recipe.
    start = code.index('    # COATNET_TRANSFORMER_BLEND_V1')
    end = code.index('    import os, sys, glob, time, json, gc', start)
    code = code[:start] + '    # V13: original pretrained CoAtNets with their documented input contracts.\n' + code[end:]
    comment_start = code.index('    # Three arms:')
    comment_end = code.index('    ARMS = [', comment_start)
    code = code[:comment_start] + '    # Same three pretrained checkpoints and equal weights as V12.\n' + code[comment_end:]
    code = code.replace('    K_EVAL = 62   # every window position the volume holds, not an evenly spaced subset\n', '')
    code = code.replace('    IMG = 336\n    CROP_MM = 140.0\n', '')
    code = code.replace('            # wide span: the collateral ligaments and lateral meniscus live in the\n'
                        '            # peripheral slices the old 0.15-0.85 crop threw away. Must match the corpus\n'
                        '            # the weights were trained on (knee_corpus_v2.py, SPAN_LO/SPAN_HI).\n',
                        '            # Match this checkpoint\'s published training-corpus slice span.\n')
    stage['source'] = code.splitlines(keepends=True)
    notebook['cells'][0]['source'] = '''# RSNA Knee Ensemble V13 — input-contract accuracy candidate

Keeps V12's pretrained weights and blend coefficients. Corrects two CoAtNet
input mismatches against the original author's releases:

| Checkpoint | Cache resolution | Slice span | Evaluation windows |
|---|---:|---:|---:|
| MaxSpan v5 SWA | 336 | 2–98% | 62 |
| Native384 v10 | 384, without an intermediate 336 resize | 2–98% | 62 |
| WideDense v4 | 336 | 6–94% | 42 |

All three use the original 140 mm center crop and 384 network resolution.
Stage 1–3 recipes are unchanged. No training or unverified new model is added.
Per-arm raw predictions, input contracts and hashes are recorded for comparison.

**Not yet scored.** These are evidence-backed input corrections, not proof of
+0.01/+0.02 AUC. V12 remains available as the paired comparison baseline.

Attach the same inputs as V12; GPU T4 x2, internet off; Save & Run All. Output:
`/kaggle/working/submission.csv`. The `v13_submission_ready.json` receipt checks
output integrity, not accuracy. Keep `v13_run_receipt.json` and `v13_diagnostics/`.

Original model/source credit: dreaddevelopment, pilkwang, mattiaangeli,
marwanmath, antoinegg1, prvsiyan, sofiaanjenje, tonylica, stevenleehans,
cf696666, romantamrazov. See README for exact author contracts and provenance.
'''.splitlines(keepends=True)
    for index, cell in enumerate(notebook['cells']):
        cell['id'] = f'v13-cell-{index:03d}'
        if cell['cell_type'] == 'code':
            compile(source(cell), f'v13-cell-{index}', 'exec')
            cell['execution_count'], cell['outputs'] = None, []
    notebook['metadata'].pop('v12_build', None)
    notebook['metadata']['v13_build'] = info
    metadata = json.loads((ROOT / 'v12/kernel-metadata.json').read_text())
    metadata.update(id=metadata['id'].split('/')[0] + '/rsna-knee-ensemble-v13',
                    title='RSNA Knee Ensemble V13 Input Contracts', code_file='rsna-knee-ensemble-v13.ipynb')
    target = HERE / metadata['code_file']
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    (HERE / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    manifest = {**info, 'notebook_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'cells': len(notebook['cells']), 'code_cells_compiled': sum(c['cell_type'] == 'code' for c in notebook['cells']),
                'kaggle_executed': False, 'kaggle_submitted': False}
    (HERE / 'build_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    with zipfile.ZipFile(HERE / 'rsna-knee-ensemble-v13.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [metadata['code_file'], 'kernel-metadata.json', 'build_manifest.json', 'README.md']:
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, (HERE / name).read_bytes())
    return notebook


if __name__ == '__main__':
    result = build()
    print(f'Built V13: {len(result["cells"])} cells; accuracy not yet measured.')
