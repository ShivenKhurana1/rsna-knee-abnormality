"""Build a private validation-only notebook from frozen V13 inference code."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one source anchor: {old[:90]!r}, found {text.count(old)}')
    return text.replace(old, new)


def build():
    parent_path = HERE.parent / 'v13/rsna-knee-ensemble-v13.ipynb'
    parent = parent_path.read_bytes()
    parent_sha = hashlib.sha256(parent).hexdigest()
    assert parent_sha == json.loads((HERE.parent / 'v13/build_manifest.json').read_text())['notebook_sha256']
    notebook = copy.deepcopy(json.loads(parent))
    adapter = (HERE / 'gold_validation.py').read_text(encoding='utf-8')
    diagnostics = (HERE / 'diagnostics.py').read_text(encoding='utf-8')
    diagnostics = diagnostics.split("if __name__ == '__main__':")[0]
    fusion = (HERE / 'fusion.py').read_text(encoding='utf-8')
    cells = notebook['cells']
    for i, cell in enumerate(cells):
        code = ''.join(cell['source']).replace('/kaggle/working', '/kaggle/working/v14_validation')
        if i == 1:
            # Remove competition-ready function entirely for this validation artifact.
            code = code.split('def v13_finalize_submission(run):')[0]
            code += "V13_PREFLIGHT = v13_preflight(input_root=GOLD_ROOT.parent, work=GOLD_WORK)\n"
        if i == 2:
            code = replace_once(code, '9 * 3600, 900', '2 * 3600, 600')
            code = code.replace('6.0 * 3600', '1.5 * 3600').replace('7.5 * 3600', '1.6 * 3600').replace('8.0 * 3600', '1.7 * 3600')
        if i == 3:
            code = replace_once(code, 'frame.to_csv(tmp, index=False)',
                                "frame.to_csv(tmp, index=False, float_format='%.17g')")
        if i == 11:
            code = replace_once(code, 'def find_root():\n', 'def find_root():\n    return GOLD_ROOT\n')
        if i == 29:
            code = replace_once(code, "COMP = _find_dir('rsna-knee-abnormality-detection')", 'COMP = GOLD_ROOT')
        if i == 33:
            code = 'GOLD_STAGE2_COMPLETE = False\n' + code
            code = replace_once(code, "        wall_log(f'stage 2 complete:", "        GOLD_STAGE2_COMPLETE = True\n        wall_log(f'stage 2 complete:")
        if i == 35:
            code = replace_once(code, '        _rad_main()\n', '        _rad_main()\n        GOLD_STAGE3_COMPLETE = True\n')
            code = replace_once(code, 'if not GPU_OK:\n', 'GOLD_STAGE3_COMPLETE = False\nif not GPU_OK:\n')
        if i == 37:
            code = replace_once(code, '    def find_test_root():\n', '    def find_test_root():\n        return str(GOLD_ROOT)\n')
        if i == 39:
            code = """# Validation-only completion gate: no leaderboard READY marker.
events = V13_RUN.receipt['events']
members = [e for e in events if e['event'] == 'stage1_member_committed']
arms = [e for e in events if e['event'] == 'stage4_arm_completed']
assert len({e['member'] for e in members}) == 24, 'Incomplete stage-1 baseline'
assert GOLD_STAGE2_COMPLETE and GOLD_STAGE3_COMPLETE, 'Incomplete DINOv3/Rad baseline'
assert len({e['arm'] for e in arms}) == 3, 'Incomplete CoAtNet baseline'
assert not any(e['event'] == 'stage1_budget_skip' for e in events), 'Budget-confounded validation'
V13_RUN.snapshot('final')
print('All model stages completed; validation only, DO NOT SUBMIT.')
"""
        cell['source'] = code.splitlines(keepends=True)
    cells[0]['source'] = ["# V14 private gold validation — exploratory, DO NOT SUBMIT\n",
        "V13 control plus six predeclared CoAtNet ablations. Gold selects pretrained checkpoints, so this is NOT independent confirmation. No model training, no leaderboard submission.\n",
        "All inference roots redirect to expert-labeled training studies; labels/reports are stripped from the pseudo-test input. Outputs stay under v14_validation/.\n",
        "Original pretrained/source credit: dreaddevelopment, pilkwang, mattiaangeli, marwanmath, antoinegg1, prvsiyan, sofiaanjenje, tonylica, stevenleehans, cf696666, romantamrazov. Original licenses apply.\n"]
    setup = adapter + """
_gold_real = next((p for p in [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), Path('/kaggle/input/rsna-knee-abnormality-detection')] if (p / 'train.csv').is_file()), None)
assert _gold_real is not None, 'Attach RSNA Knee competition data'
GOLD_ROOT, GOLD_WORK = prepare_gold(_gold_real, Path('/kaggle/working/v14_validation'), '/kaggle/temp')
GOLD_LOAD_AUDIT = install_checkpoint_audit(GOLD_WORK, pd.read_csv(GOLD_WORK / 'gold_study_ids.csv', dtype=str)[GOLD_UID])
"""
    def cell(code):
        return {'cell_type': 'code', 'metadata': {}, 'source': code.splitlines(keepends=True), 'outputs': [], 'execution_count': None}
    cells.insert(1, cell(setup))
    finish = fusion + '\nimport types as _gold_types\n_GOLD_DIAGNOSTICS = _gold_types.ModuleType("gold_diagnostics")\n'
    finish += 'exec(' + repr(diagnostics) + ', _GOLD_DIAGNOSTICS.__dict__)\n'
    finish += """
GOLD_ABLATIONS = v14_export_ablations(V13_RUN)
GOLD_RESULTS = evaluate_gold(V13_RUN, GOLD_ABLATIONS, _GOLD_DIAGNOSTICS)
# Rename only this run's primary output, so no root submission.csv can be selected.
V13_RUN.primary.rename(GOLD_WORK / 'v13_gold_predictions.csv')
gold_write(GOLD_WORK / 'validation_complete.json', {'status': 'EXPLORATORY_COMPLETE_DO_NOT_SUBMIT', 'plus_0_02_verified': False})
"""
    cells.append(cell(finish))
    for i, c in enumerate(cells):
        c['id'] = f'gold-validation-{i:03d}'
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), c['id'], 'exec')
            c['execution_count'], c['outputs'] = None, []
    info = {'parent_v13_sha256': parent_sha, 'adapter_sha256': hashlib.sha256(adapter.encode()).hexdigest(),
            'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'EXPLORATORY_GOLD_SELECTED', 'training_enabled': False, 'submission_allowed': False,
            'plus_0_02_verified': False}
    notebook['metadata']['v14_validation'] = info
    path = HERE / 'rsna-knee-v14-gold-validation.ipynb'
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    (HERE / 'gold_validation_manifest.json').write_text(json.dumps({**info, 'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'cells': len(cells)}, indent=2) + '\n')
    return notebook


if __name__ == '__main__':
    print('Built validation notebook with', len(build()['cells']), 'cells')
