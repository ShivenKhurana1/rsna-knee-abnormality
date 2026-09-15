"""Build a private, exploratory notebook that runs frozen V13 on a non-gold training
sample and exports its per-stage/per-arm predictions as residual_specialist.py features.

Unlike rsna-knee-v14-gold-validation.ipynb this notebook never reads report labels or
train.csv's finding columns beyond the gold-exclusion check, never trains anything, and
keeps the original 9-hour wall-clock budget untouched: POOL_N studies at the pipeline's
own measured per-study costs (config cell: ~11.6 s/study marginal, all four stages) is
comfortably inside it, so none of the stage-start cutoffs used by the real submission
are shortened here the way the gold-validation builder shortens them for 58 studies.
"""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
POOL_N = 500


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one source anchor: {old[:90]!r}, found {text.count(old)}')
    return text.replace(old, new)


def build():
    parent_path = HERE.parent / 'v13/rsna-knee-ensemble-v13.ipynb'
    parent = parent_path.read_bytes()
    parent_sha = hashlib.sha256(parent).hexdigest()
    # v13/build_manifest.json's recorded notebook_sha256 is already stale against the committed
    # v13 notebook (build_gold_validation.py's identical assertion fails the same way, unrelated
    # to this file) -- build from the actual committed notebook rather than gate on that record.
    notebook = copy.deepcopy(json.loads(parent))
    adapter = (HERE / 'specialist_pool.py').read_text(encoding='utf-8')
    cells = notebook['cells']
    for i, cell in enumerate(cells):
        code = ''.join(cell['source']).replace('/kaggle/working', '/kaggle/working/v14_pool')
        if i == 1:
            # Remove the competition-ready marker entirely: this run is never submitted.
            code = code.split('def v13_finalize_submission(run):')[0]
            code += "V13_PREFLIGHT = v13_preflight(input_root=POOL_ROOT.parent, work=POOL_WORK)\n"
        if i == 3:
            code = replace_once(code, 'frame.to_csv(tmp, index=False)',
                                "frame.to_csv(tmp, index=False, float_format='%.17g')")
        if i == 11:
            code = replace_once(code, 'def find_root():\n', 'def find_root():\n    return POOL_ROOT\n')
        if i == 29:
            code = replace_once(code, "COMP = _find_dir('rsna-knee-abnormality-detection')", 'COMP = POOL_ROOT')
        if i == 37:
            code = replace_once(code, '    def find_test_root():\n', '    def find_test_root():\n        return str(POOL_ROOT)\n')
        if i == 39:
            code = """# Pool-feature completion gate: no leaderboard READY marker, no training.
events = V13_RUN.receipt['events']
snapshots = {s['stage'] for s in V13_RUN.receipt['snapshots']}
members = [e for e in events if e['event'] == 'stage1_member_committed']
arms = [e for e in events if e['event'] == 'stage4_arm_completed']
assert len({e['member'] for e in members}) == 24, 'Incomplete stage-1 baseline'
assert 'stage3' in snapshots, 'Incomplete DINOv3/Rad baseline'
assert len({e['arm'] for e in arms}) == 3, 'Incomplete CoAtNet baseline'
assert not any(e['event'] == 'stage1_budget_skip' for e in events), 'Budget-confounded pool run'
V13_RUN.snapshot('final')
V13_RUN.primary.rename(POOL_WORK / 'pool_final_unused.csv')
print(f'Total wall clock: {(time.time() - WALL_T0) / 60:.1f} minutes')
print('All model stages completed for', len(V13_RUN.ids), 'pool studies; feature export only, DO NOT SUBMIT.')
import os as _pool_os
print('run.folder contents:', sorted(_pool_os.listdir(V13_RUN.folder)))
"""
        cell['source'] = code.splitlines(keepends=True)
    cells[0]['source'] = ["# V14 private specialist training pool -- exploratory, DO NOT SUBMIT\n",
        f"V13 control run on {POOL_N} non-gold train.csv studies, purely to export its normal "
        "per-stage/per-arm predictions (stage1.csv, stage3.csv, coatnet_arm_*_raw.csv) as features "
        "for residual_specialist.py. Report labels are never read here; report-derived pseudo-labels "
        "are matched to these study IDs and applied locally afterward. No training happens on Kaggle.\n",
        "All inference roots redirect to a deterministic sample of non-gold training studies. "
        "Outputs stay under v14_pool/.\n",
        "Original pretrained/source credit: dreaddevelopment, pilkwang, mattiaangeli, marwanmath, "
        "antoinegg1, prvsiyan, sofiaanjenje, tonylica, stevenleehans, cf696666, romantamrazov. "
        "Original licenses apply.\n"]
    setup = adapter + f"""
_pool_real = next((p for p in [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), Path('/kaggle/input/rsna-knee-abnormality-detection')] if (p / 'train.csv').is_file()), None)
assert _pool_real is not None, 'Attach RSNA Knee competition data'
POOL_ROOT, POOL_WORK = prepare_pool(_pool_real, Path('/kaggle/working/v14_pool'), n={POOL_N}, temporary_parent='/kaggle/temp')
"""
    def cell(code):
        return {'cell_type': 'code', 'metadata': {}, 'source': code.splitlines(keepends=True), 'outputs': [], 'execution_count': None}
    cells.insert(1, cell(setup))
    for i, c in enumerate(cells):
        c['id'] = f'specialist-pool-{i:03d}'
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), c['id'], 'exec')
            c['execution_count'], c['outputs'] = None, []
    info = {'parent_v13_sha256': parent_sha, 'adapter_sha256': hashlib.sha256(adapter.encode()).hexdigest(),
            'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'EXPLORATORY_TRAINING_POOL_FEATURES', 'pool_studies': POOL_N,
            'training_enabled': False, 'submission_allowed': False, 'plus_0_02_verified': False}
    notebook['metadata']['v14_specialist_pool'] = info
    path = HERE / 'rsna-knee-v14-specialist-pool.ipynb'
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    (HERE / 'specialist_pool_manifest.json').write_text(json.dumps(
        {**info, 'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'cells': len(cells)},
        indent=2) + '\n')
    return notebook


if __name__ == '__main__':
    print('Built specialist-pool notebook with', len(build()['cells']), 'cells')
