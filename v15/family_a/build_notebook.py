"""Builds the Kaggle notebook(s) that actually run Family A / Run 4 on real data
and a real GPU. Reuses V13's own already-tested cells for the data contract
(constants, find_root/find_dinov2, pick_slots, order_slices, read_slot) so the
slot tensors Family A trains on come from the same pipeline already validated
in production -- not a second, independently written DICOM decoder. Everything
new (the cache-writing cell, the family_a package, the training cells) is
appended after those reused cells, following the exact pattern
v14/build_image_features.py already established for this repository.

Two notebooks are built: one over the ~4,349-study report-only pool (for the
auxiliary-loss pretraining signal) and one over the 58-study gold cohort (for
expert-label supervision and evaluation). Run the pool notebook first; the gold
notebook's cache path is what train_run4.py's CLI expects for the actual
baseline-vs-auxiliary comparison.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V13_NOTEBOOK = HERE.parent.parent / 'v13' / 'rsna-knee-ensemble-v13.ipynb'
V14_DIR = HERE.parent.parent / 'v14'
V13_CELLS = [9, 11, 12, 13, 14]  # constants/helpers + slot selection + read_slot; NOT V13's own model
FAMILY_A_MODULES = ['contract.py', 'labels.py', 'model.py', 'folds.py', 'losses.py',
                     'cache.py', 'engine.py', 'compare_oof.py', 'family_c_gate.py']
KAGGLE_SRC_DIR = '/kaggle/working/family_a_src'


def cell(code):
    return {'cell_type': 'code', 'metadata': {}, 'source': code.splitlines(keepends=True),
            'outputs': [], 'execution_count': None}


def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def embed_modules_cell():
    """Writes every family_a/*.py file to KAGGLE_SRC_DIR on the Kaggle runtime,
    verbatim, using repr() so no manual quote-escaping is needed regardless of
    what each module's docstrings contain."""
    lines = ["import sys", "from pathlib import Path",
            f"_src = Path({KAGGLE_SRC_DIR!r})", "_src.mkdir(parents=True, exist_ok=True)"]
    for name in FAMILY_A_MODULES:
        content = (HERE / name).read_text(encoding='utf-8')
        lines.append(f"(_src / {name!r}).write_text({content!r}, encoding='utf-8')")
    lines.append(f"sys.path.insert(0, {KAGGLE_SRC_DIR!r})")
    lines.append("print('family_a modules written to', _src)")
    return cell('\n'.join(lines))


def cache_build_cell(cache_out_dir):
    return cell(f'''
import cache as family_a_cache

test_series = pd.read_csv(ROOT / 'test_series.csv', dtype={{'StudyInstanceUID': str, 'SeriesInstanceUID': str}})
plane_map = dict(zip(test_series['SeriesInstanceUID'], test_series['Anatomical_Plane']))
hte = annotate(walk('test_series'))
slot_map = pick_slots(hte, plane_map)
study_ids = pd.read_csv(ROOT / 'test.csv', dtype={{'StudyInstanceUID': str}})['StudyInstanceUID'].tolist()
log(f'cache build: {{len(study_ids)}} studies, {{N_SLOT}} slots')

CACHE_OUT = Path({cache_out_dir!r})
built = []
for sid in study_ids:
    chosen = slot_map.get(sid, {{}})
    imgs = np.zeros((N_SLOT, GROUP, IMG, IMG), np.uint8)
    mask = np.zeros(N_SLOT, dtype=bool)
    for k, (name, plane, fluid, fs) in enumerate(SLOTS):
        if name not in chosen:
            continue
        rec = dict(chosen[name])
        rec['ordered'], _ok = order_slices(rec)
        tile = read_slot(rec)
        if tile is None:
            continue
        imgs[k] = tile.numpy()
        mask[k] = True
    if not mask.any():
        log(f'WARNING: {{sid}} has no usable slot, skipping (no expert/aux signal possible for it)')
        continue
    family_a_cache.write_study(CACHE_OUT, sid, imgs, mask)
    built.append(sid)
    if len(built) % 200 == 0:
        log(f'cached {{len(built)}}/{{len(study_ids)}}')

family_a_cache.build_manifest(CACHE_OUT, built)
pd.Series(built, name='StudyInstanceUID').to_csv(CACHE_OUT / 'cached_ids.csv', index=False)
print(f'DONE caching {{len(built)}}/{{len(study_ids)}} studies to {{CACHE_OUT}}. Cache build only, DO NOT SUBMIT.', flush=True)
''')


def build_cache_notebook(mode, n_pool=None):
    """mode: 'pool' or 'gold'. Mirrors v14/build_image_features.py's adapter
    reuse exactly, but caches raw uint8 tensors (for a trainable encoder)
    instead of frozen pooled features."""
    if mode not in ('pool', 'gold'):
        raise ValueError("mode must be 'pool' or 'gold'")
    parent = json.loads(V13_NOTEBOOK.read_bytes())
    notebook = {'cells': [], 'metadata': {'kernelspec': parent['metadata']['kernelspec'],
                                          'language_info': parent['metadata']['language_info']},
                'nbformat': parent['nbformat'], 'nbformat_minor': parent['nbformat_minor']}
    header = (f'# Family A / Run 4 -- {mode} cache build. Exploratory, DO NOT SUBMIT.\n\n'
              'Reuses V13\'s own pick_slots/order_slices/read_slot cells verbatim against real '
              'DICOMs, then caches raw uint8 slot tensors (not frozen pooled features) for a '
              'trainable Family A encoder. No 24-member ensemble, no submission-shaped output.\n')
    notebook['cells'].append(markdown(header))

    # RSNA_DATA_ROOT lets this run unmodified on a non-Kaggle GPU device: either
    # set that env var to the real data directory, or mkdir -p a literal
    # /kaggle/input/rsna-knee-abnormality-detection (e.g. under WSL2 on Windows)
    # so the untouched, already-tested Kaggle-path search below still finds it.
    data_search = '''[Path(os.environ['RSNA_DATA_ROOT'])] if os.environ.get('RSNA_DATA_ROOT') else []) + [
    Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
    Path('/kaggle/input/rsna-knee-abnormality-detection'),
    Path('data/rsna-knee-abnormality-detection'), Path('rsna-knee-abnormality-detection')]'''
    if mode == 'pool':
        adapter = (V14_DIR / 'specialist_pool.py').read_text(encoding='utf-8')
        n = n_pool or 4349
        setup = adapter + f'''
_real = next((p for p in ({data_search} if (p / 'train.csv').is_file()), None)
assert _real is not None, 'Attach RSNA Knee competition data, or set RSNA_DATA_ROOT'
_ROOT, WORK = prepare_pool(_real, Path('/kaggle/working/family_a_pool_prep'), n={n}, temporary_parent='/kaggle/temp')
'''
        cache_out = '/kaggle/working/family_a_pool_cache'
    else:
        adapter = (V14_DIR / 'gold_validation.py').read_text(encoding='utf-8')
        setup = adapter + f'''
_real = next((p for p in ({data_search} if (p / 'train.csv').is_file()), None)
assert _real is not None, 'Attach RSNA Knee competition data, or set RSNA_DATA_ROOT'
_ROOT, WORK = prepare_gold(_real, Path('/kaggle/working/family_a_gold_prep'), '/kaggle/temp')
'''
        cache_out = '/kaggle/working/family_a_gold_cache'
    notebook['cells'].append(cell(setup))

    for idx in V13_CELLS:
        code = ''.join(parent['cells'][idx]['source'])
        if idx == 11:
            code = code.replace('def find_root():\n', 'def find_root():\n    return _ROOT\n', 1)
            if code.count('def find_root():\n    return _ROOT\n') != 1:
                raise ValueError('find_root override anchor drift; V13 cell 11 changed shape')
        notebook['cells'].append(cell(code))

    notebook['cells'].append(embed_modules_cell())
    notebook['cells'].append(cache_build_cell(cache_out))

    for i, c in enumerate(notebook['cells']):
        c['id'] = f'family-a-cache-{mode}-{i:03d}'
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), c['id'], 'exec')
            c['execution_count'], c['outputs'] = None, []

    info = {'mode': mode, 'v13_cells_reused': V13_CELLS, 'cache_out_dir': cache_out,
            'family_a_modules_embedded': FAMILY_A_MODULES,
            'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'submission_allowed': False, 'training_enabled': False}
    notebook['metadata']['family_a_cache'] = info
    path = HERE / f'rsna-knee-v15-family-a-cache-{mode}.ipynb'
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    (HERE / f'family_a_cache_{mode}_manifest.json').write_text(json.dumps(
        {**info, 'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
         'cells': len(notebook['cells'])}, indent=2) + '\n', encoding='utf-8')
    return notebook


def train_cell(pool_cache, gold_cache, gold_labels_csv, out_dir, policy_json,
              epochs, k, seed, patience, dinov2_variant, report_csv_text):
    return cell(f'''
import io
import json
import pandas as pd
from contract import TARGETS, UID
from labels import zero_policy
from engine import run_arm
from model import GeneralistModel, Dinov2Encoder

def make_model():
    return GeneralistModel(Dinov2Encoder(find_dinov2({dinov2_variant!r}), unfreeze_last=2))

pool_cache = Path({pool_cache!r})
gold_cache = Path({gold_cache!r})
gold_labels = pd.read_csv({gold_labels_csv!r}, dtype={{UID: str}})
pool_ids = pd.read_csv(pool_cache / 'cached_ids.csv', dtype=str)[UID].tolist()
gold_ids = pd.read_csv(gold_cache / 'cached_ids.csv', dtype=str)[UID].tolist()
ids = gold_ids + pool_ids  # gold first so gold rows spread across folds deterministically

# Every id must be readable from ONE cache directory; symlink pool+gold into one place.
import os
combined_cache = Path('/kaggle/working/family_a_combined_cache')
combined_cache.mkdir(parents=True, exist_ok=True)
for src_dir, id_list in ((gold_cache, gold_ids), (pool_cache, pool_ids)):
    for uid in id_list:
        dst = combined_cache / f'{{uid}}.npz'
        if not dst.exists():
            os.symlink((src_dir / f'{{uid}}.npz').resolve(), dst)

report_source = pd.read_csv(io.StringIO({report_csv_text!r}), dtype={{UID: str}})
auxiliary_policy = json.loads({policy_json!r})

out_dir = Path({out_dir!r})
for seed_value in [{seed}, {seed} + 1]:
    for arm_name, policy in (('baseline', zero_policy()), ('auxiliary', auxiliary_policy)):
        arm_dir = out_dir / f'seed{{seed_value}}' / arm_name
        run_arm(arm_name, combined_cache, ids, gold_ids, gold_labels, report_source, policy,
               arm_dir, make_model, k={k}, epochs={epochs}, seed=seed_value, patience={patience})
        print(f'seed {{seed_value}} {{arm_name}} done -> {{arm_dir}}')
print('DONE training both arms, both seeds. Run compare_oof.py and family_c_gate.py next.')
''')


def build_train_notebook(pool_cache, gold_cache, gold_labels_csv, transfer_audit_report,
                         out_dir='/kaggle/working/family_a_run4', epochs=10, k=5, seed=1400,
                         patience=4, dinov2_variant='small'):
    """A second notebook, run after both cache notebooks finish, that trains both
    arms at both seeds using a real DINOv2 encoder. Assumes family_a_src/ has
    already been written by the cache notebook(s) in this Kaggle session; if
    starting fresh, run a cache notebook first or copy that cell in."""
    from labels import load_transfer_policy
    policy = load_transfer_policy(transfer_audit_report)
    policy_json = json.dumps(policy)
    report_csv_text = (V14_DIR / 'external_labels' / 'llm_labels_v2.csv').read_text(encoding='utf-8')

    parent = json.loads(V13_NOTEBOOK.read_bytes())
    notebook = {'cells': [], 'metadata': {'kernelspec': parent['metadata']['kernelspec'],
                                          'language_info': parent['metadata']['language_info']},
                'nbformat': parent['nbformat'], 'nbformat_minor': parent['nbformat_minor']}
    notebook['cells'].append(markdown(
        '# Family A / Run 4 training -- baseline vs. auxiliary, two seeds. Exploratory, DO NOT SUBMIT.\n\n'
        'Requires the pool and gold cache notebooks to have already run in this session '
        '(or an equivalent /kaggle/working/family_a_src + cache directories).\n'))
    notebook['cells'].append(cell(''.join(parent['cells'][9]['source'])))   # constants incl. TARGETS, DEVS
    notebook['cells'].append(cell(''.join(parent['cells'][11]['source'])))  # find_dinov2
    notebook['cells'].append(embed_modules_cell())
    notebook['cells'].append(train_cell(pool_cache, gold_cache, gold_labels_csv, out_dir,
                                        policy_json, epochs, k, seed, patience, dinov2_variant,
                                        report_csv_text))
    for i, c in enumerate(notebook['cells']):
        c['id'] = f'family-a-train-{i:03d}'
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), c['id'], 'exec')
            c['execution_count'], c['outputs'] = None, []
    info = {'epochs': epochs, 'k': k, 'seed': seed, 'patience': patience, 'policy': policy,
            'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'submission_allowed': False, 'training_enabled': True}
    notebook['metadata']['family_a_train'] = info
    path = HERE / 'rsna-knee-v15-family-a-train.ipynb'
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    (HERE / 'family_a_train_manifest.json').write_text(json.dumps(
        {**info, 'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
         'cells': len(notebook['cells'])}, indent=2) + '\n', encoding='utf-8')
    return notebook


if __name__ == '__main__':
    import sys
    for m in ('pool', 'gold'):
        nb = build_cache_notebook(m)
        print(f'Built {m} cache notebook with {len(nb["cells"])} cells')
    report = HERE.parent / 'transfer_audit_report.json'
    if report.is_file():
        nb = build_train_notebook(
            pool_cache='/kaggle/working/family_a_pool_cache', gold_cache='/kaggle/working/family_a_gold_cache',
            gold_labels_csv='/kaggle/working/family_a_gold_prep/gold_labels.csv',
            transfer_audit_report=report)
        print(f'Built train notebook with {len(nb["cells"])} cells')
    else:
        print('transfer_audit_report.json not found; skipping train notebook build', file=sys.stderr)
