"""Build a minimal, standalone notebook that extracts frozen DINOv2 pooled features
per study -- genuinely new image information, not the pipeline's own fused predictions
(that was the flaw in the first residual-head attempt: it only saw the model's own
already-fused rank outputs and just re-pointed them at noisier pseudo-labels).

Reuses only the already-tested V13 utility cells verbatim (config constants, log/
find_root/find_dinov2/ROOT, probe/walk/annotate, pick_slots, order_slices/read_slot,
SlotHead/Model/build_model) -- copied by cell index, not retyped, and never modified.
Skips the 24-member ensemble and stages 2-4 entirely: this notebook never blends,
never fuses, never writes a submission-shaped output, and is not a leaderboard
candidate. It matches Stage 1's real preprocessing exactly (plane_map from
test_series.csv's Anatomical_Plane, fatsat/fluid/px from walk()+annotate(), CROP_MM/
GROUP/IMG/SLICE_BAND from the same config cell), so the extracted features reflect
what the real model actually sees, not a improvised shortcut.
"""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V13_CELLS = [9, 11, 12, 13, 14, 17, 18, 19]

EXTRACT_CELL = '''
FEATURE_MODEL = build_model(0, variant='small', pool='cls_mean').to(DEVS[0]).eval()
DIM = FEATURE_MODEL.backbone.config.hidden_size * POOL_PARTS['cls_mean']

test_series = pd.read_csv(ROOT / 'test_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
plane_map = dict(zip(test_series['SeriesInstanceUID'], test_series['Anatomical_Plane']))
hte = annotate(walk('test_series'))
slot_map = pick_slots(hte, plane_map)
test_ids = pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()
log(f'feature extraction: {len(test_ids)} studies, {N_SLOT} slots, dim {DIM}')

N = len(test_ids)
feats = np.zeros((N, N_SLOT, DIM), np.float32)
present = np.zeros((N, N_SLOT), np.float32)
BATCH = 16
with torch.no_grad():
    for start in range(0, N, BATCH):
        chunk = test_ids[start:start + BATCH]
        imgs = np.zeros((len(chunk), N_SLOT, GROUP, IMG, IMG), np.uint8)
        mask_chunk = np.zeros((len(chunk), N_SLOT), np.float32)
        for bi, sid in enumerate(chunk):
            chosen = slot_map.get(sid, {})
            for k, (name, plane, fluid, fs) in enumerate(SLOTS):
                if name not in chosen:
                    continue
                rec = dict(chosen[name])
                rec['ordered'], _ok = order_slices(rec)
                tile = read_slot(rec)
                if tile is None:
                    continue
                imgs[bi, k] = tile.numpy()
                mask_chunk[bi, k] = 1.0
        t = torch.from_numpy(imgs).to(DEVS[0])
        B, S = t.shape[:2]
        x = t.reshape(B * S, *t.shape[2:]).float().div_(255.0)
        x = (x - FEATURE_MODEL.mean) / FEATURE_MODEL.std
        with torch.autocast('cuda', enabled=DEVS[0].type == 'cuda'):
            out = FEATURE_MODEL.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = torch.cat([out[:, 0], patch.mean(1)], dim=1).float()
        feats[start:start + len(chunk)] = parts.reshape(B, S, -1).cpu().numpy()
        present[start:start + len(chunk)] = mask_chunk
        log(f'features {min(start + len(chunk), N)}/{N}')

pooled = (feats * present[..., None]).sum(axis=1) / np.clip(present.sum(axis=1, keepdims=True), 1, None)
if not np.isfinite(pooled).all():
    raise ValueError('non-finite pooled feature; refuse to save a corrupt feature file')
np.savez_compressed(WORK / 'image_features.npz', StudyInstanceUID=np.array(test_ids, dtype=str),
                    features=pooled.astype(np.float32), slot_present=present)
log(f'saved image_features.npz: {pooled.shape}, mean slots present {present.sum(1).mean():.2f}/{N_SLOT}')
print('DONE, feature extraction only, DO NOT SUBMIT.', flush=True)
'''


def cell(code):
    return {'cell_type': 'code', 'metadata': {}, 'source': code.splitlines(keepends=True),
            'outputs': [], 'execution_count': None}


def build(mode, n=None):
    if mode not in ('pool', 'gold'):
        raise ValueError('mode must be "pool" or "gold"')
    parent_path = HERE.parent / 'v13/rsna-knee-ensemble-v13.ipynb'
    parent = json.loads(parent_path.read_bytes())
    v13_cells = parent['cells']
    notebook = {'cells': [], 'metadata': {'kernelspec': parent['metadata']['kernelspec'],
                                          'language_info': parent['metadata']['language_info']},
                'nbformat': parent['nbformat'], 'nbformat_minor': parent['nbformat_minor']}
    header = ('# V14 private image-feature extraction -- exploratory, DO NOT SUBMIT\n\n'
              f'Mode: {mode}. Extracts frozen DINOv2-small pooled (CLS+mean-token) features per '
              'study for residual_specialist.py, using the real Stage-1 slot selection and slice '
              'reading verbatim. No 24-member ensemble, no stages 2-4, no submission-shaped output. '
              'Original pretrained/source credit: dreaddevelopment, pilkwang, mattiaangeli, marwanmath, '
              'antoinegg1, prvsiyan, sofiaanjenje, tonylica, stevenleehans, cf696666, romantamrazov.\n')
    notebook['cells'].append(cell('"""%s"""' % header) if False else {
        'cell_type': 'markdown', 'metadata': {}, 'source': header.splitlines(keepends=True)})

    if mode == 'pool':
        adapter = (HERE / 'specialist_pool.py').read_text(encoding='utf-8')
        n = n or 500
        setup = adapter + f"""
_real = next((p for p in [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), Path('/kaggle/input/rsna-knee-abnormality-detection')] if (p / 'train.csv').is_file()), None)
assert _real is not None, 'Attach RSNA Knee competition data'
_FEATURE_ROOT, WORK = prepare_pool(_real, Path('/kaggle/working/v14_image_features_pool'), n={n}, temporary_parent='/kaggle/temp')
"""
    else:
        adapter = (HERE / 'gold_validation.py').read_text(encoding='utf-8')
        setup = adapter + """
_real = next((p for p in [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), Path('/kaggle/input/rsna-knee-abnormality-detection')] if (p / 'train.csv').is_file()), None)
assert _real is not None, 'Attach RSNA Knee competition data'
_FEATURE_ROOT, WORK = prepare_gold(_real, Path('/kaggle/working/v14_image_features_gold'), '/kaggle/temp')
"""
    notebook['cells'].append(cell(setup))

    for idx in V13_CELLS:
        code = ''.join(v13_cells[idx]['source'])
        if idx == 11:
            code = code.replace('def find_root():\n', 'def find_root():\n    return _FEATURE_ROOT\n', 1)
            if code.count('def find_root():\n    return _FEATURE_ROOT\n') != 1:
                raise ValueError('find_root override anchor drift')
        notebook['cells'].append(cell(code))

    notebook['cells'].append(cell(EXTRACT_CELL))
    for i, c in enumerate(notebook['cells']):
        c['id'] = f'image-features-{mode}-{i:03d}'
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), c['id'], 'exec')
            c['execution_count'], c['outputs'] = None, []

    info = {'mode': mode, 'v13_cells_reused': V13_CELLS, 'pool_studies': n if mode == 'pool' else None,
            'adapter_sha256': hashlib.sha256(adapter.encode()).hexdigest(),
            'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'submission_allowed': False, 'training_enabled': False}
    notebook['metadata']['v14_image_features'] = info
    path = HERE / f'rsna-knee-v14-image-features-{mode}.ipynb'
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    (HERE / f'image_features_{mode}_manifest.json').write_text(json.dumps(
        {**info, 'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
         'cells': len(notebook['cells'])}, indent=2) + '\n')
    return notebook


if __name__ == '__main__':
    for m in ('pool', 'gold'):
        nb = build(m)
        print(f'Built {m} image-features notebook with {len(nb["cells"])} cells')
