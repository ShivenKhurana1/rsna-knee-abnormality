"""Chunked, checksum-backed slot-tensor cache: one file per study, never one
giant fragile array, per the plan's cache design. Real cache contents are built
on the GPU device by build_pool_notebook.py (which reuses V13's own
pick_slots/order_slices/read_slot cells against real DICOMs); synth() here
builds a structurally identical fake cache from random tensors so every other
Family A component can be exercised on CPU without any competition data.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from contract import GROUP, IMG, N_SLOT, UID


def _study_path(cache_dir, uid):
    if '/' in uid or '\\' in uid or uid in ('.', '..'):
        raise ValueError('Unsafe study identifier')
    return Path(cache_dir) / f'{uid}.npz'


def write_study(cache_dir, uid, imgs, mask):
    imgs = np.asarray(imgs)
    mask = np.asarray(mask)
    if imgs.shape != (N_SLOT, GROUP, IMG, IMG):
        raise ValueError(f'imgs must be shape {(N_SLOT, GROUP, IMG, IMG)}, got {imgs.shape}')
    if mask.shape != (N_SLOT,):
        raise ValueError(f'mask must be shape {(N_SLOT,)}, got {mask.shape}')
    if not mask.any():
        raise ValueError(f'{uid}: at least one slot must be present')
    if imgs.dtype != np.uint8:
        raise ValueError('imgs must be uint8 (percentile-normalized, 0-255)')
    path = _study_path(cache_dir, uid)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, imgs=imgs, mask=mask.astype(np.float32))
    return path


def build_manifest(cache_dir, ids):
    cache_dir = Path(cache_dir)
    entries = {}
    missing = []
    for uid in ids:
        path = _study_path(cache_dir, uid)
        if not path.is_file():
            missing.append(uid)
            continue
        with np.load(path) as f:
            imgs, mask = f['imgs'], f['mask']
        if imgs.shape != (N_SLOT, GROUP, IMG, IMG) or mask.shape != (N_SLOT,):
            raise ValueError(f'{uid}: cached tensor does not match the current contract shape')
        entries[uid] = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'n_slots_present': int(mask.sum())}
    if missing:
        raise FileNotFoundError(f'{len(missing)} studies missing from cache, e.g. {missing[:3]}')
    manifest = {'contract': {'n_slot': N_SLOT, 'group': GROUP, 'img': IMG},
                'studies': entries, 'n_studies': len(entries)}
    (cache_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def load_manifest(cache_dir):
    return json.loads((Path(cache_dir) / 'manifest.json').read_text(encoding='utf-8'))


def load_study(cache_dir, uid):
    with np.load(_study_path(cache_dir, uid)) as f:
        return f['imgs'], f['mask']


def load_batch(cache_dir, ids):
    imgs, masks = [], []
    for uid in ids:
        i, m = load_study(cache_dir, uid)
        imgs.append(i)
        masks.append(m)
    return np.stack(imgs), np.stack(masks)


def synth(cache_dir, ids, seed=1400, min_slots_present=3):
    """Build a structurally valid fake cache for the CPU smoke test. Not real
    image data; exercises every downstream shape/mask/manifest code path."""
    rng = np.random.default_rng(seed)
    for uid in ids:
        n_present = rng.integers(min_slots_present, N_SLOT + 1)
        mask = np.zeros(N_SLOT, dtype=bool)
        mask[rng.choice(N_SLOT, size=n_present, replace=False)] = True
        imgs = rng.integers(0, 256, size=(N_SLOT, GROUP, IMG, IMG), dtype=np.uint8)
        write_study(cache_dir, uid, imgs, mask)
    return build_manifest(cache_dir, ids)
