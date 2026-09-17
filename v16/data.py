"""Real single-frame MRI DICOM decoding and versioned, per-study disk caches."""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .common import atomic_json, child, fingerprint, sha

DECODER_VERSION = 'v16-dicom-2-compact'
TRANSFORM_KEYS = ('image_size', 'crop_mm', 'offsets', 'max_centres', 'max_series', 'percentiles')


def transform_contract(cfg):
    import pydicom
    return {'version': DECODER_VERSION, 'pydicom': pydicom.__version__,
            'torch': str(torch.__version__), 'numpy': np.__version__,
            'cache_dtype': 'float16-normalized',
            **{k: cfg[k] for k in TRANSFORM_KEYS}}


def _vector(ds, key, n):
    try:
        value = np.asarray(getattr(ds, key), float)
        return value if value.shape == (n,) and np.isfinite(value).all() else None
    except (AttributeError, ValueError, TypeError):
        return None


def _number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _flag(value):
    if pd.isna(value):
        return -1.
    text = str(value).strip().lower()
    return {'1': 1., '1.0': 1., 'true': 1., '0': 0., '0.0': 0., 'false': 0.}.get(text, -1.)


def resize_pad(array, size):
    x = torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32)
    h, w = x.shape
    scale = min(size / h, size / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    x = F.interpolate(x[None, None], size=(rh, rw), mode='bilinear',
                      align_corners=False, antialias=True)[0, 0]
    top, left = (size - rh) // 2, (size - rw) // 2
    return F.pad(x, (left, size-rw-left, top, size-rh-top))


def decode_series(directory, metadata, cfg, expected_study=None, expected_series=None):
    import pydicom
    from pydicom.pixels import apply_modality_lut
    events, records, seen = [], [], set()
    for path in sorted(p for p in Path(directory).rglob('*') if p.is_file() and p.suffix.lower() == '.dcm'):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if expected_study is not None and str(getattr(ds, 'StudyInstanceUID', '')) != expected_study:
                raise ValueError('DICOM study identity differs from containing study')
            if expected_series is not None and str(getattr(ds, 'SeriesInstanceUID', '')) != expected_series:
                raise ValueError('DICOM series identity differs from containing series')
            sop = str(getattr(ds, 'SOPInstanceUID', path))
            if sop in seen:
                events.append({'kind': 'duplicate_sop', 'file': str(path)})
                continue
            seen.add(sop)
            if int(getattr(ds, 'NumberOfFrames', 1)) != 1:
                raise ValueError('Enhanced/multiframe DICOM requires a separate adapter')
            iop, ipp = _vector(ds, 'ImageOrientationPatient', 6), _vector(ds, 'ImagePositionPatient', 3)
            records.append({'path': path, 'ds': ds, 'iop': iop, 'ipp': ipp,
                            'spacing': _vector(ds, 'PixelSpacing', 2),
                            'instance': _number(getattr(ds, 'InstanceNumber', None))})
        except Exception as exc:
            events.append({'kind': 'header_failure', 'file': str(path), 'error': str(exc)})
    if not records:
        return None, events + [{'kind': 'no_readable_headers'}]
    # Separate orientations/acquisitions rather than forming a false sequence.
    stacks = {}
    for r in records:
        orientation = tuple(np.round(r['iop'], 3)) if r['iop'] is not None else None
        key = (str(getattr(r['ds'], 'SeriesInstanceUID', '')),
               str(getattr(r['ds'], 'AcquisitionNumber', '')),
               str(getattr(r['ds'], 'EchoNumbers', '')), orientation)
        stacks.setdefault(key, []).append(r)
    records = max(stacks.values(), key=len)  # deterministic sorted-path tie order
    if len(stacks) > 1:
        events.append({'kind': 'selected_largest_consistent_stack', 'stacks': len(stacks)})
    iop = records[0]['iop']
    normal = None if iop is None else np.cross(iop[:3], iop[3:])
    geometry = normal is not None and np.linalg.norm(normal) > .99 and all(r['ipp'] is not None for r in records)
    if geometry:
        normal = normal / np.linalg.norm(normal)
        if normal[np.argmax(np.abs(normal))] < 0:
            normal = -normal
        records.sort(key=lambda r: (float(r['ipp'] @ normal), str(r['path'])))
        unique = {}
        for r in records:
            unique.setdefault(round(float(r['ipp'] @ normal), 4), r)
        if len(unique) != len(records):
            events.append({'kind': 'duplicate_position', 'removed': len(records) - len(unique)})
        records = list(unique.values())
    elif all(r['instance'] is not None for r in records):
        records.sort(key=lambda r: (r['instance'], str(r['path'])))
        events.append({'kind': 'instance_order_fallback'})
    else:
        return None, events + [{'kind': 'unorderable_stack'}]
    spacing = [r['spacing'] for r in records if r['spacing'] is not None and (r['spacing'] > 0).all()]
    shared_spacing = np.median(spacing, axis=0) if spacing else None
    count = min(len(records), cfg['max_centres'])
    centres = np.linspace(0, len(records)-1, count).round().astype(int)
    neighbors = np.clip(centres[:, None] + np.asarray(cfg['offsets'])[None], 0, len(records)-1)
    required = np.unique(neighbors)
    decoded, pixels = {}, []
    spacing_unknown = False
    for index in required:
        r = records[index]
        try:
            ds = pydicom.dcmread(r['path'])
            raw = ds.pixel_array
            if raw.ndim != 2 or not np.isfinite(raw).all():
                raise ValueError('Expected finite single-frame monochrome pixels')
            a = apply_modality_lut(raw, ds).astype(np.float32)
            valid = np.isfinite(a)
            padding = getattr(ds, 'PixelPaddingValue', None)
            if padding is not None:
                valid &= raw != padding
            if not valid.any():
                raise ValueError('All pixels missing/padding')
            if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
                a = a[valid].max() + a[valid].min() - a
            ps = r['spacing']
            if ps is None or not (ps > 0).all():
                ps = shared_spacing
                events.append({'kind': 'spacing_fallback', 'file': str(r['path']),
                               'policy': 'series_median' if ps is not None else 'full_fov'})
            if ps is not None:
                h, w = a.shape
                ch, cw = min(h, max(1, round(cfg['crop_mm']/ps[0]))), min(w, max(1, round(cfg['crop_mm']/ps[1])))
                y, x = (h-ch)//2, (w-cw)//2
                a, valid = a[y:y+ch, x:x+cw], valid[y:y+ch, x:x+cw]
            else:
                spacing_unknown = True
            if not valid.any():
                raise ValueError('Crop contains only padding')
            if iop is not None:
                if iop[:3][np.argmax(np.abs(iop[:3]))] < 0:
                    a, valid = a[:, ::-1], valid[:, ::-1]
                if iop[3:][np.argmax(np.abs(iop[3:]))] < 0:
                    a, valid = a[::-1], valid[::-1]
            # Deterministic bounded intensity sample; shared across this series.
            vals = a[valid].ravel()
            pixels.append(vals[::max(1, len(vals)//65536)])
            decoded[int(index)] = (a, valid)
        except Exception as exc:
            events.append({'kind': 'pixel_failure', 'file': str(r['path']), 'error': str(exc)})
    if not decoded:
        return None, events + [{'kind': 'no_decodable_pixels'}]
    lo, hi = np.percentile(np.concatenate(pixels), cfg['percentiles'])
    if not hi > lo:
        return None, events + [{'kind': 'constant_series'}]
    tiles = {}
    for index, (a, valid) in decoded.items():
        a = np.where(valid, np.clip((a-lo)/(hi-lo), 0, 1), 0)
        tiles[index] = resize_pad(a, cfg['image_size'])
    windows, positions, selected_neighbors = [], [], []
    for center, indices in zip(centres, neighbors):
        if not all(int(i) in tiles for i in indices):
            events.append({'kind': 'window_missing_neighbor', 'center': int(center)})
            continue
        windows.append(torch.stack([tiles[int(i)] for i in indices]))
        selected_neighbors.append([int(i) for i in indices])
        if geometry:
            positions.append(float(records[int(center)]['ipp'] @ normal))
        else:
            positions.append(float(center))
    if not windows:
        return None, events + [{'kind': 'no_complete_windows'}]
    positions = np.asarray(positions, np.float32)
    positions = (positions-positions.min()) / max(float(np.ptp(positions)), 1e-6)
    plane = np.zeros(3, np.float32)
    if normal is not None:
        plane[np.argmax(np.abs(normal))] = 1
    meta = np.r_[plane, _flag(metadata.get('Fluid_Sensitive', np.nan)),
                 _flag(metadata.get('Fat_Suppression', np.nan)), float(spacing_unknown or not geometry)]
    used = sorted({i for window in selected_neighbors for i in window})
    mapping = {index: i for i, index in enumerate(used)}
    return {'x': torch.stack(windows).half(), 'position': torch.from_numpy(positions),
            'tiles': torch.stack([tiles[i] for i in used]).half(),
            'indices': torch.tensor([[mapping[i] for i in window] for window in selected_neighbors]),
            'meta': torch.tensor(meta, dtype=torch.float32)}, events


def series_for_study(directory, uid, metadata, cfg):
    series, events = [], []
    dirs = sorted(p for p in directory.iterdir() if p.is_dir()) if directory.exists() else []
    dirs.sort(key=lambda p: (-sum(f.suffix.lower() == '.dcm' for f in p.rglob('*') if f.is_file()), p.name))
    for sdir in dirs:
        decoded, reasons = decode_series(sdir, metadata.get((uid, sdir.name), {}), cfg,
                                         expected_study=uid, expected_series=sdir.name)
        events.extend({'series': sdir.name, **r} for r in reasons)
        if decoded is not None:
            series.append(decoded)
        if len(series) >= cfg['max_series']:
            break
    return series, events


def build_cache(root, split, ids, cfg, cache_dir):
    root, cache_dir = Path(root), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    contract = transform_contract(cfg)
    meta_path = root / f'{split}_series.csv'
    metadata = {}
    if meta_path.exists():
        table = pd.read_csv(meta_path, dtype={cfg['id_col']: str, cfg['series_col']: str})
        if table[[cfg['id_col'], cfg['series_col']]].isna().any().any() or table.duplicated([cfg['id_col'], cfg['series_col']]).any():
            raise ValueError('Missing/duplicate study-series metadata')
        metadata = {(r[cfg['id_col']], r[cfg['series_col']]): r for r in table.to_dict('records')}
    entries = {}
    for number, uid in enumerate(ids):
        directory = child(root / f'{split}_series', uid)
        files = sorted(p for p in directory.rglob('*') if p.is_file()) if directory.exists() else []
        # Content fingerprints allow safe reuse across mount paths and detect changed pixels.
        source = [(str(p.relative_to(directory)), sha(p)) for p in files]
        series_meta = {k[1]: {str(c): (None if pd.isna(v) else v) for c, v in r.items()}
                       for k, r in metadata.items() if k[0] == uid}
        key = fingerprint({'uid': uid, 'source': source, 'metadata': series_meta, 'contract': contract})
        target = cache_dir / f'{key}.pt'
        log = cache_dir / f'{key}.json'
        if not target.exists() or not log.exists() or sha(target) != json.loads(log.read_text())['sha256']:
            series, events = series_for_study(directory, uid, metadata, cfg)
            # Save each normalized slice once; reconstruct overlapping triplets at load time.
            series = [{k: v for k, v in s.items() if k != 'x'} for s in series]
            estimate = sum(v.numel()*v.element_size() for s in series for v in s.values())
            if shutil.disk_usage(cache_dir).free < estimate + 1024**3:
                raise RuntimeError('Insufficient cache disk space: use larger local scratch or a smaller new preprocessing configuration')
            temp = target.with_suffix('.tmp')
            torch.save({'series': series}, temp)
            os.replace(temp, target)
            atomic_json(log, {'uid': uid, 'sha256': sha(target), 'series': len(series), 'events': events})
        info = json.loads(log.read_text())
        entries[uid] = {'file': target.name, **info}
        if (number+1) % 25 == 0 or number+1 == len(ids):
            print(f'Cached {number+1}/{len(ids)} studies', flush=True)
    manifest = {'split': split, 'contract': contract, 'entries': entries,
                'missing_studies': [u for u, e in entries.items() if e['series'] == 0]}
    atomic_json(cache_dir / 'manifest.json', manifest)
    return manifest


class CachedStudies(Dataset):
    def __init__(self, cache_dir, ids, training=False, dropout=0.):
        self.root, self.ids = Path(cache_dir), list(ids)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if not set(ids).issubset(self.manifest['entries']):
            raise ValueError('Cache missing expected IDs')
        self.training, self.dropout = training, dropout
        # Verify once when opening the dataset, rather than hash every epoch.
        for uid in ids:
            entry = self.manifest['entries'][uid]
            if sha(self.root / entry['file']) != entry['sha256']:
                raise ValueError(f'Cache checksum mismatch for {uid}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        uid = self.ids[i]
        record = torch.load(self.root / self.manifest['entries'][uid]['file'], weights_only=True)
        series = [{**s, 'x': s['tiles'][s['indices']]} for s in record['series']]
        if self.training and len(series) > 1:
            keep = torch.rand(len(series)) >= self.dropout
            if not keep.any():
                keep[torch.randint(len(series), (1,))] = True
            series = [s for s, k in zip(series, keep) if k]
        if self.training:
            # Shared small intensity transform; no anatomy-changing random flips.
            gain = .95 + .1 * torch.rand(1).item()
            series = [{**s, 'x': (s['x'].float()*gain).clamp(0, 1)} for s in series]
        return {'uid': uid, 'series': series}


class DicomStudies(Dataset):
    """Streaming hidden-set inference: never caches the entire test cohort."""
    def __init__(self, root, ids, cfg):
        self.root, self.ids, self.cfg = Path(root), list(ids), cfg
        self.metadata, self.events = {}, {}
        path = self.root/'test_series.csv'
        if path.exists():
            table = pd.read_csv(path, dtype={cfg['id_col']: str, cfg['series_col']: str})
            if table[[cfg['id_col'], cfg['series_col']]].isna().any().any() or table.duplicated([cfg['id_col'], cfg['series_col']]).any():
                raise ValueError('Missing/duplicate test series metadata')
            self.metadata = {(r[cfg['id_col']], r[cfg['series_col']]): r for r in table.to_dict('records')}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        uid = self.ids[i]
        series, events = series_for_study(child(self.root/'test_series', uid), uid, self.metadata, self.cfg)
        self.events[uid] = events
        return {'uid': uid, 'series': [{k: v for k, v in s.items() if k not in ('tiles', 'indices')} for s in series]}


def collate(items):
    series, owners = [], []
    for i, item in enumerate(items):
        series.extend(item['series'])
        owners.extend([i]*len(item['series']))
    return {'ids': [x['uid'] for x in items],
            'x': pad_sequence([s['x'].float() for s in series], batch_first=True) if series else None,
            'lengths': torch.tensor([len(s['x']) for s in series]),
            'position': pad_sequence([s['position'] for s in series], batch_first=True) if series else None,
            'metadata': torch.stack([s['meta'] for s in series]) if series else None,
            'owners': torch.tensor(owners, dtype=torch.long)}


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) and k != 'lengths' else v for k, v in batch.items()}
