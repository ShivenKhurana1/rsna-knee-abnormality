"""Frozen DINOv2 features from real MRI DICOMs; no report or model-label fitting."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from clean_specialist import PROTOCOL, TARGETS, sha


def pixels_to_rgb(ds, size=336):
    a = ds.pixel_array.astype(np.float32)
    if a.ndim != 2 or min(a.shape) < 16 or not np.isfinite(a).all():
        raise ValueError('Expected finite single-frame grayscale MRI')
    a = a * float(ds.get('RescaleSlope', 1)) + float(ds.get('RescaleIntercept', 0))
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        raise ValueError('Constant/degenerate slice')
    a = np.clip((a-lo)/(hi-lo), 0, 1)
    if str(ds.get('PhotometricInterpretation', 'MONOCHROME2')) == 'MONOCHROME1':
        a = 1-a
    # Preserve the entire FOV/aspect ratio. No learned/case-label-dependent crop.
    im = Image.fromarray(np.uint8(a*255)).convert('RGB')
    ratio = size / max(im.size)
    im = im.resize((max(1, round(im.width*ratio)), max(1, round(im.height*ratio))), Image.Resampling.BILINEAR)
    out = Image.new('RGB', (size, size))
    out.paste(im, ((size-im.width)//2, (size-im.height)//2))
    return np.asarray(out).transpose(2, 0, 1).astype(np.float32)/255


def ordered_files(folder, uid, sid):
    import pydicom
    records = []
    for file in sorted(folder.glob('*.dcm')):
        ds = pydicom.dcmread(file, stop_before_pixels=True,
                            specific_tags=['StudyInstanceUID', 'SeriesInstanceUID', 'ImagePositionPatient',
                                           'ImageOrientationPatient', 'InstanceNumber'])
        if str(ds.get('StudyInstanceUID', '')) != uid or str(ds.get('SeriesInstanceUID', '')) != sid:
            raise ValueError('DICOM/header path UID mismatch')
        orientation = np.asarray(ds.get('ImageOrientationPatient', []), float)
        position = np.asarray(ds.get('ImagePositionPatient', []), float)
        records.append((file, orientation, position, float(ds.get('InstanceNumber', len(records)))))
    if not records:
        raise ValueError('No DICOMs in selected series')
    physical = all(o.shape == (6,) and p.shape == (3,) and np.isfinite(o).all() and np.isfinite(p).all()
                   for _, o, p, _ in records)
    if physical:
        reference = records[0][1]
        physical = all(np.allclose(o, reference, atol=.01) for _, o, _, _ in records)
    if physical:
        normal = np.cross(reference[:3], reference[3:])
        physical = np.linalg.norm(normal) > .9
    records.sort(key=lambda r: (float(r[2] @ normal) if physical else r[3], r[0].name))
    return [r[0] for r in records], 'physical_position' if physical else 'instance_number_fallback'


def extract(root, groups, output, protocol=PROTOCOL):
    import pydicom
    import torch
    from transformers import AutoModel
    from huggingface_hub import snapshot_download
    start = time.perf_counter()
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for bounded extraction; do not silently run hours on CPU')
    torch.manual_seed(protocol['seed'])
    torch.set_num_threads(2)
    train = pd.read_csv(root/'train.csv', dtype={'StudyInstanceUID': str})
    series = pd.read_csv(root/'train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    expected_hash = {'train.csv': '8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33',
                     'train_series.csv': '573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3'}
    if any(sha(root/name) != digest for name, digest in expected_hash.items()):
        raise ValueError('Competition tables differ from audited dataset version')
    gold = train.loc[train[TARGETS].notna().all(axis=1), ['StudyInstanceUID']+TARGETS].sort_values('StudyInstanceUID')
    if len(gold) != 58 or set(gold.StudyInstanceUID) != set(groups):
        raise ValueError('Expected exact previously audited 58 expert studies')
    if not np.isin(gold[TARGETS].to_numpy(), [0, 1]).all():
        raise ValueError('Gold label schema changed')
    required = ['Anatomical_Plane', 'Fluid_Sensitive', 'Fat_Suppression']
    if not set(required).issubset(series):
        raise ValueError('Series metadata schema changed')
    snapshot = Path(snapshot_download(protocol['encoder'], revision=protocol['encoder_revision'],
                                     allow_patterns=['config.json', 'model.safetensors'], token=False))
    model = AutoModel.from_pretrained(snapshot, local_files_only=True, use_safetensors=True, trust_remote_code=False)
    if model.config.model_type != 'dinov2' or model.config.hidden_size != 384 or model.config.patch_size != 14:
        raise ValueError('Unexpected pretrained architecture')
    model.requires_grad_(False).eval().cuda()
    weight_hash = sha(snapshot/'model.safetensors')
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    all_base, all_region, audit, pixel_owners = [], [], [], {}
    size, hidden = protocol['image_size'], model.config.hidden_size
    grid = size//14
    def read_pixels(item):
        file, uid, sid = item
        ds = pydicom.dcmread(file)
        if str(ds.get('StudyInstanceUID', '')) != uid or str(ds.get('SeriesInstanceUID', '')) != sid:
            raise ValueError('Pixel DICOM UID mismatch')
        # Encoded pixel payload hash catches exact duplicate image payloads across groups.
        pixel_hash = hashlib.sha256(ds.PixelData).hexdigest()
        return pixels_to_rgb(ds, size), pixel_hash
    with ThreadPoolExecutor(max_workers=6) as pool:
        for ii, uid in enumerate(gold.StudyInstanceUID):
            s = series[series.StudyInstanceUID == uid].copy()
            base_parts, regional_parts, details = [], [], []
            for plane in protocol['planes']:
                rows = s[s.Anatomical_Plane.str.casefold() == plane.casefold()].copy()
                rows['priority'] = pd.to_numeric(rows.Fluid_Sensitive, errors='coerce').fillna(0)*2 + pd.to_numeric(rows.Fat_Suppression, errors='coerce').fillna(0)
                rows = rows.sort_values(['priority', 'SeriesInstanceUID'], ascending=[False, True]).head(protocol['max_series_per_plane'])
                cls_series, region_series = [], []
                for sid in rows.SeriesInstanceUID:
                    files, ordering = ordered_files(root/'train_series'/uid/sid, uid, sid)
                    idx = np.unique(np.rint(np.linspace(protocol['slice_quantiles'][0]*(len(files)-1),
                                                         protocol['slice_quantiles'][1]*(len(files)-1),
                                                         min(protocol['slices_per_series'], len(files)))).astype(int))
                    batch = list(pool.map(read_pixels, [(files[i], uid, sid) for i in idx]))
                    for _, pixel_hash in batch:
                        if pixel_hash in pixel_owners and pixel_owners[pixel_hash] != groups[uid]:
                            raise ValueError('Exact sampled image duplicate across PatientID groups; regroup before fitting')
                        pixel_owners[pixel_hash] = groups[uid]
                    x = torch.from_numpy(np.stack([a for a, _ in batch])).cuda()
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                        h = model(pixel_values=(x-mean)/std).last_hidden_state
                    if h.shape != (len(batch), 1+grid*grid, hidden):
                        raise ValueError('Unexpected patch-grid shape')
                    h = h.float().cpu().numpy()
                    cls = h[:, 0]
                    patches = h[:, 1:].reshape(len(batch), grid, grid, hidden)
                    quadrants = np.stack([patches[:, ra:rb, ca:cb].mean((1,2))
                                          for ra, rb in [(0, grid//2), (grid//2, grid)]
                                          for ca, cb in [(0, grid//2), (grid//2, grid)]], axis=1)
                    cls_series.append(np.concatenate([cls.mean(0), cls.max(0)]))
                    region_series.append(np.concatenate([quadrants.mean(0).ravel(), quadrants.max(0).ravel()]))
                    details.append({'plane': plane, 'SeriesInstanceUID': sid, 'available_slices': len(files),
                                    'used_slices': len(idx), 'ordering': ordering,
                                    'selected_instance_files': [files[i].name for i in idx]})
                present = float(bool(cls_series))
                base_parts.append(np.r_[np.mean(cls_series, axis=0) if cls_series else np.zeros(hidden*2), present])
                regional_parts.append(np.r_[np.mean(region_series, axis=0) if region_series else np.zeros(hidden*8), present])
            if not details:
                raise ValueError('Study has no usable canonical-plane series; do not silently drop it')
            all_base.append(np.concatenate(base_parts))
            # Include global features in specialist input; no artificial information removal.
            all_region.append(np.concatenate(base_parts+regional_parts))
            audit.append({'StudyInstanceUID': uid, 'series': details})
            print(f'CLEAN FEATURES {ii+1}/58 studies, elapsed={time.perf_counter()-start:.1f}s', flush=True)
    base, regional = np.asarray(all_base, np.float32), np.asarray(all_region, np.float32)
    if not np.isfinite(base).all() or not np.isfinite(regional).all():
        raise ValueError('Nonfinite features')
    feature_file = output/'clean_features.npz'
    np.savez_compressed(feature_file, StudyInstanceUID=gold.StudyInstanceUID.to_numpy(str),
                        GroupID=np.array([groups[u] for u in gold.StudyInstanceUID]),
                        base=base, regional=regional, labels=gold[TARGETS].to_numpy(np.float32))
    receipt = {'protocol': protocol, 'encoder_weights_sha256': weight_hash,
               'encoder_config_sha256': sha(snapshot/'config.json'), 'encoder_snapshot_revision': snapshot.name,
               'encoder_training_on_rsna_in_this_experiment': False, 'parameters_require_grad': sum(p.numel() for p in model.parameters() if p.requires_grad),
               'input_sha256': expected_hash, 'feature_sha256': sha(feature_file),
               'base_shape': list(base.shape), 'regional_shape': list(regional.shape),
               'sampled_images': sum(d['used_slices'] for a in audit for d in a['series']),
               'sampled_exact_pixel_duplicate_cross_group_check': 'passed',
               'scope_of_duplicate_check': 'Encoded pixel payloads in sampled slices only; not a near-duplicate or full-volume audit',
               'elapsed_seconds': time.perf_counter()-start, 'gpu': torch.cuda.get_device_name(0),
               'torch_version': torch.__version__, 'series_audit': audit}
    (output/'feature_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    return feature_file
