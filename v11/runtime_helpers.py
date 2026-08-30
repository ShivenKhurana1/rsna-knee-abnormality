"""Inference-only guards embedded in the generated V11 Kaggle notebook.

No torch, training, network access, or label-dependent tuning. Prefixed imports
avoid collisions with the inherited notebook's shared global namespace.
"""

import hashlib as _v11_hashlib
import json as _v11_json
import math as _v11_math
import os as _v11_os
import shutil as _v11_shutil
import tempfile as _v11_tempfile
import threading as _v11_threading
import time as _v11_time
import uuid as _v11_uuid
from pathlib import Path as _V11Path

import numpy as _v11_np
import pandas as _v11_pd


def v11_validate_submission(frame, expected_ids, targets):
    expected_ids = list(map(str, expected_ids))
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError('empty or duplicate expected study IDs')
    if list(frame.columns) != ['StudyInstanceUID'] + list(targets):
        raise ValueError('submission target schema changed')
    ids = frame['StudyInstanceUID'].astype(str).tolist()
    if ids != expected_ids or len(set(ids)) != len(ids):
        raise ValueError('submission study identity/order changed')
    values = frame[list(targets)].to_numpy(dtype=_v11_np.float64)
    if not _v11_np.isfinite(values).all():
        raise ValueError('non-finite prediction')
    if (values < 0).any() or (values > 1).any():
        raise ValueError('prediction outside [0,1]')
    # A one-study smoke test cannot demonstrate ranking. Do not reject it merely
    # because its ranks tie. On a larger cohort reject the all-constant fallback.
    if len(frame) > 1 and not _v11_np.any(_v11_np.ptp(values, axis=0) > 0):
        raise ValueError('all targets are constant across studies')
    return values


def v11_atomic_csv(frame, path):
    path = _V11Path(path)
    tmp = path.with_name(path.name + '.v11-' + _v11_uuid.uuid4().hex + '.tmp')
    try:
        frame.to_csv(tmp, index=False)
        # Serialization is checked before replacing the last valid artifact.
        restored = _v11_pd.read_csv(tmp, dtype={'StudyInstanceUID': str})
        if list(restored.columns) != list(frame.columns) or len(restored) != len(frame):
            raise ValueError('CSV roundtrip schema changed')
        if restored['StudyInstanceUID'].astype(str).tolist() != frame['StudyInstanceUID'].astype(str).tolist():
            raise ValueError('CSV roundtrip study IDs changed')
        columns = [column for column in frame.columns if column != 'StudyInstanceUID']
        if not _v11_np.allclose(restored[columns].to_numpy(float), frame[columns].to_numpy(float),
                               atol=1e-12, rtol=1e-12):
            raise ValueError('CSV roundtrip predictions changed')
        _v11_os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def v11_member_fits(left, remaining, devices, fixed, per_window, windows, jitter=False):
    """Admit a complete fixed-definition member, never a central-window substitute."""
    if not _v11_math.isfinite(left) or left <= 0 or windows < 1:
        return False
    if fixed is None or per_window is None:
        return True  # first complete member supplies an empirical runtime estimate
    if not all(_v11_math.isfinite(x) and x >= 0 for x in (fixed, per_window)):
        return False
    slots = max(1, _v11_math.ceil(remaining / max(1, devices)))
    cost = fixed + per_window * windows * (2 if jitter else 1)
    return cost <= max(left * 0.9, 0) / slots


def v11_combine_members(members, targets):
    """Same rank fusion on complete cohorts; normalize missing votes per study."""
    if not members:
        raise ValueError('no completed members')
    all_ids = sorted({str(uid) for member in members for uid in member['ids']})
    if not all_ids:
        raise ValueError('empty study cohort')
    pos = {uid: i for i, uid in enumerate(all_ids)}
    total = _v11_np.zeros((len(all_ids), len(targets)), _v11_np.float64)
    votes = _v11_np.zeros_like(total)
    for member in members:
        ids = list(map(str, member['ids']))
        prediction = _v11_np.asarray(member['pred'], _v11_np.float64)
        if len(set(ids)) != len(ids) or prediction.shape != (len(ids), len(targets)):
            raise ValueError('duplicate IDs or invalid member shape')
        if not _v11_np.isfinite(prediction).all():
            raise ValueError('non-finite member prediction')
        weights = member.get('target_weight')
        if weights is None:
            weights = [float(member.get('weight', 1.0))] * len(targets)
        weights = _v11_np.asarray(weights, _v11_np.float64)
        if (weights.shape != (len(targets),) or not _v11_np.isfinite(weights).all()
                or (weights < 0).any()):
            raise ValueError('invalid member weights')
        rows = [pos[uid] for uid in ids]
        ranks = _v11_pd.DataFrame(prediction).rank(pct=True).to_numpy()
        total[rows] += ranks * weights[None, :]
        votes[rows] += weights[None, :]
    if (votes <= 0).any():
        raise ValueError('study/target has no ensemble vote')
    return all_ids, total / votes


def v11_rankpct(values):
    """Preserve ties; equivalent to the inherited ordinal ranks when no ties exist."""
    values = _v11_np.asarray(values)
    if values.ndim != 2 or not _v11_np.isfinite(values).all():
        raise ValueError('invalid rank matrix')
    return (_v11_pd.DataFrame(values).rank(method='average').to_numpy() - 1) / max(1, len(values) - 1)


class V11RunAudit:
    """Current-run checkpoints only: an old submission can never be a recovery base."""

    def __init__(self, root, work, targets, build):
        self.work = _V11Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.targets = list(targets)
        test = _v11_pd.read_csv(_V11Path(root) / 'test.csv', dtype={'StudyInstanceUID': str})
        self.ids = test['StudyInstanceUID'].astype(str).tolist()
        self.run_id = _v11_uuid.uuid4().hex
        self.folder = self.work / 'v11_diagnostics' / self.run_id
        self.folder.mkdir(parents=True)
        self.primary = self.work / 'submission.csv'
        self.last_good = self.folder / 'last_good.csv'
        self.receipt = {'run_id': self.run_id, 'build': build, 'studies': len(self.ids),
                        'targets': self.targets, 'events': [], 'snapshots': []}
        self.lock = _v11_threading.RLock()
        self.started = _v11_time.monotonic()
        self.flush()

    def flush(self):
        path = self.work / 'v11_run_receipt.json'
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(_v11_json.dumps(self.receipt, indent=2, allow_nan=False) + '\n',
                       encoding='utf-8')
        _v11_os.replace(tmp, path)

    def note(self, event, **details):
        with self.lock:
            self.receipt['events'].append({'event': event,
                'elapsed_seconds': round(_v11_time.monotonic() - self.started, 3), **details})
            self.flush()

    def _validated_bytes(self, path):
        frame = _v11_pd.read_csv(path, dtype={'StudyInstanceUID': str})
        v11_validate_submission(frame, self.ids, self.targets)
        return _V11Path(path).read_bytes()

    def _save_bytes(self, data, path):
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_bytes(data)
        _v11_os.replace(tmp, path)

    def member_committed(self, member, **details):
        with self.lock:
            data = self._validated_bytes(self.primary)
            self._save_bytes(data, self.last_good)
            self.note('stage1_member_committed', member=str(member), **details)

    def recover(self, stage, error):
        with self.lock:
            if not self.last_good.is_file():
                self.note('no_current_run_recovery', stage=stage, error=str(error))
                raise RuntimeError('No valid current-run predictions; refusing a fake submission') from error
            data = self._validated_bytes(self.last_good)
            self._save_bytes(data, self.primary)
            self.note('restored_last_good', stage=stage, error=str(error))

    def snapshot(self, stage):
        with self.lock:
            restored = False
            try:
                data = self._validated_bytes(self.primary)
            except Exception as error:
                self.recover(stage, error)
                data = self._validated_bytes(self.primary)
                restored = True
            digest = _v11_hashlib.sha256(data).hexdigest()
            previous = self.receipt['snapshots'][-1]['sha256'] if self.receipt['snapshots'] else None
            path = self.folder / (stage + '.csv')
            self._save_bytes(data, path)
            self._save_bytes(data, self.last_good)
            self.receipt['snapshots'].append({'stage': stage, 'sha256': digest,
                'status': 'restored' if restored else 'unchanged' if digest == previous else 'changed_valid',
                'path': str(path)})
            self.flush()
            print(f'[V11] {stage}: {self.receipt["snapshots"][-1]["status"]}; sha256={digest}', flush=True)


class V11VolumeCache:
    """Lazy, disk-bounded cache for one Stage-4 run, retaining no volume arrays.

    Only a fresh TemporaryDirectory is managed. Cache I/O errors fall back to
    recomputation. Loader/inference errors are not swallowed. No eager futures.
    """

    def __init__(self, max_bytes=12 * 1024**3, reserve_bytes=2 * 1024**3):
        self.temp = _v11_tempfile.TemporaryDirectory(prefix='rsna-v11-volumes-')
        self.root = _V11Path(self.temp.name).resolve()
        self.max_bytes = max(0, int(max_bytes))
        self.reserve_bytes = max(0, int(reserve_bytes))
        self.paths = {}
        self.bytes = self.hits = self.misses = self.write_failures = 0

    def get(self, key, loader):
        key = str(key)
        path = self.paths.get(key)
        if path is not None:
            try:
                with _v11_np.load(path, allow_pickle=False) as saved:
                    result = saved['volume'], saved['mask']
                self.hits += 1
                return result
            except (OSError, ValueError, EOFError):
                self.paths.pop(key, None)
        self.misses += 1
        volume, mask = loader(key)
        estimated_bytes = int(volume.nbytes + mask.nbytes + 1024)
        if self.bytes + estimated_bytes > self.max_bytes:
            return volume, mask
        try:
            free = _v11_shutil.disk_usage(self.root).free
            if free < estimated_bytes + self.reserve_bytes:
                return volume, mask
            # Names are derived from keys, never interpreted as filesystem paths.
            path = self.root / (_v11_hashlib.sha256(key.encode()).hexdigest() + '.npz')
            _v11_np.savez(path, volume=volume, mask=mask)
            self.bytes += path.stat().st_size
            self.paths[key] = path
        except OSError:
            self.write_failures += 1
            if path is not None and path.parent == self.root:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        return volume, mask

    def stats(self):
        return {'hits': self.hits, 'misses': self.misses, 'bytes': self.bytes,
                'entries': len(self.paths), 'write_failures': self.write_failures}

    def close(self):
        self.paths.clear()
        # TemporaryDirectory owns this exact newly created cache directory only.
        self.temp.cleanup()
