"""Executable engineering patterns; NOT a trained submission or DICOM decoder.

Run: python research/next_tier_095/reference_patterns.py
The Dataset requires a checkpoint-compatible read_series callback. See REPORT.md.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class Schema:
    id_col: str
    targets: tuple[str, ...]

    @classmethod
    def from_sample(cls, path, id_col):
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        if id_col not in columns:
            raise ValueError(f"Missing configured identity field: {id_col}")
        targets = tuple(c for c in columns if c != id_col)
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("Empty or duplicate target schema")
        return cls(id_col, targets)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def checkpoint_permutation(self, checkpoint_targets):
        stored = list(checkpoint_targets)
        if len(stored) != len(set(stored)) or set(stored) != set(self.targets):
            raise ValueError("Checkpoint and competition target sets differ")
        return [stored.index(c) for c in self.targets]


def find_root(search_root, split):
    """Discover mounts by required files; refuse ambiguous data roots."""
    candidates = sorted({p.parent.resolve() for p in Path(search_root).rglob(f"{split}.csv")
                         if (p.parent / f"{split}_series").is_dir()
                         and (p.parent / "sample_submission.csv").is_file()})
    if len(candidates) != 1:
        raise ValueError(f"Expected one data root, found {len(candidates)}; configure explicitly")
    return candidates[0]


def safe_child(parent, uid):
    if not uid or uid in (".", "..") or "/" in uid or "\\" in uid:
        raise ValueError("Invalid path component")
    parent = Path(parent).resolve()
    child = (parent / uid).resolve()
    if not child.is_relative_to(parent):
        raise ValueError("Path escaped configured data root")
    return child


def checked_ids(frame, column):
    if column not in frame or frame[column].isna().any():
        raise ValueError(f"Missing identity: {column}")
    ids = frame[column].astype(str)
    if ids.duplicated().any() or (ids.str.len() == 0).any():
        raise ValueError("Duplicate or empty study identity")
    return ids.tolist()


class KneeDataset(Dataset):
    """One item per study; series and slice counts are variable.

    read_series(directory, metadata_dict) -> float Tensor[T,C,H,W], or None
    for an explicitly logged unusable series. Resizing/geometry/normalization
    belong to that callback and its saved preprocessing contract.
    """
    def __init__(self, root, split, schema, read_series,
                 series_col="SeriesInstanceUID", labels=None):
        self.root, self.split = Path(root), split
        self.schema, self.reader = schema, read_series
        self.series_col = series_col
        self.studies = pd.read_csv(self.root / f"{split}.csv",
                                   dtype={schema.id_col: str})
        self.ids = checked_ids(self.studies, schema.id_col)
        metadata_path = self.root / f"{split}_series.csv"
        self.metadata = {}
        if metadata_path.exists():
            table = pd.read_csv(metadata_path,
                                dtype={schema.id_col: str, series_col: str})
            if not {schema.id_col, series_col}.issubset(table.columns):
                raise ValueError("Series table lacks identity fields")
            if table[[schema.id_col, series_col]].isna().any().any():
                raise ValueError("Missing series identity")
            if table.duplicated([schema.id_col, series_col]).any():
                raise ValueError("Duplicate study/series metadata")
            self.metadata = {(r[schema.id_col], r[series_col]): r
                             for r in table.to_dict("records")}
        self.y = np.full((len(self.ids), len(schema.targets)), np.nan, np.float32)
        source = labels if labels is not None else self.studies
        if split == "train" or labels is not None:
            checked_ids(source, schema.id_col)
            absent = set(schema.targets) - set(source.columns)
            if absent:
                raise ValueError(f"Label table lacks schema fields: {sorted(absent)}")
            aligned = source.set_index(schema.id_col).reindex(self.ids)
            self.y = aligned.loc[:, list(schema.targets)].apply(
                pd.to_numeric, errors="raise").to_numpy(np.float32)
            if not (np.isnan(self.y) | ((self.y >= 0) & (self.y <= 1))).all():
                raise ValueError("Labels must be probabilities or missing")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        uid = self.ids[index]
        study_dir = safe_child(self.root / f"{self.split}_series", uid)
        # Named UID directories are the documented storage contract; no string
        # slicing, positional path parsing, sample-row count, or fixed slot array.
        series = []
        if study_dir.is_dir():
            for directory in sorted(p for p in study_dir.iterdir() if p.is_dir()):
                values = self.reader(directory, self.metadata.get((uid, directory.name), {}))
                if values is None:
                    continue
                if values.ndim != 4 or values.shape[0] == 0 or not torch.isfinite(values).all():
                    raise ValueError(f"Invalid decoded series: {directory}")
                series.append(values)
        # Retain every test study, including those with no usable images.
        return {"uid": uid, "series": series, "y": torch.from_numpy(self.y[index])}


def collate_studies(items):
    values, owners = [], []
    for owner, item in enumerate(items):
        values.extend(item["series"])
        owners.extend([owner] * len(item["series"]))
    lengths = torch.tensor([len(x) for x in values], dtype=torch.long)
    if values and len({tuple(x.shape[1:]) for x in values}) != 1:
        raise ValueError("read_series must apply the configured channel/spatial contract")
    y = torch.stack([x["y"] for x in items])
    return {"uid": [x["uid"] for x in items],
            "x": pad_sequence(values, batch_first=True) if values else None,
            "lengths": lengths, "owners": torch.tensor(owners, dtype=torch.long),
            "y": y, "mask": torch.isfinite(y)}


def adjacent_windows(volume, offsets, max_centres=None):
    """volume [D,H,W] already geometrically ordered; channels use true neighbors.

    Sparse centers DO NOT mean sparse neighboring channels. offsets and the
    optional compute cap are checkpoint/config settings, never dataset counts.
    """
    if volume.ndim != 3 or len(volume) == 0 or not offsets:
        raise ValueError("Need a nonempty volume and configured channel offsets")
    if max_centres is not None and max_centres < 1:
        raise ValueError("max_centres must be positive")
    count = len(volume) if max_centres is None else min(len(volume), max_centres)
    centres = torch.linspace(0, len(volume) - 1, count).round().long()
    indices = (centres[:, None] + torch.tensor(offsets)[None, :]).clamp(0, len(volume) - 1)
    return volume[indices]


def resize_pad(slice_2d, output_hw):
    """Infer input dimensions from pixels; output dimensions are model config."""
    x = torch.as_tensor(slice_2d, dtype=torch.float32)
    if x.ndim != 2 or min(x.shape) < 1 or not torch.isfinite(x).all():
        raise ValueError("Expected finite, nonempty 2D pixels")
    h, w = x.shape
    out_h, out_w = map(int, output_hw)
    if min(out_h, out_w) < 1:
        raise ValueError("Invalid output shape")
    scale = min(out_h / h, out_w / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    x = F.interpolate(x[None, None], size=(rh, rw), mode="bilinear",
                      align_corners=False, antialias=True)[0, 0]
    top, left = (out_h-rh)//2, (out_w-rw)//2
    return F.pad(x, (left, out_w-rw-left, top, out_h-rh-top))


class SequenceMIL(nn.Module):
    """CNN -> within-series BiGRU -> target-query attention over all study slices.

    encoder must return [N,feature_dim]. No recurrent connection crosses series.
    This reference is image-only; add approved metadata embeddings before GRU.
    """
    def __init__(self, encoder, feature_dim, n_targets, hidden=128, chunk_size=32):
        super().__init__()
        self.encoder, self.feature_dim, self.chunk_size = encoder, feature_dim, chunk_size
        self.gru = nn.GRU(feature_dim, hidden, batch_first=True, bidirectional=True)
        self.query = nn.Parameter(torch.randn(n_targets, 2 * hidden) * 0.02)
        self.weight = nn.Parameter(torch.randn(n_targets, 2 * hidden) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_targets))

    def forward(self, batch):
        if batch["x"] is None:
            return self.bias.expand(len(batch["uid"]), -1), torch.zeros(
                len(batch["uid"]), dtype=torch.bool, device=self.bias.device)
        x, lengths = batch["x"], batch["lengths"]
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths.to(x.device)[:, None]
        # Padded images never enter the encoder, including its batch statistics.
        features = torch.cat([self.encoder(c) for c in x[valid].split(self.chunk_size)])
        padded = features.new_zeros(*valid.shape, self.feature_dim)
        padded[valid] = features
        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        contextual, _ = self.gru(packed)
        contextual, _ = pad_packed_sequence(contextual, batch_first=True)
        owners = batch["owners"].to(x.device)
        output, available = [], []
        for i in range(len(batch["uid"])):
            study_tokens = contextual[(owners == i)[:, None] & valid]
            if len(study_tokens) == 0:
                output.append(self.bias)
                available.append(False)
                continue
            attention = (self.query @ study_tokens.T / study_tokens.shape[-1] ** 0.5).softmax(-1)
            context = attention @ study_tokens
            output.append((context * self.weight).sum(-1) + self.bias)
            available.append(True)
        return torch.stack(output), torch.tensor(available, device=x.device)


def masked_loss(logits, labels, mask, mode="bce", gamma_pos=0., gamma_neg=2.,
                negative_clip=0., target_weights=None):
    """FP32 BCE or simplified ASL; ASL requires observed HARD binary labels.

    Weights scale the numerator only, so an auxiliary-loss cap remains a cap.
    Returns differentiable zero for an entirely unobserved batch.
    """
    z = logits.float()
    mask = mask.bool() & torch.isfinite(labels)
    y = torch.where(mask, labels.float(), torch.zeros_like(z))
    if not ((y >= 0) & (y <= 1)).all():
        raise ValueError("Observed labels outside [0,1]")
    if mode == "bce":
        cell = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    elif mode == "asl":
        if not ((y[mask] == 0) | (y[mask] == 1)).all():
            raise ValueError("Use BCE for soft report probabilities")
        p = z.sigmoid()
        negative_p = (p - negative_clip).clamp(min=0)
        # Stable log sigmoid for positive term; unclipped negative uses logsigmoid.
        negative_log = F.logsigmoid(-z) if negative_clip == 0 else torch.log1p(-negative_p.clamp(max=1-1e-7))
        cell = -y * (1-p).pow(gamma_pos) * F.logsigmoid(z)
        cell = cell - (1-y) * negative_p.pow(gamma_neg) * negative_log
    else:
        raise ValueError(mode)
    if target_weights is not None:
        cell = cell * target_weights.to(cell)
    return (cell * mask).sum() / mask.sum().clamp_min(1)


def stratified_group_folds(y, groups, n_splits=5, seed=42):
    """SGKF using a label-only surrogate; audits ALL targets afterward.

    This is approximate multi-label stratification. For imbalanced/rare labels,
    use the group-level allocation procedure in REPORT.md; never claim exact
    multi-label balance from this surrogate.
    """
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    if len(groups) != len(y) or pd.isna(groups).any():
        raise ValueError("Missing or misaligned grouping")
    if not (np.isnan(y) | (y == 0) | (y == 1)).all():
        raise ValueError("Stratify audited binary/unknown labels, not soft probabilities")
    positives = (y == 1)
    prevalence_count = positives.sum(0)
    rarity = np.where(positives, prevalence_count[None, :], np.inf)
    strata = np.where(positives.any(1), rarity.argmin(1), y.shape[1])
    strata[np.isnan(y).all(1)] = y.shape[1] + 1
    # The seed is predeclared, never selected for model AUC.
    splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(y), -1, int)
    audit = []
    for f, (train, val) in enumerate(splitter.split(np.zeros(len(y)), strata, groups)):
        assert not set(groups[train]) & set(groups[val])
        fold[val] = f
        audit.append({"fold": f, "n_train": len(train), "n_val": len(val),
                      "positive": (y[val] == 1).sum(0).tolist(),
                      "negative": (y[val] == 0).sum(0).tolist(),
                      "unknown": np.isnan(y[val]).sum(0).tolist(),
                      "scorable": ((y[val] == 1).any(0) & (y[val] == 0).any(0)).tolist()})
    assert (fold >= 0).all()
    return fold, audit


def write_submission(test_frame, schema, predictions_by_id, destination):
    """Live test.csv owns ROWS; sample header owns COLUMNS."""
    ids = checked_ids(test_frame, schema.id_col)
    checked_ids(predictions_by_id, schema.id_col)
    if set(ids) != set(predictions_by_id[schema.id_col].astype(str)):
        raise ValueError("Prediction coverage differs from live test IDs")
    indexed = predictions_by_id.assign(**{schema.id_col: predictions_by_id[schema.id_col].astype(str)})
    out = indexed.set_index(schema.id_col).loc[ids, list(schema.targets)].reset_index()
    p = out[list(schema.targets)].to_numpy(float)
    if not (np.isfinite(p) & (p >= 0) & (p <= 1)).all():
        raise ValueError("Nonfinite/out-of-range submission")
    # No fillna: missing data must have a declared, logged fallback upstream.
    out.to_csv(destination, index=False)
    return out


def smoke():
    """Synthetic contract tests, including simulated hidden-set replacement."""
    torch.manual_seed(17)
    torch.set_num_threads(1)
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "arbitrarily" / "nested" / "mount"
        root.mkdir(parents=True)
        uid = "StudyInstanceUID"
        targets = ["finding_b", "finding_a", "finding_extra"]
        pd.DataFrame({uid: ["example"], **{t: [.5] for t in targets}}).to_csv(root / "sample_submission.csv", index=False)
        ids = ["hidden-long-identifier", "s", "missing-study"]
        pd.DataFrame({uid: ids}).to_csv(root / "test.csv", index=False)
        for study, number in zip(ids, [2, 1, 0]):
            for s in range(number):
                (root / "test_series" / study / f"series{s}").mkdir(parents=True)
        schema = Schema.from_sample(root / "sample_submission.csv", uid)
        assert find_root(tmp, "test") == root.resolve()
        assert schema.checkpoint_permutation(targets[::-1]) == [2, 1, 0]
        def reader(path, metadata):
            d = 1 if path.parent.name == "s" else (5 if path.name == "series0" else 3)
            pixels = torch.rand(d, 11 + d, 23 + d)
            volume = torch.stack([resize_pad(s, (16, 16)) for s in pixels])
            return adjacent_windows(volume, [-1, 0, 1])
        ds = KneeDataset(root, "test", schema, reader)
        batch = collate_studies([ds[i] for i in range(len(ds))])
        assert batch["lengths"].tolist() == [5, 3, 1]
        assert not batch["mask"].any()
        checks.append("live IDs, arbitrary UID lengths, dynamic targets, absent metadata, variable series/slices")
        encoder = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        model = SequenceMIL(encoder, 8, len(schema.targets), hidden=4).eval()
        logits, available = model(batch)
        assert logits.shape == (len(ids), len(targets)) and available.tolist() == [True, True, False]
        # Padding content must have no effect on predicted values.
        changed = dict(batch)
        changed["x"] = batch["x"].clone()
        padding = torch.arange(batch["x"].shape[1])[None] >= batch["lengths"][:, None]
        changed["x"][padding] = 1000
        torch.testing.assert_close(model(changed)[0], logits)
        single = collate_studies([{ "uid": ids[0], "series": [batch["x"][j, :batch["lengths"][j]]
                                  for j in range(2)], "y": batch["y"][0]}])
        torch.testing.assert_close(model(single)[0][0], logits[0], atol=1e-6, rtol=1e-5)
        checks.append("pixel-inferred resize, CNN/GRU forward, missing-study mask, padding and batch invariance")
        y = torch.tensor([[1., 0., float('nan')]]).expand_as(logits)
        loss = masked_loss(logits, y, torch.isfinite(y), mode="asl")
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        z = torch.zeros(2, len(targets), requires_grad=True)
        zero = masked_loss(z, torch.full_like(z, float('nan')), torch.zeros_like(z, dtype=torch.bool))
        zero.backward()
        assert zero.item() == 0 and torch.equal(z.grad, torch.zeros_like(z))
        checks.append("masked ASL gradients and all-unknown differentiable zero")
        pred = pd.DataFrame(logits.detach().sigmoid().numpy(), columns=targets)
        pred.insert(0, uid, ids)
        out = write_submission(ds.studies, schema, pred.iloc[::-1], root / "submission.csv")
        assert out[uid].tolist() == ids and out.columns.tolist() == [uid] + targets
        try:
            write_submission(ds.studies, schema, pred.iloc[:-1], root / "invalid.csv")
            raise AssertionError("Missing ID was not rejected")
        except ValueError:
            pass
        checks.append("keyed submission reorder and missing-ID rejection")
        groups = np.repeat(np.arange(30), 2)
        labels = np.column_stack([(groups % 2), (groups % 3 == 0), (groups % 5 == 0)]).astype(float)
        fold, audit = stratified_group_folds(labels, groups, n_splits=3)
        assert all(len(set(fold[groups == g])) == 1 for g in np.unique(groups))
        checks.append("group separation with multilabel coverage audit")
    print(json.dumps({"status": "PASS", "checks": checks,
                      "scope": "Synthetic CPU contracts only; no real MRI, training, AUC, or leaderboard run"}, indent=2))


if __name__ == "__main__":
    smoke()
