"""End-to-end tests using actual generated DICOM pixel data, never clinical data."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

from .common import config, schema, sha
from .data import decode_series, CachedStudies, collate
from .model import KneeModel, masked_loss
from .runner import prepare, train, merge, infer
from .validation import make_folds, supervision
from .evaluation import evaluate


def dicom(path, study, series, index, shape=(18, 26), spacing=True, invert=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = f'{series}.{index+1}'
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0'*128)
    ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
    ds.ImageOrientationPatient = [1., 0., 0., 0., 1., 0.]
    ds.ImagePositionPatient = [0., 0., float(index)*2]
    ds.InstanceNumber = index+1
    if spacing:
        ds.PixelSpacing = [.7, 1.1]
    ds.Rows, ds.Columns = shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME1' if invert else 'MONOCHROME2'
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 0
    yy, xx = np.indices(shape)
    pixels = (xx+yy*3+index*90+10).astype(np.uint16)
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base/'nested'/'data'
        self.root.mkdir(parents=True)
        self.cfg = config(Path(__file__).with_name('config.json'))
        self.cfg.update(encoder='tiny', image_size=16, max_centres=3, max_series=2,
                        hidden=4, encoder_chunk=3, folds=2, policy_bootstrap=30,
                        epochs=1, batch_size=3, device='cpu', gradient_checkpointing=False)
        self.uid = self.cfg['id_col']
        self.targets = ['finding_c', 'finding_a', 'finding_b']
        pd.DataFrame({self.uid: ['example'], **{t: [.5] for t in self.targets}}).to_csv(self.root/'sample_submission.csv', index=False)

    def tearDown(self):
        self.temp.cleanup()

    def build_fixture(self):
        ids = [f'1.2.826.0.1.3680043.10.16.{i+1}' for i in range(24)]
        truth = np.array([[i % 2, (i//2) % 2, (i//3) % 2] for i in range(len(ids))], float)
        expert = truth.copy()
        expert[18:] = np.nan
        frame = pd.DataFrame(expert, columns=self.targets)
        frame.insert(0, self.uid, ids)
        frame.to_csv(self.root/'train.csv', index=False)
        reports = pd.DataFrame(.15+.7*truth, columns=self.targets)
        reports.insert(0, self.uid, ids)
        reports.to_csv(self.base/'reports.csv', index=False)
        pd.DataFrame({self.uid: ids, 'group_id': [f'person{i//2}' for i in range(len(ids))]}).to_csv(self.base/'groups.csv', index=False)
        for i, study in enumerate(ids):
            for s in range(1 + (i % 3 == 0)):
                for z in range(1 if i % 5 == 0 else 3):
                    series = f'{study}.{s+1}'
                    dicom(self.root/'train_series'/study/series/f'reverse{10-z}.dcm',
                          study, series, z, shape=(18+i%4, 26), spacing=i%4 != 0)
        return ids, truth

    def test_geometry_dimensions_missing_spacing_and_corruption(self):
        directory = self.root/'series'
        for z in range(3):
            dicom(directory/f'{9-z}.dcm', '1.2.3', '1.2.3.4', z, spacing=False)
        (directory/'corrupt.dcm').write_bytes(b'not a dicom')
        result, events = decode_series(directory, {}, self.cfg)
        self.assertEqual(tuple(result['x'].shape), (3, 3, 16, 16))
        self.assertTrue(torch.all(result['position'][1:] > result['position'][:-1]))
        self.assertGreater(result['x'][2, 1].mean(), result['x'][0, 1].mean())
        self.assertTrue(any(e['kind'] == 'spacing_fallback' for e in events))
        self.assertTrue(any(e['kind'] == 'header_failure' for e in events))

    def test_fold_grouping_and_calibration_no_validation_label_leakage(self):
        ids = [str(i) for i in range(80)]
        groups = np.repeat(np.arange(40).astype(str), 2)
        y = np.c_[np.arange(80)%2, (np.arange(80)//2)%2].astype(float)
        folds, audit = make_folds(ids, y, groups, 4, 9)
        for group in set(groups):
            self.assertEqual(len(set(folds[groups == group])), 1)
        tr = np.flatnonzero(folds != 0)
        report = .1+.8*y
        report[0, 0] = .5
        first = supervision(y, report, groups, tr, self.cfg, ['x', 'y'], ids)
        changed = y.copy()
        changed[folds == 0] = 1-changed[folds == 0]
        second = supervision(changed, report, groups, tr, self.cfg, ['x', 'y'], ids)
        self.assertEqual(first[3], second[3])
        np.testing.assert_array_equal(first[0][tr], second[0][tr])
        self.assertTrue(any(v['calibrator'] == 'platt' for v in first[3].values()))
        self.assertFalse(first[1][0, 0])
        self.assertEqual(len(audit['audit']), 4)

    def test_full_prepare_train_resume_merge_infer(self):
        ids, truth = self.build_fixture()
        work = self.base/'work'
        preparation = prepare(self.root, work, self.cfg, self.base/'reports.csv', self.base/'groups.csv')
        self.assertEqual(preparation['studies'], len(ids))
        checkpoints = []
        for fold in range(self.cfg['folds']):
            receipt = train(work, fold)
            self.assertEqual(receipt['status'], 'FOLD_COMPLETE')
            self.assertFalse(set(receipt['train_ids']) & set(receipt['val_ids']))
            path = work/'runs'/f'seed{self.cfg["seed"]}'/'platt'/f'fold{fold}'/'checkpoint.pt'
            before = sha(path)
            train(work, fold)
            self.assertEqual(before, sha(path))
            checkpoints.append(path)
        result = merge(work, self.cfg['seed'], 'platt')
        self.assertTrue(result['metrics']['all_targets_scorable'])
        oof = work/'runs'/f'seed{self.cfg["seed"]}'/'platt'/'oof.csv'
        comparison = evaluate(work, oof, oof, self.base/'comparison.json', bootstrap=30)
        self.assertEqual(comparison['delta'], 0.)
        self.assertEqual(comparison['delta_ci95'], [0., 0.])
        test_ids = ['1.2.826.999', '1.2.826.2', 'missing']
        pd.DataFrame({self.uid: test_ids}).to_csv(self.root/'test.csv', index=False)
        for i, study in enumerate(test_ids[:-1]):
            series = f'{study}.5'
            dicom(self.root/'test_series'/study/series/'one.dcm', study, series, i)
        out = self.base/'submission.csv'
        first = infer(self.root, checkpoints, out, self.base/'testcache', device='cpu')
        pred = pd.read_csv(out).set_index(self.uid)
        self.assertEqual(len(pred), len(test_ids))
        self.assertEqual(first['checkpoints'][0]['fallback_ids'], ['missing'])
        # Different row order and schema order must preserve keyed model results.
        pd.DataFrame({self.uid: test_ids[::-1]}).to_csv(self.root/'test.csv', index=False)
        pd.DataFrame({self.uid: ['example'], **{t: [.5] for t in self.targets[::-1]}}).to_csv(self.root/'sample_submission.csv', index=False)
        infer(self.root, checkpoints, out, self.base/'testcache', device='cpu')
        reordered = pd.read_csv(out).set_index(self.uid).loc[pred.index, self.targets]
        np.testing.assert_allclose(pred[self.targets], reordered, atol=1e-7)
        # Frozen blending requires explicit weight and complete keyed coverage.
        with self.assertRaises(ValueError):
            infer(self.root, checkpoints, out, self.base/'testcache', device='cpu', baseline=out)
        with self.assertRaises(ValueError):
            infer(self.root, checkpoints, out, self.base/'testcache', device='cpu', baseline=out, blend_weight=.1)
        base_file = self.base/'baseline.csv'
        pred.reset_index().to_csv(base_file, index=False)
        infer(self.root, checkpoints, out, self.base/'testcache', device='cpu', baseline=base_file, blend_weight=.1)
        blended = pd.read_csv(out).set_index(self.uid).loc[pred.index, self.targets]
        np.testing.assert_allclose(pred[self.targets], blended, atol=1e-7)
        # Ensure gradients ignore all-unknown targets and padded windows.
        dataset = CachedStudies(work/'cache', ids[:2])
        batch = collate([dataset[i] for i in range(len(dataset))])
        model = KneeModel(self.cfg, len(self.targets)).eval()
        logits = model(batch)[0]
        changed = dict(batch)
        changed['x'] = batch['x'].clone()
        valid = torch.arange(batch['x'].shape[1])[None] < batch['lengths'][:, None]
        changed['x'][~valid] = 1000
        torch.testing.assert_close(model(changed)[0], logits)
        zero = masked_loss(logits, torch.full_like(logits, float('nan')),
                           torch.zeros_like(logits, dtype=torch.bool), self.cfg)
        zero.backward()
        self.assertEqual(zero.item(), 0.)
        # Cache mutation cannot be mistaken for a valid training artifact.
        entry = dataset.manifest['entries'][ids[0]]
        with (work/'cache'/entry['file']).open('ab') as stream:
            stream.write(b'tamper')
        with self.assertRaises(ValueError):
            CachedStudies(work/'cache', ids[:1])


if __name__ == '__main__':
    unittest.main()
