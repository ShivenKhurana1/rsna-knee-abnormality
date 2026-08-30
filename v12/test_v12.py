"""Submission-package tests, without network access, GPU inference, or real data."""

import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from build_v12 import HERE, ROOT, build, _source
import submission_checks as checks


def fake_competition(base, owner_qualified=False):
    root = Path(base) / ('competitions/rsna-knee-abnormality-detection'
                         if owner_qualified else 'rsna-knee-abnormality-detection')
    root.mkdir(parents=True)
    (root / 'test_series').mkdir()
    pd.DataFrame({'StudyInstanceUID': ['001', '002', '003']}).to_csv(root / 'test.csv', index=False)
    pd.DataFrame(columns=['StudyInstanceUID'] + checks.V12_SUBMISSION_TARGETS).to_csv(
        root / 'sample_submission.csv', index=False)
    pd.DataFrame(columns=['StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane',
                         'Fluid_Sensitive', 'Fat_Suppression']).to_csv(root / 'test_series.csv', index=False)
    return root


class PackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = build()
        cls.parent = json.loads((ROOT / 'v11/rsna-knee-ensemble-v11.ipynb').read_text(encoding='utf-8'))

    def test_compile_and_notebook_cell_contract(self):
        notebook = self.notebook
        self.assertEqual((notebook['nbformat'], notebook['nbformat_minor']), (4, 5))
        ids = [cell['id'] for cell in notebook['cells']]
        self.assertEqual(len(ids), len(set(ids)))
        for index, cell in enumerate(notebook['cells']):
            self.assertRegex(cell['id'], r'^[A-Za-z0-9_-]{1,64}$')
            self.assertIn(cell['cell_type'], ['code', 'markdown'])
            self.assertIsInstance(cell['metadata'], dict)
            if cell['cell_type'] == 'code':
                compile(_source(cell), f'V12-cell-{index}', 'exec')
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])

    def test_inference_recipe_unchanged_from_v11(self):
        actual = self.notebook['cells'][:1] + self.notebook['cells'][2:]
        self.assertEqual(len(actual), len(self.parent['cells']))
        for index, (old, new) in enumerate(zip(self.parent['cells'], actual)):
            if old['cell_type'] != 'code' or index == len(actual) - 1:
                continue
            before = ast.parse(_source(old))
            after = ast.parse(_source(new).replace('V12', 'V11').replace('v12', 'v11'))
            if _source(old).startswith('V11_BUILD_INFO = '):
                before.body = before.body[1:]
                after.body = after.body[1:]
            with self.subTest(cell=index):
                self.assertEqual(ast.dump(before), ast.dump(after))

    def test_inputs_same_and_metadata_points_to_v12(self):
        old = json.loads((ROOT / 'v11/kernel-metadata.json').read_text())
        new = json.loads((HERE / 'kernel-metadata.json').read_text())
        for key in old.keys() - {'id', 'title', 'code_file'}:
            self.assertEqual(old[key], new[key])
        self.assertEqual(new['code_file'], 'rsna-knee-ensemble-v12.ipynb')
        self.assertTrue(new['id'].endswith('/rsna-knee-ensemble-v12'))

    def test_release_archive_complete_and_no_weights(self):
        with zipfile.ZipFile(HERE / 'rsna-knee-ensemble-v12.zip') as archive:
            self.assertEqual(set(archive.namelist()), {'rsna-knee-ensemble-v12.ipynb',
                'kernel-metadata.json', 'build_manifest.json', 'README.md'})
            self.assertIsNone(archive.testzip())
            for name in archive.namelist():
                self.assertEqual(archive.read(name), (HERE / name).read_bytes())

    def test_hashes_and_deterministic_build(self):
        files = ['rsna-knee-ensemble-v12.ipynb', 'rsna-knee-ensemble-v12.zip',
                 'kernel-metadata.json', 'build_manifest.json']
        before = {name: (HERE / name).read_bytes() for name in files}
        build()
        for name, data in before.items():
            self.assertEqual(data, (HERE / name).read_bytes())
        manifest = json.loads(before['build_manifest.json'])
        self.assertEqual(manifest['notebook_sha256'],
                         hashlib.sha256(before['rsna-knee-ensemble-v12.ipynb']).hexdigest())
        self.assertFalse(manifest['kaggle_executed'])
        self.assertFalse(manifest['kaggle_submitted'])
        self.assertIsNone(manifest['measured_v12_auc'])


class PreflightTests(unittest.TestCase):
    def test_both_competition_mounts(self):
        for qualified in (False, True):
            with self.subTest(qualified=qualified), tempfile.TemporaryDirectory() as tmp:
                fake_competition(tmp, qualified)
                result = checks.v12_preflight(tmp, Path(tmp) / 'work', False, False)
                self.assertEqual(result['studies'], 3)
                marker = json.loads((Path(tmp) / 'work/v12_submission_ready.json').read_text())
                self.assertEqual(marker['status'], 'NOT_READY')

    def test_missing_data_resets_stale_ready_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / 'work'
            work.mkdir()
            marker = work / 'v12_submission_ready.json'
            marker.write_text('{"status":"READY_FOR_KAGGLE_SUBMISSION"}')
            with self.assertRaises(FileNotFoundError):
                checks.v12_preflight(tmp, work, False, False)
            self.assertEqual(json.loads(marker.read_text())['status'], 'NOT_READY')

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_competition(tmp)
            pd.DataFrame({'StudyInstanceUID': ['001', '001']}).to_csv(root / 'test.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                checks.v12_preflight(tmp, Path(tmp) / 'work', False, False)

    def test_schema_and_dependencies_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_competition(tmp)
            with patch.object(checks._v12_check_importlib, 'find_spec', return_value=None):
                with self.assertRaisesRegex(RuntimeError, 'packages missing'):
                    checks.v12_preflight(tmp, Path(tmp) / 'work', True, False)
            pd.DataFrame(columns=['StudyInstanceUID', 'WrongTarget']).to_csv(root / 'sample_submission.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'schema'):
                checks.v12_preflight(tmp, Path(tmp) / 'work', False, False)


class FinalGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = fake_competition(self.temp.name)
        self.work = Path(self.temp.name) / 'work'
        notebook = json.loads((HERE / 'rsna-knee-ensemble-v12.ipynb').read_text(encoding='utf-8'))
        helpers = next(_source(c) for c in notebook['cells']
                       if c['cell_type'] == 'code' and _source(c).startswith('V12_BUILD_INFO = '))
        env = {}
        exec(compile(helpers, '<embedded-V12-runtime>', 'exec'), env)
        self.run = env['V12RunAudit'](self.root, self.work, checks.V12_SUBMISSION_TARGETS, env['V12_BUILD_INFO'])
        values = np.random.default_rng(12).random((3, 12))
        self.pred = pd.DataFrame(values, columns=checks.V12_SUBMISSION_TARGETS)
        self.pred.insert(0, 'StudyInstanceUID', ['001', '002', '003'])
        self.pred.to_csv(self.run.primary, index=False)

    def test_no_member_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'No current-run'):
            checks.v12_finalize_submission(self.run)

    def test_success_matches_current_final_hash(self):
        self.run.member_committed('synthetic-test-member')
        self.run.snapshot('final')
        ready = checks.v12_finalize_submission(self.run)
        self.assertEqual(ready['status'], 'READY_FOR_KAGGLE_SUBMISSION')
        self.assertIsNone(ready['measured_auc'])
        self.assertEqual(ready['submission_sha256'], hashlib.sha256(self.run.primary.read_bytes()).hexdigest())

    def test_post_snapshot_change_rejected(self):
        self.run.member_committed('synthetic-test-member')
        self.run.snapshot('final')
        altered = self.pred.copy()
        altered['ACL'] = [0.9, 0.1, 0.3]
        altered.to_csv(self.run.primary, index=False)
        with self.assertRaisesRegex(RuntimeError, 'snapshot'):
            checks.v12_finalize_submission(self.run)

    def test_nonfinal_snapshot_rejected(self):
        self.run.member_committed('synthetic-test-member')
        self.run.snapshot('stage1')
        with self.assertRaisesRegex(RuntimeError, 'snapshot'):
            checks.v12_finalize_submission(self.run)


if __name__ == '__main__':
    unittest.main(verbosity=2)
