import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import build_gold_validation
import gold_validation as gold
import diagnostics
import verify_gold_artifacts
from fusion import v14_fuse


class GoldValidationTests(unittest.TestCase):
    def fixture(self, root):
        ids = [f'study-{i}' for i in range(30)]
        frame = pd.DataFrame({gold.GOLD_UID: ids, 'Report': ['never put this in test inputs'] * 30})
        for t in gold.GOLD_TARGETS:
            frame[t] = np.arange(30) % 2
        frame.to_csv(root / 'train.csv', index=False)
        pd.DataFrame({gold.GOLD_UID: ids, 'SeriesInstanceUID': ['series'] * 30,
                      'Anatomical_Plane': ['Sagittal'] * 30, 'Fluid_Sensitive': [1] * 30,
                      'Fat_Suppression': [1] * 30}).to_csv(root / 'train_series.csv', index=False)
        for uid in ids:
            (root / 'train_series' / uid).mkdir(parents=True)
        return ids

    def test_adapter_strips_labels_and_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ids = self.fixture(root)
            # Windows symlink privileges are irrelevant to schema correctness.
            with patch.object(gold.os, 'symlink', side_effect=lambda src, dst, **kw: Path(dst).mkdir()), contextlib.redirect_stdout(io.StringIO()):
                pseudo, work = gold.prepare_gold(root, root / 'work', root / 'absent-temp-parent')
            self.assertEqual(list(pd.read_csv(pseudo / 'test.csv')), [gold.GOLD_UID])
            self.assertEqual(list(pd.read_csv(pseudo / 'train.csv')), [gold.GOLD_UID])
            self.assertEqual(set(p.name for p in (pseudo / 'test_series').iterdir()), set(ids))
            audit = json.loads((work / 'cohort_audit.json').read_text())
            self.assertFalse(audit['plus_0_02_verified'])
            self.assertFalse(audit['leaderboard_submission_allowed'])
            with self.assertRaises(RuntimeError):
                gold.prepare_gold(root, work, root)

    def test_unknown_metadata_is_not_exclusion(self):
        result = gold.audit_metadata({'fold': 2, 'state_dict': {'train_ids': ['a']}}, ['a'])
        self.assertFalse(result['exclusion_verified'])
        self.assertEqual(result['explicit_id_sets'], [])

    def test_explicit_overlap_recorded(self):
        result = gold.audit_metadata({'metadata': {'train_ids': ['a', 'b'], 'gold_ids': ['c']}}, ['b', 'c'])
        self.assertEqual(result['explicit_id_sets'][0]['gold_overlap_ids'], ['b'])
        self.assertEqual(result['explicit_id_sets'][1]['gold_overlap_ids'], ['c'])

    def test_generated_roots_and_gate(self):
        notebook = build_gold_validation.build()
        sources = [''.join(c['source']) for c in notebook['cells']]
        text = '\n'.join(sources)
        self.assertNotIn('READY_FOR_KAGGLE_SUBMISSION', text)
        self.assertIn("e['member']", text)
        self.assertNotIn("e['member_id']", text)
        self.assertIn('COMP = GOLD_ROOT', text)
        self.assertIn('return str(GOLD_ROOT)', text)
        self.assertIn('return GOLD_ROOT', text)
        self.assertNotIn("'/kaggle/working/submission.csv'", text)
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(sources[i], str(i), 'exec')

    def test_float32_roundtrip_guard_regression(self):
        parent = json.loads((build_gold_validation.HERE.parent / 'v13/rsna-knee-ensemble-v13.ipynb').read_text(encoding='utf-8'))
        original = {}
        exec(''.join(parent['cells'][3]['source']), original)
        rng = np.random.default_rng(14)
        frame = pd.DataFrame(rng.uniform(size=(58, 12)).astype(np.float32), columns=gold.GOLD_TARGETS)
        frame.insert(0, gold.GOLD_UID, [str(i) for i in range(58)])
        notebook = build_gold_validation.build()
        fixed = {}
        exec(''.join(notebook['cells'][4]['source']), fixed)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'raw.csv'
            with self.assertRaisesRegex(ValueError, 'CSV roundtrip predictions changed'):
                original['v13_atomic_csv'](frame, path)
            fixed['v13_atomic_csv'](frame, path)
            restored = pd.read_csv(path, dtype={gold.GOLD_UID: str})
            np.testing.assert_array_equal(restored[gold.GOLD_TARGETS].to_numpy(np.float32),
                                          frame[gold.GOLD_TARGETS].to_numpy())
            np.testing.assert_allclose(restored[gold.GOLD_TARGETS].to_numpy(float),
                                       frame[gold.GOLD_TARGETS].to_numpy(float), atol=1e-15, rtol=0)

    def test_output_difference_gate_rejects_cosmetic_changes(self):
        baseline = pd.DataFrame(np.tile(np.linspace(.1, .9, 30)[:, None], (1, 12)), columns=gold.GOLD_TARGETS)
        baseline.insert(0, gold.GOLD_UID, [str(i) for i in range(30)])
        with self.assertRaisesRegex(ValueError, 'numerically identical'):
            diagnostics.require_changed_rankings(baseline, baseline)
        monotone = baseline.copy()
        monotone[gold.GOLD_TARGETS] = monotone[gold.GOLD_TARGETS] ** 2
        with self.assertRaisesRegex(ValueError, 'rankings are unchanged'):
            diagnostics.require_changed_rankings(baseline, monotone)
        changed = baseline.copy()
        changed.loc[[0, 29], 'ACL'] = [.9, .1]
        self.assertFalse(diagnostics.require_changed_rankings(baseline, changed)['all_target_rankings_equal'])

    def test_independent_capture_verifier_and_mixed_run_rejection(self):
        rng = np.random.default_rng(143)
        ids = [str(i) for i in range(58)]
        tr = rng.uniform(size=(58, 12))
        # Match the deployed arm insertion order; sorted candidate names differ.
        names = ['raptor_ft_coatnet_v5_full_swa.pt', 'raptor_ft_coatnet_v10_full.pt', 'raptor_ft_coatnet_v4_full.pt']
        arms = {name: rng.uniform(size=(58, 12)) for name in names}
        values = {'labels': np.tile((np.arange(58) % 2)[:, None], (1, 12)),
                  'baseline': v14_fuse(tr, arms), 'stage3': tr,
                  **{f'arm{i}': arms[name] for i, name in enumerate(names)}}
        with tempfile.TemporaryDirectory() as folder:
            for name, matrix in values.items():
                frame = pd.DataFrame(matrix, columns=gold.GOLD_TARGETS)
                frame.insert(0, gold.GOLD_UID, ids)
                capture = {'source': 'https://example.invalid/?scriptVersionId=test-only',
                           'heading': name, 'rows': [list(frame)] + frame.astype(str).values.tolist()}
                (Path(folder) / (name + '.json')).write_text(json.dumps(capture), encoding='utf-8')
            # Small bootstrap is sufficient for a synthetic plumbing test.
            with patch.object(verify_gold_artifacts, 'compare', side_effect=lambda *a, **kw: diagnostics.compare(*a, bootstrap=100)):
                result = verify_gold_artifacts.verify(folder, 'test-only')
            self.assertEqual(result['baseline_replay_max_error'], 0)
            self.assertLess(result['auc_sklearn_max_error'], 1e-14)
            self.assertTrue(result['all_six_candidates_have_changed_rankings'])
            self.assertFalse(result['plus_0_02_verified'])
            with self.assertRaisesRegex(ValueError, 'Mixed/stale'):
                verify_gold_artifacts.verify(folder, 'different-version')


if __name__ == '__main__':
    unittest.main()
