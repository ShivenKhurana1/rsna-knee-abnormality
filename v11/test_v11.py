"""CPU regression tests. No Kaggle data, torch, model downloads, or training."""

import ast
import contextlib
import gc
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from build_v11 import ROOT, HERE, build, find_function, function_span, source
import runtime_helpers as helpers
from runtime_helpers import (V11RunAudit, V11VolumeCache, v11_atomic_csv,
                             v11_combine_members, v11_member_fits,
                             v11_rankpct, v11_validate_submission)


TARGETS = ['A', 'B']


def frame():
    return pd.DataFrame({'StudyInstanceUID': ['001', '002', '003'],
                         'A': [0.1, 0.9, 0.3], 'B': [0.8, 0.2, 0.7]})


class ValidationTests(unittest.TestCase):
    def test_valid_and_single_study(self):
        v11_validate_submission(frame(), ['001', '002', '003'], TARGETS)
        v11_validate_submission(frame().iloc[:1], ['001'], TARGETS)

    def test_reject_invalid_contracts(self):
        cases = [frame().iloc[::-1], frame().assign(A=np.nan),
                 frame().assign(A=np.inf), frame().assign(A=1.01),
                 frame().assign(A=0.5, B=0.5),
                 frame().assign(StudyInstanceUID=['001', '001', '003']),
                 frame()[['StudyInstanceUID', 'B', 'A']]]
        for invalid in cases:
            with self.subTest(case=str(invalid)), self.assertRaises(ValueError):
                v11_validate_submission(invalid, ['001', '002', '003'], TARGETS)

    def test_atomic_roundtrip_and_failed_write_preserves_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'submission.csv'
            v11_atomic_csv(frame(), path)
            before = path.read_bytes()
            with patch.object(pd.DataFrame, 'to_csv', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    v11_atomic_csv(frame().assign(A=[0.9, 0.2, 0.1]), path)
            self.assertEqual(before, path.read_bytes())
            self.assertEqual([path], list(Path(tmp).iterdir()))


class SchedulingAndFusionTests(unittest.TestCase):
    def test_never_substitute_partial_windows(self):
        # One window fits this budget; all 10 do not.
        self.assertFalse(v11_member_fits(100, 1, 1, 20, 10, 10))
        self.assertTrue(v11_member_fits(200, 1, 1, 20, 10, 10))
        self.assertFalse(v11_member_fits(200, 1, 1, 20, 10, 10, jitter=True))
        self.assertFalse(v11_member_fits(-1, 1, 1, None, None, 10))
        self.assertTrue(v11_member_fits(200, 1, 1, None, None, 10))
        self.assertFalse(v11_member_fits(200, 1, 1, np.nan, 10, 10))

    def test_complete_cohort_matches_inherited_fusion(self):
        rng = np.random.default_rng(19)
        members = [{'ids': ['a', 'b', 'c'], 'pred': rng.random((3, 2)),
                    'target_weight': [1, 3]},
                   {'ids': ['c', 'a', 'b'], 'pred': rng.random((3, 2)), 'weight': 2}]
        ids, got = v11_combine_members(members, TARGETS)
        expected = np.zeros((3, 2))
        total = np.zeros(2)
        for member in members:
            weight = np.asarray(member.get('target_weight', [2, 2]))
            ranks = pd.DataFrame(member['pred']).rank(pct=True).to_numpy()
            expected[[ids.index(s) for s in member['ids']]] += ranks * weight
            total += weight
        np.testing.assert_array_equal(got, expected / total)

    def test_missing_member_vote_not_counted_in_denominator(self):
        ids, got = v11_combine_members([
            {'ids': ['a', 'b'], 'pred': [[0, 1], [1, 0]]},
            {'ids': ['a'], 'pred': [[0.4, 0.4]]}], TARGETS)
        self.assertEqual(ids, ['a', 'b'])
        np.testing.assert_array_equal(got[1], [1.0, 0.5])
        np.testing.assert_array_equal(got[0], [0.75, 1.0])

    def test_reject_duplicate_nonfinite_or_zero_votes(self):
        bad = [{'ids': ['a', 'a'], 'pred': [[0, 1], [1, 0]]},
               {'ids': ['a'], 'pred': [[np.nan, 1]]},
               {'ids': ['a'], 'pred': [[0, 1]], 'weight': 0},
               {'ids': ['a'], 'pred': [[0, 1]], 'target_weight': [np.inf, 1]}]
        for member in bad:
            with self.subTest(member=member), self.assertRaises(ValueError):
                v11_combine_members([member], TARGETS)

    def test_ties_are_row_order_independent(self):
        values = np.array([[0.3, 1], [0.3, 0], [0.7, 0]])
        ranks = v11_rankpct(values)
        np.testing.assert_array_equal(ranks[0:2, 0], [0.25, 0.25])
        order = np.array([2, 0, 1])
        np.testing.assert_array_equal(v11_rankpct(values[order]), ranks[order])

    def test_no_ties_match_legacy_rankpct(self):
        values = np.random.default_rng(20).random((30, 12))
        old = values.argsort(0).argsort(0).astype(float) / 29
        np.testing.assert_array_equal(v11_rankpct(values), old)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / 'work'
        frame()[['StudyInstanceUID']].to_csv(self.root / 'test.csv', index=False)
        self.audit = V11RunAudit(self.root, self.work, TARGETS, {'test': True})

    def test_stale_submission_is_never_recovered(self):
        frame().to_csv(self.audit.primary, index=False)
        with self.assertRaisesRegex(RuntimeError, 'current-run'):
            self.audit.recover('stage1', RuntimeError('no weights'))

    def test_late_stage1_failure_restores_banked_bytes(self):
        frame().to_csv(self.audit.primary, index=False)
        before = self.audit.primary.read_bytes()
        self.audit.member_committed('model1', windows=10)
        frame().assign(A=0.5, B=0.5).to_csv(self.audit.primary, index=False)
        self.audit.recover('stage1', RuntimeError('late failure'))
        self.assertEqual(before, self.audit.primary.read_bytes())

    def test_invalid_later_stage_recovers_and_records(self):
        frame().to_csv(self.audit.primary, index=False)
        self.audit.snapshot('stage1')
        before = self.audit.primary.read_bytes()
        frame().assign(A=np.nan).to_csv(self.audit.primary, index=False)
        self.audit.snapshot('stage2')
        self.assertEqual(before, self.audit.primary.read_bytes())
        self.assertEqual(self.audit.receipt['snapshots'][-1]['status'], 'restored')
        receipt = json.loads((self.work / 'v11_run_receipt.json').read_text())
        self.assertEqual(receipt['snapshots'][0]['sha256'], receipt['snapshots'][1]['sha256'])

    def test_unchanged_stage_logged_without_claiming_execution(self):
        frame().to_csv(self.audit.primary, index=False)
        self.audit.snapshot('stage1')
        self.audit.snapshot('stage2')
        self.assertEqual(self.audit.receipt['snapshots'][-1]['status'], 'unchanged')


class CacheTests(unittest.TestCase):
    def test_cache_exact_and_isolated_cleanup(self):
        cache = V11VolumeCache(max_bytes=100000, reserve_bytes=0)
        self.addCleanup(cache.close)
        volume = np.arange(128, dtype=np.uint8).reshape(2, 8, 8)
        mask = np.array([1, 0], dtype=np.uint8)
        calls = []
        def loader(key):
            calls.append(key)
            return volume.copy(), mask.copy()
        cache.get('../../outside', loader)
        got, got_mask = cache.get('../../outside', loader)
        np.testing.assert_array_equal(got, volume)
        np.testing.assert_array_equal(got_mask, mask)
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.hits, 1)
        self.assertTrue(all(path.parent == cache.root for path in cache.paths.values()))
        root = cache.root
        cache.close()
        self.assertFalse(root.exists())

    def test_zero_cap_recomputes_without_allocating_cohort_cache(self):
        cache = V11VolumeCache(max_bytes=0)
        self.addCleanup(cache.close)
        calls = []
        def loader(key):
            calls.append(key)
            return np.zeros((2, 8, 8), np.uint8), np.ones(2, np.uint8)
        cache.get('a', loader)
        cache.get('a', loader)
        self.assertEqual(calls, ['a', 'a'])
        self.assertEqual(cache.bytes, 0)
        self.assertEqual(list(cache.root.iterdir()), [])

    def test_disk_failure_keeps_decoded_result(self):
        cache = V11VolumeCache(max_bytes=100000, reserve_bytes=0)
        self.addCleanup(cache.close)
        arrays = (np.zeros((2, 8, 8), np.uint8), np.ones(2, np.uint8))
        with patch.object(helpers._v11_np, 'savez', side_effect=OSError('full')):
            got = cache.get('a', lambda key: arrays)
        self.assertIs(got[0], arrays[0])
        self.assertEqual(cache.write_failures, 1)
        self.assertEqual(cache.paths, {})

    def test_loader_failure_propagates(self):
        cache = V11VolumeCache()
        self.addCleanup(cache.close)
        def fail(key):
            raise ValueError('bad image')
        with self.assertRaisesRegex(ValueError, 'bad image'):
            cache.get('a', fail)


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = build()
        cls.v9 = json.loads((ROOT / 'v9/rsna-knee-ensemble-v9.ipynb').read_text())

    def test_all_cells_compile_and_no_saved_outputs(self):
        for index, cell in enumerate(self.notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(source(cell), f'cell-{index}', 'exec')
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def test_crops_exactly_match_v9(self):
        for name in ['read_slot', 'read_crop', 'mm_crop_resize']:
            with self.subTest(name=name):
                original = ast.dump(ast.parse(find_function(self.v9, name)))
                actual = ast.dump(ast.parse(find_function(self.notebook, name)))
                self.assertEqual(original, actual)

    def test_training_path_removed(self):
        main_cell = next(source(c) for c in self.notebook['cells']
                         if c['cell_type'] == 'code' and source(c).startswith('def write_submission'))
        self.assertNotIn('opt.step', main_cell)
        self.assertNotIn('RSNA_ALLOW_TRAIN_FALLBACK', main_cell)
        self.assertNotIn('write_benchmark_submission()\n    pkg', main_cell)
        self.assertIn('training is disabled', main_cell)
        self.assertIn('if not V11_PUBLIC_FRONTIER_READY:', main_cell)

    def test_metadata_inputs_unchanged(self):
        old = json.loads((ROOT / 'v10/kernel-metadata.json').read_text())
        new = json.loads((HERE / 'kernel-metadata.json').read_text())
        for key in ['dataset_sources', 'kernel_sources', 'model_sources', 'competition_sources',
                    'enable_gpu', 'enable_internet', 'machine_shape']:
            self.assertEqual(old[key], new[key])

    def test_stage4_no_eager_futures_or_stale_output_promotion(self):
        code = source(self.notebook['cells'][36])
        self.assertNotIn('_vol_futs', code)
        self.assertIn('if V11_COATNET_READY and _blend_coatnet_path.is_file():', code)
        self.assertIn('V11_COATNET_READY = False', code)
        self.assertIn('_vol_cache.close()', code)
        self.assertNotIn('ranks[~np.isfinite(ranks)] = 0.5', code)

    def test_stage4_failed_arm_excluded_successful_arm_preserved(self):
        # Execute the actual generated Stage-4 orchestration with synthetic
        # volumes and fake model calls, not a separate reimplementation.
        code = source(self.notebook['cells'][36])
        start, end = function_span(code, 'main')
        import textwrap
        main = textwrap.dedent(''.join(code.splitlines(keepends=True)[start:end]))
        for failed_mode in ['nonfinite', 'exception']:
            with self.subTest(mode=failed_mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'test_series').mkdir()
                frame()[['StudyInstanceUID']].to_csv(root / 'test.csv', index=False)
                pd.DataFrame({'StudyInstanceUID': ['001', '002', '003'],
                              'SeriesInstanceUID': ['s1', 's2', 's3']}).to_csv(root / 'test_series.csv', index=False)
                output = root / 'coatnet.csv'
                patched_main = main.replace('"/kaggle/working/submission_coatnet.csv"', repr(str(output)))
                notes = []
                def infer(model, value, device):
                    if model == 'bad':
                        if failed_mode == 'exception':
                            raise RuntimeError('simulated GPU error')
                        return np.array([np.nan, 0.2])
                    number = float(value[0, 0, 0])
                    return np.array([number / 4, 1 - number / 4])
                env = {'np': np, 'os': os, 'time': time, 'gc': gc, 'LAB': TARGETS,
                       'DEVS': [types.SimpleNamespace(type='cpu')],
                       'torch': types.SimpleNamespace(__version__='stub',
                           cuda=types.SimpleNamespace(device_count=lambda: 0, empty_cache=lambda: None)),
                       'find_test_root': lambda: str(root), '_make_reader': lambda: None,
                       'ARMS': [{'file': name, 'arch': 'stub', 'res': 8, 'w': 1} for name in ['bad', 'good']],
                       'V11VolumeCache': V11VolumeCache, 'V11_VOLUME_CACHE_GIB': 0,
                       'build_study': lambda sid, *a: (np.full((2, 4, 4), int(sid), np.uint8), np.ones(2, np.uint8)),
                       'wall_left': lambda: 10000, 'find_weight_file': lambda name: name,
                       'load_model': lambda path, *a: (path, 8),
                       'eval_windows': lambda vol, mask, **k: vol, 'K_EVAL': 2, 'NORM': 'stub',
                       'infer_probs': infer, 'rankpct': v11_rankpct,
                       'V11_RUN': types.SimpleNamespace(note=lambda event, **k: notes.append((event, k)))}
                exec(compile(patched_main, '<stage4-main-under-test>', 'exec'), env)
                with contextlib.redirect_stdout(io.StringIO()):
                    env['main']()
                actual = pd.read_csv(output)
                np.testing.assert_array_equal(actual[TARGETS].to_numpy(), [[0, 1], [0.5, 0.5], [1, 0]])
                completed = [details['arm'] for event, details in notes if event == 'stage4_arm_completed']
                self.assertEqual(completed, ['good'])
                self.assertTrue(any(event == 'stage4_arm_failed' for event, details in notes))

    def test_bank_is_transactional_and_retry_deduplicated(self):
        state = {'fail': True}
        per_member, frontier = [], []
        def write(*args):
            if state['fail']:
                raise OSError('save failure')
        env = {'np': np, 'WeightsError': RuntimeError, 'starts_full': [0, 1],
               'V11_STAGE1_JITTER': False, 'BANK_LOCK': threading.Lock(),
               'per_member': per_member, 'public_frontier_members': frontier,
               '_combine': lambda rows: v11_combine_members(rows, TARGETS),
               'write_submission': write, 'test_df': frame(), 'log': lambda msg: None,
               'V11_RUN': types.SimpleNamespace(member_committed=lambda *a, **k: None)}
        exec(compile(find_function(self.notebook, 'bank'), '<bank-under-test>', 'exec'), env)
        args = ({'id': 'member1'}, ['a', 'b'], np.array([[0.1, 0.8], [0.9, 0.3]]), [0, 1], False)
        with self.assertRaises(OSError):
            env['bank'](*args)
        self.assertEqual(per_member, [])
        state['fail'] = False
        env['bank'](*args)
        env['bank'](*args)
        self.assertEqual(len(per_member), 1)
        with self.assertRaisesRegex(RuntimeError, 'window'):
            env['bank'](*args[:3], [0], False)
        self.assertEqual(len(per_member), 1)

    def test_build_deterministic(self):
        path = HERE / 'rsna-knee-ensemble-v11.ipynb'
        before = path.read_bytes()
        build()
        self.assertEqual(before, path.read_bytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
