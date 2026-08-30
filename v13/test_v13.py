"""CPU regression tests; synthetic arrays/stubs are not a Kaggle accuracy test."""

import ast
import contextlib
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import textwrap
import time
import types
import unittest
import zipfile

import numpy as np
import pandas as pd

from build_v13 import HERE, ROOT, build, source, function_span
from coatnet_contracts import (V13_COATNET_CONTRACTS, v13_coatnet_contract,
                              v13_volume_key, v13_check_coatnet_checkpoint)


def extracted(code, name):
    start, end = function_span(code, name)
    return textwrap.dedent(''.join(code.splitlines(keepends=True)[start:end]))


class ContractTests(unittest.TestCase):
    def test_published_recipes(self):
        expected = [(336, .02, .98, 62), (384, .02, .98, 62), (336, .06, .94, 42)]
        for c, row in zip(V13_COATNET_CONTRACTS.values(), expected):
            self.assertEqual(tuple(c[k] for k in ['cache_img', 'span_lo', 'span_hi', 'k_eval']), row)
            self.assertEqual((c['crop_mm'], c['model_res']), (140., 384))

    def test_copy_and_unknown_filename(self):
        name = next(iter(V13_COATNET_CONTRACTS))
        c = v13_coatnet_contract(name)
        c['cache_img'] = 1
        self.assertEqual(v13_coatnet_contract(name)['cache_img'], 336)
        with self.assertRaises(ValueError):
            v13_coatnet_contract('unknown.pt')

    def test_cache_key_separates_geometry_and_studies(self):
        keys = [v13_volume_key(uid, c) for uid in ['001', '002'] for c in V13_COATNET_CONTRACTS.values()]
        self.assertEqual(len(set(keys)), 6)
        c = next(iter(V13_COATNET_CONTRACTS.values()))
        self.assertEqual(v13_volume_key('001', c), v13_volume_key('001', dict(reversed(list(c.items())))))

    def test_checkpoint_labels_and_resolution_checked(self):
        c = next(iter(V13_COATNET_CONTRACTS.values()))
        v13_check_coatnet_checkpoint({'lab': ['A', 'B'], 'res': 384}, ['A', 'B'], c)
        for ck in [{}, {'lab': ['B', 'A'], 'res': 384}, {'lab': ['A', 'B'], 'res': 336}]:
            with self.assertRaises(ValueError):
                v13_check_coatnet_checkpoint(ck, ['A', 'B'], c)


class GeneratedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = build()
        cls.parent = json.loads((ROOT / 'v12/rsna-knee-ensemble-v12.ipynb').read_text(encoding='utf-8'))
        cls.stage = source(cls.notebook['cells'][37])
        cls.helpers = {}
        exec(compile(source(cls.notebook['cells'][3]), '<embedded-runtime>', 'exec'), cls.helpers)

    def test_all_cells_compile_and_unexecuted(self):
        ids = [c['id'] for c in self.notebook['cells']]
        self.assertEqual(len(ids), len(set(ids)))
        for c in self.notebook['cells']:
            self.assertRegex(c['id'], r'^[A-Za-z0-9_-]{1,64}$')
            if c['cell_type'] == 'code':
                compile(source(c), c['id'], 'exec')
                self.assertEqual(c['outputs'], [])
                self.assertIsNone(c['execution_count'])

    def test_other_stages_and_submission_gates_unchanged(self):
        for i, (old, new) in enumerate(zip(self.parent['cells'], self.notebook['cells'])):
            if old['cell_type'] != 'code' or i in [2, 3, 37]:
                continue
            with self.subTest(cell=i):
                before = ast.parse(source(old))
                after = ast.parse(source(new).replace('V13', 'V12').replace('v13', 'v12'))
                self.assertEqual(ast.dump(before), ast.dump(after))
        old = source(self.parent['cells'][2])
        new = source(self.notebook['cells'][2]).replace('V13', 'V12').replace('v13', 'v12')
        new = new.replace('V12_VOLUME_CACHE_GIB = 0', 'V12_VOLUME_CACHE_GIB = 12')
        self.assertEqual(ast.dump(ast.parse(old)), ast.dump(ast.parse(new)))

    def test_same_weights_and_fusion(self):
        def arms(code):
            node = next(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'ARMS' for t in n.targets))
            return ast.literal_eval(node.value)
        self.assertEqual(arms(self.stage), arms(source(self.parent['cells'][37])))
        before = source(self.parent['cells'][37]).split('    # Blend two independently validated rank predictors.')[1]
        after = self.stage.split('    # Blend two independently validated rank predictors.')[1]
        self.assertEqual(before, after.replace('V13', 'V12').replace('v13', 'v12'))

    def test_native_resize_receives_original_crop_directly(self):
        arr = np.arange(400 * 400, dtype=np.float32).reshape(400, 400)
        for c in V13_COATNET_CONTRACTS.values():
            calls = []
            def resize(image, shape, interpolation):
                calls.append((image.copy(), shape, interpolation))
                return np.zeros(shape[::-1], np.float32)
            env = {'contract': c, 'cv2': types.SimpleNamespace(INTER_AREA=3, resize=resize)}
            exec(extracted(self.stage, 'mm_crop_resize'), env)
            result = env['mm_crop_resize'](arr, .5)
            self.assertEqual(result.shape, (c['cache_img'],) * 2)
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(calls[0][0], arr[60:340, 60:340])
            self.assertEqual(calls[0][1], (c['cache_img'],) * 2)

    def test_series_sampling_matches_each_published_span(self):
        for filename, contract in V13_COATNET_CONTRACTS.items():
            c = dict(contract, cache_img=4)  # size reduction only for synthetic CPU test
            seen = []
            def pixels(index):
                seen.append(index)
                return np.full((4, 4), index, np.float32)
            reader = (lambda path: ([(i, .5) for i in range(100)], .5), pixels, lambda a, ps: a)
            env = {'np': np, 'MAXS': 18, 'SLOTS': [('Sagittal', 1, 18)]}
            exec(extracted(self.stage, '_pick_series_for_slot'), env)
            exec(extracted(self.stage, 'build_study'), env)
            rows = {'001': [{'Anatomical_Plane': 'Sagittal', 'Fluid_Sensitive': 1, 'SeriesInstanceUID': 'series'}]}
            volume, mask = env['build_study']('001', rows, 'unused', reader, c)
            lo, hi = (6, 93) if filename.endswith('v4_full.pt') else (2, 97)
            np.testing.assert_array_equal(seen, np.linspace(lo, hi, 18).round().astype(int))
            self.assertEqual(volume.shape, (18, 4, 4))
            self.assertEqual(mask.shape, (18,))

    def test_maxspan_volume_exactly_matches_v12_on_synthetic_input(self):
        c = dict(next(iter(V13_COATNET_CONTRACTS.values())), cache_img=4)
        rows = {'001': [{'Anatomical_Plane': 'Sagittal', 'Fluid_Sensitive': 1, 'SeriesInstanceUID': 'series'}]}
        for count in [1, 2, 7, 20, 100]:
            reader = (lambda path: ([(i, .5) for i in range(count)], .5),
                      lambda i: np.arange(16, dtype=np.float32).reshape(4, 4) + i, lambda a, ps: a)
            outputs = []
            for code, args in [(source(self.parent['cells'][37]), ()), (self.stage, (c,))]:
                env = {'np': np, 'MAXS': 18, 'IMG': 4, 'SLOTS': [('Sagittal', 1, 18)]}
                exec(extracted(code, '_pick_series_for_slot'), env)
                exec(extracted(code, 'build_study'), env)
                outputs.append(env['build_study']('001', rows, 'unused', reader, *args))
            for old, new in zip(*outputs):
                np.testing.assert_array_equal(old, new)

    def test_actual_orchestration_routes_all_three_contracts_and_saves_raw(self):
        self.run_orchestration(fail_native=False)

    def test_failed_arm_not_blended_or_saved(self):
        self.run_orchestration(fail_native=True)

    def run_orchestration(self, fail_native):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'test_series').mkdir()
            ids = ['001', '002', '003']
            labels = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
                      'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
            pd.DataFrame({'StudyInstanceUID': ids}).to_csv(root / 'test.csv', index=False)
            pd.DataFrame({'StudyInstanceUID': ids, 'SeriesInstanceUID': ['s1', 's2', 's3']}).to_csv(root / 'test_series.csv', index=False)
            output = root / 'coatnet.csv'
            code = extracted(self.stage, 'main').replace('"/kaggle/working/submission_coatnet.csv"', repr(str(output)))
            notes, window_calls, builds = [], [], []
            names = list(V13_COATNET_CONTRACTS)
            def volume(sid, series, tsdir, reader, c):
                self.assertEqual(reader, c)
                builds.append((sid, c['cache_img'], c['span_lo']))
                return np.full((3, 4, 4), int(sid), np.uint8), np.ones(3, np.uint8)
            def windows(vol, mask, **kw):
                window_calls.append(kw)
                return vol
            def infer(model, vol, device):
                if fail_native and model == names[1]:
                    raise RuntimeError('synthetic native model error')
                return np.full(12, float(vol[0, 0, 0]) / 4)
            env = dict(self.helpers)
            env.update(np=np, os=os, time=time, gc=gc, LAB=labels,
                       DEVS=[types.SimpleNamespace(type='cpu')],
                       torch=types.SimpleNamespace(__version__='stub', cuda=types.SimpleNamespace(device_count=lambda: 0)),
                       find_test_root=lambda: str(root), _make_reader=lambda c: c,
                       ARMS=[dict(file=n, arch='stub', res=384, w=1.) for n in names],
                       V13_VOLUME_CACHE_GIB=0, build_study=volume, wall_left=lambda: 10000,
                       find_weight_file=lambda n: n, load_model=lambda n, *args: (n, 384),
                       eval_windows=windows, NORM='imagenet', infer_probs=infer,
                       rankpct=env['v13_rankpct'],
                       V13_RUN=types.SimpleNamespace(folder=root, note=lambda e, **k: notes.append((e, k))))
            exec(code, env)
            with contextlib.redirect_stdout(io.StringIO()):
                env['main']()
            self.assertEqual([c['k'] for c in window_calls], [62] * (4 if fail_native else 6) + [42] * 3)
            self.assertEqual(len(builds), 7 if fail_native else 9)
            done = [d['arm'] for e, d in notes if e == 'stage4_arm_completed']
            self.assertEqual(done, [names[0], names[2]] if fail_native else names)
            files = list(root.glob('coatnet_arm_*_raw.csv'))
            self.assertEqual(len(files), 2 if fail_native else 3)
            for path in files:
                frame = pd.read_csv(path, dtype={'StudyInstanceUID': str})
                self.assertEqual(frame.StudyInstanceUID.tolist(), ids)
                np.testing.assert_array_equal(frame['ACL'], [.25, .5, .75])
            frame = pd.read_csv(output)
            np.testing.assert_array_equal(frame['ACL'], [0., .5, 1.])
            cache = next(d for e, d in notes if e == 'stage4_volume_cache')
            self.assertEqual(cache['entries'], 0)

    def test_inputs_identical_and_archive_deterministic(self):
        before = (HERE / 'rsna-knee-ensemble-v13.zip').read_bytes()
        build()
        self.assertEqual(before, (HERE / 'rsna-knee-ensemble-v13.zip').read_bytes())
        old = json.loads((ROOT / 'v12/kernel-metadata.json').read_text())
        new = json.loads((HERE / 'kernel-metadata.json').read_text())
        for key in old.keys() - {'id', 'title', 'code_file'}:
            self.assertEqual(old[key], new[key])
        manifest = json.loads((HERE / 'build_manifest.json').read_text())
        self.assertIsNone(manifest['measured_v13_auc'])
        self.assertFalse(manifest['kaggle_executed'])
        self.assertEqual(manifest['notebook_sha256'], hashlib.sha256((HERE / new['code_file']).read_bytes()).hexdigest())
        with zipfile.ZipFile(HERE / 'rsna-knee-ensemble-v13.zip') as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(archive.namelist()), 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
