import ast
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from build_v14 import build, source, HERE, ROOT
from confirm import verify_evidence
from diagnostics import UID, TARGETS, align, aucs, compare, digest, prediction_delta, write_new
from fusion import v14_fuse, v14_export_ablations
from residual_specialist import fit, predict, crossfit


def frame(values):
    f = pd.DataFrame(values, columns=TARGETS)
    f.insert(0, UID, [f'{i:05d}' for i in range(len(f))])
    return f


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(14)
        self.y = frame(rng.integers(0, 2, (100, 12)).astype(float))
        self.p = frame(rng.uniform(.1, .9, (100, 12)))

    def test_auc_matches_sklearn_with_ties_and_missing(self):
        p = self.p[TARGETS].to_numpy().round(1)
        y = self.y[TARGETS].to_numpy(copy=True)
        self.assertTrue(np.allclose(aucs(y, p), roc_auc_score(y, p, average=None)))
        y[:10] = np.nan
        self.assertTrue(np.allclose(aucs(y, p), roc_auc_score(y[10:], p[10:], average=None)))

    def test_exact_zero_comparison_after_row_reordering(self):
        r = compare(self.y, self.p, self.p.sample(frac=1), bootstrap=100)
        self.assertEqual(r['macro_delta_ci95'], [0., 0.])
        self.assertFalse(r['ci_lower_bound_at_least_0_02'])

    def test_known_perfect_gain(self):
        baseline = self.p.copy()
        baseline[TARGETS] = .5
        r = compare(self.y, baseline, self.y, bootstrap=100, groups=np.repeat(np.arange(50), 2))
        self.assertEqual(r['macro_delta'], .5)
        self.assertEqual(r['bootstrap_unit'], 'group')

    def test_probability_changes_without_rank_changes(self):
        q = self.p.copy()
        q[TARGETS] = q[TARGETS] ** 2
        r = prediction_delta(self.p, q)
        self.assertFalse(r['values_exactly_equal'])
        self.assertTrue(r['all_target_rankings_equal'])

    def test_bad_schema_ids_coverage_and_probabilities(self):
        bad = [self.p.iloc[:-1], pd.concat([self.p, self.p.iloc[:1]])]
        for x in bad:
            with self.assertRaises(ValueError):
                align(x, self.p[UID])
        for value in [np.nan, np.inf, -1, 2]:
            q = self.p.copy()
            q.loc[0, 'ACL'] = value
            with self.assertRaises(ValueError):
                align(q)
        q = self.p.copy()
        q[UID] = range(len(q))
        with self.assertRaises(ValueError):
            align(q)

    def test_soft_and_single_class_labels_rejected(self):
        with self.assertRaises(ValueError):
            align(self.p, labels=True)
        self.y['Fracture'] = 0.
        with self.assertRaisesRegex(ValueError, 'All 12'):
            compare(self.y, self.p, self.p, bootstrap=100)

    def test_small_group_count_rejected(self):
        with self.assertRaisesRegex(ValueError, '20'):
            compare(self.y, self.p, self.p, bootstrap=100, groups=np.zeros(100))

    def test_existing_report_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'report.json'
            write_new(path, {'value': 1})
            with self.assertRaises(FileExistsError):
                write_new(path, {'value': 2})


class FusionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(22)
        self.tr = rng.uniform(size=(120, 12))
        self.arms = {str(i): rng.uniform(size=(120, 12)).round(2) for i in range(3)}

    def test_exact_v13_formula_and_serialization(self):
        # Execute the actual frozen V13 rank helper, not a second handwritten implementation.
        n = json.loads((ROOT / 'v13/rsna-knee-ensemble-v13.ipynb').read_text(encoding='utf-8'))
        code = source(n['cells'][3])
        ns = {}
        exec(compile(code, '<v13-runtime>', 'exec'), ns)
        ranks = np.tensordot(np.ones(3) / 3,
                            np.stack([ns['v13_rankpct'](a) for a in self.arms.values()]), axes=(0, 0))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'coat.csv'
            pd.DataFrame(ranks.astype(np.float32), columns=TARGETS).to_csv(path, index=False)
            cr = pd.read_csv(path).rank(method='average', pct=True)
        tr = pd.DataFrame(self.tr, columns=TARGETS).rank(method='average', pct=True)
        expected = (.5 * tr + .5 * cr).rank(method='average', pct=True).to_numpy()
        np.testing.assert_array_equal(v14_fuse(self.tr, self.arms), expected)

    def test_monotone_arm_changes_disappear(self):
        np.testing.assert_array_equal(v14_fuse(self.tr, self.arms),
                                      v14_fuse(self.tr, {k: v ** 2 for k, v in self.arms.items()}))

    def test_single_row_and_missing_arms(self):
        np.testing.assert_array_equal(v14_fuse(self.tr[:1], {'a': self.tr[:1]}), np.ones((1, 12)))
        with self.assertRaises(ValueError):
            v14_fuse(self.tr, {})
        with self.assertRaises(ValueError):
            v14_fuse(self.tr, {'bad': np.full((120, 12), np.nan)})

    def test_export_never_mutates_control(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            events = []
            frame(self.tr).to_csv(folder / 'stage3.csv', index=False)
            # Use the same serialized inputs as the actual export function.
            tr = pd.read_csv(folder / 'stage3.csv')[TARGETS].to_numpy()
            arms = {}
            for name, values in self.arms.items():
                path = folder / (name + '.csv')
                frame(values).to_csv(path, index=False)
                arms[name] = pd.read_csv(path)[TARGETS].to_numpy()
                events.append({'event': 'stage4_arm_predictions', 'arm': name,
                               'path': str(path), 'sha256': digest(path)})
            primary = folder / 'submission.csv'
            frame(v14_fuse(tr, arms)).to_csv(primary, index=False)
            before = primary.read_bytes()
            run = types.SimpleNamespace(primary=primary, folder=folder, ids=frame(tr)[UID].tolist(),
                                        receipt={'events': events}, note=lambda *a, **kw: None)
            receipt = v14_export_ablations(run)
            self.assertTrue(receipt['v13_reproduced'])
            self.assertEqual(len(receipt['candidates']), 7)
            self.assertEqual(primary.read_bytes(), before)


class SpecialistTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.x = rng.normal(size=(150, 4))
        self.y = rng.integers(0, 2, (150, 12)).astype(float)
        self.b = rng.uniform(.05, .95, (150, 12))

    def test_no_labels_exact_baseline_and_roundtrip(self):
        self.y[:] = np.nan
        head = fit(self.x, self.b, self.y)
        np.testing.assert_array_equal(predict(head, self.x, self.b), self.b)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.npz'
            np.savez(path, **head)
            with np.load(path, allow_pickle=False) as loaded:
                np.testing.assert_array_equal(predict(loaded, self.x, self.b), self.b)

    def test_holdout_label_changes_cannot_change_own_fold_predictions(self):
        groups = np.repeat(np.arange(50), 3)
        p, folds = crossfit(self.x, self.b, self.y, groups)
        y = self.y.copy()
        y[folds == 0] = 1 - y[folds == 0]
        q, again = crossfit(self.x, self.b, y, groups)
        np.testing.assert_array_equal(folds, again)
        np.testing.assert_array_equal(p[folds == 0], q[folds == 0])
        for group in set(groups):
            self.assertEqual(len(set(folds[groups == group])), 1)

    def test_bad_features_and_regularization(self):
        self.x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit(self.x, self.b, self.y)
        with self.assertRaises(ValueError):
            fit(np.nan_to_num(self.x), self.b, self.y, regularization=0)


class EvidenceTests(unittest.TestCase):
    def test_overlap_and_weak_labels_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            f = root / 'exclusion.csv'
            pd.DataFrame({UID: ['train'], 'GroupID': ['patient-a']}).to_csv(f, index=False)
            artifact = root / 'model.txt'
            artifact.write_text('frozen model')
            e = {'label_quality': 'expert_image', 'candidate_frozen_before_confirmation': True,
                 'confirmation_labels_used_for_selection': False,
                 'all_component_training_and_selection_ids_included': True, 'provenance_notes': 'unit test',
                 'exclusion_manifests': [{'path': f.name, 'sha256': digest(f)}],
                 'frozen_candidate_artifact': {'path': artifact.name, 'sha256': digest(artifact)}}
            labels = pd.DataFrame({UID: ['valid']})
            verify_evidence(e, root, labels, ['patient-b'])
            with self.assertRaisesRegex(ValueError, 'overlap'):
                verify_evidence(e, root, labels, ['patient-a'])
            e['label_quality'] = 'weak_report'
            with self.assertRaises(ValueError):
                verify_evidence(e, root, labels, ['patient-b'])


class NotebookTests(unittest.TestCase):
    def test_build_is_deterministic_and_control_unchanged(self):
        n = build()
        sha = digest(HERE / 'rsna-knee-ensemble-v14.ipynb')
        build()
        self.assertEqual(sha, digest(HERE / 'rsna-knee-ensemble-v14.ipynb'))
        parent = json.loads((ROOT / 'v13/rsna-knee-ensemble-v13.ipynb').read_text(encoding='utf-8'))
        for i, (a, b) in enumerate(zip(parent['cells'], n['cells'])):
            if a['cell_type'] != 'code' or i == 3:
                continue
            self.assertEqual(ast.dump(ast.parse(source(a))),
                             ast.dump(ast.parse(source(b).replace('V14', 'V13').replace('v14', 'v13'))))
        for cell in n['cells']:
            if cell['cell_type'] == 'code':
                compile(source(cell), cell['id'], 'exec')
                self.assertEqual(cell['outputs'], [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
