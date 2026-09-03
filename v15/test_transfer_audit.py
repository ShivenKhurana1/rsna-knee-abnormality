import unittest

import numpy as np
import pandas as pd

from transfer_audit import (TARGETS, UID, audit_target, auc, brier, brier_skill, bootstrap_auc,
                             load_aligned, require_unique_ids, run)


def brute_force_auc(y, p):
    """Reference implementation: fraction of positive/negative pairs correctly ranked."""
    pos, neg = p[y == 1], p[y == 0]
    wins = sum((pi > ni) + 0.5 * (pi == ni) for pi in pos for ni in neg)
    return wins / (len(pos) * len(neg))


def frame(values, ids=None):
    f = pd.DataFrame(values, columns=TARGETS)
    f.insert(0, UID, ids or [f'{i:05d}' for i in range(len(f))])
    return f


class AucTests(unittest.TestCase):
    def test_matches_brute_force_pair_ranking(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 60).astype(float)
        p = rng.random(60)
        self.assertAlmostEqual(auc(y, p), brute_force_auc(y, p), places=10)

    def test_nan_below_min_class_count(self):
        y = np.array([1., 1., 0., 0., 0.])
        p = np.array([.9, .8, .7, .6, .5])
        self.assertTrue(np.isnan(auc(y, p)))  # only 2 positives, below MIN_CLASS_COUNT_FOR_AUC=3

    def test_ignores_missing_labels(self):
        y = np.array([1., 0., np.nan, 1., 0., 1., 0.])
        p = np.array([.9, .1, .5, .8, .2, .7, .3])
        expected = auc(np.array([1., 0., 1., 0., 1., 0.]), np.array([.9, .1, .8, .2, .7, .3]))
        self.assertAlmostEqual(auc(y, p), expected)


class BrierTests(unittest.TestCase):
    def test_perfect_predictions_score_zero(self):
        self.assertEqual(brier(np.array([1., 0., 1.]), np.array([1., 0., 1.])), 0.0)

    def test_none_when_all_missing(self):
        self.assertIsNone(brier(np.array([np.nan, np.nan]), np.array([.5, .5])))


class BrierSkillTests(unittest.TestCase):
    def test_perfect_prediction_has_skill_one(self):
        y = np.array([1., 0., 1., 0., 1.])
        result = brier_skill(y, y.copy())
        self.assertAlmostEqual(result['skill_score'], 1.0)

    def test_always_predicting_prevalence_has_skill_zero(self):
        y = np.array([1., 0., 1., 0., 1.])
        p = np.full_like(y, y.mean())
        result = brier_skill(y, p)
        self.assertAlmostEqual(result['skill_score'], 0.0, places=10)

    def test_worse_than_baseline_is_negative(self):
        y = np.array([1., 0., 1., 0., 1.])
        p = 1 - y  # maximally wrong
        result = brier_skill(y, p)
        self.assertLess(result['skill_score'], 0.0)

    def test_none_when_no_valid_labels(self):
        result = brier_skill(np.array([np.nan, np.nan]), np.array([.5, .5]))
        self.assertIsNone(result['skill_score'])
        self.assertIsNone(result['brier'])


class BootstrapAucTests(unittest.TestCase):
    def test_ci_brackets_known_signal(self):
        rng = np.random.default_rng(1)
        n = 300
        y = rng.integers(0, 2, n).astype(float)
        p = np.clip(y * 0.7 + rng.normal(0, 0.2, n), 0, 1)
        result = bootstrap_auc(y, p, n=500, seed=1)
        self.assertIsNotNone(result['ci95'])
        self.assertLess(result['ci95'][0], result['point'])
        self.assertGreater(result['ci95'][1], result['point'])
        self.assertGreater(result['point'], 0.6)

    def test_degenerate_series_reports_no_ci(self):
        y = np.array([1., 1., 1., 1.])
        p = np.array([.9, .8, .7, .6])
        result = bootstrap_auc(y, p, n=200, seed=1)
        self.assertIsNone(result['point'])
        self.assertIsNone(result['ci95'])


class AuditTargetTests(unittest.TestCase):
    def test_separates_addressed_and_silent(self):
        y = np.array([1., 0., 1., 0., 1., 0., 1., 0., 1., 0.])
        p = np.array([.9, .1, .5, .5, .8, .2, .5, .5, .95, .05])
        result = audit_target(y, p, bootstrap=200)
        self.assertEqual(result['n_addressed'], 6)
        self.assertEqual(result['n_silent'], 4)
        self.assertEqual(result['silent_positive'], 2)
        self.assertEqual(result['silent_negative'], 2)
        self.assertEqual(result['silent_expert_prevalence'], 0.5)
        self.assertTrue(result['small_sample_warning'])  # only 3 pos/3 neg addressed

    def test_all_silent_yields_no_addressed_stats(self):
        y = np.array([1., 0., 1., 0.])
        p = np.array([.5, .5, .5, .5])
        result = audit_target(y, p, bootstrap=200)
        self.assertEqual(result['n_addressed'], 0)
        self.assertIsNone(result['addressed_expert_prevalence'])
        self.assertIsNone(result['brier_addressed_only']['skill_score'])
        self.assertIsNone(result['auc_addressed_only']['point'])


class LoadAlignedTests(unittest.TestCase):
    def test_rejects_duplicate_gold_ids(self):
        gold = frame([[0.] * 12, [1.] * 12], ids=['a', 'a'])
        with self.assertRaises(ValueError):
            require_unique_ids(gold, 'gold labels')

    def test_rejects_gold_studies_missing_from_source(self):
        gold = frame([[0.] * 12, [1.] * 12], ids=['a', 'b'])
        source = frame([[0.5] * 12], ids=['a'])
        gold.to_csv('/tmp/_v15_test_gold.csv', index=False)
        source.to_csv('/tmp/_v15_test_source.csv', index=False)
        with self.assertRaises(ValueError):
            load_aligned('/tmp/_v15_test_gold.csv', '/tmp/_v15_test_source.csv')

    def test_rejects_nonbinary_gold_labels(self):
        gold = frame([[0.5] * 12], ids=['a'])
        source = frame([[0.5] * 12], ids=['a'])
        gold.to_csv('/tmp/_v15_test_gold2.csv', index=False)
        source.to_csv('/tmp/_v15_test_source2.csv', index=False)
        with self.assertRaises(ValueError):
            load_aligned('/tmp/_v15_test_gold2.csv', '/tmp/_v15_test_source2.csv')

    def test_aligns_and_reorders_to_gold_index(self):
        gold = frame([[1.] * 12, [0.] * 12], ids=['b', 'a'])
        source = frame([[.5] * 12, [.9] * 12], ids=['a', 'b'])
        gold.to_csv('/tmp/_v15_test_gold3.csv', index=False)
        source.to_csv('/tmp/_v15_test_source3.csv', index=False)
        ids, y, p = load_aligned('/tmp/_v15_test_gold3.csv', '/tmp/_v15_test_source3.csv')
        self.assertEqual(list(ids), ['b', 'a'])
        self.assertEqual(p[0, 0], .9)
        self.assertEqual(p[1, 0], .5)


class RunTests(unittest.TestCase):
    def test_run_end_to_end_on_synthetic_cohort(self):
        rng = np.random.default_rng(3)
        n = 60
        y = rng.integers(0, 2, (n, 12)).astype(float)
        p = np.clip(y * 0.6 + rng.normal(0, 0.25, (n, 12)), 0, 1)
        silence = rng.random((n, 12)) < 0.3
        p[silence] = 0.5
        ids = [f'study-{i:03d}' for i in range(n)]
        gold = pd.DataFrame(y, columns=TARGETS)
        gold.insert(0, UID, ids)
        source = pd.DataFrame(p, columns=TARGETS)
        source.insert(0, UID, ids)
        gold.to_csv('/tmp/_v15_test_gold_run.csv', index=False)
        source.to_csv('/tmp/_v15_test_source_run.csv', index=False)
        result = run('/tmp/_v15_test_gold_run.csv', '/tmp/_v15_test_source_run.csv', bootstrap=200)
        self.assertEqual(result['cohort_studies'], n)
        self.assertEqual(set(result['targets']), set(TARGETS))
        for t in TARGETS:
            self.assertIn('auc_addressed_only', result['targets'][t])


if __name__ == '__main__':
    unittest.main()
