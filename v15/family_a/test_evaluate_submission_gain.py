import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from contract import TARGETS, UID
from evaluate_submission_gain import evaluate, labels_from_frame


def frame(ids, values):
    result = pd.DataFrame(values, columns=TARGETS)
    result.insert(0, UID, ids)
    return result


class SubmissionGainTests(unittest.TestCase):
    def test_report_silence_is_excluded_from_weak_truth(self):
        values = np.array([[0.9] * 12, [0.1] * 12, [0.5] * 12])
        labels = labels_from_frame(frame(['a', 'b', 'c'], values), 'report-soft')
        self.assertTrue((labels[0] == 1).all())
        self.assertTrue((labels[1] == 0).all())
        self.assertTrue(np.isnan(labels[2]).all())

    def test_perfect_diverse_family_improves_frozen_blend(self):
        rng = np.random.default_rng(3)
        n = 120
        ids = [f's{i:03d}' for i in range(n)]
        y = np.fromfunction(lambda i, j: (i + j) % 2, (n, len(TARGETS))).astype(float)
        baseline = rng.random((n, len(TARGETS)))
        family1 = np.clip(y * 0.9 + (1 - y) * 0.1, 0, 1)
        family2 = np.clip(family1 + rng.normal(0, 0.01, family1.shape), 0, 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / name for name in ('labels.csv', 'baseline.csv', 'f1.csv', 'f2.csv')]
            for path, values in zip(paths, (y, baseline, family1, family2)):
                frame(ids, values).to_csv(path, index=False)
            result = evaluate(paths[0], paths[1], paths[2:], weights=(0.2,),
                              bootstrap=200, seed=4)
        primary = result['primary_result']
        self.assertGreater(primary['macro_delta'], 0)
        self.assertAlmostEqual(result['gain_required_for_0_950'], 0.015)
        self.assertEqual(len(result['primary_seed_deltas']), 2)
        propagation = result['error_propagation']
        self.assertGreaterEqual(propagation['combined_se_approx'],
                                propagation['paired_bootstrap_se'])
        self.assertEqual(propagation['leaderboard_rounding']
                         ['displayed_gain_0_015_latent_interval'], [0.014, 0.016])

    def test_requires_two_training_seeds(self):
        n = 20
        ids = [f's{i}' for i in range(n)]
        y = np.fromfunction(lambda i, j: (i + j) % 2, (n, len(TARGETS))).astype(float)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels, baseline, family = (root / 'labels.csv', root / 'baseline.csv', root / 'family.csv')
            for path in (labels, baseline, family):
                frame(ids, y).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                evaluate(labels, baseline, [family], bootstrap=100)


if __name__ == '__main__':
    unittest.main()
