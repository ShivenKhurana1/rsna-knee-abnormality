"""Synthetic evaluation-tool tests; outputs are never recorded as real model AUC."""

import unittest
import numpy as np
import pandas as pd
from compare_predictions import TARGETS, UID, auc, compare, aligned_inputs


def frames():
    y = np.tile([0, 1], 30)
    labels = pd.DataFrame(np.tile(y[:, None], (1, 12)), columns=TARGETS)
    labels.insert(0, UID, [f'id-{i}' for i in range(len(y))])
    pred = labels.copy()
    pred[TARGETS] = .1 + .8 * labels[TARGETS]
    return labels, pred


class ComparisonTests(unittest.TestCase):
    def test_known_auc_and_ties(self):
        self.assertEqual(auc(np.array([0, 1]), np.array([.1, .9])), 1.)
        self.assertEqual(auc(np.array([0, 1]), np.array([.9, .1])), 0.)
        self.assertEqual(auc(np.array([0, 1]), np.array([.5, .5])), .5)
        self.assertEqual(auc(np.array([0, 1, np.nan]), np.array([.1, .9, .5])), 1.)

    def test_identical_predictions_exact_zero_paired_interval(self):
        labels, pred = frames()
        result = compare(labels, pred, pred.sample(frac=1, random_state=3), bootstrap=40)
        self.assertEqual(result['macro_delta'], 0.)
        self.assertEqual(result['macro_delta_ci95'], [0., 0.])
        self.assertFalse(result['point_estimate_gain_ge_0_01'])

    def test_known_improvement(self):
        labels, pred = frames()
        baseline = pred.copy()
        baseline[TARGETS] = .5
        result = compare(labels, baseline, pred, bootstrap=40)
        self.assertEqual(result['macro_delta'], .5)
        self.assertEqual(result['macro_delta_ci95'], [.5, .5])

    def test_partial_duplicate_soft_and_nonfinite_rejected(self):
        labels, pred = frames()
        for candidate in [pred.iloc[:-1], pd.concat([pred, pred.iloc[:1]])]:
            with self.assertRaises(ValueError):
                aligned_inputs(labels, pred, candidate)
        for value in [np.nan, 1.1, -.1]:
            bad = pred.copy()
            bad.loc[0, 'ACL'] = value
            with self.assertRaises(ValueError):
                aligned_inputs(labels, pred, bad)
        soft = labels.copy()
        soft[TARGETS] = .5
        with self.assertRaises(ValueError):
            aligned_inputs(soft, pred, pred)

    def test_single_class_target_not_silently_dropped(self):
        labels, pred = frames()
        labels['Fracture'] = 0
        with self.assertRaisesRegex(ValueError, 'Every target'):
            compare(labels, pred, pred, bootstrap=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
