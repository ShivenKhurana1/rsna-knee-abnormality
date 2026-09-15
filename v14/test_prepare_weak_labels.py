import unittest

import numpy as np
import pandas as pd

from diagnostics import TARGETS, UID
from prepare_weak_labels import threshold, report


def frame(values, ids=None):
    f = pd.DataFrame(values, columns=TARGETS)
    f.insert(0, UID, ids or [f'{i:05d}' for i in range(len(f))])
    return f


class ThresholdTests(unittest.TestCase):
    def test_masks_uncertain_and_drops_gold(self):
        raw = frame([[0.05] * 12, [0.95] * 12, [0.5] * 12, [0.2] * 12])
        pseudo = threshold(raw, gold_ids={'00001'}, lo=0.1, hi=0.9)
        self.assertEqual(pseudo[UID].tolist(), ['00000', '00002', '00003'])
        self.assertTrue((pseudo.loc[0, TARGETS] == 0).all())
        self.assertTrue(pseudo.loc[1, TARGETS].isna().all())
        self.assertTrue(pseudo.loc[2, TARGETS].isna().all())

    def test_report_counts_exclude_masked(self):
        raw = frame([[0.05, 0.95] + [0.5] * 10, [0.02, 0.98] + [0.5] * 10])
        pseudo = threshold(raw, gold_ids=set())
        counts = report(pseudo)
        self.assertEqual(counts['ACL'], {'positive': 0, 'negative': 2, 'masked': 0})
        self.assertEqual(counts['MCL'], {'positive': 2, 'negative': 0, 'masked': 0})
        self.assertEqual(counts['Medial Meniscus'], {'positive': 0, 'negative': 0, 'masked': 2})

    def test_rejects_bad_thresholds_and_bad_probabilities(self):
        raw = frame([[0.5] * 12])
        with self.assertRaises(ValueError):
            threshold(raw, set(), lo=0.6, hi=0.9)
        bad = frame([[1.5] * 12])
        with self.assertRaises(ValueError):
            threshold(bad, set())

    def test_rejects_duplicate_ids_and_missing_targets(self):
        raw = frame([[0.5] * 12, [0.5] * 12], ids=['a', 'a'])
        with self.assertRaises(ValueError):
            threshold(raw, set())
        missing = frame([[0.5] * 12]).drop(columns=['ACL'])
        with self.assertRaises(ValueError):
            threshold(missing, set())


if __name__ == '__main__':
    unittest.main()
