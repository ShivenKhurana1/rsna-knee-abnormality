import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from compare_label_sources import compare
from transfer_audit import TARGETS, UID


def frame(ids, values):
    result = pd.DataFrame(values, columns=TARGETS)
    result.insert(0, UID, ids)
    return result


class CompareLabelSourcesTests(unittest.TestCase):
    def test_paired_comparison_detects_better_candidate(self):
        rng = np.random.default_rng(3)
        n = 120
        ids = [f's{i}' for i in range(n)]
        y = np.fromfunction(lambda i, j: (i + j) % 2, (n, len(TARGETS))).astype(float)
        baseline = rng.random(y.shape)
        candidate = y * 0.9 + (1 - y) * 0.1
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [root / name for name in ('gold.csv', 'base.csv', 'candidate.csv')]
            for path, values in zip(paths, (y, baseline, candidate)):
                frame(ids, values).to_csv(path, index=False)
            result = compare(*paths, bootstrap=200, seed=4)
        self.assertGreater(result['macro_delta'], 0)
        self.assertGreater(result['probability_better'], 0.95)
        self.assertEqual(set(result['targets']), set(TARGETS))

    def test_missing_gold_id_is_rejected(self):
        n = 20
        ids = [f's{i}' for i in range(n)]
        y = np.fromfunction(lambda i, j: (i + j) % 2, (n, len(TARGETS))).astype(float)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gold, base, candidate = [root / name for name in ('g.csv', 'b.csv', 'c.csv')]
            frame(ids, y).to_csv(gold, index=False)
            frame(ids, y).to_csv(base, index=False)
            frame(ids[:-1], y[:-1]).to_csv(candidate, index=False)
            with self.assertRaisesRegex(ValueError, 'missing'):
                compare(gold, base, candidate, bootstrap=100)


if __name__ == '__main__':
    unittest.main()
