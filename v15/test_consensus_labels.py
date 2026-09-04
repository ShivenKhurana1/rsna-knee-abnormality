import unittest

import numpy as np
import pandas as pd

from consensus_labels import build_consensus
from transfer_audit import TARGETS, UID


class ConsensusLabelTests(unittest.TestCase):
    def frames(self):
        ids = ['a', 'b', 'c', 'd']
        pil = pd.DataFrame({UID: ids})
        steven = pd.DataFrame({UID: ids})
        lixin = pd.DataFrame({UID: ids})
        for target in TARGETS:
            pil[target + '__verdict'] = ['YES', 'NO', 'UNK', 'YES']
            steven[target] = [0.94, 0.08, 0.5, 0.94]
            lixin[target] = [1, 0, 1, 0]
        return pil, steven, lixin

    def test_only_three_way_addressed_agreement_becomes_label(self):
        consensus, summary = build_consensus(*self.frames())
        self.assertEqual(consensus[TARGETS[0]].tolist(), [1.0, 0.0, 0.5, 0.5])
        self.assertEqual(summary[TARGETS[0]]['unanimous_addressed'], 2)
        self.assertEqual(summary[TARGETS[0]]['unknown_or_disagreed'], 2)

    def test_universe_preserves_missing_id_as_unknown(self):
        pil, steven, lixin = self.frames()
        consensus, _ = build_consensus(pil, steven, lixin,
                                       universe_ids=['missing', 'a'])
        self.assertEqual(consensus[UID].tolist(), ['missing', 'a'])
        self.assertTrue((consensus.loc[0, TARGETS].astype(float) == 0.5).all())

    def test_duplicate_source_ids_are_rejected(self):
        pil, steven, lixin = self.frames()
        pil = pd.concat([pil, pil.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            build_consensus(pil, steven, lixin)

    def test_out_of_range_lixin_value_is_rejected(self):
        pil, steven, lixin = self.frames()
        lixin[TARGETS[0]] = lixin[TARGETS[0]].astype(float)
        lixin.loc[0, TARGETS[0]] = 1.7
        with self.assertRaises(ValueError):
            build_consensus(pil, steven, lixin)


if __name__ == '__main__':
    unittest.main()
