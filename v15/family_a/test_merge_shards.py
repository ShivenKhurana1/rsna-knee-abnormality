import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from contract import TARGETS, UID
from folds import assign_folds
from merge_shards import merge_partial_receipts


class MergeShardsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ids = [f'g{i}' for i in range(6)] + [f'p{i}' for i in range(9)]
        self.gold_ids = self.ids[:6]
        pd.DataFrame({UID: self.ids}).to_csv(self.root / 'ids.csv', index=False)
        pd.DataFrame({UID: self.gold_ids}).to_csv(self.root / 'gold.csv', index=False)
        self.fold_info = assign_folds(self.ids, k=3, seed=7, priority_ids=self.gold_ids)
        self.fold_hash = hashlib.sha256(json.dumps(
            self.fold_info['fold_assignment'], sort_keys=True).encode()).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def make_shard(self, fold, corrupt_ids=False):
        directory = self.root / f'shard{fold}'
        directory.mkdir()
        selected = [uid for uid in self.ids if self.fold_info['fold_assignment'][uid] == fold]
        if corrupt_ids:
            selected = selected[:-1]
        values = np.full((len(selected), len(TARGETS)), 0.2 + fold * 0.1)
        frame = pd.DataFrame(values, columns=TARGETS)
        frame.insert(0, UID, selected)
        prediction = directory / 'auxiliary_partial_oof.csv'
        frame.to_csv(prediction, index=False)
        receipt = {
            'status': 'COMPLETE_FOLD_SHARD_NOT_COMPLETE_ARM', 'arm': 'auxiliary',
            'seed': 1400, 'fold_seed': 7, 'k': 3, 'selected_folds': [fold],
            'epochs_requested': 2, 'head_lr': 1e-3, 'backbone_lr': 8e-6,
            'weight_decay': 0.02, 'batch_size': 2,
            'expert_fraction': 0.10,
            'policy_selection': 'cross-fitted on training-fold expert cases',
            'policy_bootstrap': 100, 'policy_seed': 1400,
            'fold_assignment_sha256': self.fold_hash,
            'partial_oof_path': str(prediction),
            'partial_oof_sha256': hashlib.sha256(prediction.read_bytes()).hexdigest(),
        }
        receipt_path = directory / 'auxiliary_partial_receipt.json'
        receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
        return receipt_path

    def test_merges_exact_disjoint_fold_coverage(self):
        receipts = [self.make_shard(fold) for fold in range(3)]
        merged, gold, result = merge_partial_receipts(
            receipts, self.root / 'ids.csv', self.root / 'gold.csv', self.root / 'merged')
        self.assertEqual(merged[UID].tolist(), self.ids)
        self.assertEqual(gold[UID].tolist(), self.gold_ids)
        self.assertEqual(result['selected_folds'], [0, 1, 2])

    def test_rejects_missing_fold(self):
        receipts = [self.make_shard(fold) for fold in range(2)]
        with self.assertRaisesRegex(ValueError, 'Incomplete folds'):
            merge_partial_receipts(receipts, self.root / 'ids.csv', self.root / 'gold.csv',
                                   self.root / 'merged')

    def test_rejects_wrong_study_coverage(self):
        receipts = [self.make_shard(0, corrupt_ids=True)] + [self.make_shard(f) for f in (1, 2)]
        with self.assertRaisesRegex(ValueError, 'coverage'):
            merge_partial_receipts(receipts, self.root / 'ids.csv', self.root / 'gold.csv',
                                   self.root / 'merged')


if __name__ == '__main__':
    unittest.main()
