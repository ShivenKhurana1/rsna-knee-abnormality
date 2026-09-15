import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from contract import TARGETS, UID
from folds import assign_folds
from labels import (MAX_AUX_WEIGHT, build_supervision, crossfit_transfer_policy,
                    load_transfer_policy, zero_policy)


def fake_transfer_report(overrides):
    """overrides: {target: (ci_lower, skill, small_sample)}"""
    targets = {}
    for t in TARGETS:
        ci_lower, skill, small_sample = overrides.get(t, (None, None, False))
        targets[t] = {
            'auc_addressed_only': {'ci95': [ci_lower, 1.0] if ci_lower is not None else None},
            'brier_addressed_only': {'skill_score': skill},
            'small_sample_warning': small_sample,
        }
    return {'targets': targets}


class LoadTransferPolicyTests(unittest.TestCase):
    def test_passes_when_ci_and_skill_both_clear_gate(self):
        report = fake_transfer_report({'ACL': (0.9, 0.2, False)})
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(report, f)
            path = f.name
        policy = load_transfer_policy(path)
        self.assertTrue(policy['ACL']['use_aux'])
        self.assertAlmostEqual(policy['ACL']['weight'], 0.2)

    def test_rejects_good_auc_with_negative_skill(self):
        report = fake_transfer_report({'Synovitis': (0.7, -0.03, False)})
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(report, f)
            path = f.name
        policy = load_transfer_policy(path)
        self.assertFalse(policy['Synovitis']['use_aux'])
        self.assertEqual(policy['Synovitis']['weight'], 0.0)

    def test_rejects_ci_lower_at_or_below_half(self):
        report = fake_transfer_report({'Fracture': (0.5, 0.3, False)})
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(report, f)
            path = f.name
        policy = load_transfer_policy(path)
        self.assertFalse(policy['Fracture']['use_aux'])

    def test_caps_weight_at_max(self):
        report = fake_transfer_report({'ACL': (0.9, 0.9, False)})
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(report, f)
            path = f.name
        policy = load_transfer_policy(path)
        self.assertAlmostEqual(policy['ACL']['weight'], MAX_AUX_WEIGHT)

    def test_small_sample_halves_weight(self):
        report = fake_transfer_report({'MCL': (0.9, 0.2, True)})
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(report, f)
            path = f.name
        policy = load_transfer_policy(path)
        self.assertAlmostEqual(policy['MCL']['weight'], 0.1)

    def test_missing_evidence_is_excluded(self):
        report = fake_transfer_report({})
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(report, f)
            path = f.name
        policy = load_transfer_policy(path)
        for t in TARGETS:
            self.assertFalse(policy[t]['use_aux'])


class ZeroPolicyTests(unittest.TestCase):
    def test_every_target_disabled(self):
        policy = zero_policy()
        self.assertTrue(all(not v['use_aux'] and v['weight'] == 0.0 for v in policy.values()))


class CrossfitPolicyTests(unittest.TestCase):
    def test_held_out_gold_cannot_change_training_fold_policy(self):
        ids = [f'g{i:02d}' for i in range(24)] + ['pool']
        values = np.fromfunction(lambda i, j: (i + j) % 2, (24, len(TARGETS)))
        gold_a = pd.DataFrame(values, columns=TARGETS)
        gold_a.insert(0, UID, ids[:24])
        report_values = values * 0.8 + 0.1
        report = pd.DataFrame(np.vstack([report_values, np.full((1, len(TARGETS)), 0.9)]),
                              columns=TARGETS)
        report.insert(0, UID, ids)
        train_idx = np.arange(20)
        policy_a, _ = crossfit_transfer_policy(ids, train_idx, gold_a, report,
                                               bootstrap=100, seed=7)

        gold_b = gold_a.copy()
        gold_b.loc[20:, TARGETS] = 1 - gold_b.loc[20:, TARGETS]
        policy_b, _ = crossfit_transfer_policy(ids, train_idx, gold_b, report,
                                               bootstrap=100, seed=7)
        self.assertEqual(policy_a, policy_b)
        self.assertTrue(all(item['use_aux'] for item in policy_a.values()))


class BuildSupervisionTests(unittest.TestCase):
    def _tables(self):
        ids = ['g1', 'g2', 'p1']
        gold = pd.DataFrame({UID: ['g1', 'g2'],
                             **{t: [1.0, 0.0] for t in TARGETS}})
        report = pd.DataFrame({UID: ids,
                               **{t: [0.9, 0.5, 0.9] for t in TARGETS}})
        return ids, gold, report

    def test_silence_is_masked_not_filled_negative(self):
        ids, gold, report = self._tables()
        policy = {t: {'use_aux': True, 'weight': 0.3} for t in TARGETS}
        y_expert, expert_mask, y_aux, aux_mask, aux_weight = build_supervision(ids, gold, report, policy)
        # g2's report value is 0.5 (silent) -> must be masked out, not treated as y_aux=0
        self.assertFalse(aux_mask[1, 0])
        self.assertTrue(np.isnan(y_aux[1, 0]))
        # p1 has no expert label and report value 0.9 -> included
        self.assertTrue(aux_mask[2, 0])
        self.assertEqual(y_aux[2, 0], 0.9)

    def test_expert_mask_only_true_for_gold_studies(self):
        ids, gold, report = self._tables()
        policy = zero_policy()
        y_expert, expert_mask, y_aux, aux_mask, aux_weight = build_supervision(ids, gold, report, policy)
        self.assertTrue(expert_mask[0].all())
        self.assertTrue(expert_mask[1].all())
        self.assertFalse(expert_mask[2].any())  # p1 has no gold label

    def test_expert_cells_override_report_auxiliary_supervision(self):
        ids, gold, report = self._tables()
        policy = {t: {'use_aux': True, 'weight': 0.3} for t in TARGETS}
        _, expert_mask, y_aux, aux_mask, _ = build_supervision(ids, gold, report, policy)
        self.assertTrue(expert_mask[:2].all())
        self.assertFalse(aux_mask[:2].any())
        self.assertTrue(np.isnan(y_aux[:2]).all())

    def test_baseline_policy_masks_out_every_aux_cell(self):
        ids, gold, report = self._tables()
        policy = zero_policy()
        _, _, y_aux, aux_mask, aux_weight = build_supervision(ids, gold, report, policy)
        self.assertFalse(aux_mask.any())
        self.assertTrue((aux_weight == 0).all())

    def test_rejects_studies_missing_from_report_source(self):
        ids, gold, report = self._tables()
        report = report.iloc[:2]
        with self.assertRaises(ValueError):
            build_supervision(ids, gold, report, zero_policy())


class AssignFoldsTests(unittest.TestCase):
    def test_every_study_assigned_exactly_once(self):
        ids = [f's{i}' for i in range(20)]
        info = assign_folds(ids, k=4, seed=1)
        self.assertEqual(set(info['fold_assignment']), set(ids))
        counts = np.bincount(list(info['fold_assignment'].values()), minlength=4)
        self.assertTrue((counts >= 4).all())

    def test_deterministic_given_seed(self):
        ids = [f's{i}' for i in range(20)]
        a = assign_folds(ids, k=4, seed=7)['fold_assignment']
        b = assign_folds(ids, k=4, seed=7)['fold_assignment']
        self.assertEqual(a, b)

    def test_priority_subset_is_balanced_across_folds(self):
        ids = [f'g{i}' for i in range(58)] + [f'p{i}' for i in range(137)]
        priority = ids[:58]
        info = assign_folds(ids, k=5, seed=7, priority_ids=priority)
        gold_counts = np.bincount([info['fold_assignment'][uid] for uid in priority], minlength=5)
        all_counts = np.bincount(list(info['fold_assignment'].values()), minlength=5)
        self.assertLessEqual(gold_counts.max() - gold_counts.min(), 1)
        self.assertLessEqual(all_counts.max() - all_counts.min(), 1)
        self.assertEqual(info['priority_studies_balanced'], 58)

    def test_rejects_unknown_priority_id(self):
        with self.assertRaises(ValueError):
            assign_folds(['a', 'b', 'c'], k=2, priority_ids=['missing'])

    def test_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            assign_folds(['a', 'a', 'b', 'c'], k=2)

    def test_rejects_more_folds_than_studies(self):
        with self.assertRaises(ValueError):
            assign_folds(['a', 'b'], k=5)

    def test_carries_patient_grouping_caveat(self):
        info = assign_folds(['a', 'b', 'c', 'd'], k=2)
        self.assertIn('singleton', info['grouping_caveat'])


if __name__ == '__main__':
    unittest.main()
