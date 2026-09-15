"""CPU end-to-end smoke test: synthetic cache + labels, no GPU, no competition
data, no external weights. Exercises the full Run 4 path -- baseline arm,
auxiliary arm, checkpoint resume, OOF export, compare_oof, and the Family C
reproducibility gate -- so a shape/contract/masking bug is caught before it
ever reaches the GPU device.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import cache
from compare_oof import compare as compare_oof
from contract import TARGETS, UID
from engine import (_epoch_training_indices, build_optimizer, resolve_device,
                    run_arm, train_one_fold)
from family_c_gate import evaluate_gate
from labels import zero_policy
from model import build_smoke_model


def make_cohort(n_gold=9, n_pool=6, seed=0):
    rng = np.random.default_rng(seed)
    gold_ids = [f'gold-{i:02d}' for i in range(n_gold)]
    pool_ids = [f'pool-{i:02d}' for i in range(n_pool)]
    ids = gold_ids + pool_ids

    # Deterministic alternating pattern guarantees both classes present per
    # target regardless of n_gold, avoiding a flaky degenerate-AUC test target.
    gold_values = np.fromfunction(lambda i, j: (i + j) % 2, (n_gold, len(TARGETS)))
    gold = pd.DataFrame(gold_values, columns=TARGETS)
    gold.insert(0, UID, gold_ids)

    report_values = np.clip(gold_values * 0.8 + 0.1 + rng.normal(0, 0.05, gold_values.shape), 0, 1)
    silent = rng.random(gold_values.shape) < 0.2
    report_values[silent] = 0.5
    pool_values = np.clip(rng.random((n_pool, len(TARGETS))), 0, 1)
    pool_silent = rng.random(pool_values.shape) < 0.2
    pool_values[pool_silent] = 0.5
    report = pd.DataFrame(np.vstack([report_values, pool_values]), columns=TARGETS)
    report.insert(0, UID, ids)
    return ids, gold_ids, gold, report


def passing_policy(weight=0.2):
    return {t: {'use_aux': True, 'weight': weight, 'reason': 'test'} for t in TARGETS}


class RunArmSmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ids, self.gold_ids, self.gold, self.report = make_cohort()
        self.cache_dir = self.tmp / 'cache'
        cache.synth(self.cache_dir, self.ids, seed=1)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_baseline_and_auxiliary_produce_full_oof_coverage(self):
        for arm, policy in (('baseline', zero_policy()), ('auxiliary', passing_policy())):
            out_dir = self.tmp / arm
            frame, receipt = run_arm(arm, self.cache_dir, self.ids, self.gold_ids, self.gold,
                                     self.report, policy, out_dir, build_smoke_model,
                                     k=3, epochs=2, batch_size=4, seed=1400,
                                     crossfit_policy=(arm == 'auxiliary'), policy_bootstrap=50)
            self.assertEqual(set(frame[UID]), set(self.gold_ids))
            self.assertFalse(frame[TARGETS].isna().any().any())
            self.assertEqual(receipt['n_studies_scored'], len(self.gold_ids))
            self.assertTrue((out_dir / f'{arm}_oof.csv').is_file())
            all_oof = pd.read_csv(out_dir / f'{arm}_all_oof.csv')
            self.assertEqual(set(all_oof[UID]), set(self.ids))
            self.assertFalse(all_oof[TARGETS].isna().any().any())
            self.assertTrue((out_dir / f'{arm}_receipt.json').is_file())
            if arm == 'auxiliary':
                self.assertEqual(receipt['policy_selection'],
                                 'cross-fitted on training-fold expert cases')
                self.assertEqual(set(receipt['fold_policies']), {'0', '1', '2'})

    def test_baseline_receipt_records_zero_aux_loss(self):
        out_dir = self.tmp / 'baseline'
        _, receipt = run_arm('baseline', self.cache_dir, self.ids, self.gold_ids, self.gold,
                             self.report, zero_policy(), out_dir, build_smoke_model,
                             k=3, epochs=1, batch_size=4, seed=1)
        for fold_history in receipt['fold_histories'].values():
            for epoch in fold_history:
                self.assertEqual(epoch['mean_aux_loss'], 0.0)

    def test_refuses_second_write_to_same_output(self):
        out_dir = self.tmp / 'baseline'
        run_arm('baseline', self.cache_dir, self.ids, self.gold_ids, self.gold, self.report,
                zero_policy(), out_dir, build_smoke_model, k=3, epochs=1, batch_size=4, seed=1)
        with self.assertRaises(FileExistsError):
            run_arm('baseline', self.cache_dir, self.ids, self.gold_ids, self.gold, self.report,
                    zero_policy(), out_dir, build_smoke_model, k=3, epochs=1, batch_size=4, seed=1,
                    resume=False)

    def test_explicit_cpu_device_is_respected(self):
        self.assertEqual(resolve_device('cpu').type, 'cpu')

    def test_single_fold_shard_writes_partial_receipt_not_fake_full_oof(self):
        out_dir = self.tmp / 'shard'
        frame, receipt = run_arm(
            'auxiliary', self.cache_dir, self.ids, self.gold_ids, self.gold,
            self.report, passing_policy(), out_dir, build_smoke_model,
            k=3, epochs=1, batch_size=4, seed=2, folds=[1])
        fold_info = __import__('folds').assign_folds(
            self.ids, k=3, seed=1400, priority_ids=self.gold_ids)
        expected = {uid for uid in self.ids if fold_info['fold_assignment'][uid] == 1}
        self.assertEqual(set(frame[UID]), expected)
        self.assertEqual(receipt['status'], 'COMPLETE_FOLD_SHARD_NOT_COMPLETE_ARM')
        self.assertTrue((out_dir / 'auxiliary_partial_oof.csv').is_file())
        self.assertFalse((out_dir / 'auxiliary_all_oof.csv').exists())

    def test_encoder_uses_discriminative_learning_rate(self):
        model = build_smoke_model()
        optimizer = build_optimizer(model, head_lr=1e-3, backbone_lr=8e-6)
        rates = {group['group_name']: group['lr'] for group in optimizer.param_groups}
        self.assertEqual(rates, {'head': 1e-3, 'encoder': 8e-6})
        parameter_ids = [id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))

    def test_auxiliary_epoch_keeps_all_pool_rows_and_anchors_experts(self):
        train_idx = np.arange(104)
        expert_mask = np.zeros((104, len(TARGETS)), dtype=bool)
        expert_mask[:4] = True
        aux_mask = np.zeros_like(expert_mask)
        aux_mask[4:] = True
        indices = _epoch_training_indices(
            train_idx, expert_mask, aux_mask, np.ones(len(TARGETS)), 0.10,
            np.random.default_rng(9))
        self.assertEqual(set(indices[~expert_mask[indices].any(axis=1)]), set(range(4, 104)))
        observed = expert_mask[indices].any(axis=1).mean()
        self.assertGreaterEqual(observed, 0.10)
        self.assertLess(observed, 0.12)


class CheckpointResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ids, self.gold_ids, self.gold, self.report = make_cohort(n_gold=9, n_pool=3)
        self.cache_dir = self.tmp / 'cache'
        cache.synth(self.cache_dir, self.ids, seed=2)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_resume_continues_from_saved_epoch_not_from_scratch(self):
        from labels import build_supervision
        y_expert, expert_mask, y_aux, aux_mask, aux_weight = build_supervision(
            self.ids, self.gold, self.report, zero_policy())
        imgs, masks = cache.load_batch(self.cache_dir, self.ids)
        train_idx, val_idx = list(range(8)), list(range(8, 12))
        ckpt = self.tmp / 'fold0.pt'

        probs1, hist1, *_ = train_one_fold(build_smoke_model, imgs, masks, y_expert, expert_mask,
                                           y_aux, aux_mask, aux_weight, train_idx, val_idx,
                                           epochs=1, lr=1e-3, seed=5, checkpoint_path=ckpt,
                                           batch_size=4, resume=True)
        self.assertEqual([h['epoch'] for h in hist1], [0])

        probs2, hist2, *_ = train_one_fold(build_smoke_model, imgs, masks, y_expert, expert_mask,
                                           y_aux, aux_mask, aux_weight, train_idx, val_idx,
                                           epochs=3, lr=1e-3, seed=5, checkpoint_path=ckpt,
                                           batch_size=4, resume=True)
        # The resumed receipt history remains complete, not just the new tail.
        self.assertEqual([h['epoch'] for h in hist2], [0, 1, 2])

    def test_outer_fold_early_stopping_is_rejected(self):
        from labels import build_supervision
        y_expert, expert_mask, y_aux, aux_mask, aux_weight = build_supervision(
            self.ids, self.gold, self.report, zero_policy())
        imgs, masks = cache.load_batch(self.cache_dir, self.ids)
        with self.assertRaises(ValueError):
            train_one_fold(build_smoke_model, imgs, masks, y_expert, expert_mask,
                           y_aux, aux_mask, aux_weight, list(range(8)), list(range(8, 12)),
                           epochs=2, lr=1e-3, seed=5, checkpoint_path=self.tmp / 'bad.pt',
                           batch_size=4, patience=1)

    def test_resume_rejects_changed_optimizer_configuration(self):
        from labels import build_supervision
        y_expert, expert_mask, y_aux, aux_mask, aux_weight = build_supervision(
            self.ids, self.gold, self.report, zero_policy())
        imgs, masks = cache.load_batch(self.cache_dir, self.ids)
        args = (build_smoke_model, imgs, masks, y_expert, expert_mask,
                y_aux, aux_mask, aux_weight, list(range(8)), list(range(8, 12)))
        ckpt = self.tmp / 'config.pt'
        train_one_fold(*args, epochs=1, lr=1e-3, seed=5, checkpoint_path=ckpt,
                       batch_size=4, resume=True)
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            train_one_fold(*args, epochs=2, lr=2e-3, seed=5, checkpoint_path=ckpt,
                           batch_size=4, resume=True)

    def test_without_resume_starts_over(self):
        from labels import build_supervision
        y_expert, expert_mask, y_aux, aux_mask, aux_weight = build_supervision(
            self.ids, self.gold, self.report, zero_policy())
        imgs, masks = cache.load_batch(self.cache_dir, self.ids)
        train_idx, val_idx = list(range(8)), list(range(8, 12))
        ckpt = self.tmp / 'fold0.pt'
        train_one_fold(build_smoke_model, imgs, masks, y_expert, expert_mask, y_aux, aux_mask,
                       aux_weight, train_idx, val_idx, epochs=1, lr=1e-3, seed=5,
                       checkpoint_path=ckpt, batch_size=4, resume=True)
        _, hist, *_ = train_one_fold(build_smoke_model, imgs, masks, y_expert, expert_mask, y_aux,
                                     aux_mask, aux_weight, train_idx, val_idx, epochs=2, lr=1e-3,
                                     seed=5, checkpoint_path=ckpt, batch_size=4, resume=False)
        self.assertEqual([h['epoch'] for h in hist], [0, 1])


class CompareAndGateSmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ids, self.gold_ids, self.gold, self.report = make_cohort(n_gold=12, n_pool=4)
        self.cache_dir = self.tmp / 'cache'
        cache.synth(self.cache_dir, self.ids, seed=3)
        labels_path = self.tmp / 'gold_labels.csv'
        self.gold.to_csv(labels_path, index=False)
        self.labels_path = labels_path

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_seed(self, seed):
        base_dir = self.tmp / f'seed{seed}_baseline'
        aux_dir = self.tmp / f'seed{seed}_auxiliary'
        base_frame, _ = run_arm('baseline', self.cache_dir, self.ids, self.gold_ids, self.gold,
                                self.report, zero_policy(), base_dir, build_smoke_model,
                                k=3, epochs=2, batch_size=4, seed=seed)
        aux_frame, _ = run_arm('auxiliary', self.cache_dir, self.ids, self.gold_ids, self.gold,
                               self.report, passing_policy(), aux_dir, build_smoke_model,
                               k=3, epochs=2, batch_size=4, seed=seed)
        base_path, aux_path = base_dir / 'baseline_oof.csv', aux_dir / 'auxiliary_oof.csv'
        return compare_oof(self.labels_path, base_path, aux_path, bootstrap=200, seed=seed)

    def test_compare_oof_runs_and_reports_all_targets(self):
        report = self._run_seed(seed=10)
        self.assertEqual(set(report['targets']), set(TARGETS))
        self.assertIn('macro_delta', report)
        self.assertIn('macro_delta_ci95', report)

    def test_family_c_gate_accepts_two_seed_reports(self):
        r1 = self._run_seed(seed=11)
        r2 = self._run_seed(seed=12)
        gate = evaluate_gate([r1, r2], min_macro_gain=-1.0)  # threshold irrelevant to shape check
        self.assertIn(gate['decision'], ('PASS_PROCEED_TO_FAMILY_C', 'FAIL_DO_NOT_START_FAMILY_C'))
        self.assertEqual(gate['targets_total'], len(TARGETS))

    def test_gate_rejects_single_report(self):
        r1 = self._run_seed(seed=13)
        with self.assertRaises(ValueError):
            evaluate_gate([r1])


if __name__ == '__main__':
    unittest.main()
