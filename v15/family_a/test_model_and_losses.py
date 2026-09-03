import unittest

import torch

from contract import GROUP, IMG, N_SLOT, TARGETS
from losses import combined_loss, masked_bce
from model import GeneralistModel, TargetQueryHead, TinyCNNEncoder, build_smoke_model


class TargetQueryHeadTests(unittest.TestCase):
    def test_output_shape(self):
        head = TargetQueryHead(dim=32, n_slot=6, n_out=12, hidden=16)
        tokens = torch.randn(4, 6, 32)
        mask = torch.ones(4, 6)
        logits, att = head(tokens, mask)
        self.assertEqual(logits.shape, (4, 12))
        self.assertEqual(att.shape, (4, 12, 6))

    def test_masked_slots_get_zero_attention(self):
        head = TargetQueryHead(dim=32, n_slot=6, n_out=12, hidden=16)
        tokens = torch.randn(2, 6, 32)
        mask = torch.zeros(2, 6)
        mask[:, :3] = 1.0
        _, att = head(tokens, mask)
        self.assertTrue(torch.allclose(att[:, :, 3:], torch.zeros_like(att[:, :, 3:]), atol=1e-6))

    def test_rejects_all_missing_study(self):
        head = TargetQueryHead(dim=32, n_slot=6, n_out=12, hidden=16)
        tokens = torch.randn(1, 6, 32)
        mask = torch.zeros(1, 6)
        with self.assertRaises(ValueError):
            head(tokens, mask)

    def test_prior_only_applies_at_matching_dims(self):
        head_default = TargetQueryHead(dim=8, n_slot=N_SLOT, n_out=len(TARGETS), hidden=8)
        self.assertTrue(head_default.use_prior)
        head_other = TargetQueryHead(dim=8, n_slot=3, n_out=5, hidden=8)
        self.assertFalse(head_other.use_prior)


class TinyCNNEncoderTests(unittest.TestCase):
    def test_output_shape(self):
        enc = TinyCNNEncoder(out_dim=24)
        x = torch.rand(5, GROUP, IMG, IMG)
        out = enc(x)
        self.assertEqual(out.shape, (5, 24))


class GeneralistModelTests(unittest.TestCase):
    def test_forward_shape_and_gradient_flow(self):
        model = build_smoke_model(hidden=16, out_dim=16)
        imgs = torch.randint(0, 256, (3, N_SLOT, GROUP, IMG, IMG), dtype=torch.uint8)
        mask = torch.ones(3, N_SLOT)
        mask[0, -1] = 0  # one missing slot for one study
        logits, att = model(imgs, mask)
        self.assertEqual(logits.shape, (3, len(TARGETS)))
        loss = logits.sum()
        loss.backward()
        grad_norms = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
        self.assertTrue(any(g > 0 for g in grad_norms))

    def test_handles_float_and_uint8_input_identically(self):
        model = build_smoke_model(hidden=16, out_dim=16)
        model.eval()
        imgs_uint8 = torch.randint(0, 256, (2, N_SLOT, GROUP, IMG, IMG), dtype=torch.uint8)
        mask = torch.ones(2, N_SLOT)
        with torch.no_grad():
            a, _ = model(imgs_uint8, mask)
            b, _ = model(imgs_uint8.float(), mask)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))


class MaskedBceTests(unittest.TestCase):
    def test_zero_mask_returns_zero_not_nan(self):
        logits = torch.randn(4, 3)
        y = torch.zeros(4, 3)
        mask = torch.zeros(4, 3, dtype=torch.bool)
        loss = masked_bce(logits, y, mask)
        self.assertEqual(float(loss), 0.0)

    def test_matches_manual_bce_on_masked_subset(self):
        logits = torch.tensor([[2.0, -2.0], [0.0, 1.0]])
        y = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        mask = torch.tensor([[True, False], [False, True]])
        loss = masked_bce(logits, y, mask)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([2.0, 1.0]), torch.tensor([1.0, 0.0]))
        self.assertAlmostEqual(float(loss), float(expected), places=5)


class CombinedLossTests(unittest.TestCase):
    def test_zero_aux_weight_matches_expert_only_loss(self):
        logits = torch.randn(3, len(TARGETS))
        y_expert = torch.zeros(3, len(TARGETS))
        expert_mask = torch.ones(3, len(TARGETS), dtype=torch.bool)
        y_aux = torch.rand(3, len(TARGETS))
        aux_mask = torch.ones(3, len(TARGETS), dtype=torch.bool)
        aux_weight = torch.zeros(len(TARGETS))
        total, parts = combined_loss(logits, y_expert, expert_mask, y_aux, aux_mask, aux_weight)
        expert_only = masked_bce(logits, y_expert, expert_mask)
        self.assertAlmostEqual(float(total), float(expert_only), places=5)
        self.assertEqual(parts['aux'], 0.0)

    def test_nonzero_weight_for_rejected_target_contributes_nothing(self):
        logits = torch.randn(3, len(TARGETS))
        y_expert = torch.zeros(3, len(TARGETS))
        expert_mask = torch.zeros(3, len(TARGETS), dtype=torch.bool)
        y_aux = torch.rand(3, len(TARGETS))
        aux_mask = torch.zeros(3, len(TARGETS), dtype=torch.bool)
        aux_mask[:, 0] = True  # only target 0 addressed
        aux_weight = torch.zeros(len(TARGETS))  # but target 0's weight is 0 (rejected by policy)
        total, parts = combined_loss(logits, y_expert, expert_mask, y_aux, aux_mask, aux_weight)
        self.assertEqual(float(total), 0.0)


if __name__ == '__main__':
    unittest.main()
