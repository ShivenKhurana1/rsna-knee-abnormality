"""Regression coverage for AMP overflow without corrupting optimizer state."""
import unittest
import torch
from v16.runner import finite_optimizer_step


class StabilityTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for AMP overflow regression')
    def test_overflow_skips_update_and_next_attempt_recovers(self):
        model = torch.nn.Linear(2, 1).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
        original = [p.detach().clone() for p in model.parameters()]
        scaler.scale(model(torch.ones(1, 2, device='cuda')).sum()).backward()
        next(model.parameters()).grad.fill_(float('inf'))
        self.assertFalse(finite_optimizer_step(model, optimizer, scaler, 1.))
        self.assertEqual(scaler.get_scale(), 512.)
        self.assertFalse(optimizer.state)
        for p, saved in zip(model.parameters(), original):
            torch.testing.assert_close(p, saved)
        scaler.scale(model(torch.ones(1, 2, device='cuda')).sum()).backward()
        self.assertTrue(finite_optimizer_step(model, optimizer, scaler, 1.))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertTrue(any(not torch.equal(p, saved) for p, saved in zip(model.parameters(), original)))

    def test_full_precision_nonfinite_is_not_silently_ignored(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler('cpu', enabled=False)
        model(torch.ones(1, 2)).sum().backward()
        next(model.parameters()).grad.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            finite_optimizer_step(model, optimizer, scaler, 1.)
        self.assertFalse(optimizer.state)


if __name__ == '__main__':
    unittest.main()
