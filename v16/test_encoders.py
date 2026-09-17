"""Offline loader/gradient tests using small random checkpoint fixtures."""
import tempfile
from pathlib import Path
import unittest

import torch

from .common import config
from .model import KneeModel
from .data import collate


class EncoderTests(unittest.TestCase):
    def test_hf_load_rebuild_and_gradients(self):
        from transformers import Dinov2Config, Dinov2Model
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            c = Dinov2Config(hidden_size=24, num_hidden_layers=2, num_attention_heads=4,
                             image_size=16, patch_size=4)
            Dinov2Model(c).save_pretrained(tmp)
            cfg = config(Path(__file__).with_name('config.json'))
            cfg.update(encoder='dinov2', encoder_path=tmp, image_size=16, hidden=4,
                        encoder_chunk=2, gradient_checkpointing=True)
            first = KneeModel(cfg, 3)
            self.assertTrue(any(p.requires_grad for p in first.encoder.parameters()))
            batch = collate([{'uid': 'one', 'series': [{'x': torch.rand(3, 3, 16, 16),
                              'position': torch.tensor([0., .5, 1.]), 'meta': torch.zeros(6)}]}])
            first.train()
            first(batch)[0].sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in first.parameters() if p.grad is not None))
            first.eval()
            cfg['encoder_path'] = '/unavailable-at-inference'
            second = KneeModel(cfg, 3, initialize=False, encoder_spec=first.encoder.spec).eval()
            second.load_state_dict(first.state_dict(), strict=True)
            torch.testing.assert_close(first(batch)[0], second(batch)[0])

    def test_cnn_single_window_training(self):
        import timm
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cnn.pt'
            model = timm.create_model('resnet18', pretrained=False, num_classes=0)
            torch.save(model.state_dict(), path)
            cfg = config(Path(__file__).with_name('config.json'))
            cfg.update(encoder='cnn', cnn_arch='resnet18', encoder_path=str(path),
                        image_size=32, hidden=4, encoder_chunk=1, gradient_checkpointing=True)
            candidate = KneeModel(cfg, 3).train()
            batch = collate([{'uid': 'one', 'series': [{'x': torch.rand(1, 3, 32, 32),
                              'position': torch.tensor([0.]), 'meta': torch.zeros(6)}]}])
            result = candidate(batch)[0]
            result.sum().backward()
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue(any(p.grad is not None for p in candidate.encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
