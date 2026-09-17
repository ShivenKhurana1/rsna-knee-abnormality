"""Offline pretrained encoders with within-series GRU and target attention."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, cfg, initialize=True, spec=None):
        super().__init__()
        self.kind = cfg['encoder']
        self.spec = spec or {}
        channels = len(cfg['offsets'])
        if self.kind == 'tiny':
            self.net = nn.Sequential(nn.Conv2d(channels, 8, 3, padding=1), nn.GELU(),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.out_dim, self.spec = 8, {'api': 'tiny'}
            return
        if channels != 3:
            raise ValueError('These pretrained backbones require three configured context channels')
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        path = Path(cfg['encoder_path'])
        if self.kind == 'dinov2':
            api = self.spec.get('api') or ('hf' if (path / 'config.json').is_file() else 'meta')
            if api == 'hf':
                from transformers import Dinov2Config, Dinov2Model
                if initialize:
                    self.net = Dinov2Model.from_pretrained(str(path), local_files_only=True)
                else:
                    self.net = Dinov2Model(Dinov2Config.from_dict(self.spec['config']))
                blocks, norm = self.net.encoder.layer, self.net.layernorm
                self.out_dim = self.net.config.hidden_size
                self.spec = {'api': 'hf', 'config': self.net.config.to_dict()}
                patch = self.net.config.patch_size
            else:
                repo = Path(cfg['dinov2_repo'])
                if not (repo / 'hubconf.py').is_file():
                    raise ValueError('Native Meta .pth needs dinov2_repo pointing to an attached local DINOv2 code repository')
                self.net = torch.hub.load(str(repo), 'dinov2_vits14', source='local', pretrained=False)
                if initialize:
                    if not path.is_file():
                        raise ValueError('encoder_path must be the native DINOv2 Small .pth file')
                    state = torch.load(path, map_location='cpu', weights_only=True)
                    self.net.load_state_dict(state, strict=True)
                blocks, norm = self.net.blocks, self.net.norm
                self.out_dim = self.net.embed_dim
                self.spec = {'api': 'meta', 'variant': 'dinov2_vits14'}
                patch = self.net.patch_size
            if cfg['image_size'] % patch:
                raise ValueError(f'image_size must be divisible by DINOv2 patch size {patch}')
            for p in self.net.parameters():
                p.requires_grad = False
            n = cfg['unfreeze_last']
            if n < 0 or n > len(blocks):
                raise ValueError('unfreeze_last exceeds encoder depth')
            if n:
                for block in blocks[-n:]:
                    for p in block.parameters():
                        p.requires_grad = True
                for p in norm.parameters():
                    p.requires_grad = True
            # Checkpoint chunks at the wrapper boundary below, also for native Meta.
        else:
            import timm
            self.net = timm.create_model(cfg['cnn_arch'], pretrained=False, num_classes=0, global_pool='avg')
            if initialize:
                if not path.is_file():
                    raise ValueError('CNN training requires an attached compatible encoder state dictionary')
                state = torch.load(path, map_location='cpu', weights_only=True)
                # A backbone-only checkpoint is required; silent key dropping is prohibited.
                self.net.load_state_dict(state, strict=True)
            self.out_dim = self.net.num_features
            self.spec = {'api': 'timm', 'arch': cfg['cnn_arch']}

    def forward(self, x):
        if self.kind == 'tiny':
            return self.net(x)
        x = (x-self.mean)/self.std
        if self.spec['api'] == 'hf':
            return self.net(pixel_values=x).last_hidden_state[:, 0]
        if self.spec['api'] == 'meta':
            return self.net.forward_features(x)['x_norm_clstoken']
        return self.net(x)


class KneeModel(nn.Module):
    def __init__(self, cfg, n_targets, initialize=True, encoder_spec=None):
        super().__init__()
        self.encoder = Encoder(cfg, initialize, encoder_spec)
        self.cfg = cfg
        dim, hidden = self.encoder.out_dim, cfg['hidden']
        self.metadata = nn.Linear(7, dim)  # patient-axis plane(3), fluid/fat(2), missing geometry, position
        self.gru = nn.GRU(dim, hidden, bidirectional=True, batch_first=True)
        self.query = nn.Parameter(torch.randn(n_targets, hidden*2)*.02)
        self.weight = nn.Parameter(torch.randn(n_targets, hidden*2)*.02)
        self.bias = nn.Parameter(torch.zeros(n_targets))

    def train(self, mode=True):
        super().train(mode)
        # Preserve pretrained BN statistics with study/microbatches as small as one.
        # ConvNeXt and DINO use layer normalization, but alternate CNNs may use BN.
        if mode:
            for module in self.encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, batch):
        n = len(batch['ids'])
        if batch['x'] is None:
            return self.bias.expand(n, -1), torch.zeros(n, dtype=torch.bool, device=self.bias.device)
        x, lengths = batch['x'], batch['lengths']
        valid = torch.arange(x.shape[1], device=x.device)[None] < lengths.to(x.device)[:, None]
        chunks = []
        for images in x[valid].split(self.cfg['encoder_chunk']):
            if self.training and self.cfg['gradient_checkpointing'] and any(p.requires_grad for p in self.encoder.parameters()):
                from torch.utils.checkpoint import checkpoint
                chunks.append(checkpoint(self.encoder, images, use_reentrant=False))
            else:
                chunks.append(self.encoder(images))
        features = torch.cat(chunks)
        # T4 FP16 has limited exponent range. Keep recurrence and finding
        # attention in FP32 while retaining mixed precision for the encoder.
        with torch.autocast(x.device.type, enabled=False):
            return self._head(batch, features.float(), valid, lengths, n)

    def _head(self, batch, features, valid, lengths, n):
        padded = features.new_zeros(*valid.shape, features.shape[-1])
        padded[valid] = features
        meta = torch.cat([batch['metadata'][:, None].expand(-1, valid.shape[1], -1),
                          batch['position'][..., None]], dim=-1)
        padded = padded + self.metadata(meta).to(padded.dtype)
        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        contextual, _ = self.gru(packed)
        contextual, _ = pad_packed_sequence(contextual, batch_first=True)
        outputs, available = [], []
        for owner in range(n):
            tokens = contextual[(batch['owners'] == owner)[:, None] & valid]
            if not len(tokens):
                outputs.append(self.bias)
                available.append(False)
            else:
                att = (self.query @ tokens.T / tokens.shape[-1]**.5).softmax(-1)
                outputs.append(((att @ tokens)*self.weight).sum(-1)+self.bias)
                available.append(True)
        return torch.stack(outputs), torch.tensor(available, device=features.device)


def masked_loss(logits, y, mask, cfg, auxiliary=False, weights=None):
    z = logits.float()
    mask = mask.bool() & torch.isfinite(y)
    safe = torch.where(mask, y.float(), torch.zeros_like(z))
    loss = F.binary_cross_entropy_with_logits(z, safe, reduction='none')
    if not auxiliary and cfg['expert_loss'] == 'asl':
        p = z.sigmoid()
        loss = -safe*(1-p).pow(cfg['gamma_pos'])*F.logsigmoid(z)
        loss -= (1-safe)*p.pow(cfg['gamma_neg'])*F.logsigmoid(-z)
    if weights is not None:
        loss = loss * weights
    return (loss*mask).sum()/mask.sum().clamp_min(1)
