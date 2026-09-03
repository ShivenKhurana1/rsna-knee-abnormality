"""Family A: six-slot 2.5D multiple-instance model with target-query attention.

The pooling head's math matches the deployed V13 SlotHead exactly (proj -> add
slot embedding -> per-target query attention over slots, masked-softmax, output
projection, optional slot-anatomy prior bias) since that mechanism is already
validated in production; what's new is the encoder is trainable here (unfrozen
last blocks, per the plan's Phase A1) rather than used only as a frozen feature
source the way the failed specialist experiments did, and the model is trained
directly (with capped auxiliary supervision), not as a post-hoc residual
correction on top of another model's output.

Encoders are pluggable so architecture correctness can be verified on CPU with
TinyCNNEncoder (no external weights, no GPU) before ever touching a real image
or the transformers/DINOv2 dependency, which Dinov2Encoder needs and which only
matters on the actual training device.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from contract import GROUP, IMG, N_SLOT, SLOT_PRIOR_STRENGTH, SLOT_PRIOR_TABLE, SLOTS, TARGETS


class TargetQueryHead(nn.Module):
    """One learned query per target attends over the N_SLOT slot tokens."""

    def __init__(self, dim, n_slot=N_SLOT, n_out=len(TARGETS), hidden=256, dropout=0.2, use_prior=True):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        self.use_prior = use_prior and n_slot == len(SLOTS) and n_out == len(TARGETS)
        prior = torch.zeros(n_out, n_slot)
        if self.use_prior:
            for t, slots in SLOT_PRIOR_TABLE.items():
                if t in TARGETS:
                    prior[TARGETS.index(t), list(slots)] = SLOT_PRIOR_STRENGTH
        self.register_buffer('slot_prior', prior)

    def forward(self, tokens, mask):
        """tokens: (B, n_slot, dim). mask: (B, n_slot) in {0, 1}, 1 = slot present.
        Returns (B, n_out) logits and (B, n_out, n_slot) attention weights."""
        h = self.proj(tokens) + self.slot_emb
        att = torch.einsum('bsh,oh->bos', h, self.query) / self.hidden ** 0.5
        if self.use_prior:
            att = att + self.slot_prior.unsqueeze(0)
        if (mask.sum(dim=1) == 0).any():
            raise ValueError('At least one slot must be present per study; an all-missing study cannot be scored')
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum('bos,bsh->boh', att, h))
        logits = (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias
        return logits, att.detach()


class TinyCNNEncoder(nn.Module):
    """CPU-only stand-in encoder for architecture/shape/gradient-flow testing.
    Not a modeling contribution -- swap in Dinov2Encoder for real training."""

    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(GROUP, 16, 5, stride=4, padding=2), nn.GELU(),
            nn.Conv2d(16, 32, 5, stride=4, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(32, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        """x: (N, GROUP, IMG, IMG) float in [0, 1]. Returns (N, out_dim)."""
        return self.proj(self.net(x).flatten(1))


class Dinov2Encoder(nn.Module):
    """Wraps a HuggingFace DINOv2 backbone with partial unfreezing, matching the
    already-tested V13 build_model() unfreeze/normalize contract exactly, so a
    checkpoint trained here is not fighting a second, different preprocessing
    convention. Only imported/instantiated on the GPU training device; the
    `transformers` dependency and pretrained weights are not needed for the CPU
    smoke test."""

    def __init__(self, source_path, unfreeze_last=2, pool='cls_mean'):
        super().__init__()
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(str(source_path))
        n_layer = len(backbone.encoder.layer)
        for p in backbone.parameters():
            p.requires_grad = False
        for blk in backbone.encoder.layer[max(0, n_layer - unfreeze_last):]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in backbone.layernorm.parameters():
            p.requires_grad = True
        self.backbone = backbone
        self.pool = pool
        parts = {'cls_mean': 2, 'cls_mean_focal': 3}[pool]
        self.out_dim = backbone.config.hidden_size * parts
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == 'cls_mean_focal':
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        return torch.cat(parts, dim=1)


class GeneralistModel(nn.Module):
    def __init__(self, encoder, n_slot=N_SLOT, n_out=len(TARGETS), head_hidden=256, dropout=0.2, use_prior=True):
        super().__init__()
        self.encoder = encoder
        self.head = TargetQueryHead(encoder.out_dim, n_slot, n_out, head_hidden, dropout, use_prior)

    def forward(self, imgs, mask, img_size=None):
        """imgs: (B, n_slot, GROUP, H, W) uint8 or float. mask: (B, n_slot)."""
        b, s = imgs.shape[:2]
        x = imgs.reshape(b * s, *imgs.shape[2:]).float()
        if x.max() > 1.5:  # tensor arrived as 0-255 uint8-range values
            x = x / 255.0
        if img_size is not None and img_size != x.shape[-1]:
            x = F.interpolate(x, size=(img_size, img_size), mode='bilinear', align_corners=False)
        tokens = self.encoder(x).reshape(b, s, -1)
        return self.head(tokens, mask)


def build_smoke_model(hidden=32, out_dim=16):
    return GeneralistModel(TinyCNNEncoder(out_dim=out_dim), head_hidden=hidden)
