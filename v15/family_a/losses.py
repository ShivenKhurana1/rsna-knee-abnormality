"""Masked BCE for expert labels, plus an optional capped, masked auxiliary term.

Both losses are computed with logits (not probabilities) via
binary_cross_entropy_with_logits for numerical stability, and both are averaged
only over the cells their own mask marks valid -- a study/target cell that is
NaN in y (no expert label, or report-silent for the auxiliary target) never
enters either loss, it is not treated as a zero.
"""
import torch
import torch.nn.functional as F


def masked_bce(logits, y, mask):
    """logits, y, mask: (B, T). Returns a scalar; 0.0 (not NaN) if mask is empty
    so a batch with no expert-labeled rows for a given fold split doesn't produce
    a NaN gradient step."""
    if mask.sum() == 0:
        return logits.new_zeros(())
    y_safe = torch.where(mask, y, torch.zeros_like(y))
    per_cell = F.binary_cross_entropy_with_logits(logits, y_safe, reduction='none')
    return (per_cell * mask.float()).sum() / mask.float().sum()


def combined_loss(logits, y_expert, expert_mask, y_aux, aux_mask, aux_weight):
    """aux_weight: (T,) capped per-target scalar, broadcast across the batch and
    multiplied into the per-cell auxiliary mask before averaging, so a target
    with weight 0 (rejected by the transfer-gate policy, or the baseline arm
    where every weight is 0) contributes exactly nothing -- not a small nonzero
    push -- to the total loss."""
    expert_term = masked_bce(logits, y_expert, expert_mask)
    if aux_weight is None or float(aux_weight.abs().sum()) == 0.0:
        return expert_term, {'expert': float(expert_term.detach()), 'aux': 0.0}
    y_safe = torch.where(aux_mask, y_aux, torch.zeros_like(y_aux))
    per_cell = F.binary_cross_entropy_with_logits(logits, y_safe, reduction='none')
    weighted_mask = aux_mask.float() * aux_weight.unsqueeze(0)
    denom = weighted_mask.sum()
    aux_term = (per_cell * weighted_mask).sum() / denom if denom > 0 else logits.new_zeros(())
    total = expert_term + aux_term
    return total, {'expert': float(expert_term.detach()), 'aux': float(aux_term.detach())}
