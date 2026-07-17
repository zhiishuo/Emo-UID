"""Numerically stable tensor operations shared by Emo-UID modules."""

from __future__ import annotations

import torch
from torch import Tensor


def masked_mean(sequence: Tensor, valid_mask: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Mean-pool `[B, T, D]` sequences over valid time steps."""

    weights = valid_mask.to(dtype=sequence.dtype).unsqueeze(-1)
    numerator = (sequence * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(epsilon)
    return numerator / denominator


def soft_ordinal_target(
    labels: Tensor,
    anchors: Tensor,
    temperature: float,
) -> Tensor:
    """Construct the soft ordinal target from Eq. (8)."""

    labels = labels.reshape(-1, 1).to(dtype=anchors.dtype, device=anchors.device)
    logits = -(labels - anchors.reshape(1, -1)).abs() / temperature
    return torch.softmax(logits, dim=-1)


def categorical_entropy(probabilities: Tensor, epsilon: float = 1e-8) -> Tensor:
    probabilities = probabilities.clamp_min(epsilon)
    return -(probabilities * probabilities.log()).sum(dim=-1, keepdim=True)


def jensen_shannon_divergence(
    first: Tensor,
    second: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    """Per-sample Jensen-Shannon divergence with shape `[B, 1]`."""

    first = first.clamp_min(epsilon)
    second = second.clamp_min(epsilon)
    midpoint = 0.5 * (first + second)
    first_kl = (first * (first.log() - midpoint.log())).sum(dim=-1, keepdim=True)
    second_kl = (second * (second.log() - midpoint.log())).sum(dim=-1, keepdim=True)
    return 0.5 * (first_kl + second_kl)
