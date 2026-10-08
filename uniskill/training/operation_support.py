from __future__ import annotations

import math

import torch


OPERATION_ACTIONS = ("NO_SKILL", "ADD_NEW_SKILL", "UPDATE_SKILL")


def operation_support_loss(
    *,
    operation_logits: torch.Tensor,
    valid_actions: torch.Tensor,
    min_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a soft lower-bound barrier to each currently legal operation."""

    if operation_logits.ndim != 2:
        raise ValueError("operation logits must have shape [batch, actions]")
    if valid_actions.shape != operation_logits.shape:
        raise ValueError("valid operation mask must match operation logits")
    if not 0 < min_probability < 1:
        raise ValueError("operation support probability must be in (0, 1)")
    valid_actions = valid_actions.bool()
    valid_counts = valid_actions.sum(dim=-1)
    if torch.any(valid_counts < 2):
        raise ValueError("each support row must expose at least two legal operations")
    if torch.any(valid_counts.to(operation_logits.dtype) * min_probability >= 1):
        raise ValueError("operation support floor leaves no probability for optimization")

    masked_logits = operation_logits.masked_fill(~valid_actions, float("-inf"))
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    probs = torch.exp(log_probs).masked_fill(~valid_actions, 0.0)
    log_floor = math.log(float(min_probability))
    violations = torch.relu(log_floor - log_probs).masked_fill(~valid_actions, 0.0)
    per_row = violations.square().sum(dim=-1) / valid_counts.to(operation_logits.dtype)
    return per_row.mean(), probs, violations
