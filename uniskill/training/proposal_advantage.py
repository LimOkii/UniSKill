from __future__ import annotations

import torch
from verl.trainer.ppo import core_algos


PROPOSAL_CHANNELS = ("format", "action", "skill")
PROPOSAL_JOINT_CHANNEL = "joint"


def nonzero_advantage_token_count(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    tolerance: float = 1e-8,
) -> int:
    if advantages.shape != response_mask.shape:
        raise ValueError("proposal advantage/mask shape mismatch")
    return int(
        ((advantages.detach().abs() > tolerance) & response_mask.detach().bool())
        .sum()
        .item()
    )


def compute_channel_reinforce_plus_plus_advantage(
    *,
    scalar_rewards: torch.Tensor,
    valid_rows: torch.Tensor,
    token_mask: torch.Tensor,
    gamma: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Whiten one scalar vote per active response, then broadcast by token mask."""

    if scalar_rewards.ndim != 2 or scalar_rewards.shape[-1] != 1:
        raise ValueError("proposal channel rewards must have shape [batch, 1]")
    if valid_rows.shape != scalar_rewards.shape:
        raise ValueError("proposal channel valid mask must match scalar rewards")
    if token_mask.ndim != 2 or token_mask.shape[0] != scalar_rewards.shape[0]:
        raise ValueError("proposal channel token mask has incompatible shape")

    active = valid_rows[:, 0].bool() & token_mask.bool().any(dim=-1)
    scalar_advantages = torch.zeros_like(scalar_rewards)
    token_advantages = torch.zeros_like(token_mask, dtype=scalar_rewards.dtype)
    token_returns = torch.zeros_like(token_mask, dtype=scalar_rewards.dtype)
    indices = torch.nonzero(active, as_tuple=False).flatten()
    if indices.numel() == 0:
        return token_advantages, token_returns, scalar_advantages

    rewards = scalar_rewards.index_select(0, indices)
    if indices.numel() == 1:
        returns = rewards
        advantages = torch.zeros_like(rewards)
    else:
        one_token_mask = torch.ones_like(rewards, dtype=torch.bool)
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=rewards,
            response_mask=one_token_mask,
            gamma=gamma,
        )
    scalar_advantages.index_copy_(0, indices, advantages)
    for local_index, row_index in enumerate(indices.tolist()):
        row_mask = token_mask[row_index].to(dtype=scalar_rewards.dtype)
        token_advantages[row_index] = advantages[local_index, 0] * row_mask
        token_returns[row_index] = returns[local_index, 0] * row_mask
    return token_advantages, token_returns, scalar_advantages


def proposal_channel_metrics(
    *,
    scalar_rewards: torch.Tensor,
    valid_rows: torch.Tensor,
    scalar_advantages: torch.Tensor,
    channel: str,
) -> dict[str, float]:
    active = valid_rows[:, 0].bool()
    rewards = scalar_rewards[:, 0][active]
    advantages = scalar_advantages[:, 0][active]
    prefix = f"proposal/{channel}"
    metrics = {f"{prefix}_active": float(active.sum().item())}
    if rewards.numel():
        metrics[f"{prefix}_reward_mean"] = float(rewards.mean().item())
        metrics[f"{prefix}_reward_std"] = float(rewards.std(unbiased=False).item())
        metrics[f"{prefix}_advantage_mean"] = float(advantages.mean().item())
        metrics[f"{prefix}_advantage_std"] = float(
            advantages.std(unbiased=False).item()
        )
    return metrics
