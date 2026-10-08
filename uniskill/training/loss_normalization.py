from __future__ import annotations


def distributed_token_mean_scale(
    *,
    local_valid_tokens: int,
    global_valid_tokens: int,
    world_size: int,
) -> float:
    """Scale a local token mean so averaged DP gradients equal a global token mean."""

    if local_valid_tokens < 0:
        raise ValueError("local_valid_tokens must be non-negative")
    if global_valid_tokens <= 0:
        raise ValueError("global_valid_tokens must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if local_valid_tokens > global_valid_tokens:
        raise ValueError("local_valid_tokens cannot exceed global_valid_tokens")
    return float(world_size * local_valid_tokens / global_valid_tokens)
