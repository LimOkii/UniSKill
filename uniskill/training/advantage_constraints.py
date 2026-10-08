from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def _invalid_row_tensor(
    invalid_rows: Sequence[object] | np.ndarray | torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(invalid_rows, torch.Tensor):
        return invalid_rows.detach().reshape(-1).to(device=device, dtype=torch.bool)
    return torch.as_tensor(
        np.asarray(invalid_rows).reshape(-1).astype(bool),
        dtype=torch.bool,
        device=device,
    )


def cap_invalid_positive_advantages(
    *,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    invalid_rows: Sequence[object] | np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Prevent interface-invalid responses from receiving positive policy credit."""

    if advantages.shape != response_mask.shape:
        raise ValueError(
            "advantage/response mask shape mismatch: "
            f"{tuple(advantages.shape)} != {tuple(response_mask.shape)}"
        )
    invalid = _invalid_row_tensor(
        invalid_rows,
        device=advantages.device,
    )
    if invalid.numel() != advantages.shape[0]:
        raise ValueError(
            "invalid-row metadata length mismatch: "
            f"{invalid.numel()} != {advantages.shape[0]}"
        )

    constrained = advantages.clone()
    invalid_response_tokens = invalid.unsqueeze(-1) & response_mask.bool()
    constrained[invalid_response_tokens] = torch.minimum(
        constrained[invalid_response_tokens],
        torch.zeros((), dtype=constrained.dtype, device=constrained.device),
    )
    return constrained


def invalid_positive_advantage_ratio(
    *,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    invalid_rows: Sequence[object] | np.ndarray | torch.Tensor,
) -> float:
    """Return the fraction of invalid response rows that still have positive credit."""

    if advantages.shape != response_mask.shape:
        raise ValueError("advantage/response mask shape mismatch")
    invalid = _invalid_row_tensor(
        invalid_rows,
        device=advantages.device,
    )
    if invalid.numel() != advantages.shape[0]:
        raise ValueError("invalid-row metadata length mismatch")

    valid_response_rows = response_mask.bool().any(dim=-1)
    denominator_rows = invalid & valid_response_rows
    denominator = int(denominator_rows.sum().item())
    if denominator == 0:
        return 0.0
    positive_rows = (
        ((advantages > 0) & response_mask.bool()).any(dim=-1)
        & denominator_rows
    )
    return float(positive_rows.sum().item() / denominator)
