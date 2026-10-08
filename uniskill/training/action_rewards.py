from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def counterfactual_action_eligibility(
    infos: Sequence[Mapping[str, Any]],
    *,
    require_overall_valid: bool = False,
) -> np.ndarray:
    """Return which action payloads may be used for counterfactual scoring.

    Existing environments retain the historical admissibility-only behavior.
    WebShop opts into ``require_overall_valid`` so structurally invalid responses
    such as generations containing two ``<action>`` blocks cannot become
    teacher-forcing anchors even when the first payload is executable.
    """

    key = (
        "is_action_overall_valid"
        if require_overall_valid
        else "is_action_admissible"
    )
    return np.asarray(
        [bool(info.get(key, False)) for info in infos],
        dtype=bool,
    )


def action_penalty_masks(
    non_tensor_batch: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strict-format-invalid, inadmissible-only, and unified masks."""

    strict_format_valid = np.asarray(
        non_tensor_batch["is_action_strict_format_valid"]
    ).reshape(-1).astype(bool)
    admissible = np.asarray(
        non_tensor_batch["is_action_admissible"]
    ).reshape(-1).astype(bool)
    if strict_format_valid.shape != admissible.shape:
        raise ValueError("action validity metadata length mismatch")

    strict_format_invalid = ~strict_format_valid
    inadmissible_only = strict_format_valid & ~admissible
    penalized = strict_format_invalid | inadmissible_only
    return strict_format_invalid, inadmissible_only, penalized


def action_penalty_metrics(non_tensor_batch: Mapping[str, Any]) -> dict[str, float]:
    """Expose which action validity failures receive the single penalty."""

    strict_format_invalid, inadmissible_only, penalized = action_penalty_masks(
        non_tensor_batch
    )
    native_format_valid = np.asarray(
        non_tensor_batch["is_action_valid"]
    ).reshape(-1).astype(bool)
    if native_format_valid.shape != strict_format_invalid.shape:
        raise ValueError("native and strict action validity metadata length mismatch")
    return {
        "episode/valid_action_ratio": float(native_format_valid.mean()),
        "episode/action_penalty_hit_ratio": float(penalized.mean()),
        "episode/action_format_invalid_ratio": float(
            1.0 - native_format_valid.mean()
        ),
        "episode/action_strict_format_valid_ratio": float(
            1.0 - strict_format_invalid.mean()
        ),
        "episode/action_structure_penalty_hit_ratio": float(
            strict_format_invalid.mean()
        ),
        "episode/action_inadmissible_only_ratio": float(
            inadmissible_only.mean()
        ),
    }
