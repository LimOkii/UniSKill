from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from uniskill.training.types import ProposalCandidate


def action_rollout_health_metrics(
    trajectories: list[Any],
    *,
    prefix: str = "episode/action",
) -> dict[str, float]:
    """Expose format, admissibility, history sanitation, and length bias."""

    steps = [step for trajectory in trajectories for step in trajectory.steps]
    if not steps:
        return {
            f"{prefix}_format_valid_ratio": 0.0,
            f"{prefix}_strict_format_valid_ratio": 0.0,
            f"{prefix}_admissible_ratio": 0.0,
            f"{prefix}_overall_valid_ratio": 0.0,
        }

    def ratio(items: list[Any], predicate) -> float:
        if not items:
            return 0.0
        return float(sum(bool(predicate(item)) for item in items) / len(items))

    success_trajectories = [trajectory for trajectory in trajectories if trajectory.success]
    failure_trajectories = [trajectory for trajectory in trajectories if not trajectory.success]
    success_steps = [step for trajectory in success_trajectories for step in trajectory.steps]
    failure_steps = [step for trajectory in failure_trajectories for step in trajectory.steps]

    def mean_length(items: list[Any]) -> float:
        if not items:
            return 0.0
        return float(sum(len(item.steps) for item in items) / len(items))

    metrics = {
        f"{prefix}_format_valid_ratio": ratio(
            steps, lambda step: step.is_action_valid
        ),
        f"{prefix}_strict_format_valid_ratio": ratio(
            steps, lambda step: step.is_action_strict_format_valid
        ),
        f"{prefix}_admissible_ratio": ratio(
            steps, lambda step: step.is_action_admissible
        ),
        f"{prefix}_overall_valid_ratio": ratio(
            steps, lambda step: step.is_action_overall_valid
        ),
        f"{prefix}_payload_parsed_ratio": ratio(
            steps, lambda step: step.action_payload_parsed
        ),
        f"{prefix}_success_format_valid_ratio": ratio(
            success_steps, lambda step: step.is_action_valid
        ),
        f"{prefix}_failure_format_valid_ratio": ratio(
            failure_steps, lambda step: step.is_action_valid
        ),
        f"{prefix}_success_strict_format_valid_ratio": ratio(
            success_steps, lambda step: step.is_action_strict_format_valid
        ),
        f"{prefix}_failure_strict_format_valid_ratio": ratio(
            failure_steps, lambda step: step.is_action_strict_format_valid
        ),
        f"{prefix}_success_admissible_ratio": ratio(
            success_steps, lambda step: step.is_action_admissible
        ),
        f"{prefix}_failure_admissible_ratio": ratio(
            failure_steps, lambda step: step.is_action_admissible
        ),
        f"{prefix}_success_overall_valid_ratio": ratio(
            success_steps, lambda step: step.is_action_overall_valid
        ),
        f"{prefix}_failure_overall_valid_ratio": ratio(
            failure_steps, lambda step: step.is_action_overall_valid
        ),
        f"{prefix}_success_trajectory_length_mean": mean_length(
            success_trajectories
        ),
        f"{prefix}_failure_trajectory_length_mean": mean_length(
            failure_trajectories
        ),
        f"{prefix}_success_step_share": float(len(success_steps) / len(steps)),
        f"{prefix}_failure_step_share": float(len(failure_steps) / len(steps)),
        f"{prefix}_history_valid_ratio": ratio(
            steps, lambda step: step.action_history_status == "valid"
        ),
        f"{prefix}_history_unavailable_ratio": ratio(
            steps, lambda step: step.action_history_status == "unavailable"
        ),
        f"{prefix}_history_omitted_ratio": ratio(
            steps, lambda step: step.action_history_status == "omitted"
        ),
        f"{prefix}_contains_chinese_ratio": ratio(
            steps, lambda step: step.contains_chinese
        ),
        f"{prefix}_missing_action_tags_ratio": ratio(
            steps, lambda step: step.missing_action_tags
        ),
        f"{prefix}_missing_think_tags_ratio": ratio(
            steps, lambda step: step.missing_think_tags
        ),
        f"{prefix}_response_hit_length_limit_ratio": ratio(
            steps, lambda step: step.response_hit_length_limit
        ),
        f"{prefix}_response_truncated_ratio": ratio(
            steps, lambda step: step.response_truncated
        ),
    }
    return metrics


def action_advantage_metrics(batch: Any) -> dict[str, float]:
    """Split scalar action advantages by outcome and action validity."""

    import numpy as np
    import torch

    if "advantages" not in batch.batch or "response_mask" not in batch.batch:
        return {}
    response_mask = batch.batch["response_mask"].to(dtype=torch.float32)
    denominators = response_mask.sum(dim=-1).clamp_min(1.0)
    row_advantages = (
        batch.batch["advantages"] * response_mask
    ).sum(dim=-1) / denominators
    row_advantages = row_advantages.detach().cpu()

    rewards = np.asarray(batch.non_tensor_batch["episode_rewards"]).reshape(-1)
    format_valids = np.asarray(
        batch.non_tensor_batch["is_action_valid"]
    ).reshape(-1).astype(bool)
    strict_format_valids = np.asarray(
        batch.non_tensor_batch["is_action_strict_format_valid"]
    ).reshape(-1).astype(bool)
    admissible = np.asarray(
        batch.non_tensor_batch["is_action_admissible"]
    ).reshape(-1).astype(bool)
    overall_valids = strict_format_valids & admissible
    successes = rewards > 0
    if len(row_advantages) != len(format_valids):
        raise ValueError("action advantage metadata length mismatch")

    metrics: dict[str, float] = {}

    def add_validity_split(valids: np.ndarray, *, overall: bool) -> None:
        values_by_group: dict[tuple[bool, bool], torch.Tensor] = {}
        for success in (True, False):
            for valid in (True, False):
                numpy_mask = (successes == success) & (valids == valid)
                mask = torch.from_numpy(numpy_mask).to(dtype=torch.bool)
                values = row_advantages[mask]
                values_by_group[(success, valid)] = values
                outcome = "success" if success else "failure"
                validity = "valid" if valid else "invalid"
                if overall:
                    prefix = (
                        f"episode/action_advantage_{outcome}_overall_{validity}"
                    )
                else:
                    prefix = f"episode/action_advantage_{outcome}_{validity}"
                metrics[f"{prefix}_count"] = float(values.numel())
                if values.numel():
                    metrics[f"{prefix}_mean"] = float(values.mean().item())

        failure_valid = values_by_group[(False, True)]
        failure_invalid = values_by_group[(False, False)]
        if failure_valid.numel() and failure_invalid.numel():
            key = (
                "episode/action_advantage_failure_overall_invalid_minus_valid"
                if overall
                else "episode/action_advantage_failure_invalid_minus_valid"
            )
            metrics[key] = float(
                failure_invalid.mean().item() - failure_valid.mean().item()
            )

    # Keep format-validity metrics for diagnostics.
    add_validity_split(format_valids, overall=False)
    # Use strict-format plus admissibility for the penalty split.
    add_validity_split(overall_valids, overall=True)
    return metrics


def action_history_metrics(trajectories: list[Any]) -> dict[str, float]:
    """Summarize the recent turns shown to the actor."""

    contexts = [
        step.context
        for trajectory in trajectories
        for step in trajectory.steps
    ]
    history_counts = [len(context.get("history") or []) for context in contexts]
    if not history_counts:
        return {
            "episode/action_history_steps_mean": 0.0,
            "episode/action_history_steps_max": 0.0,
        }
    return {
        "episode/action_history_steps_mean": float(
            sum(history_counts) / len(history_counts)
        ),
        "episode/action_history_steps_max": float(max(history_counts)),
    }


def proposal_critic_action_metrics(
    candidates: list[ProposalCandidate],
) -> dict[str, float | int]:
    """Expose compact, independently judged action/content critic signals."""

    parsed = [candidate for candidate in candidates if candidate.parse_ok]
    changed = [
        candidate
        for candidate in parsed
        if candidate.proposal_action in {"ADD_NEW_SKILL", "UPDATE_SKILL"}
    ]
    independently_judged = [
        candidate
        for candidate in changed
        if candidate.critic_action_reasonable is not None
        and candidate.critic_content_supported is not None
    ]
    disagreements = sum(
        candidate.critic_action_reasonable
        != candidate.critic_content_supported
        for candidate in independently_judged
    )
    return {
        **{
            f"proposal/raw_action_{action.lower()}": sum(
                candidate.proposal_action == action for candidate in candidates
            )
            for action in ("NO_SKILL", "ADD_NEW_SKILL", "UPDATE_SKILL")
        },
        **{
            f"proposal/action_{action.lower()}": sum(
                candidate.proposal_action == action for candidate in parsed
            )
            for action in ("NO_SKILL", "ADD_NEW_SKILL", "UPDATE_SKILL")
        },
        "proposal/action_reasonable": sum(
            candidate.critic_action_reasonable is True for candidate in parsed
        ),
        "proposal/action_unreasonable": sum(
            candidate.critic_action_reasonable is False for candidate in parsed
        ),
        "proposal/content_supported": sum(
            candidate.critic_content_supported is True for candidate in changed
        ),
        "proposal/content_unsupported": sum(
            candidate.critic_content_supported is False for candidate in changed
        ),
        "proposal/action_content_disagreement_ratio": (
            disagreements / len(independently_judged) if independently_judged else 0.0
        ),
        "proposal/critic_retried": sum(candidate.critic_attempts > 1 for candidate in parsed),
    }


def proposal_response_metrics(
    candidates: list[ProposalCandidate],
) -> dict[str, float]:
    """Report proposal completion length separately from action responses."""

    if not candidates:
        return {
            "proposal/response_length_mean": 0.0,
            "proposal/response_length_max": 0.0,
            "proposal/response_clip_ratio": 0.0,
            "proposal/truncated": 0.0,
            "proposal/chinese_ratio": 0.0,
            "proposal/chinese_clip_ratio": 0.0,
        }
    lengths = [candidate.response_token_length for candidate in candidates]
    chinese = [candidate for candidate in candidates if candidate.contains_chinese]
    return {
        "proposal/response_length_mean": float(sum(lengths) / len(lengths)),
        "proposal/response_length_max": float(max(lengths)),
        "proposal/response_clip_ratio": float(
            sum(candidate.response_hit_length_limit for candidate in candidates)
            / len(candidates)
        ),
        "proposal/truncated": float(
            sum(candidate.response_truncated for candidate in candidates)
        ),
        "proposal/chinese_ratio": float(len(chinese) / len(candidates)),
        "proposal/chinese_clip_ratio": float(
            sum(candidate.response_hit_length_limit for candidate in chinese)
            / len(chinese)
        ) if chinese else 0.0,
    }


def select_uniskill_step_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Select the UniSkill signals that must be visible on every global step."""

    compact_proposal_prefixes = (
        "proposal/action_",
        "proposal/content_",
        "proposal/critic_",
        "proposal/format_",
        "proposal/skill_",
        "proposal/joint_",
        "proposal/r_align_",
        "proposal/response_",
        "proposal/op_",
        "proposal/raw_action_",
        "proposal/exact_duplicate_",
    )
    compact_proposal_keys = {
        "proposal/candidates",
        "proposal/generated",
        "proposal/trainable",
        "proposal/parse_invalid",
        "proposal/recoverable_invalid",
        "proposal/unrecoverable_invalid",
        "proposal/truncated",
        "proposal/chinese_ratio",
        "proposal/insufficient_anchor_coverage",
        "proposal/coverage_all_mean",
        "proposal/coverage_all_min",
        "proposal/coverage_anchor_pair_min_mean",
        "proposal/r_align_warmup_active",
        "proposal/warmup_with_opposite_reference",
        "proposal/warmup_without_opposite_reference",
        "proposal/mode_scale",
        "proposal/write_eligible",
        "proposal/incomplete_feedback",
        "proposal/skills_added",
        "proposal/skills_updated",
        "proposal/no_batch_ratio",
        "proposal/op_support_active",
        "actor/proposal_support_loss",
    }
    return {
        key: value
        for key, value in metrics.items()
        if key == "training/global_step"
        or key.startswith("action/")
        or key == "episode/valid_action_ratio"
        or key.startswith("episode/action_")
        or key.startswith("episode/action_history_")
        or key in compact_proposal_keys
        or key.startswith(compact_proposal_prefixes)
    }


def format_uniskill_step_metrics(metrics: Mapping[str, Any]) -> str:
    selected = select_uniskill_step_metrics(metrics)

    def json_default(value: Any):
        if hasattr(value, "item"):
            return value.item()
        return str(value)

    return "[UNISKILL STEP METRICS] " + json.dumps(
        selected,
        sort_keys=True,
        ensure_ascii=False,
        default=json_default,
    )
