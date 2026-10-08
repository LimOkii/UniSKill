from __future__ import annotations

from collections.abc import Sequence
from statistics import median

from uniskill.training.types import ProposalCandidate, Trajectory


def scorable_action_coverage(trajectory: Trajectory) -> float:
    """Fraction of steps with a strict, non-empty action-payload token mask."""

    if not trajectory.steps:
        return 0.0
    return sum(bool(any(step.action_token_mask)) for step in trajectory.steps) / len(
        trajectory.steps
    )


def alignment_coverage_metrics(
    trajectories: Sequence[Trajectory],
    candidates: Sequence[ProposalCandidate],
) -> dict[str, float]:
    """Summarize the evidence coverage available to counterfactual scoring."""

    all_steps = [step for trajectory in trajectories for step in trajectory.steps]
    all_coverages = [scorable_action_coverage(item) for item in trajectories]
    success_coverages = [
        scorable_action_coverage(item) for item in trajectories if item.success
    ]
    failure_coverages = [
        scorable_action_coverage(item) for item in trajectories if not item.success
    ]
    pair_coverages = []
    for candidate in candidates:
        if candidate.success_anchor is None or candidate.failure_anchor is None:
            continue  # WebShop format warmup has no counterfactual anchors.
        candidate.success_anchor_coverage = scorable_action_coverage(
            candidate.success_anchor
        )
        candidate.failure_anchor_coverage = scorable_action_coverage(
            candidate.failure_anchor
        )
        pair_coverages.append(
            min(candidate.success_anchor_coverage, candidate.failure_anchor_coverage)
        )

    metrics: dict[str, float] = {}
    text_parsed = sum(bool(step.action_text.strip()) for step in all_steps)
    payload_masked = sum(bool(any(step.action_token_mask)) for step in all_steps)
    metrics.update(
        {
            "action_text_parse_success": text_parsed,
            "action_payload_mask_success": payload_masked,
            "action_payload_mask_failure": len(all_steps) - payload_masked,
            "action_text_parsed_but_mask_missing": sum(
                bool(step.action_text.strip()) and not any(step.action_token_mask)
                for step in all_steps
            ),
        }
    )

    def add_summary(name: str, values: list[float]) -> None:
        if not values:
            return
        metrics[f"coverage_{name}_mean"] = sum(values) / len(values)
        metrics[f"coverage_{name}_p50"] = float(median(values))
        metrics[f"coverage_{name}_min"] = min(values)

    def ratio_at_least(values: list[float], threshold: float) -> float:
        return sum(value >= threshold for value in values) / len(values)

    add_summary("all", all_coverages)
    add_summary("success", success_coverages)
    add_summary("failure", failure_coverages)
    add_summary("anchor_pair_min", pair_coverages)
    if all_coverages:
        metrics["coverage_trajectory_ge_50_ratio"] = ratio_at_least(
            all_coverages, 0.5
        )
        metrics["coverage_trajectory_ge_80_ratio"] = ratio_at_least(
            all_coverages, 0.8
        )
        metrics["coverage_trajectory_full_ratio"] = ratio_at_least(
            all_coverages, 1.0
        )
    if pair_coverages:
        metrics["coverage_anchor_pair_ge_50_ratio"] = ratio_at_least(
            pair_coverages, 0.5
        )
        metrics["coverage_anchor_pair_ge_80_ratio"] = ratio_at_least(
            pair_coverages, 0.8
        )
        metrics["coverage_anchor_pair_full_ratio"] = ratio_at_least(
            pair_coverages, 1.0
        )
    return metrics
