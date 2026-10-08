from __future__ import annotations

import random
import uuid
from collections import defaultdict

from uniskill.environments.prompts.registry import build_skill_proposal_prompt
from uniskill.training.types import ProposalCandidate, Trajectory


def build_proposal_candidates(
    trajectories: list[Trajectory],
    *,
    seed: int,
    expected_environment: str | None = None,
) -> tuple[list[ProposalCandidate], dict[str, int]]:
    """Build four-role proposals from mixed-outcome rollout groups.

    Proposal/Critic see a source and an opposite-outcome trajectory; R_align
    uses a separate successful and failed anchor. Selection is deterministic
    for a given iteration seed, and all four trajectory IDs must differ.
    """

    if trajectories:
        expected_family = _environment_family(
            expected_environment or trajectories[0].environment
        )
        if expected_family not in {"alfworld", "webshop"}:
            raise ValueError("proposal construction requires ALFWorld or WebShop")
        mismatched = [
            trajectory.trajectory_id
            for trajectory in trajectories
            if _environment_family(trajectory.environment) != expected_family
        ]
        if mismatched:
            preview = ", ".join(mismatched[:3])
            raise RuntimeError(
                "trajectory environment metadata is missing or "
                f"incorrect for {len(mismatched)} trajectory(s): {preview}"
            )

    grouped: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[trajectory.query_id].append(trajectory)

    rng = random.Random(seed)
    candidates: list[ProposalCandidate] = []
    stats = {
        "groups": len(grouped),
        "mixed_groups": 0,
        "homogeneous_groups": 0,
        "held_out_reference_groups": 0,
        "insufficient_held_out_reference_groups": 0,
        "skipped_sources_missing_held_out_anchors": 0,
        "held_out_opposite_mode_enabled": 1,
        "four_trajectory_candidates": 0,
    }
    for query_id in sorted(grouped):
        group = grouped[query_id]
        successes = [trajectory for trajectory in group if trajectory.success]
        failures = [trajectory for trajectory in group if not trajectory.success]
        if not successes or not failures:
            stats["homogeneous_groups"] += 1
            continue
        stats["mixed_groups"] += 1

        split_key = f"outcome_split_{len(successes)}s_{len(failures)}f_groups"
        stats[split_key] = stats.get(split_key, 0) + 1
        if len(successes) < 2 or len(failures) < 2:
            stats["insufficient_held_out_reference_groups"] += 1
            stats["skipped_sources_missing_held_out_anchors"] += len(group)
            continue
        stats["held_out_reference_groups"] += 1

        for source in group:
            opposite_pool = failures if source.success else successes
            opposite_reference = rng.choice(opposite_pool)
            excluded_ids = {source.trajectory_id, opposite_reference.trajectory_id}
            success_pool = [
                item for item in successes
                if item.trajectory_id not in excluded_ids
            ]
            failure_pool = [
                item for item in failures
                if item.trajectory_id not in excluded_ids
            ]
            if not success_pool or not failure_pool:
                stats["skipped_sources_missing_held_out_anchors"] += 1
                continue
            success_anchor = rng.choice(success_pool)
            failure_anchor = rng.choice(failure_pool)
            role_ids = {
                source.trajectory_id,
                opposite_reference.trajectory_id,
                success_anchor.trajectory_id,
                failure_anchor.trajectory_id,
            }
            if len(role_ids) != 4:
                raise AssertionError(
                    "proposal source, prompt reference, and held-out "
                    "alignment anchors must be four distinct trajectories"
                )
            stats["four_trajectory_candidates"] += 1
            candidates.append(
                ProposalCandidate(
                    proposal_id=str(uuid.uuid4()),
                    source=source,
                    success_anchor=success_anchor,
                    failure_anchor=failure_anchor,
                    prompt=build_skill_proposal_prompt(
                        source.to_record(),
                        comparison_record=opposite_reference.to_record(),
                    ),
                    prompt_reference=opposite_reference,
                )
            )

    stats["candidates"] = len(candidates)
    return candidates, stats


def _environment_family(environment: str | None) -> str:
    value = str(environment or "").lower()
    if "alfworld" in value:
        return "alfworld"
    if "webshop" in value:
        return "webshop"
    return ""


def validate_candidate_roles(
    candidates: list[ProposalCandidate],
    stats: dict[str, int],
    *,
    format_only_warmup: bool,
) -> None:
    """Validate the role contract for the active WebShop warm-up or main route."""

    if format_only_warmup:
        if stats.get("warmup_sampled") != len(candidates):
            raise RuntimeError("WebShop warm-up candidate count is inconsistent")
        if any(
            candidate.success_anchor is not None or candidate.failure_anchor is not None
            for candidate in candidates
        ):
            raise RuntimeError("WebShop format warm-up must not use alignment anchors")
        return

    if stats.get("four_trajectory_candidates") != len(candidates):
        raise RuntimeError(
            "proposal construction produced a candidate without four distinct roles"
        )
    if any(not candidate.to_record()["four_trajectory_roles_distinct"] for candidate in candidates):
        raise RuntimeError(
            "proposal construction produced overlapping trajectory roles"
        )
