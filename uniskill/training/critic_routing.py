from __future__ import annotations

from uniskill.training.types import ProposalCandidate


LOCAL_PARSE_INVALID = "proposal_parse_invalid"
LOCAL_MISSING_UPDATE_TARGET = "missing_update_target"
LOCAL_INSUFFICIENT_ANCHOR_COVERAGE = "insufficient_anchor_coverage"
WARMUP_FORMAT_VALID = "r_align_warmup_format_valid"
WARMUP_NO_SKILL_MASKED = "r_align_warmup_no_skill_masked"


def is_r_align_warmup(*, global_step: int, warmup_steps: int) -> bool:
    """The first full alignment iteration is warmup_steps + 1."""

    return global_step <= warmup_steps


def proposal_mode_scale(
    *,
    global_step: int,
    warmup_steps: int,
    warmup_scale: float,
    full_scale: float,
) -> float:
    """Use a lower proposal loss weight while only format is supervised."""

    if is_r_align_warmup(global_step=global_step, warmup_steps=warmup_steps):
        return float(warmup_scale)
    return float(full_scale)


def apply_local_critic_route(
    candidate: ProposalCandidate,
    *,
    invalid_penalty: float,
) -> bool:
    """Apply routes that do not call the critic; return whether a critic call is needed."""

    candidate.format_reward = (
        abs(float(invalid_penalty)) if candidate.parse_ok else float(invalid_penalty)
    )
    if not candidate.parse_ok:
        candidate.local_routing_reason = LOCAL_PARSE_INVALID
        return False
    return True


def apply_anchor_coverage_route(
    candidate: ProposalCandidate,
    *,
    min_anchor_coverage: float,
) -> bool:
    """Mask only the R_align/content channel when anchor coverage is low."""

    if candidate.proposal_action not in {"ADD_NEW_SKILL", "UPDATE_SKILL"}:
        return True
    candidate.anchor_coverage_threshold = float(min_anchor_coverage)
    coverage = candidate.minimum_anchor_coverage
    if coverage is None:
        raise ValueError(
            f"proposal {candidate.proposal_id} is missing anchor coverage before critic routing"
        )
    if coverage < min_anchor_coverage:
        candidate.local_routing_reason = LOCAL_INSUFFICIENT_ANCHOR_COVERAGE
        candidate.skill_reward = None
        return False
    return True


def apply_r_align_warmup_route(
    candidate: ProposalCandidate,
    *,
    invalid_penalty: float,
) -> None:
    """Train ADD/UPDATE structure without critic, alignment, or skill writes."""

    candidate.format_reward = (
        abs(float(invalid_penalty)) if candidate.parse_ok else float(invalid_penalty)
    )
    if not candidate.parse_ok:
        candidate.local_routing_reason = LOCAL_PARSE_INVALID
        return
    candidate.local_routing_reason = WARMUP_FORMAT_VALID
