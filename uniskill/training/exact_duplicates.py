from __future__ import annotations

from uniskill.environments.prompts.registry import proposal_example_skill
from uniskill.training.types import ProposalCandidate


EXACT_DUPLICATE_PROMPT = "prompt"
EXACT_DUPLICATE_BANK = "bank"
EXACT_DUPLICATE_BATCH = "batch"
EXACT_DUPLICATE_RETRIEVED = "retrieved"


def normalize_skill_text(skill_text: str) -> str:
    return " ".join(str(skill_text).split())


def apply_exact_duplicate_routing(
    candidates: list[ProposalCandidate],
    *,
    skillbank,
    invalid_penalty: float,
) -> None:
    """Reject exact content copies after Critic judgment and before R_align."""

    seen_batch_adds: set[tuple[str, str]] = set()
    for candidate in candidates:
        if (
            candidate.critic_content_supported is not True
            or candidate.proposal_action
            not in {"ADD_NEW_SKILL", "UPDATE_SKILL"}
            or candidate.local_routing_reason == "insufficient_anchor_coverage"
        ):
            continue

        normalized = normalize_skill_text(candidate.skill_text)
        prompt_example = normalize_skill_text(
            proposal_example_skill(candidate.source.environment)
        )
        duplicate_source = ""
        if candidate.proposal_action == "ADD_NEW_SKILL":
            task_type = candidate.source.task_type
            batch_key = (task_type, normalized)
            if prompt_example and normalized == prompt_example:
                duplicate_source = EXACT_DUPLICATE_PROMPT
            elif task_type and skillbank.contains_exact_skill(
                task_type, candidate.skill_text
            ):
                duplicate_source = EXACT_DUPLICATE_BANK
            elif batch_key in seen_batch_adds:
                duplicate_source = EXACT_DUPLICATE_BATCH
            else:
                seen_batch_adds.add(batch_key)
        else:
            retrieved_text = str(
                (candidate.source.retrieved_skill or {}).get("skill_text") or ""
            )
            if normalized and normalized == normalize_skill_text(retrieved_text):
                duplicate_source = EXACT_DUPLICATE_RETRIEVED

        if not duplicate_source:
            continue
        candidate.exact_duplicate_source = duplicate_source
        candidate.skill_reward = float(invalid_penalty)
        candidate.r_align = None
        candidate.delta_success = None
        candidate.delta_failure = None
