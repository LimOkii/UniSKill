from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from uniskill.training.types import ProposalCandidate

if TYPE_CHECKING:
    from uniskill.skillbank import AlfWorldSkillBank


@dataclass(frozen=True)
class CommitResult:
    proposal_id: str
    status: str
    skill: dict | None = None


def commit_candidates(
    candidates: list[ProposalCandidate],
    *,
    skillbank: AlfWorldSkillBank,
) -> list[CommitResult]:
    """Apply the iteration-level ADD/UPDATE arbitration policy."""

    eligible = [candidate for candidate in candidates if candidate.is_write_eligible]
    results: list[CommitResult] = []
    embedding_items: list[dict] = []

    seen_adds: set[tuple[str, str]] = set()
    for candidate in sorted(
        (item for item in eligible if item.proposal_action == "ADD_NEW_SKILL"),
        key=lambda item: (-float(item.r_align), item.proposal_id),
    ):
        task_type = candidate.source.task_type
        if not task_type:
            results.append(CommitResult(candidate.proposal_id, "missing_task_type"))
            continue
        key = (task_type, " ".join(candidate.skill_text.split()))
        if key in seen_adds or skillbank.contains_exact_skill(
            task_type, candidate.skill_text
        ):
            results.append(CommitResult(candidate.proposal_id, "duplicate_add"))
            continue
        seen_adds.add(key)
        written = skillbank.append_skill(
            task_type=task_type, skill_text=candidate.skill_text
        )
        results.append(CommitResult(candidate.proposal_id, "added", written))
        embedding_items.append({"task_type": task_type, "skill": written})

    updates: dict[tuple[str, str], list[ProposalCandidate]] = defaultdict(list)
    for candidate in eligible:
        if candidate.proposal_action != "UPDATE_SKILL":
            continue
        task_type = candidate.source.task_type
        skill_id = (candidate.source.retrieved_skill or {}).get("skill_id")
        if task_type and skill_id:
            updates[(task_type, skill_id)].append(candidate)
        else:
            results.append(CommitResult(candidate.proposal_id, "missing_update_target"))

    for (task_type, skill_id), group in sorted(updates.items()):
        winner = max(group, key=lambda item: (float(item.r_align), item.proposal_id))
        written = skillbank.update_skill(
            task_type=task_type,
            skill_id=skill_id,
            skill_text=winner.skill_text,
        )
        results.append(CommitResult(winner.proposal_id, "updated", written))
        embedding_items.append({"task_type": task_type, "skill": written})
        for candidate in group:
            if candidate.proposal_id != winner.proposal_id:
                results.append(CommitResult(candidate.proposal_id, "superseded_update"))

    if skillbank.retrieval_method == "embedding" and embedding_items:
        skillbank.persist_skill_embeddings(embedding_items)
    return results
