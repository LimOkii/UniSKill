"""Format-only proposal sampling; task success never gates this sampler."""
from __future__ import annotations

import random
import uuid
from collections import defaultdict

from uniskill.environments.prompts.registry import build_skill_proposal_prompt
from uniskill.training.types import ProposalCandidate, Trajectory


def build_warmup_candidates(
    trajectories: list[Trajectory], *, seed: int, sample_size: int = 8,
) -> tuple[list[ProposalCandidate], dict[str, int]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    grouped = defaultdict(list)
    for trajectory in trajectories:
        if trajectory.environment != "webshop":
            raise ValueError("independent format warmup is WebShop-only")
        if trajectory.steps:
            grouped[trajectory.query_id].append(trajectory)
    # Spread a small format batch across tasks before taking extra trajectories.
    # Preserve rollout insertion order: random UUID lexicographic order must not
    # silently control otherwise seeded sampling.
    rng = random.Random(seed)
    groups = [list(group) for group in grouped.values()]
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(group)
    sources = []
    while groups and len(sources) < sample_size:
        for group in groups:
            sources.append(group.pop())
            if len(sources) == sample_size:
                break
        groups = [group for group in groups if group]
    candidates = []
    for source in sources:
        opposite_pool = [
            trajectory
            for trajectory in grouped[source.query_id]
            if trajectory.success != source.success
        ]
        opposite_reference = rng.choice(opposite_pool) if opposite_pool else None
        candidates.append(
            ProposalCandidate(
                proposal_id=str(uuid.uuid4()),
                source=source,
                success_anchor=None,
                failure_anchor=None,
                prompt=build_skill_proposal_prompt(
                    source.to_record(),
                    comparison_record=(
                        opposite_reference.to_record()
                        if opposite_reference is not None
                        else None
                    ),
                ),
                prompt_reference=opposite_reference,
            )
        )
    with_reference = sum(
        candidate.prompt_reference is not None for candidate in candidates
    )
    return candidates, {
        "groups": len(grouped), "candidates": len(candidates),
        "warmup_sampled": len(candidates),
        "warmup_sampled_successes": sum(source.success for source in sources),
        "warmup_without_anchors": len(candidates),
        "warmup_with_opposite_reference": with_reference,
        "warmup_without_opposite_reference": len(candidates) - with_reference,
    }
