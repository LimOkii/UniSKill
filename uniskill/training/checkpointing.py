from __future__ import annotations

import math
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def normalize_milestone_steps(raw_steps: Any) -> tuple[int, ...]:
    """Return sorted, unique, positive checkpoint milestone steps."""

    if raw_steps is None:
        return ()
    if isinstance(raw_steps, (str, bytes)):
        raise TypeError("trainer.milestone_save_steps must be a list of integers")

    normalized: set[int] = set()
    for raw_step in raw_steps:
        if isinstance(raw_step, bool):
            raise TypeError("checkpoint milestone steps must be integers")
        try:
            step = int(raw_step)
        except (TypeError, ValueError) as error:
            raise TypeError("checkpoint milestone steps must be integers") from error
        if step != raw_step or step <= 0:
            raise ValueError("checkpoint milestone steps must be positive integers")
        normalized.add(step)
    return tuple(sorted(normalized))


def prune_checkpoint_to_model_only(actor_dir: str | Path) -> tuple[Path, ...]:
    """Remove resumable training state while retaining FSDP model shards."""

    actor_path = Path(actor_dir)
    removed: list[Path] = []
    for pattern in (
        "optim_world_size_*_rank_*.pt",
        "extra_state_world_size_*_rank_*.pt",
    ):
        for path in actor_path.glob(pattern):
            path.unlink()
            removed.append(path)
    return tuple(removed)


def new_best_validation_score(
    metrics: Mapping[str, Any],
    *,
    current_best: float,
    minimum: float,
) -> float | None:
    """Return a strictly better eligible validation success rate."""

    raw_score = metrics.get("val/success_rate")
    if raw_score is None:
        return None
    score = float(raw_score)
    if not math.isfinite(score):
        return None
    if score <= minimum or score <= current_best:
        return None
    return score


def snapshot_skillbank(
    source_dir: str | Path,
    destination_dir: str | Path,
) -> Path:
    """Publish a complete SkillBank snapshot without exposing a partial copy."""

    source = Path(source_dir)
    destination = Path(destination_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"SkillBank directory does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"SkillBank snapshot already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(
            source,
            temporary,
            ignore=shutil.ignore_patterns("best", "milestones"),
        )
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination
