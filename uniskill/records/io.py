from __future__ import annotations

import json
from pathlib import Path


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def episode_record_path(output_dir: str | Path, step: int) -> Path:
    return Path(output_dir) / "episode_records" / f"action_step_{step:06d}.jsonl"


def proposal_iteration_path(output_dir: str | Path, step: int) -> Path:
    return Path(output_dir) / "proposal_records" / f"proposal_iteration_{step:06d}.jsonl"
