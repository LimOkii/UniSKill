from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uniskill.skillbank.alfworld import AlfWorldSkillBank


WEBSHOP_TASK_TYPE = "webshop"
DEFAULT_SKILL_DIR = Path(__file__).resolve().parents[1] / "skills" / WEBSHOP_TASK_TYPE
DEFAULT_EMBEDDING_DIR = (
    Path(__file__).resolve().parents[2] / "skill_embedding" / WEBSHOP_TASK_TYPE
)
DEFAULT_EMBEDDING_MIN_SCORE = 0.55
INITIAL_VARIANT_SELECTION_SKILL = (
    "On a product page, treat listed size, color, style, and pack values as "
    "selectable options. Select the required options, then buy promptly once "
    "the product and price match. Return to search only when a required option "
    "is unavailable or the product clearly fails a constraint."
)


class WebshopSkillBank(AlfWorldSkillBank):
    """A single-namespace UniSkill bank for cross-task WebShop experience."""

    def __init__(
        self,
        skill_dir: str | Path | None = DEFAULT_SKILL_DIR,
        retrieval_start_step: int = 2,
        min_score: float = 1.0,
        retrieval_method: str = "lexical",
        embedding_model_path: str | None = None,
        embedding_dir: str | Path | None = DEFAULT_EMBEDDING_DIR,
        embedding_min_score: float = DEFAULT_EMBEDDING_MIN_SCORE,
        seed_initial_skill: bool = False,
    ):
        self.skill_dir = Path(skill_dir or DEFAULT_SKILL_DIR)
        self.retrieval_start_step = retrieval_start_step
        self.min_score = min_score
        self.retrieval_method = retrieval_method
        self.embedding_model_path = embedding_model_path
        self.embedding_dir = Path(embedding_dir or DEFAULT_EMBEDDING_DIR)
        self.embedding_min_score = float(embedding_min_score)
        self.seed_initial_skill = bool(seed_initial_skill)
        if not -1.0 <= self.embedding_min_score <= 1.0:
            raise ValueError("WebShop embedding_min_score must be in [-1, 1]")
        self._embedding_backend = None
        self.skill_files = {WEBSHOP_TASK_TYPE: "skills.json"}
        self.task_to_skill = {WEBSHOP_TASK_TYPE: WEBSHOP_TASK_TYPE}
        self._ensure_skill_dir_initialized()
        self._validate_payload()

    def _ensure_skill_dir_initialized(self) -> None:
        self.skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = self.skill_dir / self.skill_files[WEBSHOP_TASK_TYPE]
        payload = self._read_json(skill_path, WEBSHOP_TASK_TYPE)
        initial_skill_id = self._skill_id(
            WEBSHOP_TASK_TYPE,
            INITIAL_VARIANT_SELECTION_SKILL,
            1,
        )
        if self.seed_initial_skill and not any(
            skill.get("skill_id") == initial_skill_id
            for skill in payload["skills"]
        ):
            payload["skills"].insert(
                0,
                {
                    "skill_id": initial_skill_id,
                    "skill_text": INITIAL_VARIANT_SELECTION_SKILL,
                },
            )
        if not skill_path.exists() or self.seed_initial_skill:
            self._write_json(skill_path, payload)

    def _retrieve_embedding_batch(
        self,
        *,
        task_descriptions: list[str],
        gamefiles: list[str | None],
        task_types: list[str | None] | None = None,
    ) -> list[dict]:
        """Retrieve WebShop skills with its independently configured threshold."""

        routed = []
        results = [None] * len(task_descriptions)
        for i, (query, gamefile) in enumerate(zip(task_descriptions, gamefiles)):
            task_type = self.get_task_type(
                task_type=task_types[i] if task_types else None,
                gamefile=gamefile,
            )
            file_name = self._file_for_task(task_type)
            if file_name is None:
                results[i] = self._no_skill(
                    "task_type_unrecognized",
                    task_type=task_type,
                    query=query,
                )
                continue

            skills = self._read_skills(file_name, task_type)
            if not skills:
                results[i] = self._no_skill(
                    "empty_candidate_pool",
                    task_type=task_type,
                    query=query,
                )
                continue
            routed.append((i, query, task_type, skills))

        if not routed:
            return results

        query_embeddings = self._embedding_backend_instance().encode_queries(
            [item[1] for item in routed]
        )
        batch_embedding_cache: dict[tuple, np.ndarray] = {}
        for query_idx, (env_idx, query, task_type, skills) in enumerate(routed):
            cache_key = (
                task_type,
                tuple(
                    (skill.get("skill_id"), self._text_hash(self._skill_text(skill)))
                    for skill in skills
                ),
            )
            if cache_key not in batch_embedding_cache:
                batch_embedding_cache[cache_key] = self._load_skill_embeddings(
                    task_type, skills
                )
            skill_embeddings = batch_embedding_cache[cache_key]
            scores = skill_embeddings @ query_embeddings[query_idx]
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_skill = skills[best_idx]
            if best_score < self.embedding_min_score:
                results[env_idx] = self._no_skill(
                    "below_threshold",
                    task_type=task_type,
                    score=best_score,
                    candidate_count=len(skills),
                    query=query,
                )
                continue

            results[env_idx] = {
                "skill": best_skill,
                "score": best_score,
                "task_type": task_type,
                "candidate_count": len(skills),
                "reason": "top1",
                "method": self.retrieval_method,
                "query": query,
                "selected_skill_id": best_skill.get("skill_id"),
                "cache_hit_count": len(skills),
                "cache_miss_count": 0,
            }

        return results

    def get_task_type(
        self,
        task_type: str | None = None,
        gamefile: str | None = None,
    ) -> str:
        del gamefile
        if task_type not in (None, WEBSHOP_TASK_TYPE):
            raise ValueError(f"Unsupported WebShop task_type: {task_type}")
        return WEBSHOP_TASK_TYPE

    def _skill_id(self, task_type: str, skill_text: str, index: int) -> str:
        alfworld_id = super()._skill_id(task_type, skill_text, index)
        return alfworld_id.replace("alfworld:", "webshop:", 1)

    def _validate_payload(self) -> None:
        path = self.skill_dir / self.skill_files[WEBSHOP_TASK_TYPE]
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("task_type") != WEBSHOP_TASK_TYPE:
            raise ValueError(f"Invalid WebShop skill bank: {path}")
