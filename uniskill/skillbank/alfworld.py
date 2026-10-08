from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .embedding import QwenEmbeddingBackend
from .lexical import lexical_score


DEFAULT_SKILL_DIR = Path(__file__).resolve().parents[1] / "skills" / "alfworld"
DEFAULT_EMBEDDING_DIR = Path(__file__).resolve().parents[2] / "skill_embedding" / "alfworld"
EMBEDDING_RETRIEVAL_MIN_SCORE = 0.55
ALFWORLD_TASK_TYPES = (
    "pick_and_place_simple",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
)


class AlfWorldSkillBank:
    def __init__(
        self,
        skill_dir: str | Path | None = DEFAULT_SKILL_DIR,
        retrieval_start_step: int = 6,
        min_score: float = 1.0,
        retrieval_method: str = "lexical",
        embedding_model_path: str | None = None,
        embedding_dir: str | Path | None = DEFAULT_EMBEDDING_DIR,
    ):
        self.skill_dir = Path(skill_dir or DEFAULT_SKILL_DIR)
        self.retrieval_start_step = retrieval_start_step
        self.min_score = min_score
        self.retrieval_method = retrieval_method
        self.embedding_model_path = embedding_model_path
        self.embedding_dir = Path(embedding_dir or DEFAULT_EMBEDDING_DIR)
        self._embedding_backend = None

        self._ensure_skill_dir_initialized()
        with (self.skill_dir / "skill_mapping.json").open("r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.skill_files = mapping["skill_files"]
        self.task_to_skill = mapping["task_to_skill"]
        self._validate_task_namespaces()

    def _validate_task_namespaces(self) -> None:
        """Require one independent physical skillbank per ALFWorld task type."""

        expected_tasks = set(ALFWORLD_TASK_TYPES)
        configured_tasks = set(self.task_to_skill)
        if configured_tasks != expected_tasks:
            missing = sorted(expected_tasks - configured_tasks)
            extra = sorted(configured_tasks - expected_tasks)
            raise ValueError(
                "ALFWorld skill mapping must contain exactly the supported task types; "
                f"missing={missing}, extra={extra}"
            )

        skill_keys = [self.task_to_skill[task_type] for task_type in ALFWORLD_TASK_TYPES]
        if len(set(skill_keys)) != len(skill_keys):
            raise ValueError(
                "Each ALFWorld task type must have its own skill namespace"
            )
        if set(self.skill_files) != set(skill_keys):
            raise ValueError(
                "skill_mapping.skill_files must exactly match the task_to_skill namespaces"
            )

        file_names = [self.skill_files[skill_key] for skill_key in skill_keys]
        if len(set(file_names)) != len(file_names):
            raise ValueError(
                "Each ALFWorld task type must have its own skill file"
            )

        for task_type, file_name in zip(ALFWORLD_TASK_TYPES, file_names):
            path = self.skill_dir / file_name
            if not path.exists():
                raise ValueError(f"skill file for {task_type} does not exist: {path}")
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("task_type") != task_type:
                raise ValueError(
                    f"skill file {path} declares task_type={payload.get('task_type')!r}, "
                    f"expected {task_type!r}"
                )

    def _ensure_skill_dir_initialized(self) -> None:
        mapping_path = self.skill_dir / "skill_mapping.json"
        if mapping_path.exists():
            return
        if not DEFAULT_SKILL_DIR.exists():
            raise FileNotFoundError(f"Default skill directory not found: {DEFAULT_SKILL_DIR}")
        self.skill_dir.mkdir(parents=True, exist_ok=True)
        for src in DEFAULT_SKILL_DIR.iterdir():
            dst = self.skill_dir / src.name
            if src.is_dir():
                if not dst.exists():
                    shutil.copytree(src, dst)
            elif not dst.exists():
                shutil.copy2(src, dst)

    def retrieve(
        self,
        *,
        task_type: str | None = None,
        task_description: str = "",
        gamefile: str | None = None,
        training_step: int = 0,
    ) -> dict:
        return self.retrieve_batch(
            task_descriptions=[task_description],
            gamefiles=[gamefile],
            training_step=training_step,
            task_types=[task_type],
        )[0]

    def retrieve_batch(
        self,
        *,
        task_descriptions: list[str],
        gamefiles: list[str | None],
        training_step: int = 0,
        task_types: list[str | None] | None = None,
    ) -> list[dict]:
        if training_step < self.retrieval_start_step:
            return [
                self._no_skill("retrieval_not_started", query=query)
                for query in task_descriptions
            ]

        if self.retrieval_method == "embedding":
            return self._retrieve_embedding_batch(
                task_descriptions=task_descriptions,
                gamefiles=gamefiles,
                task_types=task_types,
            )

        return [
            self._retrieve_lexical(
                task_description=query,
                gamefile=gamefile,
                task_type=task_types[i] if task_types else None,
            )
            for i, (query, gamefile) in enumerate(zip(task_descriptions, gamefiles))
        ]

    def _retrieve_lexical(
        self,
        *,
        task_description: str,
        gamefile: str | None = None,
        task_type: str | None = None,
    ) -> dict:
        task_type = self.get_task_type(task_type=task_type, gamefile=gamefile)
        file_name = self._file_for_task(task_type)
        if file_name is None:
            return self._no_skill("task_type_unrecognized", task_type=task_type, query=task_description)

        skills = self._read_skills(file_name, task_type)
        if not skills:
            return self._no_skill("empty_candidate_pool", task_type=task_type, query=task_description)

        best_skill = None
        best_score = 0.0
        for skill in skills:
            score = lexical_score(task_description, self._skill_text(skill))
            if score > best_score:
                best_skill = skill
                best_score = score

        if best_skill is None or best_score < self.min_score:
            return self._no_skill(
                "below_threshold",
                task_type=task_type,
                score=best_score,
                candidate_count=len(skills),
                query=task_description,
            )

        return {
            "skill": best_skill,
            "score": best_score,
            "task_type": task_type,
            "candidate_count": len(skills),
            "reason": "top1",
            "method": self.retrieval_method,
            "query": task_description,
            "selected_skill_id": best_skill.get("skill_id"),
            "cache_hit_count": 0,
            "cache_miss_count": 0,
        }

    def _retrieve_embedding_batch(
        self,
        *,
        task_descriptions: list[str],
        gamefiles: list[str | None],
        task_types: list[str | None] | None = None,
    ) -> list[dict]:
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
                batch_embedding_cache[cache_key] = self._load_skill_embeddings(task_type, skills)
            skill_embeddings = batch_embedding_cache[cache_key]
            scores = skill_embeddings @ query_embeddings[query_idx]
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_skill = skills[best_idx]
            if best_score < EMBEDDING_RETRIEVAL_MIN_SCORE:
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

    def append_skill(
        self,
        *,
        task_type: str,
        skill_text: str,
    ) -> dict:
        file_name = self._file_for_task(task_type)
        if file_name is None:
            raise ValueError(f"Unsupported ALFWorld task_type: {task_type}")

        path = self.skill_dir / file_name
        payload = self._read_json(path, task_type)
        skills = payload["skills"]

        skill = {
            "skill_id": self._skill_id(task_type, skill_text, len(skills) + 1),
            "skill_text": " ".join(skill_text.split()),
        }
        skills.append(skill)
        self._write_json(path, payload)
        return skill

    def update_skill(
        self,
        *,
        task_type: str,
        skill_id: str,
        skill_text: str,
    ) -> dict:
        file_name = self._file_for_task(task_type)
        if file_name is None:
            raise ValueError(f"Unsupported ALFWorld task_type: {task_type}")
        if not skill_id:
            raise ValueError("UPDATE_SKILL requires a skill_id")

        path = self.skill_dir / file_name
        payload = self._read_json(path, task_type)
        for skill in payload["skills"]:
            if skill.get("skill_id") == skill_id:
                skill.clear()
                skill.update(
                    {
                        "skill_id": skill_id,
                        "skill_text": " ".join(skill_text.split()),
                    }
                )
                self._write_json(path, payload)
                return skill

        raise ValueError(f"Skill not found: {skill_id}")

    def contains_exact_skill(self, task_type: str, skill_text: str) -> bool:
        """Return whether the normalized plain-text skill already exists."""

        file_name = self._file_for_task(task_type)
        if not file_name:
            return False
        expected = " ".join(str(skill_text).split())
        return any(
            " ".join(self._skill_text(skill).split()) == expected
            for skill in self._read_skills(file_name, task_type)
        )

    def persist_skill_embeddings(self, items: list[dict]) -> None:
        if not items:
            return

        texts = [self._skill_text(item["skill"]) for item in items]
        embeddings = self._embedding_backend_instance().encode_skills(texts)
        for item, embedding in zip(items, embeddings):
            task_type = item["task_type"]
            skill = item["skill"]
            vector_path, metadata_path = self._embedding_paths(task_type, skill)
            vector_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(vector_path, embedding.astype(np.float32))
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "skill_id": skill.get("skill_id", ""),
                        "task_type": task_type,
                        "skill_key": self._skill_key_for_task(task_type),
                        "text_hash": self._text_hash(self._skill_text(skill)),
                        "model_path": self.embedding_model_path,
                        "skill_text": self._skill_text(skill),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
                f.write("\n")

    def get_task_type(self, task_type: str | None = None, gamefile: str | None = None) -> str | None:
        if task_type:
            return task_type
        if not gamefile:
            return None

        traj_path = Path(gamefile).parent / "traj_data.json"
        if not traj_path.exists():
            return None
        with traj_path.open("r", encoding="utf-8") as f:
            return json.load(f).get("task_type")
        return None

    def _file_for_task(self, task_type: str | None) -> str | None:
        if task_type is None:
            return None
        skill_key = self.task_to_skill.get(task_type)
        return self.skill_files.get(skill_key) if skill_key else None

    def _read_skills(self, file_name: str, task_type: str) -> list[dict]:
        return self._read_json(self.skill_dir / file_name, task_type)["skills"]

    def _read_json(self, path: Path, task_type: str) -> dict:
        if not path.exists():
            return {"task_type": task_type, "skills": []}
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("task_type", task_type)
        data.setdefault("skills", [])
        return data

    def _write_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def _no_skill(
        self,
        reason: str,
        *,
        task_type: str | None = None,
        score: float = 0.0,
        candidate_count: int = 0,
        query: str = "",
    ) -> dict:
        return {
            "skill": None,
            "score": score,
            "task_type": task_type,
            "candidate_count": candidate_count,
            "reason": reason,
            "method": self.retrieval_method,
            "query": query,
            "selected_skill_id": None,
            "cache_hit_count": 0,
            "cache_miss_count": 0,
        }

    def _skill_text(self, skill: dict) -> str:
        text = str(skill.get("skill_text") or "").strip()
        if text:
            return text
        return " ".join(
            str(skill.get(key) or "").strip()
            for key in ("title", "principle", "when_to_apply")
        ).strip()

    def _skill_id(self, task_type: str, skill_text: str, index: int) -> str:
        raw = f"{task_type}|{index}|{skill_text}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        return f"alfworld:{task_type}:{digest}"

    def _embedding_backend_instance(self) -> QwenEmbeddingBackend:
        if self._embedding_backend is None:
            self._embedding_backend = QwenEmbeddingBackend(model_path=self.embedding_model_path)
        return self._embedding_backend

    def _load_skill_embeddings(self, task_type: str, skills: list[dict]) -> np.ndarray:
        missing = []
        for skill in skills:
            vector_path, metadata_path = self._embedding_paths(task_type, skill)
            if not self._embedding_cache_is_valid(vector_path, metadata_path, skill):
                missing.append({"task_type": task_type, "skill": skill})
        if missing:
            self.persist_skill_embeddings(missing)

        embeddings = []
        for skill in skills:
            vector_path, metadata_path = self._embedding_paths(task_type, skill)
            embeddings.append(np.load(vector_path).astype(np.float32))
        return np.stack(embeddings, axis=0)

    def _embedding_cache_is_valid(
        self,
        vector_path: Path,
        metadata_path: Path,
        skill: dict,
    ) -> bool:
        if not vector_path.exists() or not metadata_path.exists():
            return False
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            if metadata.get("text_hash") != self._text_hash(self._skill_text(skill)):
                return False
            if metadata.get("model_path") != self.embedding_model_path:
                return False
            vector = np.load(vector_path, mmap_mode="r")
            return vector.ndim == 1 and vector.size > 0
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def _embedding_paths(self, task_type: str, skill: dict) -> tuple[Path, Path]:
        basename = self._embedding_basename(skill)
        task_key = self._skill_key_for_task(task_type)
        root = self.embedding_dir / task_key
        return root / "vectors" / f"{basename}.npy", root / "metadata" / f"{basename}.json"

    def _embedding_basename(self, skill: dict) -> str:
        skill_id_hash = self._text_hash(skill.get("skill_id", ""))
        text_hash = self._text_hash(self._skill_text(skill))
        return f"skill_{skill_id_hash}_{text_hash}"

    def _skill_key_for_task(self, task_type: str | None) -> str:
        if task_type is None:
            return "unknown"
        return self.task_to_skill.get(task_type, task_type)

    def _text_hash(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
