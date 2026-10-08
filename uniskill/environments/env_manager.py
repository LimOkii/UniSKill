from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory
from uniskill.environments.history import (
    build_recent_action_history_context,
    canonicalize_action_response,
)
from uniskill.environments.prompts.alfworld import (
    render_action_prompt,
)
from uniskill.response_format import contains_chinese
from uniskill.skillbank import AlfWorldSkillBank


def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        gamefile.append(info["extra.gamefile"] if "extra.gamefile" in info else None)
    return gamefile


def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        infos[i]["extra.gamefile"] = gamefile[i] if "extra.gamefile" in infos[i] else None
    return infos


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(
        self,
        envs,
        projection_f,
        config,
        skillbank: AlfWorldSkillBank | None = None,
        training_step: int = 0,
    ):
        self.memory = SimpleMemory()
        uniskill_config = config.get("uniskill", {})
        retrieval_config = uniskill_config.get("retrieval", {}) if uniskill_config else {}
        retrieval_start_step = int(retrieval_config.get("start_step", 2))
        self.skillbank = skillbank or AlfWorldSkillBank(
            skill_dir=uniskill_config.get("skill_dir"),
            retrieval_start_step=retrieval_start_step,
            retrieval_method=retrieval_config.get("method", "lexical"),
            embedding_model_path=retrieval_config.get("embedding_model_path"),
            embedding_dir=uniskill_config.get("embedding_dir"),
        )
        self.training_step = training_step
        self.retrieval_results = []
        self.retrieved_skills = []
        self.current_step_contexts: list[dict[str, Any]] = []
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        self.memory.reset(batch_size=len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)
        self.retrieve_skills()

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {"text": full_text_obs, "image": image_obs, "anchor": text_obs}, infos

    def step(self, text_actions: List[str]):
        raw_actions = list(text_actions)
        admissible_actions = [
            list(action_pool)
            for action_pool in self.envs.get_admissible_commands
        ]
        actions, valids = self.projection_f(text_actions, admissible_actions)
        canonical_actions = []
        for raw_action, action, valid, action_pool in zip(
            raw_actions,
            actions,
            valids,
            admissible_actions,
        ):
            canonical_actions.append(canonicalize_action_response(
                raw_response=raw_action,
                projected_action=action,
                format_valid=bool(valid),
                admissible_actions=action_pool,
            ))

        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store(
            {
                "text_obs": self.pre_text_obs,
                "action": [item.display for item in canonical_actions],
            }
        )
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        for i, info in enumerate(infos):
            canonical = canonical_actions[i]
            info["is_action_valid"] = to_numpy(valids[i])
            info["is_action_strict_format_valid"] = canonical.strict_format_valid
            info["action_strict_format_error"] = canonical.strict_format_error
            info["is_action_admissible"] = canonical.admissible
            info["is_action_overall_valid"] = canonical.overall_valid
            info["action_payload_parsed"] = canonical.payload_parsed
            info["canonical_action"] = canonical.payload
            info["canonical_action_display"] = canonical.display
            info["action_history_status"] = canonical.history_status
            info["action_contains_chinese"] = contains_chinese(raw_actions[i])
            lower_response = raw_actions[i].lower()
            info["action_missing_action_tags"] = not (
                "<action>" in lower_response and "</action>" in lower_response
            )
            info["action_missing_think_tags"] = not (
                "<think>" in raw_actions[i] and "</think>" in raw_actions[i]
            )

        next_observations = {"text": full_text_obs, "image": image_obs, "anchor": text_obs}
        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find("Your task is to: ")
            if task_start == -1:
                raise ValueError("Task description not found in text observation.")
            self.tasks.append(obs[task_start + len("Your task is to: "):].strip())

    def retrieve_skills(self):
        self.retrieval_results = self.skillbank.retrieve_batch(
            task_descriptions=self.tasks,
            gamefiles=self.gamefile,
            training_step=self.training_step,
        )
        self.retrieved_skills = [result["skill"] for result in self.retrieval_results]
        return self.retrieved_skills

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        postprocess_text_obs = []
        self.current_step_contexts = []

        for i in range(len(text_obs)):
            history = []
            if not init and self.config.env.history_length > 0:
                history = build_recent_action_history_context(
                    self.memory[i],
                    self.config.env.history_length,
                )
            context = {
                "task_description": self.tasks[i],
                "step_count": len(self.memory[i]),
                "current_step": len(self.memory[i]) + 1,
                "history": history,
                "current_observation": text_obs[i],
                "admissible_actions": [action for action in admissible_actions[i] if action != "help"],
                "original_skill": self.retrieved_skills[i],
            }
            self.current_step_contexts.append(context)
            postprocess_text_obs.append(render_action_prompt(context, self.retrieved_skills[i]))
        return postprocess_text_obs

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        batch_size = len(total_batch_list)

        success = defaultdict(list)
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        assert len(success["success_rate"]) == batch_size
        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                won_value = float(info["won"])
                success["success_rate"].append(won_value)

                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]

        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break
