from __future__ import annotations

from agent_system.environments.env_manager import WebshopEnvironmentManager as NativeWebshopManager
from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE, WEBSHOP_TEMPLATE_NO_HIS
from uniskill.environments.prompts.webshop_native import render_action_prompt
from uniskill.environments.webshop_actions import canonicalize_webshop_action_response
from uniskill.response_format import contains_chinese
from uniskill.webshop_state import format_selected_options


class NativeActorWebshopEnvironmentManager(NativeWebshopManager):
    """Native transitions/history with UniSkill metadata and gated skill retrieval.

    Strict validity is recorded only for diagnostics and counterfactual anchors;
    it does not reject an action, change native history, or change Actor reward.
    """

    environment_name = "webshop"
    counterfactual_requires_overall_valid_action = True

    def __init__(self, envs, projection_f, config, *, skillbank):
        self.skillbank = skillbank
        self.training_step = 0
        self.retrieved_skills = []
        self.retrieval_results = []
        self.current_step_contexts = []
        self.current_available_actions = []
        super().__init__(envs, projection_f, config)

    def retrieve_skills(self):
        if self.training_step <= int(self.config.uniskill.r_align_warmup_steps):
            # Skip skill retrieval during warmup.
            self.retrieval_results = [
                {"skill": None, "reason": "proposal_warmup", "score": None}
                for _ in self.tasks
            ]
        else:
            self.retrieval_results = self.skillbank.retrieve_batch(
                task_descriptions=self.tasks,
                gamefiles=self.gamefile,
                task_types=self.task_types,
                training_step=self.training_step,
            )
        self.retrieved_skills = [item["skill"] for item in self.retrieval_results]

    def build_text_obs(self, text_obs, infos, init=False):
        # Use the configured limit for native prompts.
        native_prompts = []
        selected_options_by_index = []
        current_observations = []
        history_length = int(self.config.env.history_length)
        if not init and history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                history_length, obs_key="text_obs", action_key="action"
            )
        char_limit = int(
            self.config.uniskill.webshop.get("native_prompt_char_limit", 13000)
        )
        for index, observation in enumerate(text_obs):
            selected_options = dict(infos[index].get("selected_options") or {})
            selected_options_text = format_selected_options(selected_options)
            if selected_options_text:
                observation = f"{observation}\n{selected_options_text}"
            selected_options_by_index.append(selected_options)
            current_observations.append(observation)
            available_actions = self.format_avail_actions(infos[index]["available_actions"])
            formatted_actions = "\n".join(f"'{action}'," for action in available_actions)
            if init or history_length <= 0:
                prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[index],
                    current_observation=observation,
                    available_actions=formatted_actions,
                )
            else:
                prompt = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[index],
                    step_count=len(self.memory[index]),
                    history_length=valid_lens[index],
                    action_history=memory_contexts[index],
                    current_step=len(self.memory[index]) + 1,
                    current_observation=observation,
                    available_actions=formatted_actions,
                )
                if char_limit > 0 and len(prompt) > char_limit:
                    print(f"Warning: WebShop prompt length {len(prompt)} exceeds {char_limit}")
                    prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[index],
                        current_observation=observation,
                        available_actions=formatted_actions,
                    )
            native_prompts.append(prompt)
        if init:
            self.task_types = ["webshop"] * len(text_obs)
            self.gamefile = [None] * len(text_obs)
            self.retrieve_skills()
        self.current_available_actions = [
            self.format_avail_actions(info["available_actions"]) for info in infos
        ]
        self.current_step_contexts = []
        for index, prompt in enumerate(native_prompts):
            memory = [] if init else self.memory[index]
            recent = memory[-int(self.config.env.history_length):]
            history = [
                {"step": len(memory) - len(recent) + offset + 1,
                 "observation": item["text_obs"], "action": item["action"]}
                for offset, item in enumerate(recent)
            ]
            self.current_step_contexts.append({
                "environment": "webshop",
                "native_action_prompt": prompt,
                "task_description": self.tasks[index],
                "step_count": len(memory),
                "current_step": len(memory) + 1,
                "current_observation": current_observations[index],
                "admissible_actions": self.current_available_actions[index],
                "history": history,
                "original_skill": self.retrieved_skills[index],
                "selected_options": selected_options_by_index[index],
            })
        return [
            render_action_prompt(context, skill)
            for context, skill in zip(self.current_step_contexts, self.retrieved_skills)
        ]

    def step(self, text_actions):
        raw_actions = list(text_actions)
        action_pools = self.current_available_actions
        actions, valids = self.projection_f(list(raw_actions))
        observations, rewards, dones, infos = super().step(list(raw_actions))
        for raw, action, valid, pool, info in zip(raw_actions, actions, valids, action_pools, infos):
            canonical = canonicalize_webshop_action_response(
                raw_response=raw, projected_action=action,
                format_valid=bool(valid), admissible_actions=pool,
            )
            info.update({
                "is_action_strict_format_valid": canonical.strict_format_valid,
                "action_strict_format_error": canonical.strict_format_error,
                "is_action_admissible": canonical.admissible,
                "is_action_overall_valid": canonical.overall_valid,
                "action_payload_parsed": canonical.payload_parsed,
                "canonical_action": canonical.payload,
                "canonical_action_display": action,
                "action_history_status": canonical.history_status,
                "action_contains_chinese": contains_chinese(raw),
                "action_missing_action_tags": not ("<action>" in raw.lower() and "</action>" in raw.lower()),
                "action_missing_think_tags": not ("<think>" in raw and "</think>" in raw),
            })
        return observations, rewards, dones, infos

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Keep native Succ/Score aggregation and add retrieval diagnostics only.
        super()._process_batch(batch_idx, total_batch_list, total_infos, success)
        result = self.retrieval_results[batch_idx]
        success["webshop_skill_retrieval_ratio"].append(float(result.get("skill") is not None))
        success["webshop_skill_retrieval_score"].append(float(result.get("score") or 0.0))
        success["webshop_skill_below_threshold_ratio"].append(float(result.get("reason") == "below_threshold"))
