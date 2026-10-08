# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from .utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from uniskill.records import episode_record_path, write_jsonl
from uniskill.response_format import (
    classify_response_termination,
    validate_tagged_response,
)
from uniskill.training.action_rewards import counterfactual_action_eligibility
from uniskill.training.token_masks import action_payload_token_mask
from uniskill.training.types import ActionStep, Trajectory

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the trajectory collector.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.step_gamma = float(config.algorithm.get("step_gamma", 0.95))
        self.enable_credit_assignment = bool(
            config.algorithm.get("credit_assignment", False)
        )
        self.tokenizer = tokenizer
        self.processor = processor
        self.iteration_trajectories: list[Trajectory] = []
        self.action_step = 0

    def save_episode_records(self) -> None:
        uniskill_config = self.config.get("uniskill", {})
        output_dir = uniskill_config.get("output_dir") if uniskill_config else None
        if not output_dir:
            return

        self.action_step += 1
        write_jsonl(
            episode_record_path(output_dir, self.action_step),
            [trajectory.to_record() for trajectory in self.iteration_trajectories],
        )

    def build_trajectories(
            self,
            total_batch_list: List[List[Dict]],
            total_infos: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            envs: EnvironmentManagerBase,
            ) -> list[Trajectory]:
        trajectories: list[Trajectory] = []
        success_values = success.get("success_rate", np.zeros(len(total_batch_list)))

        tasks = getattr(envs, "tasks", [None] * len(total_batch_list))
        gamefiles = getattr(envs, "gamefile", [None] * len(total_batch_list))
        retrieved_skills = getattr(envs, "retrieved_skills", [None] * len(total_batch_list))
        retrieval_results = getattr(envs, "retrieval_results", [None] * len(total_batch_list))
        task_types = getattr(envs, "task_types", None)
        environment_name = getattr(envs, "environment_name", None)
        if not environment_name:
            # ALFWorld's environment manager does not expose environment_name.
            # Preserve the configured name so environment-specific proposal
            # construction with held-out references is activated.
            environment_name = str(self.config.env.env_name)
        task_score_values = success.get("webshop_task_score (not success_rate)")
        skillbank = getattr(envs, "skillbank", None)

        for env_idx, steps in enumerate(total_batch_list):
            episode_steps: list[ActionStep] = []
            for step_idx, item in enumerate(steps):
                if not item["active_masks"]:
                    continue

                response_ids = item["responses"].detach().cpu().tolist()
                response_length = len(response_ids)
                response_mask = item["attention_mask"][-response_length:].detach().cpu().to(torch.int64).tolist()
                valid_response_ids = [token for token, keep in zip(response_ids, response_mask) if keep]
                action_raw = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                parsed = validate_tagged_response(action_raw)
                context = dict(item.get("step_context") or {})
                context["think"] = (
                    parsed.think
                    if bool(item.get("is_action_strict_format_valid", False))
                    else ""
                )
                canonical_action = str(item.get("canonical_action", ""))
                canonical_display = str(
                    item.get("canonical_action_display", canonical_action)
                )
                episode_steps.append(
                    ActionStep(
                        step_id=str(item["step_id"]),
                        query_id=str(item["uid"]),
                        trajectory_id=str(item["traj_uid"]),
                        step_index=step_idx + 1,
                        context=context,
                        response_ids=response_ids,
                        response_mask=response_mask,
                        action_token_mask=item["action_token_mask"].detach().cpu().to(torch.int64).tolist(),
                        response_text=action_raw,
                        action_text=canonical_action,
                        reward=float(item["rewards"]),
                        done=bool(item.get("dones", False)),
                        is_action_valid=bool(item.get("is_action_valid", True)),
                        is_action_strict_format_valid=bool(
                            item.get("is_action_strict_format_valid", False)
                        ),
                        action_strict_format_error=str(
                            item.get("action_strict_format_error", "")
                        ),
                        is_action_admissible=bool(
                            item.get("is_action_admissible", False)
                        ),
                        is_action_overall_valid=bool(
                            item.get("is_action_overall_valid", False)
                        ),
                        action_payload_parsed=bool(
                            item.get("action_payload_parsed", False)
                        ),
                        action_display=canonical_display,
                        action_history_status=str(
                            item.get("action_history_status", "")
                        ),
                        contains_chinese=bool(
                            item.get("action_contains_chinese", False)
                        ),
                        missing_action_tags=bool(
                            item.get("action_missing_action_tags", False)
                        ),
                        missing_think_tags=bool(
                            item.get("action_missing_think_tags", False)
                        ),
                        response_token_length=int(
                            item.get("action_response_token_length", 0)
                        ),
                        response_hit_length_limit=bool(
                            item.get("action_response_hit_length_limit", False)
                        ),
                        response_truncated=bool(
                            item.get("action_response_truncated", False)
                        ),
                    )
                )

            success_value = float(success_values[env_idx]) if env_idx < len(success_values) else 0.0
            is_truncation = episode_lengths[env_idx] >= self.config.env.max_steps and success_value < 1.0
            if is_truncation:
                end_reason = "truncation"
            elif success_value >= 1.0:
                end_reason = "success"
            else:
                end_reason = "failure"

            task_type = task_types[env_idx] if task_types is not None else None
            if task_type is None and skillbank is not None:
                task_type = skillbank.get_task_type(gamefile=gamefiles[env_idx])
            retrieval = dict(retrieval_results[env_idx] or {})
            retrieval.pop("skill", None)

            query_id = str(steps[0]["uid"]) if steps else ""
            trajectories.append(
                Trajectory(
                    query_id=query_id,
                    trajectory_id=str(traj_uid[env_idx]),
                    task_type=task_type,
                    task_description=tasks[env_idx],
                    gamefile=gamefiles[env_idx],
                    episode_reward=float(episode_rewards[env_idx]),
                    success=bool(success_value >= 1.0),
                    end_reason=end_reason,
                    retrieved_skill=retrieved_skills[env_idx],
                    retrieval=retrieval,
                    steps=episode_steps,
                    environment=environment_name,
                    task_score=(
                        float(task_score_values[env_idx])
                        if task_score_values is not None
                        and env_idx < len(task_score_values)
                        else None
                    ),
                )
            )

        return trajectories

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print("Warning: no text observation found")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            discounted_returns: np.ndarray | None = None,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for step_index, data in enumerate(total_batch_list[bs]):
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    if discounted_returns is None:
                        data['episode_rewards'] = episode_rewards[bs]
                    else:
                        data['episode_rewards'] = np.array(
                            episode_rewards[bs][step_index], dtype=np.float32
                        )
                        data['step_returns'] = torch.tensor(
                            float(discounted_returns[bs][step_index]),
                            dtype=torch.float32,
                        )
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    @staticmethod
    def credit_assignment(total_batch_list, step_gamma=0.95):
        """Compute backward-discounted environment-step returns."""
        total_episode_rewards = np.empty(len(total_batch_list), dtype=object)
        total_discounted_returns = np.empty(len(total_batch_list), dtype=object)
        for trajectory_index, steps in enumerate(total_batch_list):
            episode_rewards = np.zeros(len(steps), dtype=np.float64)
            discounted_returns = np.zeros(len(steps), dtype=np.float64)
            cumulative_reward = 0.0
            for step in steps:
                if step.get("active_masks", True):
                    value = step.get("rewards", 0.0)
                    cumulative_reward += float(
                        value.item() if hasattr(value, "item") else value
                    )
            running_return = 0.0
            for step_index in reversed(range(len(steps))):
                step = steps[step_index]
                if not step.get("active_masks", True):
                    continue
                value = step.get("rewards", 0.0)
                value = float(value.item() if hasattr(value, "item") else value)
                episode_rewards[step_index] = cumulative_reward
                running_return = value + float(step_gamma) * running_return
                discounted_returns[step_index] = running_return
            total_episode_rewards[trajectory_index] = episode_rewards
            total_discounted_returns[trajectory_index] = discounted_returns
        return total_episode_rewards, total_discounted_returns

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # Unpad generated sequences.
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)

            response_masks = batch.batch["attention_mask"][:, -batch.batch["responses"].shape[-1]:]
            response_limit = int(batch.batch["responses"].shape[-1])
            action_masks = []
            response_terminations = []
            for response_ids, response_mask in zip(batch.batch["responses"], response_masks):
                valid_length = int(response_mask.sum().item())
                valid_ids = response_ids[:valid_length].detach().cpu().tolist()
                response_terminations.append(
                    classify_response_termination(
                        valid_ids,
                        response_limit=response_limit,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                )
                mask = action_payload_token_mask(valid_ids, self.tokenizer)
                mask.extend([0] * (len(response_ids) - valid_length))
                action_masks.append(
                    torch.tensor(mask, dtype=torch.long, device=response_ids.device)
                    * response_mask.to(dtype=torch.long)
                )
            batch.batch["action_token_mask"] = torch.stack(action_masks, dim=0)
            batch.non_tensor_batch["step_id"] = np.array(
                [f"{traj_uid[index]}:{_step + 1}" for index in range(batch_size)],
                dtype=object,
            )
            contexts = getattr(envs, "current_step_contexts", [{} for _ in range(batch_size)])
            batch.non_tensor_batch["step_context"] = np.array(
                [dict(contexts[index]) for index in range(batch_size)],
                dtype=object,
            )
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            for info, termination in zip(infos, response_terminations):
                text_strict_valid = bool(
                    info.get("is_action_strict_format_valid", False)
                )
                strict_valid = text_strict_valid and termination.normal
                strict_error = str(info.get("action_strict_format_error", ""))
                if text_strict_valid and not termination.normal:
                    strict_error = "response reached length limit without normal termination"
                info["is_action_strict_format_valid"] = strict_valid
                info["action_strict_format_error"] = strict_error
                info["action_response_token_length"] = termination.token_length
                info["action_response_hit_length_limit"] = termination.hit_length_limit
                info["action_response_truncated"] = termination.truncated
                info["is_action_overall_valid"] = bool(
                    strict_valid and info.get("is_action_admissible", False)
                )

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)
            batch.non_tensor_batch['is_action_strict_format_valid'] = np.array(
                [info.get('is_action_strict_format_valid', False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch['action_strict_format_error'] = np.array(
                [info.get('action_strict_format_error', '') for info in infos],
                dtype=object,
            )
            batch.non_tensor_batch['is_action_admissible'] = np.array(
                [info.get('is_action_admissible', False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch['is_action_overall_valid'] = np.array(
                [info.get('is_action_overall_valid', False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch['action_payload_parsed'] = np.array(
                [info.get('action_payload_parsed', False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch['canonical_action'] = np.array(
                [info.get('canonical_action', '') for info in infos],
                dtype=object,
            )
            batch.non_tensor_batch['canonical_action_display'] = np.array(
                [info.get('canonical_action_display', '') for info in infos],
                dtype=object,
            )
            batch.non_tensor_batch['action_history_status'] = np.array(
                [info.get('action_history_status', '') for info in infos],
                dtype=object,
            )
            batch.non_tensor_batch['action_contains_chinese'] = np.array(
                [info.get('action_contains_chinese', False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch['action_missing_action_tags'] = np.array(
                [info.get('action_missing_action_tags', False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch['action_missing_think_tags'] = np.array(
                [info.get('action_missing_think_tags', False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch['action_response_token_length'] = np.array(
                [info.get('action_response_token_length', 0) for info in infos],
                dtype=np.int64,
            )
            batch.non_tensor_batch['action_response_hit_length_limit'] = np.array(
                [info.get('action_response_hit_length_limit', False) for info in infos],
                dtype=bool,
            )
            batch.non_tensor_batch['action_response_truncated'] = np.array(
                [info.get('action_response_truncated', False) for info in infos],
                dtype=bool,
            )
            payload_eligibility = counterfactual_action_eligibility(
                infos,
                require_overall_valid=bool(
                    getattr(
                        envs,
                        "counterfactual_requires_overall_valid_action",
                        False,
                    )
                ),
            )
            payload_eligibility_mask = torch.as_tensor(
                payload_eligibility,
                dtype=batch.batch["action_token_mask"].dtype,
                device=batch.batch["action_token_mask"].device,
            ).unsqueeze(-1)
            batch.batch["action_token_mask"] *= payload_eligibility_mask

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['dones'] = torch_to_numpy(dones, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )

        self.iteration_trajectories = self.build_trajectories(
                    total_batch_list=total_batch_list,
                    total_infos=total_infos,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    success=success,
                    traj_uid=traj_uid,
                    envs=envs,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        if is_train:
            self.save_episode_records()

        discounted_returns = None
        rollout_rewards = total_episode_rewards
        if self.enable_credit_assignment:
            rollout_rewards, discounted_returns = self.credit_assignment(
                total_batch_list, self.step_gamma
            )

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=rollout_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
            discounted_returns=discounted_returns,
        )
        
        return gen_batch_output
