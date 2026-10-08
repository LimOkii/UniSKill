# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import shutil
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from gigpo import core_gigpo

from uniskill.config import UniSkillSettings
from uniskill.webshop_native import uses_native_webshop_actor, strict_actor_credit_active
from uniskill.training.webshop_warmup import build_warmup_candidates
from uniskill.critic.service import CriticService
from uniskill.multi_turn_rollout import TrajectoryCollector, adjust_batch
from uniskill.records import proposal_iteration_path, write_jsonl
from uniskill.training.anchors import build_proposal_candidates, validate_candidate_roles
from uniskill.training.advantage_constraints import (
    cap_invalid_positive_advantages,
    invalid_positive_advantage_ratio,
)
from uniskill.training.action_rewards import (
    action_penalty_masks,
    action_penalty_metrics,
)
from uniskill.training.batches import (
    capture_generation_prompt_metadata,
    apply_chat_template,
    build_generation_batch,
    build_teacher_forcing_batch,
)
from uniskill.training.critic_routing import (
    apply_r_align_warmup_route,
    is_r_align_warmup,
    proposal_mode_scale,
)
from uniskill.training.coverage import alignment_coverage_metrics
from uniskill.training.checkpointing import (
    new_best_validation_score,
    normalize_milestone_steps,
    prune_checkpoint_to_model_only,
    snapshot_skillbank,
)
from uniskill.training.debug_trace import DebugTracePrinter
from uniskill.training.exact_duplicates import apply_exact_duplicate_routing
from uniskill.training.metrics import (
    action_advantage_metrics,
    action_history_metrics,
    action_rollout_health_metrics,
    format_uniskill_step_metrics,
    proposal_critic_action_metrics,
    proposal_response_metrics,
)
from uniskill.training.proposal_advantage import (
    PROPOSAL_CHANNELS,
    PROPOSAL_JOINT_CHANNEL,
    compute_channel_reinforce_plus_plus_advantage,
    proposal_channel_metrics,
)
from uniskill.training.proposal_pipeline import (
    assign_composite_policy_rewards,
    apply_alignment_scores,
    apply_critic_routing,
    build_counterfactual_samples,
    build_proposal_training_batch,
    decode_and_parse_candidates,
    original_step_logprob_scores,
)
from uniskill.training.skill_commit import commit_candidates

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name} needs {num_gpus} GPU(s) on {num_nodes} more node(s)")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def apply_invalid_action_penalty(
    data: DataProto,
    invalid_action_penalty_coef=float,
    *,
    replace_strict_format_invalid_score: bool = False,
):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    strict_format_invalid, _, action_penalties = action_penalty_masks(
        data.non_tensor_batch
    )
    if len(action_penalties) != len(data):
        raise ValueError("action penalty metadata length mismatch")
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_penalty = float(action_penalties[i])
        if replace_strict_format_invalid_score and strict_format_invalid[i]:
            # A structurally invalid response cannot claim task credit. Remove
            # every task-score contribution and assign one fixed negative score.
            reward_tensor[i].zero_()
            reward_tensor[i, valid_response_length - 1] = (
                -invalid_action_penalty_coef
            )
        else:
            # Apply one penalty when the response format is invalid or its
            # payload is outside the current admissible action list. Never
            # double-penalize.
            reward_tensor[i, valid_response_length - 1] -= (
                invalid_action_penalty_coef * action_penalty
            )

        if 'step_rewards' in data.batch.keys():
            if replace_strict_format_invalid_score and strict_format_invalid[i]:
                step_rewards[i] = -invalid_action_penalty_coef
            else:
                step_rewards[i] -= invalid_action_penalty_coef * action_penalty

    metrics = action_penalty_metrics(data.non_tensor_batch)
    metrics["episode/action_format_score_override_ratio"] = float(
        strict_format_invalid.mean()
        if replace_strict_format_invalid_score
        else 0.0
    )
    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing
            step_rewards=data.batch['step_rewards'], # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError(f"Unsupported advantage estimator: {adv_estimator}")
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class UniSkillRayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
        uniskill_settings: UniSkillSettings | None = None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector
        self.uniskill_settings = uniskill_settings or UniSkillSettings.from_config(config)
        self.debug_trace = DebugTracePrinter(
            self.uniskill_settings.debug,
            tokenizer=self.tokenizer,
            apply_chat_template_kwargs=self.config.data.get("apply_chat_template_kwargs", {}),
        )
        self._last_anchor_stats: dict[str, int] = {}
        self._last_coverage_stats: dict[str, float] = {}
        self._best_val_success_rate = (
            self.uniskill_settings.best_checkpoint_min_success_rate
        )
        self._best_checkpoint_path: str | None = None
        self._milestone_checkpoint_steps = frozenset(
            normalize_milestone_steps(
                self.config.trainer.get("milestone_save_steps", None)
            )
        )
        print(
            "[UniSkill debug] "
            f"enabled={self.uniskill_settings.debug.enabled} "
            f"samples_per_batch={self.uniskill_settings.debug.samples_per_batch} "
            f"max_action_steps={self.uniskill_settings.debug.max_action_steps} "
            f"show_token_ids={self.uniskill_settings.debug.show_token_ids}",
            flush=True,
        )
        print(
            "[UniSkill R_align warmup] "
            f"global_steps=1..{self.uniskill_settings.r_align_warmup_steps} "
            f"proposal_lambda={self.uniskill_settings.proposal_warmup_lambda} "
            f"first_full_alignment_step={self.uniskill_settings.r_align_warmup_steps + 1} "
            f"full_proposal_lambda={self.uniskill_settings.lambda_proposal}",
            flush=True,
        )
        print(
            "[UniSkill R_align coverage gate] "
            "scope=ADD_NEW_SKILL/UPDATE_SKILL "
            f"minimum_success_failure_anchor_coverage={self.uniskill_settings.min_anchor_coverage} "
            "under_threshold=mask-before-critic",
            flush=True,
        )
        print(
            "[UniSkill proposal reference mode] "
            "mode=held_out_opposite "
            "held_out_opposite_requires_four_distinct_trajectories=true",
            flush=True,
        )
        print(
            "[UniSkill best checkpoint] "
            f"metric=val/success_rate "
            f"strict_minimum={self.uniskill_settings.best_checkpoint_min_success_rate} "
            "policy=save-strict-improvement-and-replace-previous-best",
            flush=True,
        )
        print(
            "[UniSkill milestone checkpoints] "
            f"steps={sorted(self._milestone_checkpoint_steps)} "
            "contents=model-only-fsdp-and-skillbank",
            flush=True,
        )
        self.critic_service = CriticService(
            config_path=self.uniskill_settings.critic.config_path,
            max_attempts=self.uniskill_settings.critic.max_attempts,
            parse_retry_attempts=self.uniskill_settings.critic.parse_retry_attempts,
            concurrency=self.uniskill_settings.critic.concurrency,
            initial_backoff_seconds=self.uniskill_settings.critic.initial_backoff_seconds,
            max_backoff_seconds=self.uniskill_settings.critic.max_backoff_seconds,
            jitter_seconds=self.uniskill_settings.critic.jitter_seconds,
            rate_limit_backoff_seconds=(
                self.uniskill_settings.critic.rate_limit_backoff_seconds
            ),
        )

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # With LoRA, use the actor without the adapter as the reference policy.
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError(
                f"Unsupported advantage estimator: {self.config.algorithm.adv_estimator}"
            )

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] Set either '{name}.{param}' or "
                        f"'{name}.{param_per_gpu}', not both."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        if hasattr(self.val_envs, "training_step"):
            self.val_envs.training_step = self.global_steps
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}
        validation_action_health = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                                                    gen_batch=test_gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.val_envs,
                                                    is_train=False,
                                                    )
            validation_action_health.append(
                action_rollout_health_metrics(
                    self.traj_collector.iteration_trajectories,
                    prefix="val/action",
                )
            )
            print('validation generation end')
            del test_batch
            test_batch = test_output_gen_batch
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            # success rate
            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    # all success_rate should be the same
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # evaluate tool call based on data source
        # the values in tool_callings represent the tool call count for each trajectory; however, since the batch is expanded by step, we only need to take one value for each unique trajectories.
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v

        if validation_action_health:
            for key in validation_action_health[0]:
                metric_dict[key] = float(
                    np.mean(
                        [item[key] for item in validation_action_health]
                    )
                )

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError("Actor rollout requires the hybrid engine")

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _save_milestone_checkpoint(self) -> None:
        """Save evaluation-only model shards and the matching SkillBank."""

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir,
            "milestones",
            f"global_step_{self.global_steps}",
        )
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        skill_snapshot_path = os.path.join(
            self.uniskill_settings.skill_dir,
            "milestones",
            f"global_step_{self.global_steps}",
        )
        if os.path.exists(local_global_step_folder):
            raise FileExistsError(
                f"milestone checkpoint already exists: {local_global_step_folder}"
            )
        if os.path.exists(skill_snapshot_path):
            raise FileExistsError(
                f"milestone SkillBank snapshot already exists: {skill_snapshot_path}"
            )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            None,
            self.global_steps,
            max_ckpt_to_keep=None,
        )
        removed_training_state = prune_checkpoint_to_model_only(actor_local_path)
        snapshot_skillbank(
            self.uniskill_settings.skill_dir,
            skill_snapshot_path,
        )

        metadata = {
            "global_step": self.global_steps,
            "checkpoint_path": local_global_step_folder,
            "skill_snapshot_path": skill_snapshot_path,
            "model_only": True,
            "resumable": False,
            "removed_training_state_files": len(removed_training_state),
        }
        for metadata_path in (
            os.path.join(local_global_step_folder, "milestone.json"),
            os.path.join(skill_snapshot_path, "milestone.json"),
        ):
            with open(metadata_path, "w", encoding="utf-8") as file:
                json.dump(metadata, file, ensure_ascii=False, indent=2)

        print(
            "[UNISKILL MILESTONE CHECKPOINT] "
            f"step={self.global_steps} path={local_global_step_folder} "
            f"skill_snapshot={skill_snapshot_path} model_only=true",
            flush=True,
        )

    def _maybe_save_best_checkpoint(
        self,
        val_metrics: dict,
        metrics: dict,
    ) -> bool:
        score = new_best_validation_score(
            val_metrics,
            current_best=self._best_val_success_rate,
            minimum=self.uniskill_settings.best_checkpoint_min_success_rate,
        )
        metrics["checkpoint/best_val_success_rate"] = self._best_val_success_rate
        metrics["checkpoint/best_saved"] = 0.0
        if score is None:
            return False

        best_root = os.path.join(
            self.config.trainer.default_local_dir,
            "best",
        )
        new_step_path = os.path.join(
            best_root,
            f"global_step_{self.global_steps}",
        )
        skill_best_root = os.path.join(
            self.uniskill_settings.skill_dir,
            "best",
        )
        skill_snapshot_path = os.path.join(
            skill_best_root,
            f"global_step_{self.global_steps}",
        )
        actor_local_path = os.path.join(new_step_path, "actor")
        try:
            self.actor_rollout_wg.save_checkpoint(
                actor_local_path,
                None,
                self.global_steps,
                max_ckpt_to_keep=None,
            )
            os.makedirs(new_step_path, exist_ok=True)
            torch.save(
                self.train_dataloader.state_dict(),
                os.path.join(new_step_path, "data.pt"),
            )
            snapshot_skillbank(
                self.uniskill_settings.skill_dir,
                skill_snapshot_path,
            )
        except Exception:
            if os.path.isdir(new_step_path):
                shutil.rmtree(new_step_path)
            if os.path.isdir(skill_snapshot_path):
                shutil.rmtree(skill_snapshot_path)
            raise

        best_metadata = {
            "global_step": self.global_steps,
            "val_success_rate": score,
            "checkpoint_path": new_step_path,
            "skill_snapshot_path": skill_snapshot_path,
        }
        os.makedirs(best_root, exist_ok=True)
        with open(
            os.path.join(best_root, "best_validation.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(best_metadata, file, ensure_ascii=False, indent=2)
        os.makedirs(skill_best_root, exist_ok=True)
        with open(
            os.path.join(skill_best_root, "best_validation.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(best_metadata, file, ensure_ascii=False, indent=2)

        previous_path = self._best_checkpoint_path
        previous_skill_snapshot_path = None
        if previous_path:
            previous_skill_snapshot_path = os.path.join(
                skill_best_root,
                os.path.basename(previous_path),
            )
        self._best_val_success_rate = score
        self._best_checkpoint_path = new_step_path
        if (
            previous_path
            and os.path.abspath(previous_path) != os.path.abspath(new_step_path)
            and os.path.isdir(previous_path)
        ):
            shutil.rmtree(previous_path)
        if (
            previous_skill_snapshot_path
            and os.path.abspath(previous_skill_snapshot_path)
            != os.path.abspath(skill_snapshot_path)
            and os.path.isdir(previous_skill_snapshot_path)
        ):
            shutil.rmtree(previous_skill_snapshot_path)

        metrics["checkpoint/best_val_success_rate"] = score
        metrics["checkpoint/best_saved"] = 1.0
        print(
            "[UNISKILL BEST CHECKPOINT] "
            f"step={self.global_steps} val/success_rate={score:.6f} "
            f"path={new_step_path} skill_snapshot={skill_snapshot_path} "
            f"replaced={previous_path!r}",
            flush=True,
        )
        return True

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _generate_proposals(self, candidates):
        prompts = [candidate.prompt for candidate in candidates]
        prompt_batch = build_generation_batch(
            prompts,
            tokenizer=self.tokenizer,
            max_prompt_length=self.uniskill_settings.proposal_max_prompt_length,
            do_sample=True,
            ids=[candidate.proposal_id for candidate in candidates],
            apply_chat_template_kwargs=self.config.data.get("apply_chat_template_kwargs", {}),
            assistant_prefill=self.uniskill_settings.proposal_assistant_prefill,
        )
        prompt_metadata = capture_generation_prompt_metadata(prompt_batch)
        padded, pad_size = pad_dataproto_to_divisor(
            prompt_batch,
            self.actor_rollout_wg.world_size,
        )
        padded.meta_info["uniskill_response_length"] = (
            self.uniskill_settings.proposal_max_response_length
        )
        generated = self.actor_rollout_wg.generate_proposal_sequences(padded)
        return unpad_dataproto(generated, pad_size=pad_size), prompt_metadata

    def _compute_log_prob_padded(self, data: DataProto) -> DataProto:
        padded, pad_size = pad_dataproto_to_divisor(data, self.actor_rollout_wg.world_size)
        output = self.actor_rollout_wg.compute_log_prob(padded)
        return unpad_dataproto(output, pad_size=pad_size)

    def _remove_overlong_scorer_samples(self, samples, candidates):
        valid = []
        available_anchors: dict[str, set[str]] = defaultdict(set)
        chat_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        max_length = int(self.config.data.max_prompt_length)
        for sample in samples:
            text = apply_chat_template(self.tokenizer, sample.prompt, chat_kwargs)
            length = len(self.tokenizer.encode(text, add_special_tokens=False))
            if length > max_length:
                continue
            valid.append(sample)
            available_anchors[str(sample.metadata["proposal_id"])].add(
                str(sample.metadata["anchor_type"])
            )
        invalid_ids = {
            candidate.proposal_id
            for candidate in candidates
            if candidate.critic_content_supported is True
            and candidate.proposal_action in {"ADD_NEW_SKILL", "UPDATE_SKILL"}
            and available_anchors.get(candidate.proposal_id, set())
            != {"success", "failure"}
        }
        if invalid_ids:
            valid = [
                sample
                for sample in valid
                if sample.metadata["proposal_id"] not in invalid_ids
            ]
            for candidate in candidates:
                if candidate.proposal_id in invalid_ids:
                    candidate.skill_reward = None
                    if not candidate.critic_error:
                        candidate.critic_error = (
                            "one or more anchors have no scorable action prompt within "
                            "max_prompt_length"
                        )
        return valid

    def _pad_proposal_training_batch(self, batch: DataProto) -> DataProto:
        padded, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
        if pad_size:
            padding_slice = slice(len(padded) - pad_size, len(padded))
            for key in tuple(padded.batch.keys()):
                if key != "response_mask" and not key.startswith("proposal_"):
                    continue
                if key in padded.batch:
                    padded.batch[key][padding_slice] = 0
        return padded

    def _spread_proposals_across_dp_ranks(self, batch: DataProto) -> None:
        """Round-robin rows so zero-mask padding is not concentrated on one rank."""

        world_size = self.actor_rollout_wg.world_size
        if len(batch) % world_size:
            raise ValueError("proposal batch must be padded before DP distribution")
        order = [
            index
            for rank in range(world_size)
            for index in range(rank, len(batch), world_size)
        ]
        batch.reorder(torch.tensor(order, dtype=torch.long))

    def _select_proposal_candidates(self):
        if uses_native_webshop_actor(self.config) and is_r_align_warmup(
            global_step=self.global_steps,
            warmup_steps=self.uniskill_settings.r_align_warmup_steps,
        ):
            return build_warmup_candidates(
                self.traj_collector.iteration_trajectories,
                seed=self.uniskill_settings.seed + self.global_steps,
                sample_size=int(self.config.uniskill.webshop.get("warmup_samples", 8)),
            )
        return build_proposal_candidates(
            self.traj_collector.iteration_trajectories,
            seed=self.uniskill_settings.seed + self.global_steps,
            expected_environment=str(self.config.env.env_name),
        )

    def _apply_actor_action_penalty(self, batch, coefficient):
        strict_credit_active = strict_actor_credit_active(
            self.config, global_step=self.global_steps
        )
        if uses_native_webshop_actor(self.config) and not strict_credit_active:
            from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty as native_penalty

            result = native_penalty(batch, invalid_action_penalty_coef=coefficient)
        else:
            result = apply_invalid_action_penalty(
                batch,
                invalid_action_penalty_coef=coefficient,
                replace_strict_format_invalid_score=bool(
                    self.config.actor_rollout_ref.actor.get(
                        "replace_strict_format_invalid_score", False
                    )
                ),
            )
        if uses_native_webshop_actor(self.config):
            result[1]["episode/action_strict_credit_active"] = float(strict_credit_active)
        return result

    def _constrain_actor_advantages(self, batch, invalid_rows):
        strict_credit_active = strict_actor_credit_active(
            self.config, global_step=self.global_steps
        )
        if not uses_native_webshop_actor(self.config) or strict_credit_active:
            batch.batch["advantages"] = cap_invalid_positive_advantages(
                advantages=batch.batch["advantages"],
                response_mask=batch.batch["response_mask"],
                invalid_rows=invalid_rows,
            )

    def _prepare_proposal_batch(self, action_batch: DataProto, metrics: dict):
        metrics.update(
            {
                "proposal/nonzero_adv_active_ratio": 0.0,
                "proposal/zero_adv_skipped_ratio": 0.0,
                "proposal/no_batch_ratio": 1.0,
            }
        )
        metrics.update(
            action_history_metrics(self.traj_collector.iteration_trajectories)
        )
        metrics.update(
            action_rollout_health_metrics(
                self.traj_collector.iteration_trajectories
            )
        )
        in_r_align_warmup = is_r_align_warmup(
            global_step=self.global_steps,
            warmup_steps=self.uniskill_settings.r_align_warmup_steps,
        )
        candidates, anchor_stats = self._select_proposal_candidates()
        validate_candidate_roles(
            candidates,
            anchor_stats,
            format_only_warmup=(
                uses_native_webshop_actor(self.config) and in_r_align_warmup
            ),
        )
        self._last_anchor_stats = anchor_stats
        metrics.update({f"proposal/{key}": value for key, value in anchor_stats.items()})
        coverage_stats = alignment_coverage_metrics(
            self.traj_collector.iteration_trajectories,
            candidates,
        )
        self._last_coverage_stats = coverage_stats
        metrics.update({f"proposal/{key}": value for key, value in coverage_stats.items()})
        metrics["proposal/r_align_warmup_active"] = int(in_r_align_warmup)
        effective_proposal_mode_scale = proposal_mode_scale(
            global_step=self.global_steps,
            warmup_steps=self.uniskill_settings.r_align_warmup_steps,
            warmup_scale=self.uniskill_settings.proposal_warmup_lambda,
            full_scale=self.uniskill_settings.lambda_proposal,
        )
        metrics["proposal/mode_scale"] = effective_proposal_mode_scale
        if not candidates:
            return None, candidates

        generated, prompt_metadata = self._generate_proposals(candidates)
        decode_and_parse_candidates(
            generated,
            candidates,
            tokenizer=self.tokenizer,
            prompt_metadata=prompt_metadata,
            response_prefix=self.uniskill_settings.proposal_assistant_prefill,
        )
        if in_r_align_warmup:
            for candidate in candidates:
                apply_r_align_warmup_route(
                    candidate,
                    invalid_penalty=self.uniskill_settings.invalid_penalty,
                )
        else:
            apply_critic_routing(
                candidates,
                critic=self.critic_service,
                invalid_penalty=self.uniskill_settings.invalid_penalty,
                min_anchor_coverage=self.uniskill_settings.min_anchor_coverage,
            )
            apply_exact_duplicate_routing(
                candidates,
                skillbank=self.envs.skillbank,
                invalid_penalty=self.uniskill_settings.invalid_penalty,
            )

            scorer_samples, invalid_candidates = build_counterfactual_samples(candidates)
            for candidate in candidates:
                if candidate.proposal_id in invalid_candidates:
                    candidate.skill_reward = None
                    candidate.critic_error = "anchor trajectory has no scorable action steps"
            scorer_samples = self._remove_overlong_scorer_samples(scorer_samples, candidates)

            if scorer_samples:
                scorer_batch = build_teacher_forcing_batch(
                    scorer_samples,
                    tokenizer=self.tokenizer,
                    max_prompt_length=int(self.config.data.max_prompt_length),
                    apply_chat_template_kwargs=self.config.data.get("apply_chat_template_kwargs", {}),
                )
                scorer_log_probs = self._compute_log_prob_padded(scorer_batch)
                apply_alignment_scores(
                    candidates,
                    scorer_batch=scorer_batch,
                    scorer_log_probs=scorer_log_probs,
                    original_scores=original_step_logprob_scores(action_batch),
                    reward_clip=self.uniskill_settings.r_align_reward_clip,
                )

        if self.uniskill_settings.proposal_composite_reward:
            assign_composite_policy_rewards(
                candidates,
                format_only_warmup=in_r_align_warmup,
            )

        proposal_batch = build_proposal_training_batch(
            generated,
            candidates,
            full_response_channel_masks=(
                self.uniskill_settings.proposal_full_response_loss
            ),
            composite_reward=self.uniskill_settings.proposal_composite_reward,
        )
        self._update_proposal_metrics(candidates, metrics)
        if proposal_batch is None:
            return None, candidates
        proposal_reward_channels = (
            (PROPOSAL_JOINT_CHANNEL,)
            if self.uniskill_settings.proposal_composite_reward
            else PROPOSAL_CHANNELS
        )
        has_proposal_rpp = any(
            proposal_batch.batch[f"proposal_{channel}_valid"].any().item()
            for channel in proposal_reward_channels
        )
        has_operation_support = proposal_batch.batch[
            "proposal_operation_support_valid"
        ].any().item()
        if (
            not has_proposal_rpp
            and not has_operation_support
            and self.uniskill_settings.proposal_kl_loss_coef <= 0
        ):
            metrics["proposal/skipped_no_active_channel"] = 1
            return None, candidates

        response_has_tokens = proposal_batch.batch["response_mask"].bool().any(dim=-1)
        kept_indices = torch.nonzero(response_has_tokens, as_tuple=False).flatten()
        if kept_indices.numel() == 0:
            return None, candidates
        proposal_batch = proposal_batch.select_idxs(kept_indices)
        retained_proposal_ids = {
            str(proposal_id)
            for proposal_id in proposal_batch.non_tensor_batch["proposal_id"].tolist()
        }
        for candidate in candidates:
            candidate.included_in_policy_loss = (
                candidate.proposal_id in retained_proposal_ids
                and (
                    candidate.policy_reward_complete
                    if self.uniskill_settings.proposal_composite_reward
                    else not candidate.is_masked
                )
            )
        metrics["proposal/trainable"] = float(
            sum(candidate.included_in_policy_loss for candidate in candidates)
        )

        # Padding is required by verl's DP dispatch. Padding samples receive a
        # zero response mask before advantage estimation and therefore never
        # contribute to whitening or the policy loss.
        proposal_batch = self._pad_proposal_training_batch(proposal_batch)
        old_log_prob = self.actor_rollout_wg.compute_log_prob(proposal_batch)
        old_log_prob.batch.pop("entropys", None)
        proposal_batch = proposal_batch.union(old_log_prob)

        if self.use_reference_policy:
            if not self.ref_in_actor:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(proposal_batch)
            else:
                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(proposal_batch)
            proposal_batch = proposal_batch.union(ref_log_prob)

        if self.config.algorithm.use_kl_in_reward:
            raise ValueError(
                "Proposal rewards require algorithm.use_kl_in_reward=false; "
                "use the actor KL loss instead"
            )

        for channel in proposal_reward_channels:
            rewards = proposal_batch.batch[f"proposal_{channel}_rewards"]
            valid = proposal_batch.batch[f"proposal_{channel}_valid"]
            token_mask = proposal_batch.batch[f"proposal_{channel}_mask"]
            advantages, returns, scalar_advantages = (
                compute_channel_reinforce_plus_plus_advantage(
                    scalar_rewards=rewards,
                    valid_rows=valid,
                    token_mask=token_mask,
                    gamma=self.config.algorithm.gamma,
                )
            )
            proposal_batch.batch[f"proposal_{channel}_advantages"] = advantages
            proposal_batch.batch[f"proposal_{channel}_returns"] = returns
            metrics.update(
                proposal_channel_metrics(
                    scalar_rewards=rewards,
                    valid_rows=valid,
                    scalar_advantages=scalar_advantages,
                    channel=channel,
                )
            )
        self._spread_proposals_across_dp_ranks(proposal_batch)
        proposal_batch.meta_info.update(
            {
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
                "multi_turn": False,
                "lambda_proposal": effective_proposal_mode_scale,
                "proposal_entropy_coeff": self.uniskill_settings.proposal_entropy_coeff,
                "operation_support_min_probability": (
                    self.uniskill_settings.operation_support_min_probability
                ),
                "operation_support_coeff": (
                    self.uniskill_settings.operation_support_coeff
                ),
                "proposal_kl_loss_coef": (
                    self.uniskill_settings.proposal_kl_loss_coef
                ),
                "proposal_composite_reward": (
                    self.uniskill_settings.proposal_composite_reward
                ),
                "global_token_num": proposal_batch.batch["attention_mask"].sum(-1).tolist(),
            }
        )
        return proposal_batch, candidates

    def _update_proposal_metrics(self, candidates, metrics):
        if not candidates:
            return
        aligns = [candidate.r_align for candidate in candidates if candidate.r_align is not None]
        parse_valid = sum(candidate.parse_ok for candidate in candidates)
        support_valid = sum(
            candidate.operation_support_valid for candidate in candidates
        )
        metrics.update(
            {
                "proposal/generated": len(candidates),
                "proposal/trainable": sum(
                    (
                        candidate.policy_reward_complete
                        if self.uniskill_settings.proposal_composite_reward
                        else not candidate.is_masked
                    )
                    for candidate in candidates
                ),
                "proposal/incomplete_feedback": sum(
                    candidate.parse_ok
                    and not candidate.policy_reward_complete
                    for candidate in candidates
                )
                if self.uniskill_settings.proposal_composite_reward
                else 0,
                "proposal/critic_error": sum(candidate.critic_state.value == "error" for candidate in candidates),
                "proposal/parse_invalid": sum(not candidate.parse_ok for candidate in candidates),
                "proposal/recoverable_invalid": sum(
                    candidate.recoverable_invalid for candidate in candidates
                ),
                "proposal/unrecoverable_invalid": sum(
                    not candidate.parse_ok and not candidate.recoverable_invalid
                    for candidate in candidates
                ),
                "proposal/insufficient_anchor_coverage": sum(
                    candidate.local_routing_reason == "insufficient_anchor_coverage"
                    for candidate in candidates
                ),
                "proposal/warmup_format_valid": sum(
                    candidate.local_routing_reason == "r_align_warmup_format_valid"
                    for candidate in candidates
                ),
                "proposal/prompt_truncated": sum(candidate.prompt_truncated for candidate in candidates),
                "proposal/exact_duplicate_total": sum(
                    bool(candidate.exact_duplicate_source)
                    for candidate in candidates
                ),
                "proposal/exact_duplicate_prompt": sum(
                    candidate.exact_duplicate_source == "prompt"
                    for candidate in candidates
                ),
                "proposal/exact_duplicate_bank": sum(
                    candidate.exact_duplicate_source == "bank"
                    for candidate in candidates
                ),
                "proposal/exact_duplicate_batch": sum(
                    candidate.exact_duplicate_source == "batch"
                    for candidate in candidates
                ),
                "proposal/exact_duplicate_retrieved": sum(
                    candidate.exact_duplicate_source == "retrieved"
                    for candidate in candidates
                ),
                "proposal/write_eligible": sum(candidate.is_write_eligible for candidate in candidates),
                "proposal/op_support_eligible": support_valid,
                "proposal/op_support_coverage": (
                    support_valid / parse_valid if parse_valid else 0.0
                ),
            }
        )
        metrics.update(proposal_critic_action_metrics(candidates))
        metrics.update(proposal_response_metrics(candidates))
        if aligns:
            align_array = np.asarray(aligns, dtype=np.float64)
            metrics["proposal/r_align_mean"] = float(np.mean(align_array))
            metrics["proposal/r_align_std"] = float(np.std(align_array))
            metrics["proposal/r_align_min"] = float(np.min(align_array))
            metrics["proposal/r_align_max"] = float(np.max(align_array))
            metrics["proposal/r_align_p05"] = float(np.quantile(align_array, 0.05))
            metrics["proposal/r_align_p50"] = float(np.quantile(align_array, 0.50))
            metrics["proposal/r_align_p95"] = float(np.quantile(align_array, 0.95))
            metrics["proposal/r_align_positive_ratio"] = float(np.mean(np.asarray(aligns) > 0))
            if self.uniskill_settings.r_align_reward_clip is not None:
                clip = float(self.uniskill_settings.r_align_reward_clip)
                metrics["proposal/r_align_clip_ratio"] = float(
                    np.mean(np.abs(align_array) > clip)
                )

    def _commit_and_record_candidates(self, candidates):
        commit_results = commit_candidates(candidates, skillbank=self.envs.skillbank)
        result_by_id = {result.proposal_id: result for result in commit_results}
        records = []
        for candidate in candidates:
            record = candidate.to_record()
            result = result_by_id.get(candidate.proposal_id)
            if result is not None:
                record["written"] = result.status in {"added", "updated"}
                record["write_status"] = result.status
                record["written_skill"] = result.skill
            elif candidate.exact_duplicate_source:
                record["write_status"] = (
                    f"exact_duplicate_{candidate.exact_duplicate_source}"
                )
            records.append(record)
        write_jsonl(
            proposal_iteration_path(self.uniskill_settings.output_dir, self.global_steps),
            records,
        )
        return commit_results

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                proposal_batch = None
                proposal_candidates = []
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        if hasattr(self.envs, "training_step"):
                            self.envs.training_step = self.global_steps
                        self.traj_collector.action_step = self.global_steps - 1
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                                                                gen_batch=gen_batch,
                                                                actor_rollout_wg=self.actor_rollout_wg,
                                                                envs=self.envs,
                                                                is_train=True,
                                                                )
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    del batch
                    batch = gen_batch_output

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor
                    
                    batch = adjust_batch(self.config, batch)

                    credit_assignment_enabled = bool(
                        self.config.algorithm.get("credit_assignment", False)
                    )
                    metrics["episode/credit_assignment_enabled"] = float(
                        credit_assignment_enabled
                    )
                    if credit_assignment_enabled:
                        step_returns = batch.batch["step_returns"]
                        batch.non_tensor_batch["episode_rewards"] = (
                            step_returns.detach().cpu().numpy()
                        )
                        metrics.update(
                            {
                                "episode/step_return_mean": float(
                                    step_returns.float().mean().item()
                                ),
                                "episode/step_return_std": float(
                                    step_returns.float().std(unbiased=False).item()
                                ),
                                "episode/step_return_min": float(
                                    step_returns.min().item()
                                ),
                                "episode/step_return_max": float(
                                    step_returns.max().item()
                                ),
                            }
                        )

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_invalid_action_penalty if available
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = self._apply_actor_action_penalty(
                                batch, self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity= self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )
                        action_invalid_rows = ~np.asarray(
                            batch.non_tensor_batch["is_action_overall_valid"]
                        ).reshape(-1).astype(bool)
                        metrics["action/invalid_positive_adv_ratio_before_gate"] = (
                            invalid_positive_advantage_ratio(
                                advantages=batch.batch["advantages"],
                                response_mask=batch.batch["response_mask"],
                                invalid_rows=action_invalid_rows,
                            )
                        )
                        self._constrain_actor_advantages(batch, action_invalid_rows)
                        invalid_positive_after_gate = (
                            invalid_positive_advantage_ratio(
                                advantages=batch.batch["advantages"],
                                response_mask=batch.batch["response_mask"],
                                invalid_rows=action_invalid_rows,
                            )
                        )
                        metrics["action/invalid_positive_adv_ratio_after_gate"] = (
                            invalid_positive_after_gate
                        )
                        metrics["action/invalid_positive_adv_ratio"] = (
                            invalid_positive_after_gate
                        )
                        metrics.update(action_advantage_metrics(batch))

                    with _timer("proposal_pipeline", timing_raw):
                        proposal_batch, proposal_candidates = self._prepare_proposal_batch(
                            batch,
                            metrics,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            if proposal_batch is None:
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            else:
                                actor_output = self.actor_rollout_wg.update_actor_joint(
                                    batch,
                                    proposal_batch,
                                )
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    with _timer("skill_commit", timing_raw):
                        commit_results = self._commit_and_record_candidates(proposal_candidates)
                        metrics["proposal/skills_added"] = sum(
                            result.status == "added" for result in commit_results
                        )
                        metrics["proposal/skills_updated"] = sum(
                            result.status == "updated" for result in commit_results
                        )

                    with _timer("uniskill_debug_trace", timing_raw):
                        try:
                            self.debug_trace.print_iteration(
                                iteration=self.global_steps,
                                trajectories=self.traj_collector.iteration_trajectories,
                                candidates=proposal_candidates,
                                anchor_stats=self._last_anchor_stats,
                                commit_results=commit_results,
                                coverage_stats=self._last_coverage_stats,
                            )
                        except Exception as error:
                            # Observability must never invalidate an already-applied update.
                            print(
                                f"[UNISKILL DEBUG ERROR] iteration={self.global_steps} "
                                f"error={error!r}",
                                flush=True,
                            )

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                        with _timer("save_best_checkpoint", timing_raw):
                            self._maybe_save_best_checkpoint(
                                val_metrics,
                                metrics,
                            )

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                    if self.global_steps in self._milestone_checkpoint_steps:
                        with _timer("save_milestone_checkpoint", timing_raw):
                            self._save_milestone_checkpoint()
                        metrics["checkpoint/milestone_saved"] = 1.0

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # Print UniSkill metrics for each training step.
                print(format_uniskill_step_metrics(metrics), flush=True)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
