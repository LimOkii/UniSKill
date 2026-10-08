from __future__ import annotations

import logging
import psutil

from codetiming import Timer
from omegaconf import open_dict

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from uniskill.workers.actor import UniSkillDataParallelPPOActor


logger = logging.getLogger(__name__)


class UniSkillActorRolloutRefWorker(ActorRolloutRefWorker):
    """FSDP worker extension exposing a non-invasive joint actor RPC."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = self.config.model.get("use_remove_padding", False)
                self.config.actor.use_fused_kernels = self.config.model.get("use_fused_kernels", False)
            self.actor = UniSkillDataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_proposal_sequences(self, prompts: DataProto):
        """Use a proposal-only response budget without changing action rollout."""

        response_length = int(prompts.meta_info.pop("uniskill_response_length"))
        if response_length <= 0:
            raise ValueError("UniSkill proposal response length must be positive")

        rollout_config = self.rollout.config
        sampling_params = self.rollout.sampling_params
        old_response_length = int(rollout_config.response_length)
        if hasattr(sampling_params, "max_tokens"):
            sampling_key = "max_tokens"
            old_sampling_length = int(sampling_params.max_tokens)
        elif isinstance(sampling_params, dict) and "max_new_tokens" in sampling_params:
            sampling_key = "max_new_tokens"
            old_sampling_length = int(sampling_params[sampling_key])
        else:
            raise RuntimeError(
                "rollout engine does not expose a supported proposal length parameter"
            )

        try:
            with open_dict(rollout_config):
                rollout_config.response_length = response_length
            if sampling_key == "max_tokens":
                sampling_params.max_tokens = response_length
            else:
                sampling_params[sampling_key] = response_length
            return super().generate_sequences(prompts)
        finally:
            with open_dict(rollout_config):
                rollout_config.response_length = old_response_length
            if sampling_key == "max_tokens":
                sampling_params.max_tokens = old_sampling_length
            else:
                sampling_params[sampling_key] = old_sampling_length

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor_joint(self, action_data: DataProto, proposal_data: DataProto):
        action_data = action_data.to(get_torch_device().current_device())
        proposal_data = proposal_data.to(get_torch_device().current_device())
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(
                optimizer=self.actor_optimizer,
                device_id=get_torch_device().current_device(),
            )

        with self.ulysses_sharding_manager:
            action_data = self.ulysses_sharding_manager.preprocess_data(data=action_data)
            proposal_data = self.ulysses_sharding_manager.preprocess_data(data=proposal_data)
            with Timer(name="update_policy_joint", logger=None) as timer:
                metrics = self.actor.update_policy_joint(
                    action_data=action_data,
                    proposal_data=proposal_data,
                    lambda_proposal=float(proposal_data.meta_info["lambda_proposal"]),
                    proposal_entropy_coeff=float(
                        proposal_data.meta_info["proposal_entropy_coeff"]
                    ),
                    operation_support_min_probability=float(
                        proposal_data.meta_info[
                            "operation_support_min_probability"
                        ]
                    ),
                    operation_support_coeff=float(
                        proposal_data.meta_info["operation_support_coeff"]
                    ),
                    proposal_kl_loss_coef=float(
                        proposal_data.meta_info["proposal_kl_loss_coef"]
                    ),
                )
            global_tokens = list(action_data.meta_info.get("global_token_num", []))
            global_tokens.extend(proposal_data.meta_info.get("global_token_num", []))
            if global_tokens:
                estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                    global_tokens,
                    timer.last,
                )
                metrics["perf/mfu/actor"] = (
                    estimated_flops
                    * self.config.actor.ppo_epochs
                    / promised_flops
                    / self.world_size
                )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
            metrics["actor/lr"] = self.actor_lr_scheduler.get_last_lr()[0]
            self.actor_lr_scheduler.step()

            output = DataProto(meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output).to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After UniSkill joint actor update", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        return output
