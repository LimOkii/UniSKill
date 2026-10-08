from __future__ import annotations

import logging
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_policy_loss,
    compute_policy_loss_gspo,
    kl_penalty,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor.dp_actor import DataParallelPPOActor

from uniskill.training.loss_normalization import distributed_token_mean_scale
from uniskill.training.operation_support import (
    OPERATION_ACTIONS,
    operation_support_loss,
)
from uniskill.training.proposal_advantage import (
    PROPOSAL_CHANNELS,
    PROPOSAL_JOINT_CHANNEL,
)


logger = logging.getLogger(__name__)


class UniSkillDataParallelPPOActor(DataParallelPPOActor):
    """PPO actor with two mode forwards and one optimizer step."""

    @GPUMemoryLogger(role="uniskill dp actor", logger=logger)
    def update_policy_joint(
        self,
        action_data: DataProto,
        proposal_data: DataProto,
        *,
        lambda_proposal: float,
        proposal_entropy_coeff: float,
        operation_support_min_probability: float,
        operation_support_coeff: float,
        proposal_kl_loss_coef: float,
    ) -> dict:
        self.actor_module.train()
        if "multi_modal_inputs" in action_data.non_tensor_batch:
            raise NotImplementedError("UniSkill joint update supports text-only action data")
        if self.config.use_dynamic_bsz:
            raise NotImplementedError("UniSkill joint update requires use_dynamic_bsz=false")

        action_batch = self._training_tensordict(action_data, proposal=False)
        proposal_composite_reward = bool(
            proposal_data.meta_info.get("proposal_composite_reward", False)
        )
        proposal_channels = (
            (PROPOSAL_JOINT_CHANNEL,)
            if proposal_composite_reward
            else PROPOSAL_CHANNELS
        )
        proposal_batch = self._training_tensordict(
            proposal_data,
            proposal=True,
            proposal_channels=proposal_channels,
        )
        action_minibatches = list(action_batch.split(self.config.ppo_mini_batch_size))
        if not action_minibatches:
            raise ValueError("joint update received an empty action batch")

        metrics: dict[str, list[float]] = {}
        for _epoch in range(self.config.ppo_epochs):
            global_proposal_tokens = self._global_valid_token_count(
                proposal_batch["response_mask"]
            )
            global_active_rows = {
                channel: self._global_active_row_count(
                    proposal_batch[f"proposal_{channel}_mask"]
                )
                for channel in proposal_channels
            }
            global_support_rows = self._global_active_row_count(
                proposal_batch["proposal_operation_support_valid"]
            )
            proposal_slices = self._split_proposals(proposal_batch, len(action_minibatches))
            for action_minibatch, proposal_minibatch in zip(action_minibatches, proposal_slices):
                self.actor_optimizer.zero_grad()
                self._backward_minibatch(
                    action_minibatch,
                    temperature=float(action_data.meta_info["temperature"]),
                    mode="action",
                    mode_scale=1.0,
                    metrics=metrics,
                )
                action_grad_norm = self._current_grad_norm()
                append_to_dict(
                    metrics,
                    {
                        "actor/action_grad_norm_before_proposal": action_grad_norm.detach().item(),
                        "actor/proposal_entropy_coeff": float(proposal_entropy_coeff),
                        "actor/proposal_mode_scale": float(lambda_proposal),
                        "actor/proposal_support_coeff": float(operation_support_coeff),
                        "actor/proposal_kl_loss_coef": float(proposal_kl_loss_coef),
                        "actor/proposal_support_min_probability": float(
                            operation_support_min_probability
                        ),
                    },
                )
                if proposal_minibatch is not None:
                    if global_proposal_tokens > 0:
                        self._backward_proposal_minibatch(
                            proposal_minibatch,
                            temperature=float(proposal_data.meta_info["temperature"]),
                            mode_scale=float(lambda_proposal),
                            entropy_coeff=float(proposal_entropy_coeff),
                            global_valid_tokens=global_proposal_tokens,
                            global_active_rows=global_active_rows,
                            proposal_channels=proposal_channels,
                            global_support_rows=global_support_rows,
                            support_min_probability=float(
                                operation_support_min_probability
                            ),
                            support_coeff=float(operation_support_coeff),
                            proposal_kl_loss_coef=float(proposal_kl_loss_coef),
                            metrics=metrics,
                        )
                        append_to_dict(
                            metrics,
                            {
                                "actor/proposal_active_step": 1.0,
                                "proposal/no_batch_ratio": 0.0,
                            },
                        )
                    else:
                        append_to_dict(
                            metrics,
                            {
                                "actor/proposal_active_step": 0.0,
                                "proposal/nonzero_adv_active_ratio": 0.0,
                                "proposal/zero_adv_skipped_ratio": 0.0,
                                "proposal/no_batch_ratio": 1.0,
                            },
                        )
                else:
                    append_to_dict(
                        metrics,
                        {
                            "actor/proposal_active_step": 0.0,
                            "proposal/nonzero_adv_active_ratio": 0.0,
                            "proposal/zero_adv_skipped_ratio": 0.0,
                            "proposal/no_batch_ratio": 1.0,
                        },
                    )

                grad_norm = self._optimizer_step()
                append_to_dict(
                    metrics,
                    {
                        "actor/grad_norm": grad_norm.detach().item(),
                        "actor/joint_grad_norm": grad_norm.detach().item(),
                    },
                )

        self.actor_optimizer.zero_grad()
        return metrics

    def _training_tensordict(
        self,
        data: DataProto,
        *,
        proposal: bool,
        proposal_channels=PROPOSAL_CHANNELS,
    ):
        keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "response_mask",
        ]
        if proposal:
            for channel in proposal_channels:
                keys.extend(
                    [
                        f"proposal_{channel}_mask",
                        f"proposal_{channel}_advantages",
                    ]
                )
            keys.extend(
                [
                    "proposal_operation_token_ids",
                    "proposal_operation_valid_mask",
                    "proposal_operation_decision_position",
                    "proposal_operation_support_valid",
                ]
            )
        else:
            keys.append("advantages")
        if self.config.use_kl_loss:
            keys.append("ref_log_prob")
        return data.select(batch_keys=keys).batch

    @staticmethod
    def _split_proposals(proposal_batch, number_of_steps: int):
        if len(proposal_batch) == 0:
            return [None] * number_of_steps
        permutation = torch.randperm(
            len(proposal_batch),
            device=proposal_batch["responses"].device,
        )
        index_slices = torch.tensor_split(permutation, number_of_steps)
        return [proposal_batch[indexes] if indexes.numel() else None for indexes in index_slices]

    @staticmethod
    def _global_valid_token_count(response_mask: torch.Tensor) -> int:
        token_count = response_mask.sum().detach().to(dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        return int(token_count.item())

    @staticmethod
    def _global_active_row_count(
        token_mask: torch.Tensor,
    ) -> int:
        row_count = torch.tensor(
            int(token_mask.bool().any(dim=-1).sum().item()),
            dtype=torch.long,
            device=token_mask.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(row_count, op=dist.ReduceOp.SUM)
        return int(row_count.item())

    def _backward_proposal_minibatch(
        self,
        minibatch,
        *,
        temperature: float,
        mode_scale: float,
        entropy_coeff: float,
        global_valid_tokens: int,
        global_active_rows: dict[str, int],
        proposal_channels: tuple[str, ...],
        global_support_rows: int,
        support_min_probability: float,
        support_coeff: float,
        proposal_kl_loss_coef: float,
        metrics: dict,
    ) -> None:
        if entropy_coeff != 0:
            raise ValueError("Proposal entropy must remain disabled")
        microbatches = list(minibatch.split(self.config.ppo_micro_batch_size_per_gpu))
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        for microbatch in microbatches:
            microbatch = microbatch.to(get_torch_device().current_device())
            response_mask = microbatch["response_mask"]
            (
                log_prob,
                support_loss,
                support_probabilities,
                support_violations,
                support_valid_actions,
                local_support_active,
            ) = self._forward_proposal_micro_batch(
                microbatch,
                temperature=temperature,
                min_probability=support_min_probability,
            )
            clip_ratio = self.config.clip_ratio
            total_loss = log_prob.sum() * 0.0
            for channel in proposal_channels:
                channel_mask = microbatch[f"proposal_{channel}_mask"].bool()
                active = channel_mask.any(dim=-1)
                local_active = int(active.sum().item())
                global_active = int(global_active_rows[channel])
                if local_active == 0 or global_active == 0:
                    continue
                pg_loss, clipfrac, ppo_kl, clipfrac_lower = compute_policy_loss(
                    old_log_prob=microbatch["old_log_probs"][active],
                    log_prob=log_prob[active],
                    advantages=microbatch[f"proposal_{channel}_advantages"][active],
                    response_mask=channel_mask[active],
                    cliprange=clip_ratio,
                    cliprange_low=(
                        self.config.clip_ratio_low
                        if self.config.clip_ratio_low is not None
                        else clip_ratio
                    ),
                    cliprange_high=(
                        self.config.clip_ratio_high
                        if self.config.clip_ratio_high is not None
                        else clip_ratio
                    ),
                    clip_ratio_c=self.config.get("clip_ratio_c", 3.0),
                    loss_agg_mode="seq-mean-token-mean",
                )
                row_weight = world_size * local_active / global_active
                total_loss = total_loss + pg_loss * row_weight
                append_to_dict(
                    metrics,
                    {
                        f"actor/proposal_{channel}_pg_loss": pg_loss.detach().item(),
                        f"actor/proposal_{channel}_pg_clipfrac": clipfrac.detach().item(),
                        f"actor/proposal_{channel}_ppo_kl": ppo_kl.detach().item(),
                        f"actor/proposal_{channel}_pg_clipfrac_lower": clipfrac_lower.detach().item(),
                        f"actor/proposal_{channel}_loss_weight": row_weight,
                    },
                )

            token_weight = distributed_token_mean_scale(
                local_valid_tokens=int(response_mask.sum().item()),
                global_valid_tokens=global_valid_tokens,
                world_size=world_size,
            )
            if self.config.use_kl_loss:
                kld = kl_penalty(
                    logprob=log_prob,
                    ref_logprob=microbatch["ref_log_prob"],
                    kl_penalty=self.config.kl_loss_type,
                )
                kl_loss = agg_loss(
                    loss_mat=kld,
                    loss_mask=response_mask,
                    loss_agg_mode="token-mean",
                )
                total_loss = (
                    total_loss
                    + kl_loss * proposal_kl_loss_coef * token_weight
                )
                append_to_dict(
                    metrics, {"actor/proposal_kl_loss": kl_loss.detach().item()}
                )
            if support_loss is not None:
                support_row_weight = (
                    world_size * local_support_active / global_support_rows
                )
                total_loss = (
                    total_loss
                    + support_coeff * support_loss * support_row_weight
                )
                self._record_operation_support_metrics(
                    support_loss=support_loss,
                    probabilities=support_probabilities,
                    violations=support_violations,
                    valid_actions=support_valid_actions,
                    local_active=local_support_active,
                    row_weight=support_row_weight,
                    metrics=metrics,
                )
            (mode_scale * total_loss).backward()

    def _forward_proposal_micro_batch(
        self,
        microbatch,
        *,
        temperature: float,
        min_probability: float,
    ) -> tuple:
        """Return sampled log-probs and operation support from one model forward."""

        if self.use_fused_kernels:
            raise NotImplementedError(
                "operation support requires actor_rollout_ref.model.use_fused_kernels=false"
            )
        if self.use_ulysses_sp:
            raise NotImplementedError(
                "operation support currently requires actor ulysses sequence parallel size 1"
            )

        input_ids = microbatch["input_ids"]
        attention_mask = microbatch["attention_mask"]
        position_ids = microbatch["position_ids"]
        if position_ids.dim() != 2:
            raise NotImplementedError(
                "Proposal support requires text-only position ids"
            )
        response_length = microbatch["responses"].size(-1)
        active = microbatch["proposal_operation_support_valid"][:, 0].bool()
        local_active = int(active.sum().item())

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if self.use_remove_padding:
                keep = attention_mask.bool()
                input_ids_rmpad = input_ids[keep].unsqueeze(0)
                position_ids_rmpad = position_ids[keep].unsqueeze(0)
                input_ids_rmpad_rolled = torch.roll(
                    input_ids_rmpad, shifts=-1, dims=1
                ).squeeze(0)
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                )
                logits_rmpad = output.logits.squeeze(0)
                logits_rmpad.div_(float(temperature))
                log_probs_rmpad = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=False,
                )
                full_log_probs = torch.zeros_like(
                    input_ids, dtype=log_probs_rmpad.dtype
                ).masked_scatter(keep, log_probs_rmpad)
                log_prob = full_log_probs[:, -response_length - 1 : -1]
                all_logits = logits_rmpad
            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                all_logits = output.logits / float(temperature)
                response_logits = all_logits[:, -response_length - 1 : -1]
                log_prob = logprobs_from_logits(
                    logits=response_logits,
                    labels=microbatch["responses"],
                    inplace_backward=False,
                )

            if local_active == 0:
                return log_prob, None, None, None, None, 0

            decision_positions = microbatch[
                "proposal_operation_decision_position"
            ][active, 0]
            context_positions = (
                input_ids.size(-1) - response_length - 1 + decision_positions
            )
            if not attention_mask[active].bool().gather(
                1, context_positions[:, None]
            ).all():
                raise ValueError("operation decision context points to padding")
            if self.use_remove_padding:
                keep = attention_mask.bool()
                row_offsets = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.long, device=input_ids.device),
                        keep.sum(dim=-1).cumsum(dim=0)[:-1],
                    ]
                )
                context_unpadded = (
                    keep[active].to(torch.long).cumsum(dim=-1).gather(
                        1, context_positions[:, None]
                    )[:, 0]
                    - 1
                )
                selected_logits = all_logits[
                    row_offsets[active] + context_unpadded
                ]
            else:
                batch_indices = torch.arange(
                    input_ids.size(0), device=input_ids.device
                )[active]
                selected_logits = all_logits[
                    batch_indices, context_positions
                ]

            candidate_ids = microbatch["proposal_operation_token_ids"][active]
            operation_logits = selected_logits.gather(
                dim=-1, index=candidate_ids
            ).float()
            valid_actions = microbatch["proposal_operation_valid_mask"][active]
            support_loss, probabilities, violations = operation_support_loss(
                operation_logits=operation_logits,
                valid_actions=valid_actions,
                min_probability=min_probability,
            )
        return (
            log_prob,
            support_loss,
            probabilities,
            violations,
            valid_actions,
            local_active,
        )

    @staticmethod
    def _record_operation_support_metrics(
        *,
        support_loss,
        probabilities,
        violations,
        valid_actions,
        local_active: int,
        row_weight: float,
        metrics: dict,
    ) -> None:
        metric_values = {
            "actor/proposal_support_loss": support_loss.detach().item(),
            "actor/proposal_support_loss_weight": row_weight,
            "proposal/op_support_active": float(local_active),
            "proposal/op_support_violation_ratio": (
                (violations > 0).sum().to(torch.float32)
                / valid_actions.sum().clamp_min(1)
            ).detach().item(),
            "proposal/op_prob_min_mean": probabilities.masked_fill(
                ~valid_actions, 1.0
            ).min(dim=-1).values.mean().detach().item(),
        }
        for action_index, action_name in enumerate(OPERATION_ACTIONS):
            legal = valid_actions[:, action_index]
            if legal.any():
                metric_values[f"proposal/op_prob_{action_name.lower()}"] = (
                    probabilities[legal, action_index].mean().detach().item()
                )
        append_to_dict(metrics, metric_values)

    def _current_grad_norm(self) -> torch.Tensor:
        """Read the distributed gradient norm without clipping the gradients."""

        if isinstance(self.actor_module, FSDP):
            return self.actor_module.clip_grad_norm_(max_norm=float("inf"))
        if isinstance(self.actor_module, FSDPModule):
            return fsdp2_clip_grad_norm_(
                self.actor_module.parameters(),
                max_norm=float("inf"),
            )
        return torch.nn.utils.clip_grad_norm_(
            self.actor_module.parameters(),
            max_norm=float("inf"),
        )

    def _backward_minibatch(
        self,
        minibatch,
        *,
        temperature: float,
        mode: str,
        mode_scale: float,
        metrics: dict,
        global_valid_tokens: int | None = None,
        entropy_coeff: float | None = None,
    ) -> None:
        microbatches = list(minibatch.split(self.config.ppo_micro_batch_size_per_gpu))
        if not microbatches:
            return
        gradient_accumulation = len(microbatches)
        for microbatch in microbatches:
            microbatch = microbatch.to(get_torch_device().current_device())
            response_mask = microbatch["response_mask"]
            effective_entropy_coeff = (
                float(self.config.entropy_coeff)
                if entropy_coeff is None
                else float(entropy_coeff)
            )
            calculate_entropy = effective_entropy_coeff != 0
            entropy, log_prob = self._forward_micro_batch(
                micro_batch=microbatch,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
            )
            loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
            if loss_mode == "vanilla":
                policy_loss_fn = compute_policy_loss
            elif loss_mode == "gspo":
                policy_loss_fn = compute_policy_loss_gspo
            else:
                raise ValueError(f"Unsupported loss_mode: {loss_mode}")

            clip_ratio = self.config.clip_ratio
            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                old_log_prob=microbatch["old_log_probs"],
                log_prob=log_prob,
                advantages=microbatch["advantages"],
                response_mask=response_mask,
                cliprange=clip_ratio,
                cliprange_low=(
                    self.config.clip_ratio_low
                    if self.config.clip_ratio_low is not None
                    else clip_ratio
                ),
                cliprange_high=(
                    self.config.clip_ratio_high
                    if self.config.clip_ratio_high is not None
                    else clip_ratio
                ),
                clip_ratio_c=self.config.get("clip_ratio_c", 3.0),
                loss_agg_mode=self.config.loss_agg_mode,
            )
            policy_loss = pg_loss
            if calculate_entropy:
                entropy_loss = agg_loss(
                    loss_mat=entropy,
                    loss_mask=response_mask,
                    loss_agg_mode=self.config.loss_agg_mode,
                )
                policy_loss = policy_loss - entropy_loss * effective_entropy_coeff
                append_to_dict(metrics, {f"actor/{mode}_entropy": entropy_loss.detach().item()})

            if self.config.use_kl_loss:
                kld = kl_penalty(
                    logprob=log_prob,
                    ref_logprob=microbatch["ref_log_prob"],
                    kl_penalty=self.config.kl_loss_type,
                )
                kl_loss = agg_loss(
                    loss_mat=kld,
                    loss_mask=response_mask,
                    loss_agg_mode=self.config.loss_agg_mode,
                )
                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                append_to_dict(metrics, {f"actor/{mode}_kl_loss": kl_loss.detach().item()})

            if global_valid_tokens is None:
                # Accumulate action loss across local microbatches.
                loss_weight = 1.0 / gradient_accumulation
            else:
                local_valid_tokens = int(response_mask.sum().detach().item())
                world_size = (
                    dist.get_world_size()
                    if dist.is_available() and dist.is_initialized()
                    else 1
                )
                loss_weight = distributed_token_mean_scale(
                    local_valid_tokens=local_valid_tokens,
                    global_valid_tokens=global_valid_tokens,
                    world_size=world_size,
                )
            loss = mode_scale * policy_loss * loss_weight
            loss.backward()
            append_to_dict(
                metrics,
                {
                    f"actor/{mode}_pg_loss": pg_loss.detach().item(),
                    f"actor/{mode}_pg_clipfrac": pg_clipfrac.detach().item(),
                    f"actor/{mode}_ppo_kl": ppo_kl.detach().item(),
                    f"actor/{mode}_pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    f"actor/{mode}_loss_weight": loss_weight,
                },
            )
