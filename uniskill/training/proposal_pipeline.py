from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch

from verl import DataProto

from uniskill.critic.prompts import build_critic_prompt
from uniskill.critic.service import CriticService
from uniskill.environments.prompts.registry import render_action_prompt
from uniskill.proposal.parser import parse_proposal
from uniskill.response_format import classify_response_termination
from uniskill.training.batches import TeacherForcingSample, generated_response_mask
from uniskill.training.critic_routing import (
    apply_anchor_coverage_route,
    apply_local_critic_route,
)
from uniskill.training.operation_support import OPERATION_ACTIONS
from uniskill.training.token_masks import (
    character_spans_token_mask,
    masked_mean,
    proposal_channel_token_masks,
)
from uniskill.training.types import CriticState, ProposalCandidate


def _first_operation_token_ids(
    *,
    tokenizer,
    response_text: str,
    action_span: tuple[int, int],
    generated_text_offset: int = 0,
) -> tuple[list[int], list[int]]:
    """Resolve the first semantic token for every operation under one prefix."""

    action_open_end = response_text.find("<action>") + len("<action>")
    if action_open_end < len("<action>") or action_span[0] < action_open_end:
        raise ValueError("cannot locate proposal action prefix")
    prefix = response_text[: action_span[0]]
    fixed_prefix = response_text[:generated_text_offset]
    fixed_prefix_ids = tokenizer.encode(fixed_prefix, add_special_tokens=False)
    token_ids = []
    tokenized_prefix = None
    for action in OPERATION_ACTIONS:
        candidate_text = prefix + action
        candidate_ids = tokenizer.encode(candidate_text, add_special_tokens=False)
        semantic_mask = character_spans_token_mask(
            candidate_ids,
            tokenizer,
            [(len(prefix), len(candidate_text))],
        )
        selected_positions = [
            index for index, keep in enumerate(semantic_mask) if keep
        ]
        if not selected_positions:
            raise ValueError(f"cannot resolve first token for operation {action}")
        first_position = selected_positions[0]
        if candidate_ids[: len(fixed_prefix_ids)] != fixed_prefix_ids:
            raise ValueError("operation prefix tokenization changed across prefill boundary")
        candidate_prefix = candidate_ids[len(fixed_prefix_ids) : first_position]
        if tokenized_prefix is None:
            tokenized_prefix = candidate_prefix
        elif candidate_prefix != tokenized_prefix:
            raise ValueError("operation alternatives do not share one token prefix")
        token_ids.append(int(candidate_ids[first_position]))
    if len(set(token_ids)) != len(OPERATION_ACTIONS):
        raise ValueError("operation labels do not have distinct first tokens")
    return token_ids, list(tokenized_prefix or [])


def decode_and_parse_candidates(
    generated: DataProto,
    candidates: list[ProposalCandidate],
    *,
    tokenizer,
    prompt_metadata: Mapping[str, Sequence],
    response_prefix: str = "",
) -> None:
    if len(generated) != len(candidates):
        raise ValueError(f"proposal output size mismatch: {len(generated)} != {len(candidates)}")
    response_mask = generated_response_mask(generated)
    response_limit = int(generated.batch["responses"].shape[-1])
    for index, candidate in enumerate(candidates):
        candidate.response_ids = generated.batch["responses"][index].detach().cpu().tolist()
        candidate.response_mask = response_mask[index].detach().cpu().tolist()
        ids = [
            token_id
            for token_id, keep in zip(candidate.response_ids, candidate.response_mask)
            if keep
        ]
        termination = classify_response_termination(
            ids,
            response_limit=response_limit,
            eos_token_id=tokenizer.eos_token_id,
        )
        candidate.response_token_length = termination.token_length
        candidate.response_hit_length_limit = termination.hit_length_limit
        candidate.response_truncated = termination.truncated
        generated_text = tokenizer.decode(ids, skip_special_tokens=True)
        candidate.response_text = response_prefix + generated_text
        parsed = parse_proposal(candidate.response_text)
        candidate.contains_chinese = bool(parsed["contains_chinese"])
        candidate.proposal_action = parsed["proposal_action"]
        candidate.skill_text = parsed["skill_text"]
        candidate.parse_ok = parsed["parse_ok"]
        candidate.parse_error = parsed["parse_error"]
        candidate.recoverable_invalid = bool(parsed["recoverable_invalid"])
        if candidate.response_truncated:
            candidate.parse_ok = False
            candidate.recoverable_invalid = False
            candidate.parse_error = (
                "proposal response reached length limit without normal termination"
            )
        compact_ids = list(ids)
        eos_position = None
        if compact_ids and compact_ids[-1] == tokenizer.eos_token_id:
            eos_position = len(compact_ids) - 1
            visible_ids = compact_ids[:-1]
        else:
            visible_ids = compact_ids
        generated_offset = len(response_prefix)
        parsed_generated = dict(parsed)
        if parsed["action_span"] is not None and parsed["skill_span"] is not None:
            for key in ("action_span", "skill_span"):
                start, end = parsed[key]
                parsed_generated[key] = (
                    max(0, start - generated_offset),
                    max(0, end - generated_offset),
                )
            parsed_generated["format_spans"] = [
                (max(0, start - generated_offset), end - generated_offset)
                for start, end in parsed["format_spans"]
                if end > generated_offset
            ]
        try:
            (
                compact_format_mask,
                compact_action_mask,
                compact_skill_mask,
            ) = proposal_channel_token_masks(
                compact_ids,
                visible_ids,
                tokenizer=tokenizer,
                parsed=parsed_generated,
                parse_ok=candidate.parse_ok,
                eos_position=eos_position,
            )
        except ValueError as exc:
            candidate.parse_ok = False
            candidate.recoverable_invalid = False
            candidate.parse_error = f"proposal token-mask mapping failed: {exc}"
            compact_format_mask = [1] * len(compact_ids)
            compact_action_mask = [0] * len(compact_ids)
            compact_skill_mask = [0] * len(compact_ids)

        valid_positions = [
            position for position, keep in enumerate(candidate.response_mask) if keep
        ]
        candidate.format_token_mask = [0] * len(candidate.response_ids)
        candidate.action_token_mask = [0] * len(candidate.response_ids)
        candidate.skill_token_mask = [0] * len(candidate.response_ids)
        for compact_index, full_index in enumerate(valid_positions):
            candidate.format_token_mask[full_index] = compact_format_mask[compact_index]
            candidate.action_token_mask[full_index] = compact_action_mask[compact_index]
            candidate.skill_token_mask[full_index] = compact_skill_mask[compact_index]
        candidate.operation_token_ids = []
        candidate.operation_valid_mask = []
        candidate.operation_decision_position = -1
        candidate.operation_support_valid = False
        candidate.operation_support_error = ""
        if candidate.parse_ok:
            try:
                action_positions = [
                    position
                    for position, keep in enumerate(candidate.action_token_mask)
                    if keep
                ]
                if not action_positions:
                    raise ValueError("proposal action has no token position")
                operation_token_ids, operation_prefix_ids = _first_operation_token_ids(
                    tokenizer=tokenizer,
                    response_text=candidate.response_text,
                    action_span=parsed["action_span"],
                    generated_text_offset=generated_offset,
                )
                decision_position = action_positions[0]
                if (
                    candidate.response_ids[:decision_position]
                    != operation_prefix_ids
                ):
                    raise ValueError(
                        "operation alternatives do not match the sampled token prefix"
                    )
                sampled_action_index = OPERATION_ACTIONS.index(
                    candidate.proposal_action
                )
                if (
                    int(candidate.response_ids[decision_position])
                    != operation_token_ids[sampled_action_index]
                ):
                    raise ValueError(
                        "sampled action token does not match canonical operation token"
                    )
                candidate.operation_token_ids = operation_token_ids
                candidate.operation_valid_mask = [
                    True,
                    True,
                    candidate.source.retrieved_skill is not None,
                ]
                candidate.operation_decision_position = decision_position
                candidate.operation_support_valid = True
            except (ValueError, IndexError) as exc:
                # Format and factorized rewards remain valid even when the
                # tokenizer cannot expose a reliable operation decision point.
                candidate.operation_support_valid = False
                candidate.operation_support_error = str(exc)
        candidate.train_index = index
        candidate.prompt_token_length = int(prompt_metadata["prompt_token_length"][index])
        candidate.prompt_truncated = bool(prompt_metadata["prompt_truncated"][index])
        candidate.model_prompt_ids = [
            int(token_id) for token_id in prompt_metadata["raw_prompt_ids"][index]
        ]


def apply_critic_routing(
    candidates: list[ProposalCandidate],
    *,
    critic: CriticService,
    invalid_penalty: float,
    min_anchor_coverage: float,
) -> None:
    """Route proposals through local validation and one concurrent critic batch."""

    prompts: dict[str, str] = {}
    for candidate in candidates:
        if not apply_local_critic_route(candidate, invalid_penalty=invalid_penalty):
            continue
        candidate.critic_prompt = build_critic_prompt(
            candidate.source.to_record(),
            {
                "proposal_action": candidate.proposal_action,
                "skill_text": candidate.skill_text,
            },
            comparison_record=(
                candidate.opposite_reference.to_record()
                if candidate.prompt_reference is not None
                else None
            ),
        )
        prompts[candidate.proposal_id] = candidate.critic_prompt

    results = critic.evaluate_many(
        prompts,
        proposal_actions={
            candidate.proposal_id: candidate.proposal_action
            for candidate in candidates
            if candidate.proposal_id in prompts
        },
    )
    for candidate in candidates:
        result = results.get(candidate.proposal_id)
        if result is None:
            continue
        candidate.critic_state = result.state
        candidate.critic_action_reason = result.action_reason
        candidate.critic_content_reason = result.content_reason
        candidate.critic_error = result.error
        candidate.critic_attempts = result.attempts
        candidate.critic_raw_response = result.raw_response
        candidate.critic_action_reasonable = result.action_reasonable
        candidate.critic_content_supported = result.content_supported
        if candidate.proposal_action == "UPDATE_SKILL" and not (
            candidate.source.retrieved_skill or {}
        ).get("skill_id"):
            candidate.local_routing_reason = "missing_update_target"
            candidate.critic_action_reasonable = False
            candidate.critic_action_reason = "UPDATE_SKILL requires a retrieved skill."
        if result.state is CriticState.ERROR:
            candidate.action_reward = None
            candidate.skill_reward = None
            continue
        candidate.action_reward = (
            abs(float(invalid_penalty))
            if candidate.critic_action_reasonable
            else float(invalid_penalty)
        )
        if candidate.proposal_action == "NO_SKILL":
            candidate.skill_reward = None
        elif candidate.contains_chinese:
            candidate.local_routing_reason = "contains_chinese_skill"
            candidate.critic_content_supported = False
            candidate.skill_reward = float(invalid_penalty)
        elif candidate.critic_content_supported is False:
            candidate.skill_reward = float(invalid_penalty)
        elif candidate.critic_content_supported is True:
            apply_anchor_coverage_route(
                candidate,
                min_anchor_coverage=min_anchor_coverage,
            )

        if candidate.critic_action_reasonable is True and (
            candidate.proposal_action == "NO_SKILL"
            or candidate.critic_content_supported is True
        ):
            candidate.critic_state = CriticState.ACCEPT
        else:
            candidate.critic_state = CriticState.REJECT


def build_counterfactual_samples(
    candidates: list[ProposalCandidate],
) -> tuple[list[TeacherForcingSample], set[str]]:
    samples: list[TeacherForcingSample] = []
    invalid_candidates: set[str] = set()
    for candidate in candidates:
        if candidate.critic_content_supported is not True or candidate.proposal_action not in {
            "ADD_NEW_SKILL",
            "UPDATE_SKILL",
        }:
            continue
        if (
            candidate.contains_chinese
            or candidate.local_routing_reason == "insufficient_anchor_coverage"
            or candidate.exact_duplicate_source
        ):
            continue
        for anchor_type, trajectory in (
            ("success", candidate.success_anchor),
            ("failure", candidate.failure_anchor),
        ):
            anchor_samples = 0
            for step in trajectory.steps:
                if not any(step.action_token_mask):
                    continue
                anchor_samples += 1
                samples.append(
                    TeacherForcingSample(
                        sample_id=f"{candidate.proposal_id}:{anchor_type}:{step.step_id}",
                        prompt=render_action_prompt(
                            step.context, {"skill_text": candidate.skill_text}
                        ),
                        response_ids=step.response_ids,
                        response_mask=step.response_mask,
                        action_token_mask=step.action_token_mask,
                        metadata={
                            "proposal_id": candidate.proposal_id,
                            "anchor_type": anchor_type,
                            "step_id": step.step_id,
                        },
                    )
                )
            if anchor_samples == 0:
                invalid_candidates.add(candidate.proposal_id)
    if invalid_candidates:
        samples = [
            sample
            for sample in samples
            if sample.metadata["proposal_id"] not in invalid_candidates
        ]
    return samples, invalid_candidates


def original_step_logprob_scores(action_batch: DataProto) -> dict[str, float]:
    required = {"old_log_probs", "action_token_mask"}
    missing = required.difference(action_batch.batch.keys())
    if missing:
        raise KeyError(f"action batch missing counterfactual baseline fields: {sorted(missing)}")
    scores: dict[str, float] = {}
    for index, step_id in enumerate(action_batch.non_tensor_batch["step_id"]):
        if str(step_id) in scores:
            continue
        score = masked_mean(
            action_batch.batch["old_log_probs"][index].tolist(),
            action_batch.batch["action_token_mask"][index].tolist(),
        )
        if score is not None:
            scores[str(step_id)] = score
    return scores


def apply_alignment_scores(
    candidates: list[ProposalCandidate],
    *,
    scorer_batch: DataProto,
    scorer_log_probs: DataProto,
    original_scores: dict[str, float],
    reward_clip: float | None = None,
) -> None:
    candidate_step_scores: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    log_probs = scorer_log_probs.batch["old_log_probs"]
    for index, proposal_id in enumerate(scorer_batch.non_tensor_batch["proposal_id"]):
        score = masked_mean(
            log_probs[index].tolist(),
            scorer_batch.batch["action_token_mask"][index].tolist(),
        )
        if score is not None:
            key = (str(proposal_id), str(scorer_batch.non_tensor_batch["anchor_type"][index]))
            candidate_step_scores[key][str(scorer_batch.non_tensor_batch["step_id"][index])] = score

    for candidate in candidates:
        if candidate.critic_content_supported is not True or candidate.proposal_action not in {
            "ADD_NEW_SKILL",
            "UPDATE_SKILL",
        }:
            continue
        if (
            candidate.contains_chinese
            or candidate.local_routing_reason == "insufficient_anchor_coverage"
            or candidate.exact_duplicate_source
        ):
            continue
        deltas: dict[str, float] = {}
        failed = False
        for anchor_type, trajectory in (
            ("success", candidate.success_anchor),
            ("failure", candidate.failure_anchor),
        ):
            candidate_scores = candidate_step_scores.get((candidate.proposal_id, anchor_type), {})
            step_deltas = []
            for step in trajectory.steps:
                if not any(step.action_token_mask):
                    continue
                if step.step_id not in candidate_scores or step.step_id not in original_scores:
                    failed = True
                    break
                step_deltas.append(candidate_scores[step.step_id] - original_scores[step.step_id])
            if failed or not step_deltas:
                failed = True
                break
            deltas[anchor_type] = sum(step_deltas) / len(step_deltas)
        if failed:
            candidate.skill_reward = None
            candidate.critic_error = "counterfactual scoring missing one or more anchor steps"
            continue
        candidate.delta_success = deltas["success"]
        candidate.delta_failure = deltas["failure"]
        candidate.r_align = candidate.delta_success - candidate.delta_failure
        candidate.skill_reward = candidate.r_align
        if reward_clip is not None:
            candidate.skill_reward = max(
                -float(reward_clip),
                min(float(reward_clip), candidate.skill_reward),
            )


def assign_composite_policy_rewards(
    candidates: list[ProposalCandidate],
    *,
    format_only_warmup: bool,
) -> None:
    """Assign one complete Proposal reward or leave the row out of Proposal PG."""

    for candidate in candidates:
        candidate.policy_reward = None
        candidate.policy_reward_complete = False
        if format_only_warmup:
            if candidate.format_reward is not None:
                candidate.policy_reward = float(candidate.format_reward)
                candidate.policy_reward_complete = True
            continue
        if not candidate.parse_ok:
            if candidate.format_reward is not None:
                candidate.policy_reward = float(candidate.format_reward)
                candidate.policy_reward_complete = True
            continue
        if candidate.proposal_action == "NO_SKILL":
            required = (candidate.format_reward, candidate.action_reward)
        elif candidate.proposal_action in {"ADD_NEW_SKILL", "UPDATE_SKILL"}:
            required = (
                candidate.format_reward,
                candidate.action_reward,
                candidate.skill_reward,
            )
        else:
            continue
        if any(reward is None for reward in required):
            continue
        candidate.policy_reward = sum(float(reward) for reward in required)
        candidate.policy_reward_complete = True


def build_proposal_training_batch(
    generated: DataProto,
    candidates: list[ProposalCandidate],
    *,
    full_response_channel_masks: bool = False,
    composite_reward: bool = False,
) -> DataProto | None:
    all_response_masks = generated_response_mask(generated)
    for candidate in candidates:
        if candidate.train_index is not None and not all_response_masks[int(candidate.train_index)].any():
            candidate.format_reward = None
            candidate.action_reward = None
            candidate.skill_reward = None
            candidate.policy_reward = None
            candidate.policy_reward_complete = False
            candidate.critic_error = "proposal generation returned an empty response"
    trainable = [
        candidate
        for candidate in candidates
        if (
            candidate.train_index is not None
            and all_response_masks[int(candidate.train_index)].any()
            and (composite_reward or not candidate.is_masked)
        )
    ]
    if not trainable:
        return None
    indices = torch.tensor([int(candidate.train_index) for candidate in trainable], dtype=torch.long)
    batch = generated.select_idxs(indices)
    response_mask = generated_response_mask(batch)
    channel_names = ("format", "action", "skill")
    for channel in channel_names:
        batch.batch[f"proposal_{channel}_mask"] = torch.zeros_like(
            response_mask, dtype=torch.bool
        )
        batch.batch[f"proposal_{channel}_rewards"] = torch.zeros(
            (len(trainable), 1), dtype=torch.float32, device=response_mask.device
        )
        batch.batch[f"proposal_{channel}_valid"] = torch.zeros(
            (len(trainable), 1), dtype=torch.bool, device=response_mask.device
        )
    batch.batch["proposal_joint_mask"] = torch.zeros_like(
        response_mask, dtype=torch.bool
    )
    batch.batch["proposal_joint_rewards"] = torch.zeros(
        (len(trainable), 1), dtype=torch.float32, device=response_mask.device
    )
    batch.batch["proposal_joint_valid"] = torch.zeros(
        (len(trainable), 1), dtype=torch.bool, device=response_mask.device
    )
    operation_count = len(OPERATION_ACTIONS)
    batch.batch["proposal_operation_token_ids"] = torch.zeros(
        (len(trainable), operation_count),
        dtype=torch.long,
        device=response_mask.device,
    )
    batch.batch["proposal_operation_valid_mask"] = torch.zeros(
        (len(trainable), operation_count),
        dtype=torch.bool,
        device=response_mask.device,
    )
    batch.batch["proposal_operation_decision_position"] = torch.zeros(
        (len(trainable), 1),
        dtype=torch.long,
        device=response_mask.device,
    )
    batch.batch["proposal_operation_support_valid"] = torch.zeros(
        (len(trainable), 1),
        dtype=torch.bool,
        device=response_mask.device,
    )
    for index, candidate in enumerate(trainable):
        valid_positions = torch.nonzero(response_mask[index], as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            raise ValueError(f"proposal {candidate.proposal_id} has an empty generated response")
        for channel in channel_names:
            token_mask = torch.tensor(
                getattr(candidate, f"{channel}_token_mask"),
                dtype=torch.bool,
                device=response_mask.device,
            )
            reward = getattr(candidate, f"{channel}_reward")
            if reward is not None and not token_mask.any():
                raise ValueError(
                    f"proposal {candidate.proposal_id} has {channel} reward "
                    "but no trainable token"
                )
            if reward is not None:
                batch.batch[f"proposal_{channel}_mask"][index] = (
                    response_mask[index]
                    if full_response_channel_masks
                    else token_mask
                )
                batch.batch[f"proposal_{channel}_rewards"][index, 0] = float(reward)
                batch.batch[f"proposal_{channel}_valid"][index, 0] = True
        if composite_reward and candidate.policy_reward_complete:
            if candidate.policy_reward is None:
                raise ValueError(
                    f"proposal {candidate.proposal_id} has complete feedback "
                    "without reward"
                )
            batch.batch["proposal_joint_mask"][index] = response_mask[index]
            batch.batch["proposal_joint_rewards"][index, 0] = float(
                candidate.policy_reward
            )
            batch.batch["proposal_joint_valid"][index, 0] = True
        if candidate.operation_support_valid:
            batch.batch["proposal_operation_token_ids"][index] = torch.tensor(
                candidate.operation_token_ids,
                dtype=torch.long,
                device=response_mask.device,
            )
            batch.batch["proposal_operation_valid_mask"][index] = torch.tensor(
                candidate.operation_valid_mask,
                dtype=torch.bool,
                device=response_mask.device,
            )
            batch.batch["proposal_operation_decision_position"][index, 0] = int(
                candidate.operation_decision_position
            )
            batch.batch["proposal_operation_support_valid"][index, 0] = True
    batch.batch["response_mask"] = response_mask
    batch.non_tensor_batch["proposal_id"] = torch_to_object_array(
        [candidate.proposal_id for candidate in trainable]
    )
    batch.non_tensor_batch["proposal_action"] = torch_to_object_array(
        [candidate.proposal_action for candidate in trainable]
    )
    batch.non_tensor_batch["proposal_contains_chinese"] = torch_to_object_array(
        [candidate.contains_chinese for candidate in trainable]
    )
    batch.meta_info["multi_turn"] = False
    return batch


def torch_to_object_array(items: list[str]):
    # Local import keeps NumPy out of the hot scoring path above.
    import numpy as np

    return np.asarray(items, dtype=object)
