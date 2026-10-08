from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CriticState(str, Enum):
    NOT_RUN = "not_run"
    ACCEPT = "accept"
    REJECT = "reject"
    ERROR = "error"


@dataclass
class ActionStep:
    """One actor decision together with the exact context used to render it."""

    step_id: str
    query_id: str
    trajectory_id: str
    step_index: int
    context: dict[str, Any]
    response_ids: list[int]
    response_mask: list[int]
    action_token_mask: list[int]
    response_text: str
    action_text: str
    reward: float
    done: bool
    is_action_valid: bool
    is_action_strict_format_valid: bool = False
    action_strict_format_error: str = ""
    is_action_admissible: bool = False
    is_action_overall_valid: bool = False
    action_payload_parsed: bool = False
    action_display: str = ""
    action_history_status: str = ""
    contains_chinese: bool = False
    missing_action_tags: bool = False
    missing_think_tags: bool = False
    response_token_length: int = 0
    response_hit_length_limit: bool = False
    response_truncated: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step": self.step_index,
            "observation": self.context.get("current_observation", ""),
            "think": self.context.get("think", ""),
            "action_raw": self.response_text,
            "action_parsed": self.action_text,
            "action_display": self.action_display,
            "reward": self.reward,
            "done": self.done,
            "is_action_valid": self.is_action_valid,
            "is_action_strict_format_valid": self.is_action_strict_format_valid,
            "action_strict_format_error": self.action_strict_format_error,
            "is_action_admissible": self.is_action_admissible,
            "is_action_overall_valid": self.is_action_overall_valid,
            "action_payload_parsed": self.action_payload_parsed,
            "action_history_status": self.action_history_status,
            "contains_chinese": self.contains_chinese,
            "missing_action_tags": self.missing_action_tags,
            "missing_think_tags": self.missing_think_tags,
            "response_token_length": self.response_token_length,
            "response_hit_length_limit": self.response_hit_length_limit,
            "response_truncated": self.response_truncated,
        }


@dataclass
class Trajectory:
    query_id: str
    trajectory_id: str
    task_type: str | None
    task_description: str
    gamefile: str | None
    episode_reward: float
    success: bool
    end_reason: str
    retrieved_skill: dict[str, Any] | None
    retrieval: dict[str, Any]
    steps: list[ActionStep] = field(default_factory=list)
    environment: str | None = None
    task_score: float | None = None

    def to_record(self) -> dict[str, Any]:
        record = {
            "query_id": self.query_id,
            "episode_id": self.trajectory_id,
            "task_type": self.task_type,
            "task_description": self.task_description,
            "gamefile": self.gamefile,
            "episode_reward": self.episode_reward,
            "success": self.success,
            "end_reason": self.end_reason,
            "retrieved_skill": self.retrieved_skill,
            "retrieval": self.retrieval,
            "steps": [step.to_record() for step in self.steps],
        }
        if self.environment is not None:
            record["environment"] = self.environment
        if self.task_score is not None:
            record["task_score"] = self.task_score
        return record


@dataclass
class ProposalCandidate:
    proposal_id: str
    source: Trajectory
    success_anchor: Trajectory | None
    failure_anchor: Trajectory | None
    prompt: str
    prompt_reference: Trajectory | None = None
    response_text: str = ""
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    model_prompt_ids: list[int] = field(default_factory=list)
    proposal_action: str = ""
    skill_text: str = ""
    parse_ok: bool = False
    parse_error: str = ""
    recoverable_invalid: bool = False
    local_routing_reason: str = ""
    critic_prompt: str = ""
    critic_state: CriticState = CriticState.NOT_RUN
    critic_action_reason: str = ""
    critic_content_reason: str = ""
    critic_error: str = ""
    critic_attempts: int = 0
    critic_raw_response: str = ""
    critic_action_reasonable: bool | None = None
    critic_content_supported: bool | None = None
    exact_duplicate_source: str = ""
    format_token_mask: list[int] = field(default_factory=list)
    action_token_mask: list[int] = field(default_factory=list)
    skill_token_mask: list[int] = field(default_factory=list)
    format_reward: float | None = None
    action_reward: float | None = None
    skill_reward: float | None = None
    policy_reward: float | None = None
    policy_reward_complete: bool = False
    r_align: float | None = None
    delta_success: float | None = None
    delta_failure: float | None = None
    success_anchor_coverage: float | None = None
    failure_anchor_coverage: float | None = None
    anchor_coverage_threshold: float | None = None
    train_index: int | None = None
    prompt_token_length: int = 0
    prompt_truncated: bool = False
    response_token_length: int = 0
    response_hit_length_limit: bool = False
    response_truncated: bool = False
    contains_chinese: bool = False
    included_in_policy_loss: bool = False
    operation_token_ids: list[int] = field(default_factory=list)
    operation_valid_mask: list[bool] = field(default_factory=list)
    operation_decision_position: int = -1
    operation_support_valid: bool = False
    operation_support_error: str = ""

    @property
    def is_masked(self) -> bool:
        return all(
            reward is None
            for reward in (self.format_reward, self.action_reward, self.skill_reward)
        )

    @property
    def minimum_anchor_coverage(self) -> float | None:
        if self.success_anchor_coverage is None or self.failure_anchor_coverage is None:
            return None
        return min(self.success_anchor_coverage, self.failure_anchor_coverage)

    @property
    def opposite_reference(self) -> Trajectory | None:
        if self.prompt_reference is not None:
            return self.prompt_reference
        if self.source.success:
            return self.failure_anchor
        return self.success_anchor

    @property
    def is_write_eligible(self) -> bool:
        return (
            self.critic_action_reasonable is True
            and self.critic_content_supported is True
            and self.r_align is not None
            and self.r_align > 0
            and self.delta_success is not None
            and self.delta_success > 0
            and self.proposal_action in {"ADD_NEW_SKILL", "UPDATE_SKILL"}
        )

    def to_record(self) -> dict[str, Any]:
        role_ids = {
            self.source.trajectory_id,
            getattr(self.opposite_reference, "trajectory_id", None),
            getattr(self.success_anchor, "trajectory_id", None),
            getattr(self.failure_anchor, "trajectory_id", None),
        }
        return {
            "proposal_id": self.proposal_id,
            "query_id": self.source.query_id,
            "source_trajectory_id": self.source.trajectory_id,
            "success_anchor_id": getattr(self.success_anchor, "trajectory_id", None),
            "failure_anchor_id": getattr(self.failure_anchor, "trajectory_id", None),
            "opposite_reference_id": getattr(
                self.opposite_reference, "trajectory_id", None
            ),
            "opposite_reference_success": getattr(self.opposite_reference, "success", None),
            "proposal_reference_mode": (
                "held_out_opposite"
                if self.prompt_reference is not None
                else "source_only"
            ),
            "four_trajectory_roles_distinct": None not in role_ids and len(role_ids) == 4,
            "task_type": self.source.task_type,
            "task_description": self.source.task_description,
            "retrieved_skill": self.source.retrieved_skill,
            "proposal_raw": self.response_text,
            "prompt_token_length": self.prompt_token_length,
            "prompt_truncated": self.prompt_truncated,
            "response_token_length": self.response_token_length,
            "response_hit_length_limit": self.response_hit_length_limit,
            "response_truncated": self.response_truncated,
            "contains_chinese": self.contains_chinese,
            "included_in_policy_loss": self.included_in_policy_loss,
            "operation_support_valid": self.operation_support_valid,
            "operation_support_error": self.operation_support_error,
            "proposal_action": self.proposal_action,
            "skill_text": self.skill_text,
            "parse_ok": self.parse_ok,
            "parse_error": self.parse_error,
            "recoverable_invalid": self.recoverable_invalid,
            "local_routing_reason": self.local_routing_reason,
            "critic_called": bool(self.critic_prompt),
            "critic_state": self.critic_state.value,
            "critic_action_reason": self.critic_action_reason,
            "critic_content_reason": self.critic_content_reason,
            "critic_error": self.critic_error,
            "critic_attempts": self.critic_attempts,
            "critic_raw_response": self.critic_raw_response,
            "critic_action_reasonable": self.critic_action_reasonable,
            "critic_content_supported": self.critic_content_supported,
            "exact_duplicate_source": self.exact_duplicate_source,
            "channel_rewards": {
                "format": self.format_reward,
                "action": self.action_reward,
                "skill": self.skill_reward,
            },
            "policy_reward": self.policy_reward,
            "policy_reward_complete": self.policy_reward_complete,
            "r_align": self.r_align,
            "delta_success": self.delta_success,
            "delta_failure": self.delta_failure,
            "success_anchor_coverage": self.success_anchor_coverage,
            "failure_anchor_coverage": self.failure_anchor_coverage,
            "minimum_anchor_coverage": self.minimum_anchor_coverage,
            "anchor_coverage_threshold": self.anchor_coverage_threshold,
            "written": False,
            "write_status": "not_committed",
        }
