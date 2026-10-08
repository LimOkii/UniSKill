from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CriticSettings:
    config_path: str
    max_attempts: int
    parse_retry_attempts: int
    concurrency: int
    initial_backoff_seconds: float
    max_backoff_seconds: float
    jitter_seconds: float
    rate_limit_backoff_seconds: float = 60.0


@dataclass(frozen=True)
class DebugSettings:
    enabled: bool
    samples_per_batch: int
    max_action_steps: int
    show_token_ids: bool


@dataclass(frozen=True)
class UniSkillSettings:
    output_dir: str
    skill_dir: str
    embedding_dir: str
    proposal_max_prompt_length: int
    proposal_max_response_length: int
    best_checkpoint_min_success_rate: float
    lambda_proposal: float
    proposal_entropy_coeff: float
    operation_support_min_probability: float
    operation_support_coeff: float
    invalid_penalty: float
    r_align_warmup_steps: int
    min_anchor_coverage: float
    seed: int
    critic: CriticSettings
    debug: DebugSettings
    proposal_full_response_loss: bool = False
    proposal_kl_loss_coef: float = 0.01
    proposal_warmup_lambda: float = 0.5
    proposal_assistant_prefill: str = ""
    proposal_composite_reward: bool = False
    r_align_reward_clip: float | None = None

    @classmethod
    def from_config(cls, config) -> "UniSkillSettings":
        raw = config.get("uniskill")
        if not raw:
            raise ValueError("missing required +uniskill configuration")
        critic = raw.get("critic", {})
        debug = raw.get("debug", {})
        default_critic_path = Path(__file__).resolve().parent / "critic" / "config.yaml"
        settings = cls(
            output_dir=str(raw.get("output_dir", "uniskill/artifacts/records/alfworld/main")),
            skill_dir=str(raw.get("skill_dir", "uniskill/artifacts/runtime_skills/alfworld/main")),
            embedding_dir=str(raw.get("embedding_dir", "uniskill/artifacts/skill_embedding/alfworld/main")),
            proposal_max_prompt_length=int(raw.get("proposal_max_prompt_length", 4096)),
            proposal_max_response_length=int(
                raw.get(
                    "proposal_max_response_length",
                    config.data.max_response_length,
                )
            ),
            best_checkpoint_min_success_rate=float(
                raw.get("best_checkpoint_min_success_rate", 0.85)
            ),
            lambda_proposal=float(raw.get("lambda_proposal", 0.5)),
            proposal_entropy_coeff=float(raw.get("proposal_entropy_coeff", 0.0)),
            operation_support_min_probability=float(
                raw.get("operation_support_min_probability", 0.1)
            ),
            operation_support_coeff=float(
                raw.get("operation_support_coeff", 0.005)
            ),
            invalid_penalty=float(raw.get("invalid_penalty", -0.05)),
            r_align_warmup_steps=int(raw.get("r_align_warmup_steps", 3)),
            min_anchor_coverage=float(raw.get("min_anchor_coverage", 0.8)),
            seed=int(raw.get("seed", config.env.seed)),
            critic=CriticSettings(
                config_path=str(critic.get("config_path", default_critic_path)),
                max_attempts=int(critic.get("max_attempts", 5)),
                parse_retry_attempts=int(critic.get("parse_retry_attempts", 1)),
                concurrency=int(critic.get("concurrency", 4)),
                initial_backoff_seconds=float(critic.get("initial_backoff_seconds", 1.0)),
                max_backoff_seconds=float(critic.get("max_backoff_seconds", 30.0)),
                jitter_seconds=float(critic.get("jitter_seconds", 0.5)),
                rate_limit_backoff_seconds=float(
                    critic.get("rate_limit_backoff_seconds", 60.0)
                ),
            ),
            debug=DebugSettings(
                enabled=bool(debug.get("enabled", True)),
                samples_per_batch=int(debug.get("samples_per_batch", 1)),
                max_action_steps=int(debug.get("max_action_steps", 2)),
                show_token_ids=bool(debug.get("show_token_ids", True)),
            ),
            proposal_full_response_loss=bool(
                raw.get("proposal_full_response_loss", False)
            ),
            proposal_kl_loss_coef=float(
                raw.get(
                    "proposal_kl_loss_coef",
                    config.actor_rollout_ref.actor.get("kl_loss_coef", 0.01),
                )
            ),
            proposal_warmup_lambda=float(
                raw.get("proposal_warmup_lambda", raw.get("lambda_proposal", 0.5))
            ),
            proposal_assistant_prefill=str(
                raw.get("proposal_assistant_prefill", "")
            ),
            proposal_composite_reward=bool(
                raw.get("proposal_composite_reward", False)
            ),
            r_align_reward_clip=(
                float(raw["r_align_reward_clip"])
                if raw.get("r_align_reward_clip") is not None
                else None
            ),
        )
        settings.validate(config)
        return settings

    def validate(self, config) -> None:
        from uniskill.webshop_native import validate_native_webshop

        validate_native_webshop(config)
        if str(config.algorithm.adv_estimator).lower() != "grpo":
            raise ValueError("UniSkill requires algorithm.adv_estimator=grpo")
        environment = str(config.env.env_name)
        is_alfworld = environment == "alfworld/AlfredTWEnv"
        is_webshop = environment.lower() == "webshop"
        if not (is_alfworld or is_webshop):
            raise ValueError(
                "UniSkill supports env.env_name=alfworld/AlfredTWEnv or Webshop"
            )
        required_max_steps = 50 if is_alfworld else 15
        if int(config.env.max_steps) != required_max_steps:
            raise ValueError(
                f"UniSkill {environment} requires env.max_steps={required_max_steps}"
            )
        required_history_length = 2
        if int(config.env.history_length) != required_history_length:
            raise ValueError(
                f"UniSkill {environment} requires "
                f"env.history_length={required_history_length}"
            )
        retrieval_method = str(
            config.get("uniskill", {}).get("retrieval", {}).get("method", "")
        )
        if retrieval_method != "embedding":
            raise ValueError("UniSkill requires uniskill.retrieval.method=embedding")
        if int(config.env.rollout.n) != 8:
            raise ValueError("UniSkill requires env.rollout.n=8")
        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise ValueError("UniSkill joint updates require the FSDP or FSDP2 actor strategy")
        if config.actor_rollout_ref.actor.loss_agg_mode != "token-mean":
            raise ValueError("UniSkill joint updates require actor.loss_agg_mode=token-mean")
        if config.actor_rollout_ref.actor.use_dynamic_bsz:
            raise ValueError("UniSkill requires actor.use_dynamic_bsz=false")
        if config.actor_rollout_ref.rollout.mode == "async":
            raise ValueError("UniSkill requires synchronous rollout so action and proposal share pi_old")
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            raise ValueError("UniSkill uses the external environment loop and requires rollout.multi_turn.enable=false")
        if int(config.actor_rollout_ref.actor.ppo_epochs) != 1:
            raise ValueError("UniSkill requires actor_rollout_ref.actor.ppo_epochs=1")
        if config.algorithm.filter_groups.enable:
            raise ValueError("UniSkill requires algorithm.filter_groups.enable=false")
        total_epochs = int(config.trainer.total_epochs)
        if is_alfworld and total_epochs != 250:
            raise ValueError(
                f"UniSkill {environment} requires trainer.total_epochs=250"
            )
        if is_webshop and total_epochs <= 0:
            raise ValueError("UniSkill WebShop requires trainer.total_epochs > 0")
        if self.proposal_max_prompt_length <= 0:
            raise ValueError("uniskill.proposal_max_prompt_length must be positive")
        if self.proposal_max_response_length <= 0:
            raise ValueError("uniskill.proposal_max_response_length must be positive")
        if (
            self.proposal_max_prompt_length + self.proposal_max_response_length
            > int(config.actor_rollout_ref.rollout.max_model_len)
        ):
            raise ValueError(
                "proposal prompt/response budget exceeds "
                "actor_rollout_ref.rollout.max_model_len"
            )
        if (
            int(config.data.max_prompt_length) + int(config.data.max_response_length)
            > int(config.actor_rollout_ref.rollout.max_model_len)
        ):
            raise ValueError(
                "action prompt/response budget exceeds "
                "actor_rollout_ref.rollout.max_model_len"
            )
        if not 0 <= self.best_checkpoint_min_success_rate <= 1:
            raise ValueError(
                "uniskill.best_checkpoint_min_success_rate must be in [0, 1]"
            )
        if abs(self.lambda_proposal - 0.5) > 1e-12:
            raise ValueError("UniSkill requires uniskill.lambda_proposal=0.5")
        if not 0 < self.proposal_warmup_lambda <= self.lambda_proposal:
            raise ValueError(
                "uniskill.proposal_warmup_lambda must be positive and no greater "
                "than uniskill.lambda_proposal"
            )
        if abs(self.proposal_entropy_coeff) > 1e-12:
            raise ValueError("UniSkill requires uniskill.proposal_entropy_coeff=0")
        if abs(self.operation_support_min_probability - 0.1) > 1e-12:
            raise ValueError(
                "UniSkill requires "
                "uniskill.operation_support_min_probability=0.1"
            )
        if self.operation_support_coeff <= 0:
            raise ValueError("uniskill.operation_support_coeff must be positive")
        if self.proposal_kl_loss_coef < 0:
            raise ValueError("uniskill.proposal_kl_loss_coef must be non-negative")
        if self.r_align_reward_clip is not None and self.r_align_reward_clip <= 0:
            raise ValueError("uniskill.r_align_reward_clip must be positive")
        if self.proposal_composite_reward:
            if not self.proposal_full_response_loss:
                raise ValueError(
                    "composite Proposal reward requires "
                    "proposal_full_response_loss=true"
                )
            if is_alfworld and self.proposal_assistant_prefill != "<action>":
                raise ValueError(
                    "composite ALFWorld Proposal training requires <action> prefill"
                )
            if self.r_align_reward_clip is None:
                raise ValueError(
                    "composite Proposal reward requires r_align_reward_clip"
                )
        if is_webshop and self.proposal_assistant_prefill:
            raise ValueError(
                "WebShop Proposal training must not use the ALFWorld action prefill"
            )
        if self.invalid_penalty >= 0:
            raise ValueError("uniskill.invalid_penalty must be negative")
        if self.r_align_warmup_steps < 0:
            raise ValueError("uniskill.r_align_warmup_steps must be non-negative")
        if abs(self.min_anchor_coverage - 0.8) > 1e-12:
            raise ValueError("UniSkill requires uniskill.min_anchor_coverage=0.8")
        if self.critic.concurrency <= 0 or self.critic.max_attempts <= 0:
            raise ValueError("critic concurrency and max_attempts must be positive")
        if self.critic.rate_limit_backoff_seconds < 60:
            raise ValueError("critic rate_limit_backoff_seconds must be at least 60")
        if self.debug.samples_per_batch < 0:
            raise ValueError("uniskill.debug.samples_per_batch must be non-negative")
        if self.debug.enabled and self.debug.samples_per_batch == 0:
            raise ValueError("enabled debug tracing requires samples_per_batch > 0")
