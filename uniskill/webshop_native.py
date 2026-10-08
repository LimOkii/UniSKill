"""WebShop training settings and actor credit timing."""
from __future__ import annotations


def uses_native_webshop_actor(config) -> bool:
    return str(config.env.env_name).lower() == "webshop"


def strict_actor_credit_active(config, *, global_step: int) -> bool:
    if not uses_native_webshop_actor(config):
        return False
    start_step = int(config.uniskill.webshop.get("strict_actor_credit_start_step", 31))
    return int(global_step) >= start_step


def validate_native_webshop(config) -> None:
    if not uses_native_webshop_actor(config):
        return
    settings = config.uniskill
    webshop = settings.webshop
    if webshop.get("seed_initial_skill", False):
        raise ValueError("WebShop training requires an empty initial skill bank")
    if webshop.get("task_score_reward_scale") is not None:
        raise ValueError("WebShop uses binary task rewards")
    if not bool(config.algorithm.get("credit_assignment", False)):
        raise ValueError("WebShop requires credit_assignment=true")
    if abs(float(config.algorithm.get("step_gamma", 0.95)) - 0.95) > 1e-12:
        raise ValueError("WebShop requires algorithm.step_gamma=0.95")
    if not bool(webshop.get("selected_options_state", False)):
        raise ValueError("WebShop requires selected_options_state=true")
    if not bool(config.actor_rollout_ref.actor.get("replace_strict_format_invalid_score", False)):
        raise ValueError("WebShop requires replace_strict_format_invalid_score=true")
    strict_start = int(webshop.get("strict_actor_credit_start_step", 31))
    if strict_start != int(settings.r_align_warmup_steps) + 1:
        raise ValueError("WebShop strict actor credit must start after warmup")
    samples = int(webshop.get("warmup_samples", 8))
    if not 0 < samples <= int(config.data.train_batch_size) * int(config.env.rollout.n):
        raise ValueError("WebShop warmup_samples must fit the rollout batch")
    if int(settings.retrieval.get("start_step", 0)) != int(settings.r_align_warmup_steps) + 1:
        raise ValueError("WebShop retrieval must start after proposal warmup")
    if abs(float(settings.retrieval.get("embedding_min_score", 0.55)) - 0.55) > 1e-12:
        raise ValueError("WebShop requires retrieval.embedding_min_score=0.55")
    if not bool(settings.get("proposal_composite_reward", False)):
        raise ValueError("WebShop requires proposal_composite_reward=true")
    reward_clip = settings.get("r_align_reward_clip")
    if reward_clip is None or abs(float(reward_clip) - 0.05) > 1e-12:
        raise ValueError("WebShop requires r_align_reward_clip=0.05")
