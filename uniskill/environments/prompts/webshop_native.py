"""Render the exact native prompt captured at rollout, plus an optional skill."""
from __future__ import annotations

from uniskill.environments.prompts.webshop import format_skill_context


def render_action_prompt(context: dict, skill: dict | None) -> str:
    # Keep the skill-free prompt verbatim, including native history truncation.
    # Counterfactual scoring uses this same base, never a skill-augmented copy.
    prompt = context["native_action_prompt"]
    if not skill:
        return prompt
    return prompt.replace(
        "Your task is to:",
        format_skill_context(skill) + "\nYour task is to:",
        1,
    )
