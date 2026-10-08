from __future__ import annotations

from uniskill.environments.prompts import alfworld, webshop


def _environment_name(payload: dict) -> str:
    return str(payload.get("environment") or "alfworld").lower()


def render_action_prompt(context: dict, skill: dict | None) -> str:
    if "webshop" in _environment_name(context):
        if "native_action_prompt" in context:
            from uniskill.environments.prompts.webshop_native import render_action_prompt as render_native

            return render_native(context, skill)
        return webshop.render_action_prompt(context, skill)
    return alfworld.render_action_prompt(context, skill)


def build_skill_proposal_prompt(
    episode_record: dict,
    comparison_record: dict | None = None,
) -> str:
    if "webshop" in _environment_name(episode_record):
        return webshop.build_skill_proposal_prompt(
            episode_record,
            comparison_record=comparison_record,
        )
    return alfworld.build_skill_proposal_prompt(
        episode_record,
        comparison_record=comparison_record,
    )


def proposal_example_skill(environment: str | None) -> str:
    if "webshop" in str(environment or "").lower():
        return webshop.PROPOSAL_EXAMPLE_SKILL
    return alfworld.PROPOSAL_EXAMPLE_SKILL
