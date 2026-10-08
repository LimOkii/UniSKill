from __future__ import annotations

NO_RETRIEVED_SKILL = "No retrieved skill is available."
PROPOSAL_EXAMPLE_SKILL = ""


def format_skill_context(skill: dict | None) -> str:
    if not skill:
        return NO_RETRIEVED_SKILL
    skill_text = str(skill.get("skill_text") or "").strip()
    if not skill_text:
        skill_text = " ".join(
            str(skill.get(key) or "").strip()
            for key in ("title", "principle", "when_to_apply")
        ).strip()
    return "\n".join(["## Retrieved Relevant Experience for Reference", skill_text])


WEBSHOP_ACTION_TEMPLATE_NO_HISTORY = """
You are in ACTION MODE. You are an expert autonomous agent operating in the WebShop e-commerce environment. Your task is to: {task_description}
{skill_context}
Your current observation is: {current_observation}
Your admissible actions of the current situation are:
[
{admissible_actions}
]

Now it is your turn to take one action for the current step.
First reason step-by-step about the current situation. This reasoning MUST be enclosed within <think> </think> tags.
Then choose one admissible action and present it within <action> </action> tags. For search, replace <your query> with concrete keywords.
"""


WEBSHOP_ACTION_TEMPLATE = """
You are in ACTION MODE. You are an expert autonomous agent operating in the WebShop e-commerce environment. Your task is to: {task_description}
{skill_context}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and corresponding actions: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are:
[
{admissible_actions}
]

Now it is your turn to take one action for the current step.
First reason step-by-step about the current situation. This reasoning MUST be enclosed within <think> </think> tags.
Then choose one admissible action and present it within <action> </action> tags. For search, replace <your query> with concrete keywords.
"""


WEBSHOP_SKILL_PROPOSAL_TEMPLATE = """
You are in SKILL PROPOSAL MODE. You are not acting in the environment.
Your job is to compare a completed WebShop episode with an opposite-outcome reference and propose one skill-management action.

CURRENT_EPISODE:
{source_context}

OPPOSITE_OUTCOME_REFERENCE:
{comparison_context}

Use the reference only to identify reusable behavioral differences that are supported by the recorded actions and outcomes. Do not infer hidden causes, memorize product identifiers, or copy an exact action path.

Now decide one skill-management action based on the comparison above.

Available actions:
{available_actions}

Rules:
- Choose "NO_SKILL" if the comparison does not reveal concrete reusable guidance, or if the retrieved skill already contains that guidance.
- Choose "ADD_NEW_SKILL" only if the comparison reveals distinct reusable guidance that should stand as a separate skill.
{update_rule}
- A skill must be concrete, supported by the recorded actions and outcomes, and reusable across shopping tasks, rather than merely restating this request, copying the retrieved skill, paraphrasing another skill, or memorizing a product.
- For "NO_SKILL", write exactly NONE in the skill block.
{skill_rule}

Your response must contain exactly two tagged blocks:
1. <action> containing only {allowed_action_values}.
2. <skill> containing only the plain-text skill, or NONE for NO_SKILL.

Do not output reasoning, JSON, Markdown fences, or any text outside these blocks.
End immediately after </skill>.
"""


def _compact_episode_context(record: dict, *, include_retrieved_skill: bool) -> str:
    lines = [
        "Task:",
        f"- Type: {record.get('task_type') or ''}",
        f"- Description: {record.get('task_description') or ''}",
    ]
    if include_retrieved_skill:
        lines.extend(
            [
                "Retrieved Skill:",
                _format_proposal_skill(record.get("retrieved_skill")),
            ]
        )
    lines.extend(
        [
            "Outcome:",
            f"- Success: {bool(record.get('success', False))}",
            f"- End Reason: {record.get('end_reason') or ''}",
        ]
    )
    if record.get("task_score") is not None:
        lines.append(f"- WebShop Task Score: {float(record['task_score']):.6f}")
    lines.append("Action Sequence:")
    actions = [
        _proposal_context_action(step)
        for step in record.get("steps", [])
    ]
    if actions:
        lines.extend(
            f"- Step {index}: {action}"
            for index, action in enumerate(actions, start=1)
        )
    else:
        lines.append("- No recorded action.")
    return "\n".join(lines)


def _proposal_context_action(step: dict) -> str:
    display = " ".join(str(step.get("action_display") or "").split())
    if display:
        return display
    parsed = " ".join(str(step.get("action_parsed") or "").split())
    if parsed:
        return parsed
    return "[INVALID ACTION OMITTED]"


def _format_proposal_skill(skill: dict | None) -> str:
    if not skill:
        return "- No retrieved skill is available."
    skill_text = str(skill.get("skill_text") or "").strip()
    if not skill_text:
        skill_text = " ".join(
            str(skill.get(key) or "").strip()
            for key in ("title", "principle", "when_to_apply")
        ).strip()
    return "\n".join(
        [
            f"- Skill ID: {skill.get('skill_id', '')}",
            f"- Skill: {skill_text}",
        ]
    )


def _comparison_context(comparison_record: dict | None) -> str:
    if comparison_record is None:
        return (
            "No opposite-outcome reference is available during format warmup. "
            "Follow the required output schema."
        )
    return _compact_episode_context(
        comparison_record,
        include_retrieved_skill=False,
    )


def render_action_prompt(context: dict, skill: dict | None) -> str:
    history = list(context.get("history") or [])
    admissible_actions = "\n".join(
        f"'{action}'," for action in context.get("admissible_actions", [])
    )
    common = {
        "task_description": context.get("task_description", ""),
        "skill_context": format_skill_context(skill),
        "current_observation": context.get("current_observation", ""),
        "admissible_actions": admissible_actions,
    }
    if not history:
        return WEBSHOP_ACTION_TEMPLATE_NO_HISTORY.format(**common)

    history_text = "\n".join(
        "[Observation "
        f"{item.get('step', index + 1)}: '{item.get('observation', '')}', "
        f"Action {item.get('step', index + 1)}: '{item.get('action', '')}']"
        for index, item in enumerate(history)
    )
    return WEBSHOP_ACTION_TEMPLATE.format(
        **common,
        step_count=int(context.get("step_count", len(history))),
        history_length=len(history),
        action_history=history_text,
        current_step=int(context.get("current_step", len(history) + 1)),
    )


def build_skill_proposal_prompt(
    episode_record: dict,
    *,
    comparison_record: dict | None = None,
) -> str:
    retrieved_skill = episode_record.get("retrieved_skill") or {}
    has_retrieved_skill = bool(retrieved_skill.get("skill_id"))
    if has_retrieved_skill:
        available_actions = "\n".join(
            ["- NO_SKILL", "- ADD_NEW_SKILL", "- UPDATE_SKILL"]
        )
        update_rule = (
            '- Choose "UPDATE_SKILL" if the retrieved skill should be improved '
            "based on this comparison."
        )
        skill_rule = (
            '- For "ADD_NEW_SKILL" or "UPDATE_SKILL", write one self-contained '
            "reusable skill in plain text."
        )
        allowed_action_values = "NO_SKILL, ADD_NEW_SKILL, or UPDATE_SKILL"
    else:
        available_actions = "\n".join(["- NO_SKILL", "- ADD_NEW_SKILL"])
        update_rule = ""
        skill_rule = (
            '- For "ADD_NEW_SKILL", write one self-contained reusable skill in '
            "plain text."
        )
        allowed_action_values = "NO_SKILL or ADD_NEW_SKILL"

    return WEBSHOP_SKILL_PROPOSAL_TEMPLATE.format(
        source_context=_compact_episode_context(
            episode_record,
            include_retrieved_skill=True,
        ),
        comparison_context=_comparison_context(comparison_record),
        available_actions=available_actions,
        update_rule=update_rule,
        skill_rule=skill_rule,
        allowed_action_values=allowed_action_values,
    )
