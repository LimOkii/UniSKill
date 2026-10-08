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
    return "\n".join(
        ["## Retrieved Relevant Experience for Reference", skill_text]
    )


ALFWORLD_ACTION_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
{skill_context}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


ALFWORLD_ACTION_TEMPLATE = """
You are in ACTION MODE. You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
{skill_context}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


def _compact_opposite_context(record: dict) -> str:
    """Keep the comparison useful without doubling a long ALFWorld prompt."""

    lines = [
        "Task:",
        f"- Type: {record.get('task_type') or ''}",
        f"- Description: {record.get('task_description') or ''}",
        "Outcome:",
        f"- Success: {bool(record.get('success', False))}",
        f"- End Reason: {record.get('end_reason') or ''}",
        "Action Sequence:",
    ]
    actions = []
    for step in record.get("steps", []):
        action = " ".join(str(step.get("action_display") or "").split())
        if not action:
            action = " ".join(str(step.get("action_parsed") or "").split())
        actions.append(action or "[INVALID ACTION OMITTED]")
    if actions:
        lines.extend(
            f"- Step {index}: {action}"
            for index, action in enumerate(actions, start=1)
        )
    else:
        lines.append("- No recorded action.")
    return "\n".join(lines)


ALFWORLD_SKILL_PROPOSAL_TEMPLATE = """
You are in SKILL PROPOSAL MODE. You are not acting in the environment.
Your job is to compare a completed ALFWorld episode with an opposite-outcome reference and propose one skill management action.

CURRENT_EPISODE:
{source_context}

OPPOSITE_OUTCOME_REFERENCE:
{comparison_context}

Use the reference only to identify reusable behavioral differences that are supported by the recorded observations, actions, and outcomes. Do not infer hidden causes, memorize instance-specific identifiers, or copy an exact action path.

Now decide one skill-management action based on the comparison above.

Available actions:
{available_actions}

Rules:
- Choose "NO_SKILL" if the comparison does not reveal concrete reusable guidance, or if the retrieved skill already contains that guidance.
- Choose "ADD_NEW_SKILL" if the comparison reveals distinct reusable guidance that should stand as a separate skill.
{update_rule}
- A skill must be concrete, supported by the trajectory comparison, and usable from the actor's available prompt, rather than merely restating the task, copying the retrieved skill, or memorizing this episode or the reference.
- For "NO_SKILL", write exactly NONE in the skill block.
{skill_rule}

Your response must contain exactly two tagged blocks:
1. <action> containing only {allowed_action_values}.
2. <skill> containing only the plain-text skill, or NONE for NO_SKILL.

Do not output reasoning, JSON, Markdown fences, or any text outside these blocks.
End immediately after </skill>.

"""


def render_action_prompt(context: dict, skill: dict | None) -> str:
    """Render the exact action-mode user message with a replaceable skill slot."""

    skill_context = format_skill_context(skill)
    history = list(context.get("history") or [])
    admissible_actions = "\n ".join(
        f"'{action}'" for action in context.get("admissible_actions", []) if action != "help"
    )
    if not history:
        return ALFWORLD_ACTION_TEMPLATE_NO_HIS.format(
            skill_context=skill_context,
            current_observation=context.get("current_observation", ""),
            admissible_actions=admissible_actions,
        )

    history_text = "\n".join(
        "[Observation "
        f"{item.get('step', index + 1)}: '{item.get('observation', '')}', "
        f"Action {item.get('step', index + 1)}: '{item.get('action', '')}']"
        for index, item in enumerate(history)
    )
    return ALFWORLD_ACTION_TEMPLATE.format(
        task_description=context.get("task_description", ""),
        skill_context=skill_context,
        step_count=int(context.get("step_count", len(history))),
        history_length=len(history),
        action_history=history_text,
        current_step=int(context.get("current_step", len(history) + 1)),
        current_observation=context.get("current_observation", ""),
        admissible_actions=admissible_actions,
    )


def build_skill_proposal_prompt(
    episode_record: dict,
    *,
    comparison_record: dict | None = None,
) -> str:
    # Imported lazily to keep prompt rendering independent from proposal I/O.
    from uniskill.proposal.episode_context import build_episode_context

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

    return ALFWORLD_SKILL_PROPOSAL_TEMPLATE.format(
        source_context=build_episode_context(episode_record),
        comparison_context=(
            _compact_opposite_context(comparison_record)
            if comparison_record is not None
            else "No opposite-outcome reference is available."
        ),
        available_actions=available_actions,
        update_rule=update_rule,
        skill_rule=skill_rule,
        allowed_action_values=allowed_action_values,
    )
