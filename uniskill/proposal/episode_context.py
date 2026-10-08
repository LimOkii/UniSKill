from __future__ import annotations

def build_episode_context(record: dict) -> str:
    lines = [
        "## Task",
        f"Task Type: {record.get('task_type') or ''}",
        f"Task Description: {record.get('task_description') or ''}",
        "",
        "## Retrieved Skill",
        _format_skill(record.get("retrieved_skill")),
        "",
        "## Outcome",
        f"Success: {bool(record.get('success', False))}",
        f"End Reason: {record.get('end_reason') or ''}",
    ]
    if record.get("task_score") is not None:
        lines.append(f"WebShop Task Score: {float(record['task_score']):.6f}")
    lines.extend(["", "## Trajectory"])

    for step in record.get("steps", []):
        action = _proposal_context_action(step)
        think = step.get("think") or ""
        lines.extend(
            [
                f"Step {step.get('step')}",
                f"Observation: {step.get('observation') or ''}",
                f"Reasoning: {think}",
                f"Action: {action}",
                "",
            ]
        )

    return "\n".join(lines).strip()


def _proposal_context_action(step: dict) -> str:
    """Render the same canonical action representation used by action history."""

    display = " ".join(str(step.get("action_display") or "").split())
    if display:
        return display

    # Use the parsed action if no display form is available.
    parsed = " ".join(str(step.get("action_parsed") or "").split())
    if parsed:
        return parsed
    return "[INVALID ACTION OMITTED]"


def _format_skill(skill: dict | None) -> str:
    if not skill:
        return "No retrieved skill is available."
    skill_text = str(skill.get("skill_text") or "").strip()
    if not skill_text:
        skill_text = " ".join(
            str(skill.get(key) or "").strip()
            for key in ("title", "principle", "when_to_apply")
        ).strip()
    return f"Skill ID: {skill.get('skill_id', '')}\nSkill: {skill_text}"
