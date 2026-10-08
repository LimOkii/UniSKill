from __future__ import annotations

import re

from uniskill.response_format import contains_chinese

VALID_ACTIONS = {"NO_SKILL", "ADD_NEW_SKILL", "UPDATE_SKILL"}

_PROPOSAL_PATTERN = re.compile(
    r"\A\s*<action>(?P<action>.*?)</action>\s*"
    r"<skill>(?P<skill>.*?)</skill>\s*\Z",
    re.DOTALL,
)
_ACTION_BLOCK_PATTERN = re.compile(r"<action>(?P<action>.*?)</action>", re.DOTALL)
_SKILL_BLOCK_PATTERN = re.compile(r"<skill>(?P<skill>.*?)</skill>", re.DOTALL)


def _trimmed_span(match: re.Match[str], group: str) -> tuple[int, int]:
    value = match.group(group)
    leading = len(value) - len(value.lstrip())
    trailing = len(value) - len(value.rstrip())
    return match.start(group) + leading, match.end(group) - trailing


def _non_whitespace_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    value = text[start:end]
    leading = len(value) - len(value.lstrip())
    trailing = len(value) - len(value.rstrip())
    span = (start + leading, end - trailing)
    return span if span[0] < span[1] else None


def _canonical_action(value: str) -> str:
    """Accept only the three exact, case-sensitive protocol labels."""

    stripped = value.strip()
    return stripped if stripped in VALID_ACTIONS else ""


def _semantic_error(
    *,
    proposal_action: str,
    raw_action: str,
    skill_text: str,
) -> tuple[str, str]:
    if not proposal_action:
        return f"invalid proposal action: {raw_action}", "action"
    if proposal_action == "NO_SKILL":
        if skill_text != "NONE":
            return "NO_SKILL requires <skill>NONE</skill>", "skill"
    elif not skill_text or skill_text == "NONE":
        return f"{proposal_action} requires non-empty skill text", "skill"
    return "", ""


def parse_proposal(text: str) -> dict:
    """Parse the strict two-block protocol and describe recoverable errors.

    Language validity is deliberately separate from structural validity. A
    Chinese skill remains structurally parseable so the critic can still train
    the management-action tokens, while the skill channel receives its language
    penalty. Character spans are diagnostic only: every structurally
    invalid response receives one full-response format penalty.
    """

    result = {
        "proposal_action": "",
        "skill_text": "",
        "contains_chinese": False,
        "parse_ok": False,
        "parse_error": "",
        "recoverable_invalid": False,
        "action_span": None,
        "skill_span": None,
        "format_spans": [],
        "penalty_spans": [],
    }
    tag_counts = {
        "action_open": text.count("<action>"),
        "action_close": text.count("</action>"),
        "skill_open": text.count("<skill>"),
        "skill_close": text.count("</skill>"),
    }
    if any(count != 1 for count in tag_counts.values()):
        result["contains_chinese"] = contains_chinese(text)
        result["parse_error"] = (
            "proposal response must contain exactly one <action> block and one "
            "<skill> block"
        )
        return result

    match = _PROPOSAL_PATTERN.fullmatch(text)
    outside_penalty_spans: list[tuple[int, int]] = []
    if match is None:
        action_matches = list(_ACTION_BLOCK_PATTERN.finditer(text))
        skill_matches = list(_SKILL_BLOCK_PATTERN.finditer(text))
        if (
            len(action_matches) != 1
            or len(skill_matches) != 1
            or action_matches[0].end() > skill_matches[0].start()
        ):
            result["contains_chinese"] = contains_chinese(text)
            result["parse_error"] = (
                "proposal response must contain only one <action> block followed by "
                "one <skill> block"
            )
            return result
        action_match = action_matches[0]
        skill_match = skill_matches[0]
        for start, end in (
            (0, action_match.start()),
            (action_match.end(), skill_match.start()),
            (skill_match.end(), len(text)),
        ):
            span = _non_whitespace_span(text, start, end)
            if span is not None:
                outside_penalty_spans.append(span)
        if not outside_penalty_spans:
            result["contains_chinese"] = contains_chinese(text)
            result["parse_error"] = (
                "proposal response must contain only one <action> block followed by "
                "one <skill> block"
            )
            return result
    else:
        action_match = match
        skill_match = match

    raw_action = action_match.group("action").strip()
    proposal_action = _canonical_action(raw_action)
    skill_text = " ".join(skill_match.group("skill").split())
    action_span = _trimmed_span(action_match, "action")
    skill_span = _trimmed_span(skill_match, "skill")
    semantic_error, error_group = _semantic_error(
        proposal_action=proposal_action,
        raw_action=raw_action,
        skill_text=skill_text,
    )
    penalty_spans = list(outside_penalty_spans)
    if semantic_error:
        penalty_spans.append(action_span if error_group == "action" else skill_span)
    parse_error = semantic_error
    if outside_penalty_spans:
        parse_error = (
            "proposal response must contain only one <action> block followed by "
            "one <skill> block"
        )

    result.update(
        {
            "proposal_action": proposal_action,
            "skill_text": "" if proposal_action == "NO_SKILL" else skill_text,
            "contains_chinese": contains_chinese(text),
            "parse_ok": not parse_error,
            "parse_error": parse_error,
            "recoverable_invalid": bool(parse_error),
            "action_span": action_span,
            "skill_span": skill_span,
            "format_spans": (
                [
                    (match.start(), match.start("action")),
                    (match.end("action"), match.start("skill")),
                    (match.end("skill"), match.end()),
                ]
                if match is not None
                else []
            ),
            "penalty_spans": penalty_spans,
        }
    )
    return result
