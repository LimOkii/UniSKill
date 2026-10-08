from __future__ import annotations

import json
import re


def _invalid(error: str) -> dict:
    return {
        "action_reasonable": None,
        "action_reason": "",
        "content_supported": None,
        "content_reason": "",
        "parse_ok": False,
        "parse_error": error,
    }


def parse_critic(text: str, *, proposal_action: str | None = None) -> dict:
    cleaned_text = _extract_json_text(text)
    try:
        payload = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        return _invalid(f"critic output parse failed: {exc}")

    if not isinstance(payload, dict):
        return _invalid("critic output must be a JSON object")
    if not isinstance(payload.get("action_reasonable"), bool):
        return _invalid("critic output requires boolean action_reasonable")

    action_reason = payload.get("action_reason")
    if not isinstance(action_reason, str) or not action_reason.strip():
        return _invalid("critic output requires non-empty action_reason")

    content_supported = payload.get("content_supported")
    if content_supported is not None and not isinstance(content_supported, bool):
        return _invalid("critic output requires boolean or null content_supported")

    content_reason = payload.get("content_reason")
    if not isinstance(content_reason, str) or not content_reason.strip():
        return _invalid("critic output requires non-empty content_reason")

    if proposal_action == "NO_SKILL" and content_supported is not None:
        return _invalid("NO_SKILL requires null content_supported")
    if proposal_action in {"ADD_NEW_SKILL", "UPDATE_SKILL"} and not isinstance(
        content_supported, bool
    ):
        return _invalid(f"{proposal_action} requires boolean content_supported")

    return {
        "action_reasonable": payload["action_reasonable"],
        "action_reason": action_reason.strip(),
        "content_supported": content_supported,
        "content_reason": content_reason.strip(),
        "parse_ok": True,
        "parse_error": "",
    }


def _extract_json_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("critic output must be a string")
    stripped = text.strip()

    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE
    )
    if fenced:
        return fenced.group(1).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and start < end:
        return stripped[start : end + 1].strip()

    return stripped
