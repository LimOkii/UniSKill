from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from uniskill.response_format import validate_tagged_response


HISTORY_ACTION_VALID = "valid"
HISTORY_ACTION_UNAVAILABLE = "unavailable"
HISTORY_ACTION_OMITTED = "omitted"
INVALID_ACTION_PLACEHOLDER = "[INVALID ACTION OMITTED]"


@dataclass(frozen=True)
class CanonicalAction:
    """One shared interpretation of an actor response for every downstream consumer."""

    payload: str
    payload_parsed: bool
    admissible: bool
    format_valid: bool
    strict_format_valid: bool
    strict_format_error: str
    display: str
    history_status: str

    @property
    def overall_valid(self) -> bool:
        return self.strict_format_valid and self.admissible


def canonicalize_action_response(
    *,
    raw_response: str,
    projected_action: str,
    format_valid: bool,
    admissible_actions: list[str],
) -> CanonicalAction:
    """Parse one canonical action while preserving native format-valid semantics.

    The environment still receives ``projected_action`` unchanged.  This helper
    determines the action identity shared by history, proposal context, metrics,
    and counterfactual scoring.  A missing ``<think>`` or Chinese text may make
    ``format_valid`` false without erasing a complete, admissible action payload.
    """

    response = str(raw_response)
    strict = validate_tagged_response(response)
    lower_response = response.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_index = lower_response.find(start_tag)
    end_index = lower_response.find(
        end_tag,
        start_index + len(start_tag) if start_index >= 0 else 0,
    )
    payload = str(projected_action).strip()
    payload_parsed = bool(
        start_index >= 0
        and end_index >= start_index + len(start_tag)
        and payload
    )
    normalized_payload = payload.lower()
    normalized_admissible = {
        str(action).strip().lower()
        for action in admissible_actions
    }
    admissible = payload_parsed and normalized_payload in normalized_admissible

    if admissible:
        display = payload
        history_status = HISTORY_ACTION_VALID
    elif payload_parsed:
        display = f"[UNAVAILABLE ACTION: {payload}]"
        history_status = HISTORY_ACTION_UNAVAILABLE
    else:
        display = INVALID_ACTION_PLACEHOLDER
        history_status = HISTORY_ACTION_OMITTED

    return CanonicalAction(
        payload=payload if payload_parsed else "",
        payload_parsed=payload_parsed,
        admissible=admissible,
        format_valid=bool(format_valid),
        strict_format_valid=strict.valid,
        strict_format_error=strict.error,
        display=display,
        history_status=history_status,
    )


def build_recent_action_history_context(
    full_history: list[dict[str, Any]],
    history_length: int,
) -> list[dict[str, Any]]:
    """Return the most recent observation/action pairs with absolute step ids."""

    recent_count = min(max(int(history_length), 0), len(full_history))
    start_at = len(full_history) - recent_count
    return [
        {
            "step": start_at + offset + 1,
            "observation": item["text_obs"],
            "action": item["action"],
        }
        for offset, item in enumerate(full_history[start_at:])
    ]
