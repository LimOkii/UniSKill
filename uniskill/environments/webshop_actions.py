from __future__ import annotations

import re

from uniskill.environments.history import (
    CanonicalAction,
    HISTORY_ACTION_OMITTED,
    HISTORY_ACTION_UNAVAILABLE,
    HISTORY_ACTION_VALID,
    INVALID_ACTION_PLACEHOLDER,
)
from uniskill.response_format import validate_tagged_response


SEARCH_ACTION = re.compile(r"search\[(.+)\]", re.IGNORECASE | re.DOTALL)


def canonicalize_webshop_action_response(
    *,
    raw_response: str,
    projected_action: str,
    format_valid: bool,
    admissible_actions: list[str],
) -> CanonicalAction:
    """Canonicalize WebShop while treating search text as a free-form argument."""

    response = str(raw_response)
    strict = validate_tagged_response(response)
    lower_response = response.lower()
    start_index = lower_response.find("<action>")
    end_index = lower_response.find(
        "</action>", start_index + len("<action>") if start_index >= 0 else 0
    )
    payload = str(projected_action).strip()
    payload_parsed = bool(
        start_index >= 0
        and end_index >= start_index + len("<action>")
        and payload
    )

    normalized_pool = {str(action).strip().lower() for action in admissible_actions}
    normalized_payload = payload.lower()
    search_match = SEARCH_ACTION.fullmatch(payload)
    search_admissible = bool(
        search_match
        and search_match.group(1).strip()
        and "search[<your query>]" in normalized_pool
    )
    click_admissible = normalized_payload in normalized_pool
    admissible = payload_parsed and (search_admissible or click_admissible)

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
