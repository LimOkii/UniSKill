from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


CHINESE_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
TAGGED_RESPONSE_PATTERN = re.compile(
    r"\s*<think>(.*?)</think>\s*<action>(.*?)</action>\s*",
    re.DOTALL,
)


@dataclass(frozen=True)
class TaggedResponseValidation:
    think: str
    action: str
    valid: bool
    error: str
    contains_chinese: bool


@dataclass(frozen=True)
class ResponseTermination:
    token_length: int
    hit_length_limit: bool
    ended_with_eos: bool
    truncated: bool

    @property
    def normal(self) -> bool:
        return self.token_length > 0 and not self.truncated


def contains_chinese(text: str) -> bool:
    return bool(CHINESE_PATTERN.search(str(text)))


def validate_tagged_response(text: str) -> TaggedResponseValidation:
    """Require exactly one non-empty think block followed by one action block."""

    response = str(text)
    has_chinese = contains_chinese(response)
    tag_counts = {
        "think_open": response.count("<think>"),
        "think_close": response.count("</think>"),
        "action_open": response.count("<action>"),
        "action_close": response.count("</action>"),
    }
    expected = {
        "think_open": 1,
        "think_close": 1,
        "action_open": 1,
        "action_close": 1,
    }
    if tag_counts != expected:
        return TaggedResponseValidation(
            think="",
            action="",
            valid=False,
            error="response must contain exactly one <think> block and one <action> block",
            contains_chinese=has_chinese,
        )

    match = TAGGED_RESPONSE_PATTERN.fullmatch(response)
    if match is None:
        return TaggedResponseValidation(
            think="",
            action="",
            valid=False,
            error="response must contain only <think> then <action> blocks",
            contains_chinese=has_chinese,
        )

    think = match.group(1).strip()
    action = match.group(2).strip()
    if not think:
        return TaggedResponseValidation(
            think="",
            action=action,
            valid=False,
            error="response must contain a non-empty <think> block",
            contains_chinese=has_chinese,
        )
    if not action:
        return TaggedResponseValidation(
            think=think,
            action="",
            valid=False,
            error="response must contain a non-empty <action> block",
            contains_chinese=has_chinese,
        )
    if has_chinese:
        return TaggedResponseValidation(
            think=think,
            action=action,
            valid=False,
            error="response contains Chinese characters",
            contains_chinese=True,
        )
    return TaggedResponseValidation(
        think=think,
        action=action,
        valid=True,
        error="",
        contains_chinese=False,
    )


def classify_response_termination(
    token_ids: Sequence[int],
    *,
    response_limit: int,
    eos_token_id: int | None,
) -> ResponseTermination:
    """Detect a length-limited response while allowing EOS exactly at the limit."""

    ids = [int(token_id) for token_id in token_ids]
    hit_limit = len(ids) >= int(response_limit)
    ended_with_eos = bool(
        ids
        and eos_token_id is not None
        and ids[-1] == int(eos_token_id)
    )
    return ResponseTermination(
        token_length=len(ids),
        hit_length_limit=hit_limit,
        ended_with_eos=ended_with_eos,
        truncated=bool(hit_limit and not ended_with_eos),
    )
