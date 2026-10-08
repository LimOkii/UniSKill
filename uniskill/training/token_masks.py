from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


def find_subsequence(sequence: Sequence[int], needle: Sequence[int], start: int = 0) -> int:
    if not needle:
        return -1
    last = len(sequence) - len(needle) + 1
    for index in range(max(0, start), max(0, last)):
        if list(sequence[index : index + len(needle)]) == list(needle):
            return index
    return -1


def _decode(tokenizer, ids: Sequence[int]) -> str:
    """Decode original rollout ids without normalizing their tokenization."""

    try:
        return tokenizer.decode(
            list(ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        # Small test tokenizers and some custom tokenizers expose only decode(ids).
        return tokenizer.decode(list(ids))


def _trimmed_payload_span(text: str) -> tuple[int, int] | None:
    # ALFWorld's native projection lowercases the response before locating the
    # action tags, so payload masking must accept the same tag casing.
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    payload = match.group(1)
    leading = len(payload) - len(payload.lstrip())
    trailing = len(payload) - len(payload.rstrip())
    start = match.start(1) + leading
    end = match.end(1) - trailing
    if start >= end:
        return None
    return start, end


def _first_prefix_count(
    ids: list[int],
    tokenizer,
    target: int,
    *,
    strictly_greater: bool,
) -> tuple[int, dict[int, int]]:
    """Find a decoded character boundary using prefixes of the original ids."""

    cache: dict[int, int] = {0: 0}

    def prefix_length(count: int) -> int:
        if count not in cache:
            cache[count] = len(_decode(tokenizer, ids[:count]))
        return cache[count]

    low, high = 0, len(ids)
    while low < high:
        middle = (low + high) // 2
        length = prefix_length(middle)
        reached = length > target if strictly_greater else length >= target
        if reached:
            high = middle
        else:
            low = middle + 1
    return low, cache


def action_payload_token_mask(response_ids: Sequence[int], tokenizer) -> list[int]:
    """Mask the first non-whitespace action payload in original rollout ids.

    Generated ids need not be the canonical encoding of their decoded text, and BPE
    tokens may merge tag boundaries with adjacent newlines. Character offsets are
    therefore mapped through decoded prefixes of the original ids; the text is never
    re-encoded to infer rollout token positions.
    """

    ids = list(response_ids)
    if hasattr(tokenizer, "decode"):
        try:
            span = _trimmed_payload_span(_decode(tokenizer, ids))
            if span is None:
                return [0] * len(ids)
            payload_start, payload_end = span
            start_count, start_cache = _first_prefix_count(
                ids,
                tokenizer,
                payload_start,
                strictly_greater=True,
            )
            end_count, end_cache = _first_prefix_count(
                ids,
                tokenizer,
                payload_end,
                strictly_greater=False,
            )
            start_token = start_count - 1

            def cached_prefix_length(count: int) -> int:
                cache = start_cache if count in start_cache else end_cache
                if count not in cache:
                    cache[count] = len(_decode(tokenizer, ids[:count]))
                return cache[count]

            boundaries_are_valid = (
                0 <= start_token < end_count <= len(ids)
                and cached_prefix_length(start_token) <= payload_start
                and cached_prefix_length(start_token + 1) > payload_start
                and cached_prefix_length(end_count - 1) < payload_end
                and cached_prefix_length(end_count) >= payload_end
            )
            if boundaries_are_valid:
                mask = [0] * len(ids)
                for index in range(start_token, end_count):
                    mask[index] = 1
                return mask
        except (TypeError, ValueError, UnicodeError):
            pass

    # Compatibility fallback for tokenizers without decode(). It cannot handle
    # context-dependent BPE merges and is not used by the Qwen training runtime.
    open_ids = tokenizer.encode("<action>", add_special_tokens=False)
    close_ids = tokenizer.encode("</action>", add_special_tokens=False)
    open_at = find_subsequence(ids, open_ids)
    if open_at < 0:
        return [0] * len(ids)
    payload_start = open_at + len(open_ids)
    close_at = find_subsequence(ids, close_ids, start=payload_start)
    if close_at < payload_start:
        return [0] * len(ids)
    mask = [0] * len(ids)
    for index in range(payload_start, close_at):
        mask[index] = 1
    return mask


def character_spans_token_mask(
    response_ids: Sequence[int],
    tokenizer,
    spans: Sequence[tuple[int, int]],
) -> list[int]:
    """Map decoded character spans onto the original rollout tokenization."""

    ids = list(response_ids)
    mask = [0] * len(ids)
    if not ids:
        return mask
    for start, end in spans:
        if start >= end:
            continue
        start_count, start_cache = _first_prefix_count(
            ids, tokenizer, int(start), strictly_greater=True
        )
        end_count, end_cache = _first_prefix_count(
            ids, tokenizer, int(end), strictly_greater=False
        )
        start_token = start_count - 1

        def prefix_length(count: int) -> int:
            if count in start_cache:
                return start_cache[count]
            if count in end_cache:
                return end_cache[count]
            return len(_decode(tokenizer, ids[:count]))

        valid = (
            0 <= start_token < end_count <= len(ids)
            and prefix_length(start_token) <= start
            and prefix_length(start_token + 1) > start
            and prefix_length(end_count - 1) < end
            and prefix_length(end_count) >= end
        )
        if not valid:
            raise ValueError(f"cannot map character span {(start, end)} to rollout tokens")
        for index in range(start_token, end_count):
            mask[index] = 1
    return mask


def proposal_channel_token_masks(
    compact_ids: list[int],
    visible_ids: list[int],
    *,
    tokenizer,
    parsed: Mapping,
    parse_ok: bool,
    eos_position: int | None,
) -> tuple[list[int], list[int], list[int]]:
    """Build disjoint masks for valid proposals and a full mask for invalid ones.

    A malformed proposal is unusable as one autoregressive response.  Its
    format penalty therefore covers every sampled token instead of trying to
    localize the error to a character span.  In particular, an empty required
    skill has a zero-width semantic span; localizing that error would silently
    produce an empty training mask and let the model evade the penalty.
    """

    format_mask = [0] * len(compact_ids)
    action_mask = [0] * len(compact_ids)
    skill_mask = [0] * len(compact_ids)
    if parse_ok:
        visible_format = character_spans_token_mask(
            visible_ids, tokenizer, parsed["format_spans"]
        )
        action_spans = [parsed["action_span"]]
        if parsed["proposal_action"] == "NO_SKILL":
            action_spans.append(parsed["skill_span"])
        visible_action = character_spans_token_mask(
            visible_ids, tokenizer, action_spans
        )
        visible_skill = (
            character_spans_token_mask(
                visible_ids, tokenizer, [parsed["skill_span"]]
            )
            if parsed["proposal_action"] != "NO_SKILL"
            else [0] * len(visible_ids)
        )
        action_mask[: len(visible_ids)] = visible_action
        skill_mask[: len(visible_ids)] = visible_skill
        format_mask[: len(visible_ids)] = [
            int(bool(format_keep) and not action_keep and not skill_keep)
            for format_keep, action_keep, skill_keep in zip(
                visible_format, visible_action, visible_skill
            )
        ]
        if eos_position is not None:
            format_mask[eos_position] = 1
    else:
        format_mask = [1] * len(compact_ids)
    return format_mask, action_mask, skill_mask


def masked_mean(values: Sequence[float], mask: Sequence[int]) -> float | None:
    selected = [float(value) for value, keep in zip(values, mask) if keep]
    if not selected:
        return None
    return sum(selected) / len(selected)
