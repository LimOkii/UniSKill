from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


class PromptTooLongError(ValueError):
    pass


@dataclass(frozen=True)
class TeacherForcingSample:
    sample_id: str
    prompt: str
    response_ids: list[int]
    response_mask: list[int]
    action_token_mask: list[int]
    metadata: dict[str, Any]


GENERATION_PROMPT_METADATA_KEYS = (
    "raw_prompt_ids",
    "prompt_token_length",
    "prompt_truncated",
)


def capture_generation_prompt_metadata(batch: DataProto) -> dict[str, np.ndarray]:
    """Snapshot driver-owned prompt metadata before rollout drops object arrays."""

    missing = [
        key for key in GENERATION_PROMPT_METADATA_KEYS if key not in batch.non_tensor_batch
    ]
    if missing:
        raise KeyError(f"generation batch missing prompt metadata: {missing}")
    metadata = {
        key: np.asarray(batch.non_tensor_batch[key]).copy()
        for key in GENERATION_PROMPT_METADATA_KEYS
    }
    for key, values in metadata.items():
        if len(values) != len(batch):
            raise ValueError(
                f"generation prompt metadata size mismatch for {key}: "
                f"{len(values)} != {len(batch)}"
            )
    return metadata


def apply_chat_template(
    tokenizer,
    prompt: str,
    apply_chat_template_kwargs: dict | None = None,
    *,
    assistant_prefill: str = "",
) -> str:
    kwargs = dict(apply_chat_template_kwargs or {})
    if not assistant_prefill:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            **kwargs,
        )

    # The prefix is model context, not a sampled response token.  Continuing
    # the final assistant message keeps training and rollout conditioned on the
    # exact same fixed prefix without adding a demonstration to the user prompt.
    kwargs.pop("add_generation_prompt", None)
    kwargs.pop("continue_final_message", None)
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_prefill},
        ],
        add_generation_prompt=False,
        continue_final_message=True,
        tokenize=False,
        **kwargs,
    )


def build_generation_batch(
    prompts: list[str],
    *,
    tokenizer,
    max_prompt_length: int,
    do_sample: bool,
    ids: list[str] | None = None,
    apply_chat_template_kwargs: dict | None = None,
    assistant_prefill: str = "",
) -> DataProto:
    if not prompts:
        raise ValueError("cannot build an empty generation batch")
    texts = [
        apply_chat_template(
            tokenizer,
            prompt,
            apply_chat_template_kwargs,
            assistant_prefill=assistant_prefill,
        )
        for prompt in prompts
    ]
    raw_prompt_ids = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    old_truncation_side = tokenizer.truncation_side
    old_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.truncation_side = "left"
    if assistant_prefill:
        tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_prompt_length),
        )
    finally:
        tokenizer.truncation_side = old_truncation_side
        if assistant_prefill:
            if old_padding_side is None:
                del tokenizer.padding_side
            else:
                tokenizer.padding_side = old_padding_side
    attention_mask = encoded["attention_mask"]
    position_ids = compute_position_id_with_mask(attention_mask)
    non_tensors: dict[str, np.ndarray] = {
        "raw_prompt_ids": np.array(
            [tokens[-int(max_prompt_length) :] for tokens in raw_prompt_ids],
            dtype=object,
        ),
        "raw_prompt": np.array(
            [
                (
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant_prefill},
                    ]
                    if assistant_prefill
                    else [{"role": "user", "content": prompt}]
                )
                for prompt in prompts
            ],
            dtype=object,
        ),
        "prompt_token_length": np.asarray([len(tokens) for tokens in raw_prompt_ids], dtype=np.int64),
        "prompt_truncated": np.asarray(
            [len(tokens) > int(max_prompt_length) for tokens in raw_prompt_ids],
            dtype=bool,
        ),
    }
    if ids is not None:
        non_tensors["sample_id"] = np.asarray(ids, dtype=object)
    return DataProto.from_dict(
        tensors={
            "input_ids": encoded["input_ids"],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        non_tensors=non_tensors,
        meta_info={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": bool(do_sample),
        },
    )


def build_teacher_forcing_batch(
    samples: list[TeacherForcingSample],
    *,
    tokenizer,
    max_prompt_length: int,
    apply_chat_template_kwargs: dict | None = None,
) -> DataProto:
    if not samples:
        raise ValueError("cannot build an empty teacher-forcing batch")

    prompt_ids: list[list[int]] = []
    trimmed_responses: list[list[int]] = []
    trimmed_action_masks: list[list[int]] = []
    for sample in samples:
        text = apply_chat_template(tokenizer, sample.prompt, apply_chat_template_kwargs)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_prompt_length:
            raise PromptTooLongError(
                f"teacher-forcing prompt {sample.sample_id} has {len(tokens)} tokens; "
                f"max_prompt_length={max_prompt_length}"
            )
        prompt_ids.append(tokens)
        response = [token for token, keep in zip(sample.response_ids, sample.response_mask) if keep]
        action_mask = [mask for mask, keep in zip(sample.action_token_mask, sample.response_mask) if keep]
        if not response:
            raise ValueError(f"teacher-forcing sample {sample.sample_id} has an empty response")
        if not any(action_mask):
            raise ValueError(f"teacher-forcing sample {sample.sample_id} has no <action> payload tokens")
        trimmed_responses.append(response)
        trimmed_action_masks.append(action_mask)

    max_prompt = max(len(tokens) for tokens in prompt_ids)
    max_response = max(len(tokens) for tokens in trimmed_responses)
    pad_id = int(tokenizer.pad_token_id)
    input_rows = []
    prompt_rows = []
    response_rows = []
    attention_rows = []
    response_mask_rows = []
    action_mask_rows = []
    for prompt, response, action_mask in zip(prompt_ids, trimmed_responses, trimmed_action_masks):
        left_padding = [pad_id] * (max_prompt - len(prompt))
        right_padding = [pad_id] * (max_response - len(response))
        padded_prompt = left_padding + prompt
        padded_response = response + right_padding
        prompt_attention = [0] * len(left_padding) + [1] * len(prompt)
        response_attention = [1] * len(response) + [0] * len(right_padding)
        input_rows.append(padded_prompt + padded_response)
        prompt_rows.append(padded_prompt)
        response_rows.append(padded_response)
        attention_rows.append(prompt_attention + response_attention)
        response_mask_rows.append(response_attention)
        action_mask_rows.append(action_mask + [0] * len(right_padding))

    attention_mask = torch.tensor(attention_rows, dtype=torch.long)
    metadata_keys = sorted({key for sample in samples for key in sample.metadata})
    non_tensors = {
        "sample_id": np.asarray([sample.sample_id for sample in samples], dtype=object),
        **{
            key: np.asarray([sample.metadata.get(key) for sample in samples], dtype=object)
            for key in metadata_keys
        },
    }
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor(prompt_rows, dtype=torch.long),
            "responses": torch.tensor(response_rows, dtype=torch.long),
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
            "response_mask": torch.tensor(response_mask_rows, dtype=torch.long),
            "action_token_mask": torch.tensor(action_mask_rows, dtype=torch.long),
        },
        non_tensors=non_tensors,
        meta_info={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": True,
        },
    )


def generated_response_mask(batch: DataProto) -> torch.Tensor:
    response_length = batch.batch["responses"].shape[-1]
    return batch.batch["attention_mask"][:, -response_length:].to(dtype=torch.long)
