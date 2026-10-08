from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from transformers.utils import is_flash_attn_2_available


DEFAULT_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "Qwen/Qwen3-Embedding-0.6B")
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 512
QUERY_INSTRUCTION = "Retrieve a reusable skill that can help solve the given task."


class QwenEmbeddingBackend:
    def __init__(
        self,
        model_path: str | None = None,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        queries = [f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}" for text in texts]
        return self.encode(queries)

    def encode_skills(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        outputs = []
        for batch in self._batches(texts):
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.no_grad():
                model_outputs = self.model(**inputs)
                embeddings = self._last_token_pool(
                    model_outputs.last_hidden_state,
                    inputs["attention_mask"],
                )
                embeddings = F.normalize(embeddings, p=2, dim=1)
            outputs.append(embeddings.cpu().float().numpy())
        if not outputs:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def _load(self) -> None:
        if self.model is not None:
            return

        kwargs = {"trust_remote_code": True}
        if self.device.type == "cuda":
            kwargs["torch_dtype"] = torch.float16
            if is_flash_attn_2_available():
                kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModel.from_pretrained(self.model_path, **kwargs).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",
        )

    @staticmethod
    def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def _batches(self, texts: list[str]) -> Iterable[list[str]]:
        for start in range(0, len(texts), self.batch_size):
            yield texts[start:start + self.batch_size]
