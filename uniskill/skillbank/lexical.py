from __future__ import annotations

import math
from collections import Counter


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "with",
    "you",
    "your",
}


def tokenize(text: str) -> list[str]:
    normalized = []
    for char in text:
        normalized.append(char.lower() if char.isalnum() or char in "_'-" else " ")
    return [token for token in "".join(normalized).split() if token not in _STOPWORDS]


def lexical_score(query: str, candidate: str) -> float:
    query_tokens = tokenize(query)
    candidate_tokens = tokenize(candidate)
    if not query_tokens or not candidate_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    candidate_counts = Counter(candidate_tokens)
    overlap = set(query_counts) & set(candidate_counts)
    if not overlap:
        return 0.0

    exact_overlap = sum(min(query_counts[token], candidate_counts[token]) for token in overlap)
    jaccard = len(overlap) / len(set(query_counts) | set(candidate_counts))
    tf_component = sum(
        (1.0 + math.log(candidate_counts[token])) * query_counts[token]
        for token in overlap
    )
    return exact_overlap + jaccard + 0.25 * tf_component
