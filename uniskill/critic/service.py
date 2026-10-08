from __future__ import annotations

import random
import threading
import time
import urllib.error
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable

from uniskill.critic.client import DEFAULT_CONFIG_PATH, call_openai_compatible
from uniskill.critic.parser import parse_critic
from uniskill.training.types import CriticState


@dataclass(frozen=True)
class CriticResult:
    state: CriticState
    action_reasonable: bool | None = None
    content_supported: bool | None = None
    action_reason: str = ""
    content_reason: str = ""
    error: str = ""
    attempts: int = 0
    raw_response: str = ""


class CriticService:
    """Bounded concurrent critic calls with explicit reject/error semantics."""

    def __init__(
        self,
        *,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        max_attempts: int = 5,
        parse_retry_attempts: int = 1,
        concurrency: int = 4,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
        jitter_seconds: float = 0.5,
        rate_limit_backoff_seconds: float = 60.0,
        call_fn: Callable[[str, str | Path], str] = call_openai_compatible,
    ) -> None:
        self.config_path = config_path
        self.max_attempts = max(1, int(max_attempts))
        self.parse_retry_attempts = max(0, int(parse_retry_attempts))
        self.concurrency = max(1, int(concurrency))
        self.initial_backoff_seconds = max(0.0, float(initial_backoff_seconds))
        self.max_backoff_seconds = max(self.initial_backoff_seconds, float(max_backoff_seconds))
        self.jitter_seconds = max(0.0, float(jitter_seconds))
        self.rate_limit_backoff_seconds = max(
            60.0, float(rate_limit_backoff_seconds)
        )
        self.call_fn = call_fn
        self._rate_limit_lock = threading.Lock()
        self._rate_limit_until = 0.0

    def evaluate_many(
        self,
        prompts: dict[str, str],
        *,
        proposal_actions: dict[str, str] | None = None,
    ) -> dict[str, CriticResult]:
        if not prompts:
            return {}
        results: dict[str, CriticResult] = {}
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(prompts))) as executor:
            futures = {
                executor.submit(
                    self._evaluate_one,
                    prompt,
                    proposal_action=(proposal_actions or {}).get(key),
                ): key
                for key, prompt in prompts.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as exc:  # Defensive: infrastructure errors never become rejects.
                    results[key] = CriticResult(CriticState.ERROR, error=str(exc))
        return results

    def _evaluate_one(
        self,
        prompt: str,
        *,
        proposal_action: str | None = None,
    ) -> CriticResult:
        backoff = self.initial_backoff_seconds
        parse_failures = 0
        last_error = ""
        last_raw = ""
        for attempt in range(1, self.max_attempts + 1):
            self._wait_for_rate_limit_cooldown()
            try:
                raw = self.call_fn(prompt, self.config_path)
                last_raw = raw
                parsed = parse_critic(raw, proposal_action=proposal_action)
                if parsed["parse_ok"]:
                    action_reasonable = parsed["action_reasonable"]
                    content_supported = parsed["content_supported"]
                    state = (
                        CriticState.ACCEPT
                        if action_reasonable and content_supported is not False
                        else CriticState.REJECT
                    )
                    return CriticResult(
                        state,
                        action_reasonable=action_reasonable,
                        content_supported=content_supported,
                        action_reason=parsed["action_reason"],
                        content_reason=parsed["content_reason"],
                        attempts=attempt,
                        raw_response=raw,
                    )
                parse_failures += 1
                last_error = parsed["parse_error"]
                if parse_failures > self.parse_retry_attempts:
                    return CriticResult(
                        CriticState.ERROR,
                        error=last_error,
                        attempts=attempt,
                        raw_response=last_raw,
                    )
            except Exception as exc:
                last_error = str(exc)
                if not _is_retryable_error(exc):
                    return CriticResult(
                        CriticState.ERROR,
                        error=last_error,
                        attempts=attempt,
                        raw_response=last_raw,
                    )
                if _is_rate_limit_error(exc):
                    self._start_rate_limit_cooldown(_retry_after_seconds(exc))
                    continue

            if attempt < self.max_attempts:
                time.sleep(backoff + random.uniform(0.0, self.jitter_seconds))
                backoff = min(max(backoff * 2, self.initial_backoff_seconds), self.max_backoff_seconds)

        return CriticResult(
            CriticState.ERROR,
            error=last_error,
            attempts=self.max_attempts,
            raw_response=last_raw,
        )

    def _start_rate_limit_cooldown(self, retry_after_seconds: float | None) -> None:
        delay = self.rate_limit_backoff_seconds
        if retry_after_seconds is not None:
            delay = max(delay, retry_after_seconds)
        with self._rate_limit_lock:
            self._rate_limit_until = max(
                self._rate_limit_until,
                time.monotonic() + delay,
            )

    def _wait_for_rate_limit_cooldown(self) -> None:
        with self._rate_limit_lock:
            remaining = self._rate_limit_until - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(remaining + random.uniform(0.0, self.jitter_seconds))


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 429} or 500 <= exc.code < 600
    if isinstance(
        exc,
        (
            TimeoutError,
            ConnectionError,
            urllib.error.URLError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ),
    ):
        return True
    message = str(exc).lower()
    markers = ("timeout", "timed out", "429", "rate limit", "connection", "502", "503", "504")
    return any(marker in message for marker in markers)


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429
    message = str(exc).lower()
    return "429" in message or "rate limit" in message


def _retry_after_seconds(exc: Exception) -> float | None:
    if not isinstance(exc, urllib.error.HTTPError) or exc.headers is None:
        return None
    value = exc.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None
