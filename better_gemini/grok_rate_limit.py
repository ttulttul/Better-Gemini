from __future__ import annotations

import itertools
import logging
import os
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

from .grok_core import BetterGrokError

logger = logging.getLogger(__name__)

DEFAULT_MAX_REQUESTS_PER_SECOND = 5
DEFAULT_MAX_IN_FLIGHT = 5
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE_SECONDS = 1.0
DEFAULT_BACKOFF_MAX_SECONDS = 30.0
RECENT_ATTEMPT_TTL_SECONDS = 60.0


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d.", name, raw_value, default)
        return default
    if value < minimum:
        logger.warning("Ignoring %s=%r below %d; using %d.", name, raw_value, minimum, default)
        return default
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %.3f.", name, raw_value, default)
        return default
    if value < minimum:
        logger.warning("Ignoring %s=%r below %.3f; using %.3f.", name, raw_value, minimum, default)
        return default
    return value


class GrokRateLimitError(BetterGrokError):
    """Internal signal for an xAI 429 response that is safe to retry."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class GrokRequestCancelled(Exception):
    """Internal signal that prevents a queued or retrying request from being sent."""


@dataclass(frozen=True)
class RateLimitSnapshot:
    model: str
    active_requests: int
    attempts_last_minute: int
    blocked_for_seconds: float


class GrokRateLimitCoordinator:
    """Process-wide admission control and coordinated 429 retry handling."""

    def __init__(
        self,
        *,
        max_requests_per_second: int = DEFAULT_MAX_REQUESTS_PER_SECOND,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
        backoff_max_seconds: float = DEFAULT_BACKOFF_MAX_SECONDS,
        jitter_ratio: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        if max_requests_per_second < 1:
            raise ValueError("max_requests_per_second must be at least 1")
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if backoff_base_seconds < 0 or backoff_max_seconds < backoff_base_seconds:
            raise ValueError("invalid backoff bounds")
        if jitter_ratio < 0:
            raise ValueError("jitter_ratio cannot be negative")

        self.max_requests_per_second = max_requests_per_second
        self.max_in_flight = max_in_flight
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.jitter_ratio = jitter_ratio
        self._clock = clock
        self._sleep = sleep
        self._random_value = random_value
        self._condition = threading.Condition()
        self._request_ids = itertools.count(1)
        self._active_requests: dict[str, set[int]] = defaultdict(set)
        self._recent_attempts: dict[str, deque[float]] = defaultdict(deque)
        self._next_attempt_at: dict[str, float] = defaultdict(float)
        self._blocked_until: dict[str, float] = defaultdict(float)

    def execute(
        self,
        *,
        model: str,
        operation: Callable[[], Any],
        cancel_event: threading.Event | None = None,
    ) -> Any:
        request_id = self._admit_request(model, cancel_event)
        try:
            for retry_index in range(self.max_retries + 1):
                self._wait_for_attempt_slot(model, cancel_event)
                self._raise_if_cancelled(cancel_event)
                try:
                    return operation()
                except GrokRateLimitError as error:
                    if retry_index >= self.max_retries:
                        raise BetterGrokError(
                            f"xAI rate limit retries exhausted for model {model} "
                            f"after {retry_index + 1} attempt(s): {error}"
                        ) from error

                    delay = self._retry_delay(
                        retry_index=retry_index,
                        retry_after_seconds=error.retry_after_seconds,
                    )
                    self._block_model(model, delay)
                    logger.warning(
                        "xAI rate limited model=%s; retrying attempt %d/%d in %.2fs",
                        model,
                        retry_index + 2,
                        self.max_retries + 1,
                        delay,
                    )
                    self._sleep_before_retry(delay, cancel_event)
        finally:
            self._release_request(model, request_id)

        raise AssertionError("unreachable")

    def snapshot(self, model: str) -> RateLimitSnapshot:
        with self._condition:
            now = self._clock()
            self._prune_recent_attempts(model, now)
            return RateLimitSnapshot(
                model=model,
                active_requests=len(self._active_requests[model]),
                attempts_last_minute=len(self._recent_attempts[model]),
                blocked_for_seconds=max(0.0, self._blocked_until[model] - now),
            )

    def _admit_request(
        self,
        model: str,
        cancel_event: threading.Event | None,
    ) -> int:
        with self._condition:
            while len(self._active_requests[model]) >= self.max_in_flight:
                self._raise_if_cancelled(cancel_event)
                self._condition.wait(timeout=0.1)
            self._raise_if_cancelled(cancel_event)
            request_id = next(self._request_ids)
            self._active_requests[model].add(request_id)
            logger.debug(
                "Admitted xAI request id=%d model=%s active=%d",
                request_id,
                model,
                len(self._active_requests[model]),
            )
            return request_id

    def _release_request(self, model: str, request_id: int) -> None:
        with self._condition:
            self._active_requests[model].discard(request_id)
            logger.debug(
                "Released xAI request id=%d model=%s active=%d",
                request_id,
                model,
                len(self._active_requests[model]),
            )
            self._condition.notify_all()

    def _wait_for_attempt_slot(
        self,
        model: str,
        cancel_event: threading.Event | None,
    ) -> None:
        minimum_interval = 1.0 / self.max_requests_per_second
        with self._condition:
            while True:
                self._raise_if_cancelled(cancel_event)
                now = self._clock()
                eligible_at = max(self._next_attempt_at[model], self._blocked_until[model])
                delay = eligible_at - now
                if delay <= 0:
                    self._recent_attempts[model].append(now)
                    self._prune_recent_attempts(model, now)
                    self._next_attempt_at[model] = now + minimum_interval
                    return
                self._condition.wait(timeout=min(delay, 0.1))

    def _block_model(self, model: str, delay: float) -> None:
        with self._condition:
            self._blocked_until[model] = max(
                self._blocked_until[model],
                self._clock() + delay,
            )
            self._condition.notify_all()

    def _retry_delay(
        self,
        *,
        retry_index: int,
        retry_after_seconds: float | None,
    ) -> float:
        exponential = min(
            self.backoff_max_seconds,
            self.backoff_base_seconds * (2**retry_index),
        )
        jittered = min(
            self.backoff_max_seconds,
            exponential * (1.0 + self.jitter_ratio * self._random_value()),
        )
        return max(jittered, retry_after_seconds or 0.0)

    def _sleep_before_retry(
        self,
        delay: float,
        cancel_event: threading.Event | None,
    ) -> None:
        if cancel_event is None:
            self._sleep(delay)
            return
        if cancel_event.wait(delay):
            raise GrokRequestCancelled("xAI request was cancelled during rate-limit backoff")

    @staticmethod
    def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise GrokRequestCancelled("xAI request was cancelled before it was sent")

    def _prune_recent_attempts(self, model: str, now: float) -> None:
        cutoff = now - RECENT_ATTEMPT_TTL_SECONDS
        attempts = self._recent_attempts[model]
        while attempts and attempts[0] <= cutoff:
            attempts.popleft()


_configured_backoff_base_seconds = _env_float(
    "BETTER_GROK_BACKOFF_BASE_SECONDS",
    DEFAULT_BACKOFF_BASE_SECONDS,
    minimum=0.0,
)
_configured_backoff_max_seconds = _env_float(
    "BETTER_GROK_BACKOFF_MAX_SECONDS",
    DEFAULT_BACKOFF_MAX_SECONDS,
    minimum=0.0,
)
if _configured_backoff_max_seconds < _configured_backoff_base_seconds:
    logger.warning(
        "BETTER_GROK_BACKOFF_MAX_SECONDS is below the configured base; using %.3f.",
        _configured_backoff_base_seconds,
    )
    _configured_backoff_max_seconds = _configured_backoff_base_seconds


grok_rate_limit_coordinator = GrokRateLimitCoordinator(
    max_requests_per_second=_env_int(
        "BETTER_GROK_MAX_RPS",
        DEFAULT_MAX_REQUESTS_PER_SECOND,
        minimum=1,
    ),
    max_in_flight=_env_int(
        "BETTER_GROK_MAX_IN_FLIGHT",
        DEFAULT_MAX_IN_FLIGHT,
        minimum=1,
    ),
    max_retries=_env_int(
        "BETTER_GROK_MAX_RETRIES",
        DEFAULT_MAX_RETRIES,
        minimum=0,
    ),
    backoff_base_seconds=_configured_backoff_base_seconds,
    backoff_max_seconds=_configured_backoff_max_seconds,
)
