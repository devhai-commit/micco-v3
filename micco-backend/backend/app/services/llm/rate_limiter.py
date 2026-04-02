"""
OpenAI Rate Limiter
===================
Combines three layers of protection against OpenAI 429 errors:

1. Semaphore  — limits max concurrent in-flight requests
2. Token bucket — limits requests-per-minute (sliding 60-second window)
3. Tenacity retry — exponential backoff + jitter on RateLimitError / timeouts
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Callable, TypeVar, AsyncGenerator

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AsyncRateLimiter:
    """Async-safe rate limiter: semaphore + sliding-window RPM bucket."""

    def __init__(self, max_concurrent: int = 5, requests_per_minute: int = 60):
        self._max_concurrent = max_concurrent
        self._rpm = requests_per_minute
        # Lazily initialised inside the running event loop
        self._semaphore: asyncio.Semaphore | None = None
        self._lock: asyncio.Lock | None = None
        self._request_times: deque[float] = deque()

    def _ensure_primitives(self) -> None:
        """Create asyncio primitives if not yet created (must run inside event loop)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        if self._lock is None:
            self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        self._ensure_primitives()

        # 1. Concurrency gate
        await self._semaphore.acquire()  # type: ignore[union-attr]

        # 2. RPM gate (sliding 60-second window)
        async with self._lock:  # type: ignore[union-attr]
            now = time.monotonic()
            # Evict timestamps older than 60 s
            while self._request_times and now - self._request_times[0] >= 60.0:
                self._request_times.popleft()

            if len(self._request_times) >= self._rpm:
                wait_sec = 60.0 - (now - self._request_times[0]) + 0.1
                logger.warning(
                    f"[rate_limiter] RPM limit ({self._rpm}) reached — "
                    f"sleeping {wait_sec:.1f}s before next request"
                )
                await asyncio.sleep(wait_sec)

            self._request_times.append(time.monotonic())

    def release(self) -> None:
        if self._semaphore is not None:
            self._semaphore.release()

    async def __aenter__(self) -> "AsyncRateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, *_: object) -> None:
        self.release()


def make_retry(max_attempts: int = 5):
    """Return a configured AsyncRetrying context manager for OpenAI calls."""
    from openai import RateLimitError, APITimeoutError, APIConnectionError

    return AsyncRetrying(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=2, max=60, jitter=4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# ── Module-level singleton ──────────────────────────────────────────────────
# Lazily created once from settings on first use.
_limiter: AsyncRateLimiter | None = None


def get_rate_limiter() -> AsyncRateLimiter:
    """Return the global OpenAI rate limiter (created from settings on first call)."""
    global _limiter
    if _limiter is None:
        from app.core.config import settings
        _limiter = AsyncRateLimiter(
            max_concurrent=settings.OPENAI_MAX_CONCURRENT,
            requests_per_minute=settings.OPENAI_REQUESTS_PER_MINUTE,
        )
    return _limiter
