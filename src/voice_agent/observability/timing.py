"""Debug-only performance timing decorator.

This module provides a @debug_timed decorator that measures function execution time
when enabled via the LOG_TIMING environment variable. When disabled, it adds zero
overhead by returning the original function unchanged.

Usage:
    @debug_timed("stt_feed")
    async def feed(self, audio: bytes) -> None:
        ...

    # Enable timing:
    # LOG_TIMING=1 python app.py
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, cast

from ..config import SETTINGS
from .metrics import Trace

# Global flag set at import time based on environment
_TIMING_ENABLED = SETTINGS.log_timing

# Statistics tracking per operation
@dataclass
class TimingStats:
    """Aggregate timing statistics for an operation."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        """Average execution time in milliseconds."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    def record(self, elapsed_ms: float) -> None:
        """Record a new timing sample."""
        self.count += 1
        self.total_ms += elapsed_ms
        self.min_ms = min(self.min_ms, elapsed_ms)
        self.max_ms = max(self.max_ms, elapsed_ms)


_TIMING_STATS: dict[str, TimingStats] = defaultdict(TimingStats)


def enable_timing() -> None:
    """Enable timing globally. Used for testing."""
    global _TIMING_ENABLED
    _TIMING_ENABLED = True


def disable_timing() -> None:
    """Disable timing globally. Used for testing."""
    global _TIMING_ENABLED
    _TIMING_ENABLED = False


def reset_stats() -> None:
    """Reset all timing statistics. Used for testing."""
    _TIMING_STATS.clear()


def get_stats(operation: str) -> TimingStats | None:
    """Get timing statistics for an operation."""
    return _TIMING_STATS.get(operation)


def get_all_stats() -> dict[str, TimingStats]:
    """Get all timing statistics."""
    return dict(_TIMING_STATS)


F = TypeVar("F", bound=Callable[..., Any])


def debug_timed(operation: str) -> Callable[[F], F]:
    """Decorator to measure function execution time when LOG_TIMING is enabled.

    Args:
        operation: Name of the operation being timed (e.g., "stt_feed")

    Returns:
        The original function if timing is disabled, or a wrapped version if enabled.

    Examples:
        @debug_timed("stt_feed")
        async def feed(self, audio: bytes) -> None:
            await process(audio)

        @debug_timed("llm_stream")
        def stream_tokens(self) -> Iterator[str]:
            yield from tokens
    """

    def decorator(func: F) -> F:
        # If timing is disabled, return the original function unchanged
        # This ensures ZERO overhead in production
        if not _TIMING_ENABLED:
            return func

        # Timing is enabled, wrap the function
        if asyncio.iscoroutinefunction(func):
            # Async function wrapper
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    stats = _TIMING_STATS[operation]
                    stats.record(elapsed_ms)

                    # Emit trace every 10 samples to avoid log spam
                    if stats.count % 10 == 0:
                        _emit_timing_trace(operation, elapsed_ms, stats)

            return cast(F, async_wrapper)
        else:
            # Sync function wrapper
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    stats = _TIMING_STATS[operation]
                    stats.record(elapsed_ms)

                    # Emit trace every 10 samples to avoid log spam
                    if stats.count % 10 == 0:
                        _emit_timing_trace(operation, elapsed_ms, stats)

            return cast(F, sync_wrapper)

    return decorator


def _emit_timing_trace(operation: str, elapsed_ms: float, stats: TimingStats) -> None:
    """Emit a timing trace with aggregate statistics."""
    Trace(
        call_sid="DEBUG",
        turn_id=0,
        event=f"perf_timing:{operation}",
        trace_id="debug",
        extra={
            "elapsed_ms": round(elapsed_ms, 2),
            "samples": stats.count,
            "avg_ms": round(stats.avg_ms, 2),
            "min_ms": round(stats.min_ms, 2),
            "max_ms": round(stats.max_ms, 2),
            "total_ms": round(stats.total_ms, 2),
        },
    ).log()
