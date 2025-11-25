"""Utility helpers for lightweight request rate limiting."""

from __future__ import annotations

from dataclasses import dataclass

from django.core.cache import cache


@dataclass(frozen=True)
class RateLimitResult:
    """Represents the outcome of a rate limit check."""

    limited: bool
    retry_after: int


def _client_identifier(request) -> str:
    """Return a stable identifier for the caller (user id or IP)."""
    if request.user.is_authenticated:
        return f"user:{request.user.pk}"

    forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.META.get("REMOTE_ADDR", "unknown")
    return f"ip:{client_ip}"


def check_rate_limit(
    request,
    scope: str,
    limit: int,
    window: int,
) -> RateLimitResult:
    """
    Check if the caller has exceeded the configured limit for the scope.

    The first call within the window will create the counter; subsequent calls
    increment it until the limit is reached.
    """
    identifier = _client_identifier(request)
    cache_key = f"rate-limit:{scope}:{identifier}"

    current = cache.get(cache_key, 0)
    if current >= limit:
        retry_after = _cache_ttl(cache_key, fallback=window)
        return RateLimitResult(True, retry_after)

    _increment_counter(cache_key, window)
    remaining_ttl = _cache_ttl(cache_key, fallback=window)
    return RateLimitResult(False, remaining_ttl)


def _increment_counter(cache_key: str, window: int) -> None:
    """Increment the cached counter, initializing if required."""
    added = cache.add(cache_key, 1, timeout=window)
    if added:
        return

    try:
        cache.incr(cache_key)
    except ValueError:
        cache.set(cache_key, 1, timeout=window)


def _cache_ttl(cache_key: str, fallback: int) -> int:
    """Best-effort TTL lookup with sensible fallback if unsupported."""
    ttl = fallback
    if hasattr(cache, "ttl"):
        try:
            val = cache.ttl(cache_key)
            if isinstance(val, int) and val >= 0:
                ttl = val
        except NotImplementedError:
            ttl = fallback
    return max(int(ttl), 1)