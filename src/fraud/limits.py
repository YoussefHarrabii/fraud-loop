"""In-process sliding-window limits for the write endpoints.

One uvicorn worker, so a dict is enough. Behind a proxy the caller is taken from the first
`X-Forwarded-For` entry. That header is spoofable by a direct caller, which is why this is
an abuse brake, not an authorisation control: the LLM cap and the per-key uses are the
controls that hold even if someone bypasses this.
"""

import os
import time

WINDOW = 60.0
MAX_BUCKETS = 4096
DEFAULTS = {"score": 30, "decision": 30, "drift": 3, "access": 20, "report": 10}

BUCKETS: dict[tuple[str, str], list[float]] = {}


def limit_for(scope: str) -> int:
    raw = os.getenv(f"FRAUD_LIMIT_{scope.upper()}")
    if raw is None:
        return DEFAULTS.get(scope, 0)
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULTS.get(scope, 0)


def client_of(request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    return request.client.host if request.client else "unknown"


def _prune(cutoff: float) -> None:
    """Drop stale buckets, then the oldest ones, so a flood of clients cannot grow memory."""
    for key in [k for k, v in BUCKETS.items() if not v or v[-1] < cutoff]:
        BUCKETS.pop(key, None)
    overflow = len(BUCKETS) - MAX_BUCKETS
    if overflow > 0:
        for key in sorted(BUCKETS, key=lambda k: BUCKETS[k][-1])[:overflow]:
            BUCKETS.pop(key, None)


def allow(scope: str, client: str) -> bool:
    """True when the caller may proceed. A limit of 0 blocks the scope outright."""
    limit = limit_for(scope)
    if limit <= 0:
        return False
    now = time.monotonic()
    cutoff = now - WINDOW
    key = (scope, client)
    hits = BUCKETS.setdefault(key, [])
    if hits and hits[0] < cutoff:
        hits = BUCKETS[key] = [stamp for stamp in hits if stamp >= cutoff]
    if len(hits) >= limit:
        return False
    hits.append(now)
    if len(BUCKETS) > MAX_BUCKETS:
        _prune(cutoff)
    return True


def reset() -> None:
    BUCKETS.clear()
