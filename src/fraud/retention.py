"""Age-based cleanup for the demo rows.

Cases and requests are visible to whoever opens the site next, so they expire. Key usage
never does: it is the only record of a spent invitation.
"""

import os
import time
from datetime import datetime, timedelta, timezone

from fraud import store

DEFAULT_HOURS = 24.0
INTERVAL = 3600.0
_last = 0.0


def hours() -> float:
    try:
        return max(0.0, float(os.getenv("FRAUD_RETENTION_HOURS", DEFAULT_HOURS)))
    except ValueError:
        return DEFAULT_HOURS


def maybe_purge(force: bool = False) -> dict | None:
    """Purge at most once an hour. 0 hours disables it."""
    global _last
    now = time.monotonic()
    if not force and now - _last < INTERVAL:
        return None
    _last = now
    span = hours()
    if span <= 0:
        return None
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=span)).isoformat(timespec="seconds")
    return store.purge_older_than(cutoff)


def reset_clock() -> None:
    global _last
    _last = 0.0
