"""Per-application access keys for the LLM report.

The registry holds SHA-256 hashes only, so it can live in the repo and in the image.
Plaintext keys exist once, in the CSV the mint script writes. A key is spent per report
attempt and refunded when the provider does not deliver a summary.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(os.getenv("FRAUD_KEYS", ROOT / "configs" / "access_keys.json"))
CACHE: dict = {"mtime": None, "keys": {}}

REASON_STATUS = {"unknown": 401, "expired": 403, "exhausted": 429}


def digest(key: str) -> str:
    return hashlib.sha256(key.strip().encode("utf-8")).hexdigest()


def _parse(value):
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def load_registry() -> dict[str, dict]:
    """Hash -> record. Reloaded when the file changes, so a redeploy is not required locally."""
    try:
        mtime = REGISTRY.stat().st_mtime
    except OSError:
        CACHE["keys"], CACHE["mtime"] = {}, None
        return {}
    if CACHE["mtime"] != mtime:
        try:
            doc = json.loads(REGISTRY.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        CACHE["keys"] = {entry["hash"]: entry for entry in doc.get("keys", [])}
        CACHE["mtime"] = mtime
    return CACHE["keys"]


def resolve(raw: str | None) -> dict | None:
    if not raw:
        return None
    return load_registry().get(digest(raw))


def _expired(record: dict) -> bool:
    expires = _parse(record.get("expires_at"))
    return bool(expires and expires <= datetime.now(timezone.utc))


def status(raw: str | None) -> dict:
    """Report the key without spending anything."""
    record = resolve(raw)
    if record is None:
        return {"valid": False, "reason": "unknown", "remaining": 0, "max_uses": 0, "expires_at": None}
    if _expired(record):
        return {
            "valid": False, "reason": "expired", "remaining": 0,
            "max_uses": int(record["max_uses"]), "expires_at": record["expires_at"],
        }
    from fraud import store

    used = store.key_use_count(record["hash"])
    remaining = max(0, int(record["max_uses"]) - used)
    return {
        "valid": remaining > 0,
        "reason": "ok" if remaining else "exhausted",
        "remaining": remaining,
        "max_uses": int(record["max_uses"]),
        "expires_at": record.get("expires_at"),
    }


def consume(raw: str | None) -> tuple[dict | None, str, int]:
    """Take one use. Returns (record, reason, remaining)."""
    record = resolve(raw)
    if record is None:
        return None, "unknown", 0
    if _expired(record):
        return None, "expired", 0
    from fraud import store

    if not store.consume_key_use(record["hash"], int(record["max_uses"])):
        return None, "exhausted", 0
    remaining = max(0, int(record["max_uses"]) - store.key_use_count(record["hash"]))
    return record, "ok", remaining


def refund(record: dict) -> None:
    from fraud import store

    store.refund_key_use(record["hash"])
