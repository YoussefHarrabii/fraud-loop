"""Mint invitation keys for the LLM report.

Writes two files:

- `configs/access_keys.json` â€” hashes, labels, expiry, use limit. Committed, safe to ship.
- `secrets/keys.csv`         â€” plaintext keys, ignored. The only copy. Back it up.

Re-running with `--force` replaces the registry and invalidates every key already sent out.
"""

import argparse
import csv
import json
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud.access import digest  # noqa: E402

REGISTRY = ROOT / "configs" / "access_keys.json"
PLAIN = ROOT / "secrets" / "keys.csv"


def show(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--uses", type=int, default=5)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--label", default="Application")
    parser.add_argument("--force", action="store_true", help="replace an existing registry")
    args = parser.parse_args()

    if REGISTRY.exists() and not args.force:
        sys.exit(f"{show(REGISTRY)} exists. --force replaces it and invalidates issued keys.")
    if args.count < 1 or args.uses < 1:
        sys.exit("count and uses must be positive")

    expires = (datetime.now(timezone.utc) + timedelta(days=args.days)).replace(microsecond=0).isoformat()
    rows = []
    for i in range(1, args.count + 1):
        key = "fr_" + secrets.token_urlsafe(24)
        rows.append({
            "id": f"app-{i:03d}",
            "label": f"{args.label} {i:03d}",
            "hash": digest(key),
            "max_uses": args.uses,
            "expires_at": expires,
            "key": key,
        })

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps({
        "version": 1,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "keys": [{k: v for k, v in row.items() if k != "key"} for row in rows],
    }, indent=2) + "\n", encoding="utf-8")

    PLAIN.parent.mkdir(parents=True, exist_ok=True)
    with PLAIN.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "key", "uses", "expires_at"])
        for row in rows:
            writer.writerow([row["id"], row["label"], row["key"], row["max_uses"], row["expires_at"]])

    print(f"{len(rows)} keys, {args.uses} reports each, expire {expires}")
    print(f"hashes  {show(REGISTRY)}  (commit this)")
    print(f"plain   {show(PLAIN)}  (ignored, the only copy, back it up)")
    print("keys are hashed in the registry; losing the CSV means minting again with --force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

