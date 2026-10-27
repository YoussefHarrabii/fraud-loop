"""Clear the demo rows from Postgres, keep the transaction history.

    uv run python scripts/reset_demo.py
    uv run python scripts/reset_demo.py --keep-keys     # keep invitation counters

Key counters are the one thing worth keeping on a redeploy, because deleting them
hands every issued invitation key its uses back.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud import store  # noqa: E402

TABLES = ("cases", "requests", "key_usage", "llm_usage")


def counts() -> dict:
    from psycopg.errors import UndefinedTable

    out = {"transactions": 0}
    with store.connect() as con:
        try:
            out["transactions"] = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        except UndefinedTable:
            pass
        for table in TABLES:
            try:
                out[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except UndefinedTable:
                out[table] = 0
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-keys", action="store_true", help="leave key_usage alone")
    args = parser.parse_args()

    before = counts()
    store.clear_demo(keep_keys=args.keep_keys)
    after = counts()

    print(f"database {store.dsn().rsplit('@', 1)[-1]}")
    for table in ("transactions", "cases", "requests", "key_usage", "llm_usage"):
        print(f"  {table:<13} {before.get(table, 0):>8} -> {after.get(table, 0)}")
    print("transactions are never touched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
