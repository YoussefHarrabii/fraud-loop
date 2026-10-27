"""Clear the demo rows from the app database, keep the transaction history.

    uv run python scripts/reset_demo.py
    uv run python scripts/reset_demo.py --keep-keys     # keep invitation counters

Run this before `publish_data.py` so test cases never ship inside the image.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "app.db"

TABLES = ("cases", "requests", "key_usage", "llm_usage")


def counts(con) -> dict:
    out = {"transactions": 0}
    try:
        out["transactions"] = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    except sqlite3.OperationalError:
        pass
    for table in TABLES:
        try:
            out[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.OperationalError:
            out[table] = 0
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--keep-keys", action="store_true", help="leave key_usage alone")
    args = parser.parse_args()

    path = Path(args.db)
    if not path.exists():
        sys.exit(f"{path} does not exist")
    con = sqlite3.connect(path)
    before = counts(con)

    targets = [t for t in TABLES if not (args.keep_keys and t == "key_usage")]
    for table in targets:
        try:
            con.execute(f"DELETE FROM {table}")
        except sqlite3.OperationalError:
            pass
    con.commit()
    con.execute("VACUUM")
    after = counts(con)
    con.close()

    print(f"database {path}")
    for table in ("transactions", "cases", "requests", "key_usage", "llm_usage"):
        print(f"  {table:<13} {before.get(table, 0):>8} -> {after.get(table, 0)}")
    print("transactions are never touched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
