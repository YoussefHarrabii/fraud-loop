"""Load the transaction history into Postgres.

Run once against a fresh database, locally or with `heroku run`:

    uv run python scripts/seed_pg.py --reset
    heroku run python scripts/seed_pg.py --reset -a <app>

Source is `data/processed/transactions.parquet` (1.79M rows). The index is dropped for
the COPY and rebuilt afterwards, which is several times faster than loading with it in
place. Nothing else in the app writes to `transactions`.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud import store  # noqa: E402

DEFAULT_PARQUET = ROOT / "data" / "processed" / "transactions.parquet"
COLS = (
    "cc_num", "ts_epoch", "trans_num", "amt", "category", "merchant", "state", "city_pop", "is_fraud",
)
SOURCE_COLS = ("cc_num", "unix_time", "trans_num", "amt", "category", "merchant", "state", "city_pop", "is_fraud")


def rows(chunk: pd.DataFrame):
    for r in chunk.itertuples(index=False):
        yield (
            str(int(r.cc_num)), int(r.unix_time), r.trans_num, float(r.amt),
            r.category, r.merchant, r.state, int(r.city_pop), int(r.is_fraud),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    parser.add_argument("--url", default=None, help="overrides DATABASE_URL")
    parser.add_argument("--reset", action="store_true", help="drop and recreate every table first")
    parser.add_argument("--batch", type=int, default=50_000)
    args = parser.parse_args()

    if args.url:
        os.environ["DATABASE_URL"] = args.url

    source = Path(args.parquet)
    if not source.exists():
        print(f"{source} does not exist; run the prepare step first", file=sys.stderr)
        return 2

    frame = pd.read_parquet(source, columns=list(SOURCE_COLS))
    total = len(frame)
    print(f"source {source} rows {total:,}")

    if args.reset:
        store.reset_db()
        print("schema recreated")
    else:
        store.init_db()

    started = time.time()
    with store.connect() as con:
        con.execute("DROP INDEX IF EXISTS idx_tx_card_ts")

    with store.connect() as con:
        copy_sql = f"COPY transactions ({', '.join(COLS)}) FROM STDIN"
        with con.cursor().copy(copy_sql) as copy:
            for start in range(0, total, args.batch):
                for row in rows(frame.iloc[start:start + args.batch]):
                    copy.write_row(row)
                done = min(start + args.batch, total)
                elapsed = time.time() - started
                print(f"  {done:>9,}/{total:,}  {done / max(elapsed, 1e-6):,.0f} rows/s", end="\r")

    print()
    with store.connect() as con:
        con.execute(store.TX_INDEX_DDL)
        con.execute("ANALYZE transactions")
        loaded = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    print(f"loaded {loaded:,} rows in {time.time() - started:.0f}s")
    if loaded != total:
        print(f"mismatch: expected {total:,}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
