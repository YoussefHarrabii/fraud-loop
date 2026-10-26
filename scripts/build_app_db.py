"""Build the SQLite store used by the API: transactions with a card/time index, plus request log."""

import sqlite3
from pathlib import Path

import pandas as pd

from fraud import store

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "processed" / "transactions.parquet"
DB = ROOT / "data" / "app.db"


def main() -> None:
    if DB.exists():
        DB.unlink()
    tx = pd.read_parquet(RAW)[["cc_num", "ts", "trans_num", "amt", "category", "merchant", "state", "city_pop", "is_fraud"]]
    tx["cc_num"] = tx["cc_num"].astype(str)
    tx["ts_epoch"] = tx["ts"].astype("datetime64[s]").astype("int64")
    tx = tx[["cc_num", "ts_epoch", "trans_num", "amt", "category", "merchant", "state", "city_pop", "is_fraud"]]
    con = sqlite3.connect(DB)
    tx.to_sql("transactions", con, index=False, chunksize=100_000)
    con.execute("CREATE INDEX idx_tx_card_ts ON transactions(cc_num, ts_epoch)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            cc_num TEXT NOT NULL,
            trans_num TEXT,
            ts_epoch INTEGER NOT NULL,
            amount REAL NOT NULL,
            score REAL NOT NULL,
            raw_score REAL NOT NULL,
            decision TEXT NOT NULL,
            threshold REAL NOT NULL,
            model_version TEXT NOT NULL,
            top_drivers TEXT NOT NULL,
            features_json TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'live'
        )
    """)
    con.commit()
    con.execute("ANALYZE")
    con.commit()
    con.close()
    store.init_db()
    print(f"wrote {DB} ({DB.stat().st_size / 1e6:.0f} MB), {len(tx):,} transactions")


if __name__ == "__main__":
    main()
