"""Postgres access: card history for scoring, request log, and analyst cases.

One schema, one connection pool. `DATABASE_URL` is the only configuration; Heroku sets
it when the Postgres add-on is attached. Nothing here talks to a local file any more,
so cases and invitation counters survive a dyno restart.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd
from psycopg import sql
from psycopg.errors import UndefinedTable
from psycopg_pool import ConnectionPool

DEFAULT_URL = "postgresql://postgres:postgres@127.0.0.1:5432/fraud"

TABLES = ("cases", "requests", "key_usage", "llm_usage", "transactions")

_pool: ConnectionPool | None = None


def dsn() -> str:
    """`DATABASE_URL`, with Heroku's old `postgres://` prefix accepted."""
    url = os.getenv("DATABASE_URL", DEFAULT_URL)
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def pool() -> ConnectionPool:
    """One pool per process. Essential-0 allows 20 connections, so 8 is safe."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(dsn(), min_size=1, max_size=8, timeout=10, open=False)
        _pool.open()
        # Fail at boot, not on the first request, when the database is unreachable.
        _pool.wait(timeout=15)
    return _pool


@contextmanager
def connect():
    """A pooled connection. Commits on success, rolls back when the block raises."""
    with pool().connection() as con:
        yield con


LLM_USAGE_DDL = """
    CREATE TABLE IF NOT EXISTS llm_usage (
        day TEXT PRIMARY KEY,
        used INTEGER NOT NULL
    )
"""

KEY_USAGE_DDL = """
    CREATE TABLE IF NOT EXISTS key_usage (
        key_hash TEXT PRIMARY KEY,
        used INTEGER NOT NULL
    )
"""

TRANSACTIONS_DDL = """
    CREATE TABLE IF NOT EXISTS transactions (
        cc_num TEXT NOT NULL,
        ts_epoch BIGINT NOT NULL,
        trans_num TEXT,
        amt DOUBLE PRECISION NOT NULL,
        category TEXT,
        merchant TEXT,
        state TEXT,
        city_pop INTEGER,
        is_fraud SMALLINT NOT NULL DEFAULT 0
    )
"""

REQUESTS_DDL = """
    CREATE TABLE IF NOT EXISTS requests (
        id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL,
        cc_num TEXT NOT NULL,
        trans_num TEXT,
        ts_epoch BIGINT NOT NULL,
        amount DOUBLE PRECISION NOT NULL,
        score DOUBLE PRECISION NOT NULL,
        raw_score DOUBLE PRECISION NOT NULL,
        decision TEXT NOT NULL,
        threshold DOUBLE PRECISION NOT NULL,
        model_version TEXT NOT NULL,
        top_drivers TEXT NOT NULL,
        features_json TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'live'
    )
"""

CASES_DDL = """
    CREATE TABLE IF NOT EXISTS cases (
        id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        request_id INTEGER NOT NULL,
        cc_num TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        status TEXT NOT NULL,
        facts_json TEXT NOT NULL,
        summary TEXT NOT NULL,
        summary_source TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        facts_hash TEXT NOT NULL,
        model_version TEXT NOT NULL,
        decision TEXT,
        decided_at TIMESTAMPTZ,
        analyst TEXT
    )
"""

TX_INDEX_DDL = "CREATE INDEX IF NOT EXISTS idx_tx_card_ts ON transactions (cc_num, ts_epoch)"
CASES_INDEX_DDL = "CREATE INDEX IF NOT EXISTS idx_cases_status ON cases (status, id DESC)"
REQUESTS_INDEX_DDL = "CREATE INDEX IF NOT EXISTS idx_requests_id ON requests (id DESC)"


def init_db() -> None:
    """Create anything missing. Safe to run on every boot."""
    with connect() as con:
        for ddl in (
            TRANSACTIONS_DDL, TX_INDEX_DDL,
            REQUESTS_DDL, REQUESTS_INDEX_DDL,
            CASES_DDL, CASES_INDEX_DDL,
            LLM_USAGE_DDL, KEY_USAGE_DDL,
        ):
            con.execute(ddl)


def reset_db() -> None:
    """Drop and rebuild every table. Used by the seeder and by tests."""
    with connect() as con:
        for table in TABLES:
            con.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table)))
    init_db()


def clear_demo(keep_keys: bool = False) -> None:
    """Empty the demo tables, keep the transaction history."""
    targets = [t for t in ("cases", "requests", "key_usage", "llm_usage") if not (keep_keys and t == "key_usage")]
    with connect() as con:
        con.execute(
            sql.SQL("TRUNCATE {} RESTART IDENTITY").format(
                sql.SQL(", ").join(sql.Identifier(t) for t in targets)
            )
        )


def _epoch(ts) -> int:
    return int(pd.Timestamp(ts).value // 10**9)


def _iso(value) -> str | None:
    """Timestamps leave the store as ISO strings, the shape the UI already parses."""
    return value.isoformat() if isinstance(value, datetime) else value


HISTORY_COLS = ["ts_epoch", "amt", "category", "merchant", "state", "city_pop"]


def get_history(cc_num: str, ts) -> pd.DataFrame:
    with connect() as con:
        rows = con.execute(
            "SELECT ts_epoch, amt, category, merchant, state, city_pop FROM transactions "
            "WHERE cc_num = %s AND ts_epoch < %s ORDER BY ts_epoch",
            (str(cc_num), _epoch(ts)),
        ).fetchall()
    df = pd.DataFrame(rows, columns=HISTORY_COLS)
    df["ts_epoch"] = df["ts_epoch"].astype("int64")
    df["amt"] = df["amt"].astype("float64")
    df["ts"] = pd.to_datetime(df["ts_epoch"], unit="s")
    return df


def log_request(tx, out: dict, row: dict, model_version: str, source: str = "live") -> int:
    with connect() as con:
        cur = con.execute(
            "INSERT INTO requests (created_at, cc_num, trans_num, ts_epoch, amount, score, raw_score, "
            "decision, threshold, model_version, top_drivers, features_json, source) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (
                datetime.now(timezone.utc),
                str(tx.cc_num),
                tx.trans_num,
                _epoch(tx.ts),
                float(tx.amt),
                float(out["score"]),
                float(out["raw_score"]),
                out["decision"],
                float(out["threshold"]),
                model_version,
                json.dumps(out["top_drivers"]),
                json.dumps(row, default=str),
                source,
            ),
        ).fetchone()
    return int(cur[0])


def recent_requests(limit: int = 20) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT id, created_at, cc_num, trans_num, amount, score, decision, top_drivers "
            "FROM requests ORDER BY id DESC LIMIT %s", (limit,)
        ).fetchall()
    return [
        {
            "id": r[0], "created_at": _iso(r[1]), "cc_num": r[2], "trans_num": r[3],
            "amount": r[4], "score": r[5], "decision": r[6], "top_drivers": json.loads(r[7]),
        }
        for r in rows
    ]


def create_case(request_id: int, cc_num: str, facts: list[dict], summary: str, source: str, model_version: str) -> int:
    facts_json = json.dumps(facts)
    facts_hash = hashlib.sha256(facts_json.encode()).hexdigest()[:16]
    with connect() as con:
        cur = con.execute(
            "INSERT INTO cases (request_id, cc_num, created_at, status, facts_json, summary, summary_source, "
            "prompt_version, facts_hash, model_version) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (request_id, str(cc_num), datetime.now(timezone.utc), "open", facts_json,
             summary, source, "v1", facts_hash, model_version),
        ).fetchone()
    return int(cur[0])


def update_case_summary(case_id: int, summary: str, source: str) -> None:
    with connect() as con:
        con.execute(
            "UPDATE cases SET summary = %s, summary_source = %s WHERE id = %s",
            (summary, source, case_id),
        )


def _case_dict(row) -> dict:
    return {
        "id": row[0], "request_id": row[1], "cc_num": row[2], "created_at": _iso(row[3]), "status": row[4],
        "facts": json.loads(row[5]), "summary": row[6], "summary_source": row[7],
        "prompt_version": row[8], "facts_hash": row[9], "model_version": row[10],
        "decision": row[11], "decided_at": _iso(row[12]), "analyst": row[13],
    }


def get_case(case_id: int) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM cases WHERE id = %s", (case_id,)).fetchone()
        if row is None:
            return None
        out = _case_dict(row)
        req = con.execute(
            "SELECT amount, score, threshold FROM requests WHERE id = %s", (out["request_id"],)
        ).fetchone()
    if req:
        out.update({"amount": req[0], "score": req[1], "threshold": req[2]})
    return out


def list_cases(status: str | None = "open", limit: int = 50) -> list[dict]:
    with connect() as con:
        if status:
            rows = con.execute(
                "SELECT * FROM cases WHERE status = %s ORDER BY id DESC LIMIT %s", (status, limit)
            ).fetchall()
        else:
            rows = con.execute("SELECT * FROM cases ORDER BY id DESC LIMIT %s", (limit,)).fetchall()
        cases = [_case_dict(r) for r in rows]
        if cases:
            ids = [c["request_id"] for c in cases]
            found = {
                r[0]: (r[1], r[2])
                for r in con.execute(
                    "SELECT id, amount, score FROM requests WHERE id = ANY(%s)", (ids,)
                ).fetchall()
            }
    for c in cases:
        if c["request_id"] in found:
            c.update({"amount": found[c["request_id"]][0], "score": found[c["request_id"]][1]})
    return cases


def decide_case(case_id: int, decision: str, analyst: str) -> dict | None:
    with connect() as con:
        updated = con.execute(
            "UPDATE cases SET status = 'closed', decision = %s, decided_at = %s, analyst = %s "
            "WHERE id = %s RETURNING id",
            (decision, datetime.now(timezone.utc), analyst, case_id),
        ).fetchone()
    if updated is None:
        return None
    return get_case(case_id)


def request_features(limit: int = 500) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT score, decision, features_json, created_at, source FROM requests "
            "ORDER BY id DESC LIMIT %s", (limit,)
        ).fetchall()
    return [
        {"score": r[0], "decision": r[1], "features": json.loads(r[2]), "created_at": _iso(r[3]), "source": r[4]}
        for r in rows
    ]


SAMPLE_COLS = ["cc_num", "trans_num", "ts_epoch", "amt", "category", "merchant", "state", "city_pop"]


def sample_transactions(limit: int = 8, start: str = "2020-12-01") -> list[dict]:
    start_epoch = _epoch(start)
    half = max(1, limit // 2)
    with connect() as con:
        rows = con.execute(
            "SELECT " + ", ".join(SAMPLE_COLS) + " FROM transactions "
            "WHERE is_fraud = 1 AND ts_epoch >= %s ORDER BY random() LIMIT %s",
            (start_epoch, half),
        ).fetchall()
        rows += con.execute(
            "SELECT " + ", ".join(SAMPLE_COLS) + " FROM transactions "
            "WHERE is_fraud = 0 AND ts_epoch >= %s ORDER BY random() LIMIT %s",
            (start_epoch, limit - half),
        ).fetchall()
    return [
        {
            "cc_num": r[0], "trans_num": r[1],
            "ts": pd.to_datetime(r[2], unit="s").isoformat(),
            "amt": r[3], "category": r[4], "merchant": r[5], "state": r[6], "city_pop": r[7],
        }
        for r in rows
    ]


def case_stats() -> dict:
    with connect() as con:
        rows = con.execute(
            "SELECT decision, COUNT(*) FROM cases WHERE status = 'closed' GROUP BY decision"
        ).fetchall()
        open_count = con.execute("SELECT COUNT(*) FROM cases WHERE status = 'open'").fetchone()[0]
    counts = {k: int(v) for k, v in rows}
    confirmed = counts.get("confirmed_fraud", 0)
    false_alarm = counts.get("false_alarm", 0)
    reviewed = confirmed + false_alarm
    return {
        "open": int(open_count),
        "confirmed_fraud": confirmed,
        "false_alarm": false_alarm,
        "reviewed": reviewed,
        "precision": confirmed / reviewed if reviewed else None,
    }


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def consume_llm_credit(limit: int) -> bool:
    """Spend one of today's LLM credits. False when the day is used up.

    The UPDATE is a single statement, so two workers cannot both take the last
    credit. One credit covers a call plus its single retry.
    """
    day = _today()
    with connect() as con:
        con.execute("INSERT INTO llm_usage (day, used) VALUES (%s, 0) ON CONFLICT DO NOTHING", (day,))
        row = con.execute(
            "UPDATE llm_usage SET used = used + 1 WHERE day = %s AND used < %s RETURNING used",
            (day, max(0, int(limit))),
        ).fetchone()
    return row is not None


def llm_usage_today() -> dict:
    day = _today()
    with connect() as con:
        row = con.execute("SELECT used FROM llm_usage WHERE day = %s", (day,)).fetchone()
    return {"day": day, "used": int(row[0]) if row else 0}


def consume_key_use(key_hash: str, max_uses: int) -> bool:
    """Atomically spend one use of an access key. False when the key is exhausted."""
    with connect() as con:
        con.execute("INSERT INTO key_usage (key_hash, used) VALUES (%s, 0) ON CONFLICT DO NOTHING", (key_hash,))
        row = con.execute(
            "UPDATE key_usage SET used = used + 1 WHERE key_hash = %s AND used < %s RETURNING used",
            (key_hash, max(0, int(max_uses))),
        ).fetchone()
    return row is not None


def refund_key_use(key_hash: str) -> None:
    with connect() as con:
        con.execute(
            "UPDATE key_usage SET used = used - 1 WHERE key_hash = %s AND used > 0", (key_hash,)
        )


def key_use_count(key_hash: str) -> int:
    with connect() as con:
        row = con.execute("SELECT used FROM key_usage WHERE key_hash = %s", (key_hash,)).fetchone()
    return int(row[0]) if row else 0


def purge_older_than(cutoff) -> dict:
    """Delete demo rows older than a timestamp. Never touches key_usage.

    Key usage is the one thing that must not age out: deleting it would hand a spent key
    its uses back.
    """
    when = cutoff if isinstance(cutoff, datetime) else datetime.fromisoformat(str(cutoff))
    with connect() as con:
        cases = con.execute("DELETE FROM cases WHERE created_at < %s", (when,)).rowcount
        requests = con.execute("DELETE FROM requests WHERE created_at < %s", (when,)).rowcount
        days = con.execute("DELETE FROM llm_usage WHERE day < %s", (when.date().isoformat(),)).rowcount
    return {"cases": int(cases), "requests": int(requests), "llm_days": int(days)}


def counts() -> dict:
    with connect() as con:
        def one(sql: str) -> int:
            try:
                return int(con.execute(sql).fetchone()[0])
            except UndefinedTable:
                return 0

        return {
            "transactions": one("SELECT COUNT(*) FROM transactions"),
            "requests": one("SELECT COUNT(*) FROM requests"),
            "cases_open": one("SELECT COUNT(*) FROM cases WHERE status = 'open'"),
            "cases_total": one("SELECT COUNT(*) FROM cases"),
        }
