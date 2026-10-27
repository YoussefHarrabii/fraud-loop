"""SQLite access: card history for scoring, request log, and analyst cases."""

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = Path(os.getenv("FRAUD_DB", ROOT / "data" / "app.db"))


def connect() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB, check_same_thread=False, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=10000")
    return con


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


def init_db() -> None:
    con = connect()
    con.execute(LLM_USAGE_DDL)
    con.execute(KEY_USAGE_DDL)
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
            features_json TEXT NOT NULL
        )
    """)
    cols = {r[1] for r in con.execute("PRAGMA table_info(requests)")}
    if "source" not in cols:
        con.execute("ALTER TABLE requests ADD COLUMN source TEXT NOT NULL DEFAULT 'live'")
    con.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER NOT NULL,
            cc_num TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            facts_json TEXT NOT NULL,
            summary TEXT NOT NULL,
            summary_source TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            facts_hash TEXT NOT NULL,
            model_version TEXT NOT NULL,
            decision TEXT,
            decided_at TEXT,
            analyst TEXT
        )
    """)
    con.commit()
    con.close()


def get_history(cc_num: str, ts) -> pd.DataFrame:
    ts_epoch = int(pd.Timestamp(ts).value // 10**9)
    con = connect()
    df = pd.read_sql_query(
        "SELECT ts_epoch, amt, category, merchant, state, city_pop FROM transactions "
        "WHERE cc_num = ? AND ts_epoch < ? ORDER BY ts_epoch",
        con, params=(str(cc_num), ts_epoch),
    )
    con.close()
    df["ts"] = pd.to_datetime(df["ts_epoch"], unit="s")
    return df


def log_request(tx, out: dict, row: dict, model_version: str, source: str = "live") -> int:
    con = connect()
    cur = con.execute(
        "INSERT INTO requests (created_at, cc_num, trans_num, ts_epoch, amount, score, raw_score, "
        "decision, threshold, model_version, top_drivers, features_json, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            datetime.now(timezone.utc).isoformat(),
            str(tx.cc_num),
            tx.trans_num,
            int(pd.Timestamp(tx.ts).value // 10**9),
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
    )
    con.commit()
    rid = int(cur.lastrowid)
    con.close()
    return rid


def recent_requests(limit: int = 20) -> list[dict]:
    con = connect()
    rows = con.execute(
        "SELECT id, created_at, cc_num, trans_num, amount, score, decision, top_drivers "
        "FROM requests ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    out = []
    for r in rows:
        out.append({
            "id": r[0], "created_at": r[1], "cc_num": r[2], "trans_num": r[3],
            "amount": r[4], "score": r[5], "decision": r[6],
            "top_drivers": json.loads(r[7]),
        })
    return out


def create_case(request_id: int, cc_num: str, facts: list[dict], summary: str, source: str, model_version: str) -> int:
    facts_json = json.dumps(facts)
    facts_hash = hashlib.sha256(facts_json.encode()).hexdigest()[:16]
    con = connect()
    cur = con.execute(
        "INSERT INTO cases (request_id, cc_num, created_at, status, facts_json, summary, summary_source, "
        "prompt_version, facts_hash, model_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (request_id, str(cc_num), datetime.now(timezone.utc).isoformat(), "open", facts_json,
         summary, source, "v1", facts_hash, model_version),
    )
    con.commit()
    cid = int(cur.lastrowid)
    con.close()
    return cid


def update_case_summary(case_id: int, summary: str, source: str) -> None:
    con = connect()
    con.execute("UPDATE cases SET summary = ?, summary_source = ? WHERE id = ?", (summary, source, case_id))
    con.commit()
    con.close()


def _case_dict(row) -> dict:
    return {
        "id": row[0], "request_id": row[1], "cc_num": row[2], "created_at": row[3], "status": row[4],
        "facts": json.loads(row[5]), "summary": row[6], "summary_source": row[7],
        "prompt_version": row[8], "facts_hash": row[9], "model_version": row[10],
        "decision": row[11], "decided_at": row[12], "analyst": row[13],
    }


def get_case(case_id: int) -> dict | None:
    con = connect()
    row = con.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    if row is None:
        con.close()
        return None
    out = _case_dict(row)
    req = con.execute("SELECT amount, score, threshold FROM requests WHERE id = ?", (out["request_id"],)).fetchone()
    con.close()
    if req:
        out.update({"amount": req[0], "score": req[1], "threshold": req[2]})
    return out


def list_cases(status: str | None = "open", limit: int = 50) -> list[dict]:
    con = connect()
    if status:
        rows = con.execute(
            "SELECT * FROM cases WHERE status = ? ORDER BY id DESC LIMIT ?", (status, limit)
        ).fetchall()
    else:
        rows = con.execute("SELECT * FROM cases ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    cases = [_case_dict(r) for r in rows]
    for c in cases:
        req = con.execute("SELECT amount, score FROM requests WHERE id = ?", (c["request_id"],)).fetchone()
        if req:
            c.update({"amount": req[0], "score": req[1]})
    con.close()
    return cases


def decide_case(case_id: int, decision: str, analyst: str) -> dict | None:
    con = connect()
    con.execute(
        "UPDATE cases SET status = 'closed', decision = ?, decided_at = ?, analyst = ? WHERE id = ?",
        (decision, datetime.now(timezone.utc).isoformat(), analyst, case_id),
    )
    con.commit()
    con.close()
    return get_case(case_id)


def request_features(limit: int = 500) -> list[dict]:
    con = connect()
    rows = con.execute(
        "SELECT score, decision, features_json, created_at, source FROM requests ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [
        {"score": r[0], "decision": r[1], "features": json.loads(r[2]), "created_at": r[3], "source": r[4]}
        for r in rows
    ]


def sample_transactions(limit: int = 8, start: str = "2020-12-01") -> list[dict]:
    start_epoch = int(pd.Timestamp(start).value // 10**9)
    half = max(1, limit // 2)
    con = connect()
    rows = con.execute(
        "SELECT cc_num, trans_num, ts_epoch, amt, category, merchant, state, city_pop FROM transactions "
        "WHERE is_fraud = 1 AND ts_epoch >= ? ORDER BY RANDOM() LIMIT ?", (start_epoch, half)
    ).fetchall()
    rows += con.execute(
        "SELECT cc_num, trans_num, ts_epoch, amt, category, merchant, state, city_pop FROM transactions "
        "WHERE is_fraud = 0 AND ts_epoch >= ? ORDER BY RANDOM() LIMIT ?", (start_epoch, limit - half)
    ).fetchall()
    con.close()
    out = []
    for r in rows:
        out.append({
            "cc_num": r[0], "trans_num": r[1],
            "ts": pd.to_datetime(r[2], unit="s").isoformat(),
            "amt": r[3], "category": r[4], "merchant": r[5], "state": r[6], "city_pop": r[7],
        })
    return out


def case_stats() -> dict:
    con = connect()
    rows = con.execute(
        "SELECT decision, COUNT(*) FROM cases WHERE status = 'closed' GROUP BY decision"
    ).fetchall()
    open_count = con.execute("SELECT COUNT(*) FROM cases WHERE status = 'open'").fetchone()[0]
    con.close()
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
    con = connect()
    try:
        con.execute(LLM_USAGE_DDL)
        con.execute("INSERT OR IGNORE INTO llm_usage (day, used) VALUES (?, 0)", (day,))
        row = con.execute(
            "UPDATE llm_usage SET used = used + 1 WHERE day = ? AND used < ? RETURNING used",
            (day, max(0, int(limit))),
        ).fetchone()
        con.commit()
    finally:
        con.close()
    return row is not None


def llm_usage_today() -> dict:
    day = _today()
    con = connect()
    try:
        con.execute(LLM_USAGE_DDL)
        row = con.execute("SELECT used FROM llm_usage WHERE day = ?", (day,)).fetchone()
    finally:
        con.close()
    return {"day": day, "used": int(row[0]) if row else 0}


def consume_key_use(key_hash: str, max_uses: int) -> bool:
    """Atomically spend one use of an access key. False when the key is exhausted."""
    con = connect()
    try:
        con.execute(KEY_USAGE_DDL)
        con.execute("INSERT OR IGNORE INTO key_usage (key_hash, used) VALUES (?, 0)", (key_hash,))
        row = con.execute(
            "UPDATE key_usage SET used = used + 1 WHERE key_hash = ? AND used < ? RETURNING used",
            (key_hash, max(0, int(max_uses))),
        ).fetchone()
        con.commit()
    finally:
        con.close()
    return row is not None


def refund_key_use(key_hash: str) -> None:
    con = connect()
    try:
        con.execute(KEY_USAGE_DDL)
        con.execute("UPDATE key_usage SET used = used - 1 WHERE key_hash = ? AND used > 0", (key_hash,))
        con.commit()
    finally:
        con.close()


def key_use_count(key_hash: str) -> int:
    con = connect()
    try:
        con.execute(KEY_USAGE_DDL)
        row = con.execute("SELECT used FROM key_usage WHERE key_hash = ?", (key_hash,)).fetchone()
    finally:
        con.close()
    return int(row[0]) if row else 0


def purge_older_than(cutoff: str) -> dict:
    """Delete demo rows older than an ISO timestamp. Never touches key_usage.

    Key usage is the one thing that must not age out: deleting it would hand a spent key
    its uses back.
    """
    con = connect()
    try:
        cases = con.execute("DELETE FROM cases WHERE created_at < ?", (cutoff,)).rowcount
        requests = con.execute("DELETE FROM requests WHERE created_at < ?", (cutoff,)).rowcount
        days = con.execute("DELETE FROM llm_usage WHERE day < ?", (cutoff[:10],)).rowcount
        con.commit()
    finally:
        con.close()
    return {"cases": int(cases), "requests": int(requests), "llm_days": int(days)}


def counts() -> dict:
    con = connect()
    def one(sql: str) -> int:
        try:
            return int(con.execute(sql).fetchone()[0])
        except sqlite3.OperationalError:
            return 0
    out = {
        "transactions": one("SELECT COUNT(*) FROM transactions"),
        "requests": one("SELECT COUNT(*) FROM requests"),
        "cases_open": one("SELECT COUNT(*) FROM cases WHERE status = 'open'"),
        "cases_total": one("SELECT COUNT(*) FROM cases"),
    }
    con.close()
    return out
