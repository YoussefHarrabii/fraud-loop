from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from fraud import retention, store

OUT = {"score": 0.9, "raw_score": 1.0, "decision": "flag", "threshold": 0.1, "top_drivers": []}


def _case():
    tx = SimpleNamespace(cc_num="C1", trans_num="t1", ts=datetime(2020, 12, 15), amt=10.0)
    rid = store.log_request(tx, OUT, {"amt": 10.0}, "model")
    cid = store.create_case(rid, "C1", [{"id": "f1", "text": "Amount is $10.00."}], "s", "template", "model")
    return rid, cid


def _age(rid, cid, stamp):
    con = store.connect()
    con.execute("UPDATE cases SET created_at = ? WHERE id = ?", (stamp, cid))
    con.execute("UPDATE requests SET created_at = ? WHERE id = ?", (stamp, rid))
    con.commit()
    con.close()


def test_purge_removes_only_old_rows(db):
    old_rid, old_cid = _case()
    new_rid, new_cid = _case()
    _age(old_rid, old_cid, "2000-01-01T00:00:00+00:00")

    result = store.purge_older_than("2020-01-01T00:00:00+00:00")

    assert result["cases"] == 1
    assert result["requests"] == 1
    assert store.get_case(old_cid) is None
    assert store.get_case(new_cid) is not None
    assert [r["id"] for r in store.recent_requests(10)] == [new_rid]


def test_purge_never_touches_key_usage(db):
    store.consume_key_use("somehash", 5)
    store.purge_older_than("2999-01-01T00:00:00+00:00")
    assert store.key_use_count("somehash") == 1, "a purge must not hand a spent key its uses back"


def test_purge_drops_past_llm_days(db):
    store.consume_llm_credit(5)
    con = store.connect()
    con.execute("UPDATE llm_usage SET day = '2000-01-01'")
    con.commit()
    con.close()
    result = store.purge_older_than("2020-01-01T00:00:00+00:00")
    assert result["llm_days"] == 1
    assert store.llm_usage_today()["used"] == 0


def test_hours_default_and_override(monkeypatch):
    monkeypatch.delenv("FRAUD_RETENTION_HOURS", raising=False)
    assert retention.hours() == retention.DEFAULT_HOURS
    monkeypatch.setenv("FRAUD_RETENTION_HOURS", "6")
    assert retention.hours() == 6.0
    monkeypatch.setenv("FRAUD_RETENTION_HOURS", "nonsense")
    assert retention.hours() == retention.DEFAULT_HOURS


def test_zero_hours_disables_the_purge(db, monkeypatch):
    rid, cid = _case()
    _age(rid, cid, "2000-01-01T00:00:00+00:00")
    monkeypatch.setenv("FRAUD_RETENTION_HOURS", "0")
    retention.reset_clock()
    assert retention.maybe_purge(force=True) is None
    assert store.get_case(cid) is not None


def test_maybe_purge_runs_once_per_interval(db, monkeypatch):
    rid, cid = _case()
    _age(rid, cid, "2000-01-01T00:00:00+00:00")
    monkeypatch.setenv("FRAUD_RETENTION_HOURS", str(24 * 365))
    retention.reset_clock()
    first = retention.maybe_purge(force=True)
    assert first is not None
    assert retention.maybe_purge() is None


def test_retention_window_respects_the_cutoff(db, monkeypatch):
    retention.reset_clock()
    old_rid, old_cid = _case()
    _age(old_rid, old_cid, (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat())
    fresh_rid, fresh_cid = _case()

    monkeypatch.setenv("FRAUD_RETENTION_HOURS", "24")
    retention.maybe_purge(force=True)

    assert store.get_case(old_cid) is None
    assert store.get_case(fresh_cid) is not None


def test_reset_script_clears_rows_and_keeps_transactions(db, insert, monkeypatch):
    import importlib.util
    import sys

    insert("C1", "2020-12-10 10:00:00", 10)
    _case()
    store.consume_key_use("hash", 5)

    spec = importlib.util.spec_from_file_location(
        "reset_demo", Path(__file__).resolve().parents[1] / "scripts" / "reset_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["reset_demo.py", "--db", str(store.DB)])
    assert module.main() == 0

    counts = store.counts()
    assert counts["transactions"] == 1
    assert counts["cases_total"] == 0
    assert counts["requests"] == 0
    assert store.key_use_count("hash") == 0
