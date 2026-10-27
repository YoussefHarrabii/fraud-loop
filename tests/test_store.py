from datetime import datetime
from types import SimpleNamespace

from fraud import store


def test_history_and_samples(db, insert):
    insert("C1", "2020-12-10 10:00:00", 10)
    insert("C1", "2020-12-12 10:00:00", 30)
    insert("C2", "2020-12-05 10:00:00", 40, is_fraud=1)
    insert("C3", "2020-12-06 10:00:00", 50, is_fraud=1)

    history = store.get_history("C1", "2020-12-15")
    assert len(history) == 2
    assert history.amt.tolist() == [10.0, 30.0]

    samples = store.sample_transactions(4)
    assert len(samples) == 4
    assert {s["cc_num"] for s in samples} <= {"C1", "C2", "C3"}


def test_case_flow(db):
    out = {"score": 0.9, "raw_score": 1.0, "decision": "flag", "threshold": 0.1,
           "top_drivers": [{"feature": "amt", "contribution": 1.0}]}
    tx = SimpleNamespace(cc_num="C1", trans_num="t1", ts=datetime(2020, 12, 15), amt=10.0)
    rid = store.log_request(tx, out, {"amt": 10.0}, "test-model")
    facts = [{"id": "f1", "text": "Amount is $10.00."}]
    cid = store.create_case(rid, "C1", facts, "summary", "template", "test-model",
                            recommendation="check merchant")

    case = store.get_case(cid)
    assert case["status"] == "open"
    assert case["amount"] == 10.0
    assert case["recommendation"] == "check merchant"
    assert len(store.list_cases("open")) == 1
    assert store.case_stats()["open"] == 1

    closed = store.decide_case(cid, "false_alarm", "tester")
    assert closed["status"] == "closed"
    assert closed["decision"] == "false_alarm"
    assert store.case_stats()["reviewed"] == 1
    assert store.request_features(10)[0]["source"] == "live"
    assert store.counts()["cases_total"] == 1


def test_missing_cases_and_empty_history(db):
    assert store.get_case(999) is None
    assert store.decide_case(999, "false_alarm", "tester") is None
    assert store.list_cases("open") == []
    assert store.case_stats()["precision"] is None
    assert store.get_history("nope", "2020-12-15").empty


def test_history_excludes_same_timestamp(db, insert):
    insert("C1", "2020-12-15 12:00:00", 10)
    assert len(store.get_history("C1", "2020-12-15 12:00:00")) == 0
    assert len(store.get_history("C1", "2020-12-15 12:00:01")) == 1


def test_duplicate_decision_updates_one_case(db):
    out = {"score": 0.9, "raw_score": 1.0, "decision": "flag", "threshold": 0.1,
           "top_drivers": [{"feature": "amt", "contribution": 1.0}]}
    tx = SimpleNamespace(cc_num="C1", trans_num="t1", ts=datetime(2020, 12, 15), amt=10.0)
    rid = store.log_request(tx, out, {"amt": 10.0}, "test-model")
    cid = store.create_case(rid, "C1", [{"id": "f1", "text": "Amount is $10.00."}], "s", "template", "m")

    store.decide_case(cid, "false_alarm", "first")
    store.decide_case(cid, "confirmed_fraud", "second")
    stats = store.case_stats()
    assert stats["reviewed"] == 1
    assert stats["confirmed_fraud"] == 1
    assert stats["false_alarm"] == 0
    assert store.get_case(cid)["analyst"] == "second"


def test_init_db_is_idempotent(db):
    store.init_db()
    store.init_db()
    assert store.counts() == {
        "transactions": 0, "requests": 0, "cases_open": 0, "cases_total": 0,
    }


def test_recent_requests_and_summary_update(db):
    out = {"score": 0.9, "raw_score": 1.0, "decision": "flag", "threshold": 0.1,
           "top_drivers": [{"feature": "amt", "contribution": 1.0}]}
    tx = SimpleNamespace(cc_num="C1", trans_num="t1", ts=datetime(2020, 12, 15), amt=10.0)
    rid = store.log_request(tx, out, {"amt": 10.0}, "test-model")
    cid = store.create_case(rid, "C1", [{"id": "f1", "text": "Amount is $10.00."}], "old", "template", "m")
    store.update_case_summary(cid, "new summary", "llm")
    assert store.get_case(cid)["summary"] == "new summary"
    assert store.recent_requests(5)[0]["id"] == rid


def test_llm_credit_stops_at_the_limit(db):
    assert store.llm_usage_today()["used"] == 0
    assert store.consume_llm_credit(2) is True
    assert store.consume_llm_credit(2) is True
    assert store.consume_llm_credit(2) is False
    assert store.llm_usage_today()["used"] == 2


def test_llm_credit_zero_never_grants(db):
    assert store.consume_llm_credit(0) is False
    assert store.llm_usage_today()["used"] == 0
