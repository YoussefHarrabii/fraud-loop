import time
from types import SimpleNamespace

from starlette.datastructures import Headers

from fraud import limits


def test_window_allows_then_blocks(monkeypatch):
    monkeypatch.setenv("FRAUD_LIMIT_SCORE", "2")
    assert limits.allow("score", "1.2.3.4") is True
    assert limits.allow("score", "1.2.3.4") is True
    assert limits.allow("score", "1.2.3.4") is False


def test_clients_have_separate_buckets(monkeypatch):
    monkeypatch.setenv("FRAUD_LIMIT_SCORE", "1")
    assert limits.allow("score", "1.1.1.1") is True
    assert limits.allow("score", "2.2.2.2") is True
    assert limits.allow("score", "1.1.1.1") is False


def test_scopes_have_separate_buckets(monkeypatch):
    monkeypatch.setenv("FRAUD_LIMIT_SCORE", "1")
    monkeypatch.setenv("FRAUD_LIMIT_DRIFT", "1")
    assert limits.allow("score", "1.1.1.1") is True
    assert limits.allow("drift", "1.1.1.1") is True
    assert limits.allow("score", "1.1.1.1") is False


def test_window_rolls_over(monkeypatch):
    monkeypatch.setattr(limits, "WINDOW", 0.05)
    monkeypatch.setenv("FRAUD_LIMIT_SCORE", "1")
    assert limits.allow("score", "1.2.3.4") is True
    assert limits.allow("score", "1.2.3.4") is False
    time.sleep(0.08)
    assert limits.allow("score", "1.2.3.4") is True


def test_zero_blocks_and_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv("FRAUD_LIMIT_SCORE", "0")
    assert limits.allow("score", "1.2.3.4") is False
    monkeypatch.setenv("FRAUD_LIMIT_SCORE", "lots")
    assert limits.limit_for("score") == limits.DEFAULTS["score"]


def test_unknown_scope_has_no_default_and_blocks(monkeypatch):
    monkeypatch.delenv("FRAUD_LIMIT_NOPE", raising=False)
    assert limits.limit_for("nope") == 0
    assert limits.allow("nope", "1.2.3.4") is False


def test_prune_keeps_the_dict_bounded(monkeypatch):
    monkeypatch.setenv("FRAUD_LIMIT_SCORE", "1")
    monkeypatch.setattr(limits, "MAX_BUCKETS", 2)
    for i in range(20):
        limits.allow("score", f"10.0.0.{i}")
    assert len(limits.BUCKETS) <= 2
    limits.reset()
    assert limits.BUCKETS == {}


def test_client_of_prefers_forwarded_for():
    request = SimpleNamespace(
        headers=Headers({"x-forwarded-for": "9.9.9.9, 10.0.0.1"}),
        client=SimpleNamespace(host="10.0.0.1"),
    )
    assert limits.client_of(request) == "9.9.9.9"


def test_client_of_falls_back_to_the_socket():
    request = SimpleNamespace(headers=Headers({}), client=SimpleNamespace(host="10.0.0.5"))
    assert limits.client_of(request) == "10.0.0.5"
    assert limits.client_of(SimpleNamespace(headers=Headers({}), client=None)) == "unknown"
