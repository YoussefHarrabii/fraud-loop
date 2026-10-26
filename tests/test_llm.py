import httpx
import pytest

from fraud import case_facts, store

FACTS = [
    {"id": "f1", "text": "Amount is $90.00."},
    {"id": "f2", "text": "Local time is 03:00."},
]


class Response:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


@pytest.fixture
def key(db, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_DAILY_CAP", raising=False)
    yield


def test_llm_valid_summary(key, monkeypatch):
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        return Response("Amount is $90.00. [f1]")

    monkeypatch.setattr(case_facts.httpx, "post", post)
    text, source = case_facts.summarize(FACTS)
    assert source == "llm"
    assert "90.00" in text
    assert len(calls) == 1


def test_llm_retry_after_bad_number(key, monkeypatch):
    responses = iter([Response("Amount is $999.00. [f1]"), Response("Amount is $90.00. [f1]")])
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(case_facts.httpx, "post", post)
    text, source = case_facts.summarize(FACTS)
    assert source == "llm"
    assert "90.00" in text
    assert len(calls) == 2


def test_llm_timeout_falls_back(key, monkeypatch):
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(case_facts.httpx, "post", post)
    text, source = case_facts.summarize(FACTS)
    assert source == "template_fallback"
    assert text
    assert len(calls) == 1


def test_daily_cap_stops_calling_provider(key, monkeypatch):
    monkeypatch.setenv("LLM_DAILY_CAP", "1")
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        return Response("Amount is $90.00. [f1]")

    monkeypatch.setattr(case_facts.httpx, "post", post)
    first, first_source = case_facts.summarize(FACTS)
    second, second_source = case_facts.summarize(FACTS)
    assert first_source == "llm"
    assert second_source == "template_capped"
    assert "90.00" in first
    assert second == case_facts.template_summary(FACTS)
    assert len(calls) == 1


def test_cap_zero_never_reaches_provider(key, monkeypatch):
    monkeypatch.setenv("LLM_DAILY_CAP", "0")
    calls = []
    monkeypatch.setattr(case_facts.httpx, "post", lambda *a, **k: calls.append(1) or Response("x [f1]"))
    text, source = case_facts.summarize(FACTS)
    assert source == "template_capped"
    assert calls == []
    assert text


def test_refresh_case_summary_updates_store(key, monkeypatch):
    monkeypatch.setattr(
        case_facts.httpx, "post",
        lambda *a, **k: Response("Amount is $90.00. [f1]"),
    )
    updated = {}
    monkeypatch.setattr(store, "update_case_summary", lambda cid, text, source: updated.update(
        {"cid": cid, "text": text, "source": source}
    ))
    case_facts.refresh_case_summary(7, FACTS)
    assert updated["cid"] == 7
    assert updated["source"] == "llm"
