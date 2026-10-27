def test_page_and_score_flow(client, insert):
    insert("C1", "2020-12-10 10:00:00", 10)
    insert("C1", "2020-12-12 10:00:00", 30)
    insert("C2", "2020-12-05 10:00:00", 40, is_fraud=1)
    insert("C3", "2020-12-06 10:00:00", 50, is_fraud=1)

    page = client.get("/")
    assert page.status_code == 200
    assert "Fraud / Loop" in page.text

    samples = client.get("/samples?limit=4").json()
    assert len(samples) == 4

    payload = {
        "cc_num": "C1",
        "ts": "2020-12-15T12:00:00",
        "amt": 120.0,
        "category": "shopping_net",
        "merchant": "merchant_a",
        "state": "CA",
        "city_pop": 1000,
        "trans_num": "api-test",
    }
    scored = client.post("/score", json=payload)
    assert scored.status_code == 200, scored.text
    out = scored.json()
    assert 0 <= out["score"] <= 1
    assert out["case_id"]

    case = client.get(f"/cases/{out['case_id']}").json()
    assert case["facts"]
    assert case["summary"]

    decided = client.post(
        f"/cases/{out['case_id']}/decision",
        json={"decision": "confirmed_fraud", "analyst": "tester"},
    ).json()
    assert decided["status"] == "closed"
    assert decided["decision"] == "confirmed_fraud"

    mon = client.get("/monitoring").json()
    assert mon["n"] >= 1
    assert len(mon["score_histogram"]["counts"]) == 10

    injected = client.post("/demo/inject-drift?n=2").json()
    assert injected["injected"] == 2

    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert client.get("/requests").json()


def test_cold_start_without_transactions(client):
    payload = {
        "cc_num": "C-new", "ts": "2020-12-15T12:00:00", "amt": 120.0,
        "category": "shopping_net", "merchant": "merchant_a", "state": "CA", "city_pop": 1000,
    }
    out = client.post("/score", json=payload).json()
    assert out["case_id"]
    assert client.get("/samples").json() == []
    assert client.post("/demo/inject-drift?n=0").json() == {"injected": 0, "flagged": 0}


def test_validation_and_missing_paths(client):
    assert client.post("/score", json={}).status_code == 422
    bad = {
        "cc_num": "C1", "ts": "2020-12-15T12:00:00", "amt": 0,
        "category": "shopping_net", "merchant": "merchant_a", "state": "CA", "city_pop": 1000,
    }
    assert client.post("/score", json=bad).status_code == 422
    assert client.post("/score", json={**bad, "amt": -1}).status_code == 422
    assert client.get("/cases?limit=0").status_code == 422
    assert client.get("/cases?limit=500").status_code == 422
    assert client.get("/samples?limit=0").status_code == 422
    assert client.get("/monitoring?limit=0").status_code == 422
    assert client.get("/requests?limit=0").status_code == 422
    assert client.get("/cases?status=open&limit=200").status_code == 200

    assert client.get("/cases/999").status_code == 404
    assert client.post("/cases/999/decision", json={"decision": "nope"}).status_code == 400
    assert client.post("/cases/999/decision", json={"decision": "confirmed_fraud"}).status_code == 404

    mon = client.get("/monitoring").json()
    assert mon["n"] == 0
    assert mon["drift_note"]
    assert client.get("/samples?limit=4").json() == []

def _payload(cc="C1"):
    return {
        "cc_num": cc, "ts": "2020-12-15T12:00:00", "amt": 120.0,
        "category": "shopping_net", "merchant": "merchant_a", "state": "CA", "city_pop": 1000,
    }


def test_access_endpoint_reports_the_key(client, keys):
    good = client.get("/access", headers={"X-Access-Key": keys["good"]}).json()
    assert good["valid"] is True
    assert good["remaining"] == good["max_uses"] == 2
    assert client.get("/access").json()["valid"] is False
    assert client.get("/access", headers={"X-Access-Key": keys["expired"]}).json()["reason"] == "expired"


def test_report_needs_a_valid_key(client, keys):
    case_id = client.post("/score", json=_payload()).json()["case_id"]
    assert client.post(f"/cases/{case_id}/report").status_code == 401
    assert client.post(f"/cases/{case_id}/report", headers={"X-Access-Key": "fr_nope"}).status_code == 401
    assert client.post(f"/cases/{case_id}/report", headers={"X-Access-Key": keys["expired"]}).status_code == 403
    assert client.post("/cases/99999/report", headers={"X-Access-Key": keys["good"]}).status_code == 404


def test_report_spends_a_use_and_refuses_when_empty(client, keys, monkeypatch):
    from fraud import case_facts

    monkeypatch.setattr(case_facts, "refresh_case_summary", lambda case_id, facts: "llm")
    case_id = client.post("/score", json=_payload()).json()["case_id"]

    first = client.post(f"/cases/{case_id}/report", headers={"X-Access-Key": keys["good"]})
    assert first.status_code == 202
    assert first.json()["status"] == "queued"
    assert first.json()["remaining"] == 1

    second = client.post(f"/cases/{case_id}/report", headers={"X-Access-Key": keys["good"]})
    assert second.json()["remaining"] == 0

    third = client.post(f"/cases/{case_id}/report", headers={"X-Access-Key": keys["good"]})
    assert third.status_code == 429
    assert third.json()["detail"] == "exhausted"


def test_failed_report_gives_the_use_back(client, keys, monkeypatch):
    from fraud import case_facts

    monkeypatch.setattr(case_facts, "refresh_case_summary", lambda case_id, facts: "template_fallback")
    case_id = client.post("/score", json=_payload()).json()["case_id"]
    response = client.post(f"/cases/{case_id}/report", headers={"X-Access-Key": keys["good"]})
    assert response.status_code == 202
    assert client.get("/access", headers={"X-Access-Key": keys["good"]}).json()["remaining"] == 2


def test_score_never_calls_the_model_without_a_report(client, keys, monkeypatch):
    from fraud import case_facts

    def forbidden(*_args, **_kwargs):
        raise AssertionError("scoring must not generate a report")

    monkeypatch.setattr(case_facts, "refresh_case_summary", forbidden)
    out = client.post("/score", json=_payload()).json()
    assert out["case_id"]
    case = client.get(f"/cases/{out['case_id']}").json()
    assert case["summary_source"] == "template"

def test_expired_key_never_reaches_the_provider(client, keys, monkeypatch):
    from fraud import access, case_facts, store

    calls = []

    def spy(*args, **kwargs):
        calls.append(args)
        return "llm"

    monkeypatch.setattr(case_facts, "refresh_case_summary", spy)
    case_id = client.post("/score", json=_payload()).json()["case_id"]

    response = client.post(f"/cases/{case_id}/report", headers={"X-Access-Key": keys["expired"]})

    assert response.status_code == 403
    assert response.json()["detail"] == "expired"
    assert calls == [], "an expired key must not queue a report"
    assert store.key_use_count(access.digest(keys["expired"])) == 0


def test_unknown_key_never_reaches_the_provider(client, keys, monkeypatch):
    from fraud import case_facts

    calls = []
    monkeypatch.setattr(case_facts, "refresh_case_summary", lambda *a, **k: calls.append(a) or "llm")
    case_id = client.post("/score", json=_payload()).json()["case_id"]

    assert client.post(f"/cases/{case_id}/report", headers={"X-Access-Key": "fr_made_up"}).status_code == 401
    assert calls == []


def test_exhausted_key_never_reaches_the_provider(client, keys, monkeypatch):
    from fraud import access, case_facts, store

    calls = []
    monkeypatch.setattr(case_facts, "refresh_case_summary", lambda *a, **k: calls.append(a) or "llm")
    case_id = client.post("/score", json=_payload()).json()["case_id"]

    assert client.post(f"/cases/{case_id}/report", headers={"X-Access-Key": keys["good"]}).status_code == 202
    assert client.post(f"/cases/{case_id}/report", headers={"X-Access-Key": keys["good"]}).status_code == 202
    calls.clear()

    assert client.post(f"/cases/{case_id}/report", headers={"X-Access-Key": keys["good"]}).status_code == 429
    assert calls == []
    assert store.key_use_count(access.digest(keys["good"])) == 2

def test_score_is_rate_limited(client, monkeypatch):
    monkeypatch.setenv("FRAUD_LIMIT_SCORE", "2")
    assert client.post("/score", json=_payload()).status_code == 200
    assert client.post("/score", json=_payload()).status_code == 200
    blocked = client.post("/score", json=_payload())
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "60"
    assert blocked.json()["detail"] == "rate limited"


def test_drift_and_decision_are_rate_limited(client, monkeypatch):
    monkeypatch.setenv("FRAUD_LIMIT_DRIFT", "1")
    assert client.post("/demo/inject-drift?n=1").status_code == 200
    assert client.post("/demo/inject-drift?n=1").status_code == 429

    case_id = client.post("/score", json=_payload()).json()["case_id"]
    monkeypatch.setenv("FRAUD_LIMIT_DECISION", "1")
    body = {"decision": "false_alarm", "analyst": "t"}
    assert client.post(f"/cases/{case_id}/decision", json=body).status_code == 200
    assert client.post(f"/cases/{case_id}/decision", json=body).status_code == 429


def test_access_probing_is_rate_limited(client, monkeypatch):
    monkeypatch.setenv("FRAUD_LIMIT_ACCESS", "2")
    assert client.get("/access", headers={"X-Access-Key": "x"}).status_code == 200
    assert client.get("/access", headers={"X-Access-Key": "x"}).status_code == 200
    assert client.get("/access", headers={"X-Access-Key": "x"}).status_code == 429


def test_security_headers(client):
    response = client.get("/")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert "Strict-Transport-Security" not in response.headers

def test_sql_injection_payloads_are_inert(client, insert):
    insert("C1", "2020-12-10 10:00:00", 10)
    before = client.get("/health").json()["counts"]

    payloads = [
        "' OR 1=1 --",
        "C1'; DROP TABLE cases; --",
        "C1' UNION SELECT 1,2,3,4,5,6 --",
        'C1" OR ""="',
    ]
    for value in payloads:
        body = {**_payload(value), "merchant": value}
        response = client.post("/score", json=body)
        assert response.status_code == 200, value
        out = response.json()
        assert out["cc_num"] == value
        if out["case_id"]:
            case = client.get(f"/cases/{out['case_id']}").json()
            assert case["cc_num"] == value, "the payload must be stored as text, not executed"

    assert client.get("/cases", params={"status": "' OR 1=1 --"}).json() == []
    assert client.get("/cases", params={"status": "open"}).status_code == 200
    assert client.get("/cases", params={"status": "x' UNION SELECT * FROM cases --"}).json() == []
    assert client.get("/cases", params={"limit": "1 OR 1=1"}).status_code == 422

    after = client.get("/health").json()["counts"]
    assert after["transactions"] == before["transactions"]
    assert after["cases_total"] >= before["cases_total"]
    assert client.get("/cases").status_code == 200, "the cases table still exists"
