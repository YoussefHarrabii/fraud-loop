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
