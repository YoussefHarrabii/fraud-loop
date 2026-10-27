import pandas as pd

from fraud import case_facts


def _facts():
    history = pd.DataFrame({"amt": [10.0, 20.0, 30.0]})
    out = {
        "score": 0.91,
        "threshold": 0.071,
        "top_drivers": [{"feature": "amt"}, {"feature": "hour"}, {"feature": "merchant"}],
    }
    row = {
        "hour": 3, "is_night": 1, "card_age_days": 400,
        "tx_count_1h": 2, "tx_count_24h": 6, "tx_count_7d": 35,
        "new_merchant": 1, "new_category": 0, "secs_since_same_merchant": 3600,
    }
    return case_facts.build_facts({"amt": 90.0}, history, out, row)


def test_facts_have_ids_and_numbers(monkeypatch):
    monkeypatch.setattr(case_facts, "_amount_sample", lambda: [])
    facts = _facts()
    assert [f["id"] for f in facts] == [f"f{i}" for i in range(1, 9)]
    assert "$90.00" in facts[0]["text"]
    assert "4.5x" in facts[0]["text"]
    assert "0.910" in facts[6]["text"]


def test_peer_percentile_fact(monkeypatch):
    monkeypatch.setattr(case_facts, "_amount_sample", lambda: [float(i) for i in range(100)])
    facts = _facts()
    assert [f["id"] for f in facts][-1] == "f9"
    assert "91%" in facts[-1]["text"]
    assert "trained on" in facts[-1]["text"]


def test_no_peer_fact_without_a_reference(monkeypatch):
    monkeypatch.setattr(case_facts, "_amount_sample", lambda: [])
    assert [f["id"] for f in _facts()] == [f"f{i}" for i in range(1, 9)]


def test_driver_direction_comes_from_the_contribution():
    facts = case_facts.build_facts(
        {"amt": 90.0},
        pd.DataFrame({"amt": [10.0, 20.0]}),
        {
            "score": 0.9,
            "threshold": 0.1,
            "top_drivers": [
                {"feature": "amt", "contribution": 0.4},
                {"feature": "hour", "contribution": -0.2},
                {"feature": "city_pop", "contribution": 0.0},
            ],
        },
        {"hour": 3, "is_night": 1, "card_age_days": 400, "tx_count_1h": 1, "tx_count_24h": 2,
         "tx_count_7d": 9, "new_merchant": 0, "new_category": 0, "secs_since_same_merchant": -1},
    )
    drivers = [f for f in facts if f["id"] == "f8"][0]["text"]
    assert "amt (raises the score)" in drivers
    assert "hour (lowers the score)" in drivers
    assert "city_pop," not in drivers


def test_recommendation_follows_score_and_merchant_novelty():
    assert case_facts.recommend({"score": 0.9, "threshold": 0.071}, {"new_merchant": 1}) == "check merchant"
    assert case_facts.recommend({"score": 0.9, "threshold": 0.071}, {"new_merchant": 0}) == "contact cardholder"
    assert case_facts.recommend({"score": 0.08, "threshold": 0.071}, {"new_merchant": 1}) == "likely benign"
    assert case_facts.recommend({"score": 0.9, "threshold": 0.0}, {"new_merchant": 1}) == "likely benign"


def test_template_uses_facts():
    facts = _facts()
    summary = case_facts.template_summary(facts)
    assert "$90.00" in summary
    assert "0.910" in summary


def test_validator_accepts_cited_summary():
    facts = _facts()
    good = f"{facts[0]['text']} [f1] {facts[1]['text']} [f2]"
    assert case_facts.validate_summary(good, facts)


def test_validator_rejects_invented_number():
    facts = _facts()
    good = f"{facts[0]['text']} [f1]"
    assert not case_facts.validate_summary(good + " Extra $999.99 [f1]", facts)


def test_validator_rejects_missing_citation():
    facts = _facts()
    assert not case_facts.validate_summary(facts[0]["text"], facts)


def test_validator_rejects_verdict_phrase():
    facts = _facts()
    text = f"{facts[0]['text']} This is fraud. [f1]"
    assert not case_facts.validate_summary(text, facts)


def test_validator_rejects_unknown_citation():
    facts = _facts()
    assert not case_facts.validate_summary(f"{facts[0]['text']} [f99]", facts)


def test_summarize_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(case_facts, "_load_env", lambda: None)
    text, source = case_facts.summarize(_facts())
    assert source == "template"
    assert text


def test_zero_median_is_not_no_history():
    facts = case_facts.build_facts(
        {"amt": 10.0},
        pd.DataFrame({"amt": [0.0, 0.0]}),
        {"score": 0.7, "threshold": 0.1, "top_drivers": [{"feature": "amt"}]},
        {"hour": 12, "is_night": 0, "card_age_days": 10, "tx_count_1h": 0, "tx_count_24h": 0,
         "tx_count_7d": 2, "new_merchant": 0, "new_category": 0, "secs_since_same_merchant": -1},
    )
    assert "no prior transactions" not in facts[0]["text"]
    assert "median is $0.00" in facts[0]["text"]
    assert "2 prior transactions" in facts[0]["text"]


def test_empty_history_facts():
    facts = case_facts.build_facts(
        {"amt": 12.0},
        pd.DataFrame({"amt": []}),
        {"score": 0.7, "threshold": 0.1, "top_drivers": [{"feature": "amt"}]},
        {"hour": 12, "is_night": 0, "card_age_days": 0, "tx_count_1h": 0, "tx_count_24h": 0,
         "tx_count_7d": 0, "new_merchant": 1, "new_category": 1, "secs_since_same_merchant": -1},
    )
    assert "no prior transactions" in facts[0]["text"]
    assert "no prior transaction" in facts[5]["text"]
    assert "0 days old" in facts[2]["text"]
    assert case_facts.template_summary(facts)
