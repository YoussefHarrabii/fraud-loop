import pandas as pd

from fraud import serving


def _history():
    return pd.DataFrame({
        "ts": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "amt": [10.0, 30.0],
        "merchant": ["merchant_a", "merchant_b"],
        "category": ["shopping_net", "grocery_pos"],
    })


def _tx():
    return {
        "ts": pd.Timestamp("2020-01-03"),
        "amt": 50.0,
        "category": "shopping_net",
        "merchant": "merchant_c",
        "state": "CA",
        "city_pop": 1000,
    }


def test_build_one_uses_history():
    row = serving.build_one(_history(), _tx())
    assert row["tx_count_prior"] == 2
    assert row["amt_prior_mean"] == 20.0
    assert row["new_merchant"] == 1
    assert row["card_age_days"] == 2.0
    assert row["tx_count_7d"] == 2


def test_bundle_scores(tiny_bundle):
    row = serving.build_one(_history(), _tx())
    out = tiny_bundle.score(row)
    assert 0 <= out["score"] <= 1
    assert out["decision"] == "flag"
    assert len(out["top_drivers"]) == 5


def test_cold_start_and_unknown_merchant(tiny_bundle):
    empty = pd.DataFrame({
        "ts": pd.to_datetime([]), "amt": [], "merchant": [], "category": [],
    })
    tx = {**_tx(), "merchant": "merchant_unknown"}
    row = serving.build_one(empty, tx)
    assert row["tx_count_prior"] == 0
    assert row["secs_since_prev"] == -1
    assert row["amt_ratio_mean"] == 1.0
    assert row["new_merchant"] == 1
    out = tiny_bundle.score(row)
    assert 0 <= out["score"] <= 1
