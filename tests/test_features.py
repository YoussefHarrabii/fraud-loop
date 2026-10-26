import pandas as pd

from fraud import features


def test_window_count():
    df = pd.DataFrame({
        "cc_num": ["a", "a", "a"],
        "ts": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:30", "2020-01-01 02:00"]),
    })
    assert features._window_count(df, 3600).tolist() == [0, 1, 0]


def test_split_name():
    ts = pd.Series(pd.to_datetime(["2020-09-30", "2020-10-01", "2020-11-01", "2020-12-01"]))
    assert features.split_name(ts).tolist() == ["train", "val", "holdout", "stream"]


def test_split_exact_boundaries():
    ts = pd.Series(pd.to_datetime(["2020-09-30 23:59:59", "2020-10-01 00:00:00",
                                   "2020-11-01 00:00:00", "2020-12-01 00:00:00"]))
    assert features.split_name(ts).tolist() == ["train", "val", "holdout", "stream"]


def test_same_timestamp_counts_as_prior_row():
    df = pd.DataFrame({
        "cc_num": ["a", "a"],
        "ts": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:00"]),
    })
    assert features._window_count(df, 3600).tolist() == [0, 1]


def test_load_reads_and_writes_cache(tmp_path, monkeypatch):
    cache = tmp_path / "features.parquet"
    frame = pd.DataFrame({"x": [1, 2]})
    monkeypatch.setattr(features, "CACHE", cache)
    monkeypatch.setattr(features, "build", lambda: frame)
    out = features.load(rebuild=True)
    assert cache.exists()
    assert out.equals(features.load())


def test_build_uses_only_past_rows(tmp_path, monkeypatch):
    rows = [
        ("t1", "C1", "2020-09-30 23:50:00", 10, "c1", "m1", "CA", 1000, 0),
        ("t2", "C1", "2020-10-01 00:30:00", 20, "c1", "m2", "CA", 1000, 0),
        ("t3", "C1", "2020-10-01 02:00:00", 30, "c2", "m1", "NY", 2000, 1),
        ("t4", "C2", "2020-11-05 12:00:00", 40, "c1", "m1", "CA", 1000, 0),
        ("t5", "C3", "2020-12-05 12:00:00", 50, "c1", "m1", "CA", 1000, 0),
    ]
    raw = pd.DataFrame(rows, columns=[
        "trans_num", "cc_num", "ts", "amt", "category", "merchant", "state", "city_pop", "is_fraud",
    ])
    raw["ts"] = pd.to_datetime(raw["ts"])
    path = tmp_path / "raw.parquet"
    raw.to_parquet(path, index=False)
    monkeypatch.setattr(features, "RAW", path)

    out = features.build()
    c1 = out[out.cc_num == "C1"].sort_values("ts")

    assert c1.iloc[0]["tx_count_prior"] == 0
    assert c1.iloc[1]["tx_count_prior"] == 1
    assert c1.iloc[1]["tx_count_1h"] == 1
    assert c1.iloc[2]["tx_count_1h"] == 0
    assert c1.iloc[2]["tx_count_24h"] == 2
    assert c1.iloc[1]["new_merchant"] == 1
    assert c1.iloc[2]["new_merchant"] == 0
    assert c1.iloc[2]["amt_prior_mean"] == 15.0
    assert set(out.split) == {"train", "val", "holdout", "stream"}
