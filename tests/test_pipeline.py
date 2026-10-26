import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraud import features, serving


def _raw():
    rows = [
        ("t1", "C1", "2020-09-01 10:00:00", 10, "c1", "m1", "CA", 1000, 0),
        ("t2", "C1", "2020-09-02 10:00:00", 20, "c1", "m2", "CA", 1000, 1),
        ("t3", "C2", "2020-09-03 10:00:00", 30, "c2", "m1", "NY", 2000, 0),
        ("t4", "C2", "2020-09-04 10:00:00", 40, "c2", "m2", "NY", 2000, 1),
        ("t5", "C1", "2020-10-05 10:00:00", 50, "c1", "m3", "CA", 1000, 0),
        ("t6", "C2", "2020-10-06 10:00:00", 60, "c2", "m3", "NY", 2000, 1),
        ("t7", "C1", "2020-11-05 10:00:00", 70, "c1", "m1", "CA", 1000, 0),
        ("t8", "C2", "2020-11-06 10:00:00", 80, "c2", "m1", "NY", 2000, 1),
        ("t9", "C1", "2020-12-05 10:00:00", 90, "c1", "m2", "CA", 1000, 0),
        ("t10", "C2", "2020-12-06 10:00:00", 100, "c2", "m2", "NY", 2000, 1),
    ]
    raw = pd.DataFrame(rows, columns=[
        "trans_num", "cc_num", "ts", "amt", "category", "merchant", "state", "city_pop", "is_fraud",
    ])
    raw["ts"] = pd.to_datetime(raw["ts"])
    return raw


def test_no_leak_columns():
    used = set(features.NUMERIC + features.CATEGORICAL)
    assert not used & {"cc_num", "trans_num", "is_fraud", "unix_time", "profile"}


def test_build_splits_are_chronological(tmp_path, monkeypatch):
    path = tmp_path / "raw.parquet"
    _raw().to_parquet(path, index=False)
    monkeypatch.setattr(features, "RAW", path)
    data = features.build()
    spans = {name: (group.ts.min(), group.ts.max()) for name, group in data.groupby("split")}
    assert spans["train"][1] < spans["val"][0]
    assert spans["val"][1] < spans["holdout"][0]
    assert spans["holdout"][1] < spans["stream"][0]


def test_batch_and_serving_features_match(tmp_path, monkeypatch):
    raw = _raw()
    path = tmp_path / "raw.parquet"
    raw.to_parquet(path, index=False)
    monkeypatch.setattr(features, "RAW", path)
    data = features.build().sort_values("ts").reset_index(drop=True)

    row = data.iloc[4]
    history = raw[(raw.cc_num == row.cc_num) & (raw.ts < row.ts)]
    built = serving.build_one(history, row)

    for col in features.NUMERIC:
        a, b = float(row[col]), float(built[col])
        if np.isnan(a) and np.isnan(b):
            continue
        assert np.isclose(a, b, atol=1e-4), f"{col}: {a} != {b}"


def test_train_calibrate_bundle_score(tmp_path, monkeypatch):
    raw = _raw()
    path = tmp_path / "raw.parquet"
    raw.to_parquet(path, index=False)
    monkeypatch.setattr(features, "RAW", path)
    data = features.build()
    X = data[features.NUMERIC + features.CATEGORICAL]
    y = data.is_fraud.to_numpy()

    model = lgb.train(
        {"objective": "binary", "verbosity": -1, "num_threads": 2, "seed": 1, "min_data_in_leaf": 1},
        lgb.Dataset(X, label=y, categorical_feature=features.CATEGORICAL),
        num_boost_round=5,
    )
    cal = LogisticRegression().fit(model.predict(X).reshape(-1, 1), y)

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    model.save_model(str(bundle_dir / "model.txt"))
    joblib.dump(cal, bundle_dir / "calibrator.joblib")
    (bundle_dir / "model_card.json").write_text(json.dumps({"created_at": "test", "threshold": 0.5}))

    bundle = serving.Bundle(bundle_dir)
    first = data.iloc[0]
    built = serving.build_one(pd.DataFrame(columns=["ts", "amt", "merchant", "category"]), first)
    out = bundle.score(built)
    assert 0 <= out["score"] <= 1
    assert out["decision"] in {"flag", "pass"}
    assert len(out["top_drivers"]) == 5
