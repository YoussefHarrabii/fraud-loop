import json
import os
import sqlite3
import tempfile
import uuid
from pathlib import Path

os.environ.setdefault("FRAUD_DB", str(Path(tempfile.mkdtemp(prefix="fraud-tests-")) / "app.db"))

# Tests must never reach a real provider. `load_dotenv` does not override values that are
# already set, so an empty string here also blocks `secrets/llm.env`.
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENCODE_KEY"] = ""
os.environ["OPENCODE_API_KEY"] = ""
os.environ["OPENROUTER_API_KEY"] = ""

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fraud import features, serving


def _tx_table(db: Path) -> None:
    con = sqlite3.connect(db)
    con.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            cc_num TEXT, ts_epoch INTEGER, trans_num TEXT, amt REAL,
            category TEXT, merchant TEXT, state TEXT, city_pop REAL, is_fraud INTEGER
        )
    """)
    con.commit()
    con.close()


def insert_tx(db, cc, ts, amt, category="shopping_net", merchant="merchant_a",
              state="CA", city_pop=1000, is_fraud=0):
    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?)",
        (cc, int(pd.Timestamp(ts).value // 10**9), f"tx-{uuid.uuid4().hex[:10]}",
         float(amt), category, merchant, state, float(city_pop), int(is_fraud)),
    )
    con.commit()
    con.close()


def _write_bundle(path: Path) -> Path:
    rng = np.random.default_rng(7)
    n = 240
    cats = {
        "category": ["shopping_net", "grocery_pos", "misc_net"],
        "merchant": ["merchant_a", "merchant_b", "merchant_c"],
        "state": ["CA", "NY", "TX"],
    }
    data = {col: rng.random(n) for col in features.NUMERIC}
    data["hour"] = rng.integers(0, 24, n)
    data["weekday"] = rng.integers(0, 7, n)
    data["is_weekend"] = (data["weekday"] >= 5).astype(int)
    data["is_night"] = ((data["hour"] >= 22) | (data["hour"] <= 5)).astype(int)
    for col, values in cats.items():
        data[col] = pd.Categorical(rng.choice(values, n), categories=values)
    df = pd.DataFrame(data)[features.NUMERIC + features.CATEGORICAL]
    y = rng.integers(0, 2, n)
    model = lgb.train(
        {"objective": "binary", "verbosity": -1, "num_threads": 2, "seed": 1, "min_data_in_leaf": 5},
        lgb.Dataset(df, label=y, categorical_feature=features.CATEGORICAL),
        num_boost_round=8,
    )
    model.save_model(str(path / "model.txt"))
    raw = model.predict(df)
    cal = LogisticRegression().fit(raw.reshape(-1, 1), y)
    joblib.dump(cal, path / "calibrator.joblib")
    (path / "model_card.json").write_text(
        json.dumps({"created_at": "test-model", "threshold": 0.0}), encoding="utf-8"
    )
    return path


@pytest.fixture(scope="session")
def tiny_bundle_path(tmp_path_factory):
    return _write_bundle(tmp_path_factory.mktemp("bundle"))


@pytest.fixture(scope="session")
def tiny_bundle(tiny_bundle_path):
    return serving.Bundle(tiny_bundle_path)


@pytest.fixture
def db(tmp_path, monkeypatch):
    from fraud import store
    monkeypatch.setattr(store, "DB", tmp_path / "test.db")
    store.init_db()
    _tx_table(store.DB)
    return store.DB


@pytest.fixture
def insert(db):
    def _insert(cc, ts, amt, category="shopping_net", merchant="merchant_a",
                state="CA", city_pop=1000, is_fraud=0):
        return insert_tx(db, cc, ts, amt, category, merchant, state, city_pop, is_fraud)
    return _insert


@pytest.fixture
def client(db, tiny_bundle, monkeypatch):
    import fraud.app as app_module
    monkeypatch.setattr(app_module, "_bundle", tiny_bundle)
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)
