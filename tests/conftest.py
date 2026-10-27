import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

# Tests get their own database. Assigning (not setdefault) guarantees a stray
# DATABASE_URL in the shell can never point the suite at real data.
TEST_URL = os.environ.get(
    "FRAUD_TEST_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5432/fraud_test"
)
os.environ["DATABASE_URL"] = TEST_URL

# Tests must never reach a real provider. `load_dotenv` does not override values that are
# already set, so an empty string here also blocks `secrets/llm.env`.
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENCODE_KEY"] = ""
os.environ["OPENCODE_API_KEY"] = ""
os.environ["OPENROUTER_API_KEY"] = ""

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psycopg  # noqa: E402
import pytest  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from fraud import features, serving  # noqa: E402


def ensure_database(url: str) -> None:
    """Create the database named in `url` when the server does not have it yet."""
    parts = urlsplit(url)
    name = parts.path.lstrip("/")
    admin = url.rsplit("/", 1)[0] + "/postgres"
    with psycopg.connect(admin, autocommit=True, connect_timeout=5) as con:
        if not con.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            con.execute(f'CREATE DATABASE "{name}"')


@pytest.fixture(scope="session", autouse=True)
def _database():
    from fraud import store

    ensure_database(TEST_URL)
    store.reset_db()
    yield


@pytest.fixture(autouse=True)
def _clean_tables():
    """Every test starts against empty tables, and the transaction history too."""
    from fraud import store

    store.clear_demo()
    with store.connect() as con:
        con.execute("TRUNCATE transactions")
    yield


def insert_tx(cc, ts, amt, category="shopping_net", merchant="merchant_a",
              state="CA", city_pop=1000, is_fraud=0):
    from fraud import store

    with store.connect() as con:
        con.execute(
            "INSERT INTO transactions (cc_num, ts_epoch, trans_num, amt, category, merchant, state, "
            "city_pop, is_fraud) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (cc, int(pd.Timestamp(ts).value // 10**9), f"tx-{uuid.uuid4().hex[:10]}",
             float(amt), category, merchant, state, int(city_pop), int(is_fraud)),
        )


def _write_bundle(path):
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
def db():
    """The connection string under test. Kept so existing tests read the same."""
    return TEST_URL


@pytest.fixture
def insert(db):
    def _insert(cc, ts, amt, category="shopping_net", merchant="merchant_a",
                state="CA", city_pop=1000, is_fraud=0):
        return insert_tx(cc, ts, amt, category, merchant, state, city_pop, is_fraud)
    return _insert


@pytest.fixture
def client(db, tiny_bundle, monkeypatch):
    import fraud.app as app_module
    monkeypatch.setattr(app_module, "_bundle", tiny_bundle)
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


GOOD_KEY = "fr_test_good_key"
EXPIRED_KEY = "fr_test_expired_key"


@pytest.fixture
def keys(tmp_path, monkeypatch):
    """A two-key registry: one usable twice, one already expired."""
    from fraud import access

    path = tmp_path / "key_registry.json"
    now = datetime.now(timezone.utc)
    path.write_text(json.dumps({
        "version": 1,
        "keys": [
            {"id": "a1", "label": "Test", "hash": access.digest(GOOD_KEY),
             "max_uses": 2, "expires_at": (now + timedelta(days=1)).isoformat()},
            {"id": "a2", "label": "Old", "hash": access.digest(EXPIRED_KEY),
             "max_uses": 2, "expires_at": (now - timedelta(days=1)).isoformat()},
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(access, "REGISTRY", path)
    monkeypatch.setattr(access, "CACHE", {"mtime": None, "keys": {}})
    return {"good": GOOD_KEY, "expired": EXPIRED_KEY}

@pytest.fixture(autouse=True)
def _fresh_limits():
    from fraud import limits
    limits.reset()
    yield
    limits.reset()
