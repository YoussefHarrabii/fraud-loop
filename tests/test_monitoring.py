import numpy as np
import pandas as pd

from fraud import monitoring


def test_psi_identical_is_zero():
    a = np.arange(100, dtype=float)
    assert monitoring.psi(a, a) < 1e-6


def test_psi_detects_shift():
    a = np.arange(100, dtype=float)
    assert monitoring.psi(a, a + 1000) > 0.1


def test_report_includes_histogram_and_drift():
    rng = np.random.default_rng(3)
    n = 40
    columns = {col: rng.random(n) for col in monitoring.DRIFT_FEATURES}
    rows = [
        {
            "score": float(rng.random()),
            "decision": "flag" if i % 5 == 0 else "pass",
            "features": {col: float(values[i]) for col, values in columns.items()},
            "source": "live",
        }
        for i in range(n)
    ]
    profile = {
        "n": 200,
        "features": {col: {"sample": rng.random(100).tolist()} for col in monitoring.DRIFT_FEATURES},
    }
    cases = {"open": 1, "reviewed": 0, "precision": None}
    out = monitoring.report(rows, cases, profile)
    assert out["n"] == n
    assert sum(out["score_histogram"]["counts"]) == n
    assert len(out["score_histogram"]["counts"]) == 10
    assert set(out["drift"]) == set(monitoring.DRIFT_FEATURES)
    assert out["injected"] == 0
    assert out["live"] == n


def test_insufficient_sample_has_note():
    rows = [{"score": 0.2, "decision": "pass", "features": {}, "source": "live"} for _ in range(5)]
    out = monitoring.report(rows, {"open": 0, "reviewed": 0, "precision": None}, None)
    assert out["n"] == 5
    assert out["drift_note"]
    assert out["drift"] == {}


def test_psi_constant_distribution_is_undefined():
    assert monitoring.psi(np.ones(50), np.ones(50)) is None


def _rows(n, source="live"):
    rng = np.random.default_rng(4)
    columns = {col: rng.random(n) for col in monitoring.DRIFT_FEATURES}
    return [
        {"score": float(rng.random()), "decision": "pass", "source": source,
         "features": {col: float(values[i]) for col, values in columns.items()}}
        for i in range(n)
    ]


def _profile():
    rng = np.random.default_rng(5)
    return {"n": 200, "features": {col: {"sample": rng.random(100).tolist()} for col in monitoring.DRIFT_FEATURES}}


def test_drift_starts_at_thirty_requests():
    for n, has_note in [(29, True), (30, False)]:
        out = monitoring.report(_rows(n), {"open": 0, "reviewed": 0, "precision": None}, _profile())
        assert ("drift_note" in out) is has_note


def test_all_injected_traffic():
    rows = _rows(30, source="injected")
    out = monitoring.report(rows, {"open": 0, "reviewed": 0, "precision": None}, _profile())
    assert out["injected"] == 30
    assert out["live"] == 0


def test_short_arrays_are_undefined():
    assert monitoring.psi([1, 2], [1, 2]) is None
    assert monitoring.ks([1, 2], [1, 2]) is None


def test_profile_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring, "PROFILE", tmp_path / "reference.json")
    df = pd.DataFrame({col: np.arange(50, dtype=float) for col in monitoring.DRIFT_FEATURES})
    monitoring.save_profile(df)
    profile = monitoring.load_profile()
    assert profile["n"] == 50
    assert len(profile["features"]["amt"]["sample"]) == 50
