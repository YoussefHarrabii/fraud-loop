import pytest

from fraud import monitoring, serving, store


def test_counts_on_an_empty_database(db):
    assert store.counts() == {
        "transactions": 0, "requests": 0, "cases_open": 0, "cases_total": 0,
    }


def test_missing_profile_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring, "PROFILE", tmp_path / "missing.json")
    assert monitoring.load_profile() is None


def test_missing_bundle_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        serving.Bundle(tmp_path)
