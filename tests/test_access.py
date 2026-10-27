import json
from datetime import datetime, timezone
from pathlib import Path

from fraud import access, store

ROOT = Path(__file__).resolve().parents[1]

def test_unknown_key_is_rejected(db, keys):
    result = access.status("fr_not_a_key")
    assert result["valid"] is False
    assert result["reason"] == "unknown"
    record, reason, remaining = access.consume("fr_not_a_key")
    assert record is None and reason == "unknown" and remaining == 0


def test_status_does_not_spend_a_use(db, keys):
    assert access.status(keys["good"])["remaining"] == 2
    assert access.status(keys["good"])["remaining"] == 2


def test_consume_counts_down_and_stops(db, keys):
    record, reason, remaining = access.consume(keys["good"])
    assert reason == "ok" and remaining == 1
    assert record["id"] == "a1"
    assert access.consume(keys["good"])[2] == 0
    assert access.consume(keys["good"])[1] == "exhausted"
    assert access.status(keys["good"])["reason"] == "exhausted"


def test_expired_key_is_rejected(db, keys):
    assert access.status(keys["expired"])["reason"] == "expired"
    record, reason, _ = access.consume(keys["expired"])
    assert record is None and reason == "expired"


def test_expired_key_never_touches_the_counter(db, keys):
    access.consume(keys["expired"])
    access.consume(keys["expired"])
    assert store.key_use_count(access.digest(keys["expired"])) == 0
    assert access.status(keys["expired"])["remaining"] == 0


def test_refund_gives_the_use_back(db, keys):
    record, _, remaining = access.consume(keys["good"])
    assert remaining == 1
    access.refund(record)
    access.refund(record)  # never goes below zero
    assert access.status(keys["good"])["remaining"] == 2


def test_blank_and_missing_header_are_unknown(db, keys):
    assert access.status("")["reason"] == "unknown"
    assert access.status(None)["reason"] == "unknown"


def test_missing_registry_means_no_keys(tmp_path, monkeypatch, db):
    monkeypatch.setattr(access, "REGISTRY", tmp_path / "absent.json")
    monkeypatch.setattr(access, "CACHE", {"mtime": None, "keys": {}})
    assert access.status("anything")["reason"] == "unknown"


def test_committed_registry_holds_hashes_only():
    path = ROOT / "configs" / "access_keys.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    assert registry["keys"], "mint at least one key"
    for entry in registry["keys"]:
        assert set(entry) == {"id", "label", "hash", "max_uses", "expires_at"}
        assert len(entry["hash"]) == 64
        int(entry["hash"], 16)
        assert entry["max_uses"] >= 1
        assert datetime.fromisoformat(entry["expires_at"]).tzinfo is not None
    assert "fr_" not in path.read_text(encoding="utf-8")


def test_mint_output_hashes_match_the_plaintext(tmp_path, monkeypatch):
    """If the script hashed something other than what it prints, every handed-out key is dead."""
    import csv
    import importlib.util
    import sys

    registry = tmp_path / "keys.json"
    plain = tmp_path / "keys.csv"
    spec = importlib.util.spec_from_file_location("mint_keys", ROOT / "scripts" / "mint_keys.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REGISTRY = registry
    module.PLAIN = plain

    monkeypatch.setattr(sys, "argv", ["mint_keys.py", "--count", "3", "--uses", "5"])
    assert module.main() == 0

    with plain.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    stored = json.loads(registry.read_text(encoding="utf-8"))["keys"]
    assert len(rows) == len(stored) == 3
    for row, entry in zip(rows, stored):
        assert access.digest(row["key"]) == entry["hash"]
        assert row["uses"] == "5"
        assert row["key"].startswith("fr_")

