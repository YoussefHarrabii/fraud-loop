from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from types import SimpleNamespace

from fraud import store


def test_concurrent_request_and_case_writes(db):
    def worker(i):
        out = {"score": 0.9, "raw_score": 1.0, "decision": "flag", "threshold": 0.1,
               "top_drivers": [{"feature": "amt", "contribution": 1.0}]}
        tx = SimpleNamespace(cc_num=f"C{i}", trans_num=f"t{i}", ts=datetime(2020, 12, 15), amt=10.0)
        rid = store.log_request(tx, out, {"amt": 10.0}, "test-model")
        store.create_case(rid, f"C{i}", [{"id": "f1", "text": "Amount is $10.00."}], "s", "template", "m")

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(worker, range(6)))

    counts = store.counts()
    assert counts["requests"] == 6
    assert counts["cases_total"] == 6
    assert store.case_stats()["open"] == 6
