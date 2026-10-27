import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
PORT = 8765
URL = f"http://127.0.0.1:{PORT}"

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_UI"),
    reason="set RUN_UI=1 and install the Playwright browser",
)


def _rows():
    categories = ["shopping_net", "grocery_pos", "misc_net"]
    merchants = ["merchant_a", "merchant_b", "merchant_c"]
    states = ["CA", "NY", "TX"]
    rows = []
    for i in range(60):
        ts = datetime(2020, 12, 10, 10) + timedelta(minutes=i)
        rows.append((
            f"4090000000{i:06d}", f"ui-tx-{i}", ts.strftime("%Y-%m-%d %H:%M:%S"),
            20.0 + i, categories[i % 3], merchants[i % 3], states[i % 3], 1000 + i, i % 2,
        ))
    return rows


def _epoch(value: str) -> int:
    return int(time.mktime(time.strptime(value, "%Y-%m-%d %H:%M:%S")))


@pytest.fixture(scope="module")
def live_server(tmp_path_factory, tiny_bundle_path):
    db = tmp_path_factory.mktemp("ui") / "app.db"
    con = sqlite3.connect(db)
    con.execute("""
        CREATE TABLE transactions (
            cc_num TEXT, ts_epoch INTEGER, trans_num TEXT, amt REAL,
            category TEXT, merchant TEXT, state TEXT, city_pop REAL, is_fraud INTEGER
        )
    """)
    con.executemany(
        "INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?)",
        [(cc, _epoch(ts), tn, amt, cat, mer, st, pop, fraud)
         for cc, tn, ts, amt, cat, mer, st, pop, fraud in _rows()],
    )
    con.commit()
    con.close()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["FRAUD_DB"] = str(db)
    env["FRAUD_ARTIFACTS"] = str(tiny_bundle_path)
    env["OPENAI_API_KEY"] = ""
    env["OPENROUTER_API_KEY"] = ""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "fraud.app:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=ROOT, env=env,
    )
    try:
        for _ in range(60):
            try:
                if httpx.get(URL + "/health", timeout=1).status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not start")
        yield URL
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_score_case_and_drift(live_server):
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        errors = []
        csp = []
        fonts = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("console", lambda msg: csp.append(msg.text[:160]) if "Content Security Policy" in msg.text else None)
        page.on("response", lambda res: fonts.append(res.url) if "/static/vendor/fonts/" in res.url else None)
        page.goto(live_server)
        page.wait_for_selector(".sample")

        page.get_by_role("button", name="Score transaction").click()
        page.wait_for_selector("#decision.flag")
        assert "Flag" in page.inner_text("#decision")

        page.click("#open-case")
        page.wait_for_selector("#confirm")
        page.click("#confirm")
        page.wait_for_timeout(800)

        page.locator('button[data-view="monitor"]').click()
        page.wait_for_selector(".hist-bar")
        assert page.locator(".hist-bar").count() == 10
        page.wait_for_selector("#hairline-figure svg")

        page.click("#inject")
        deadline = time.time() + 20
        while time.time() < deadline:
            if "60" in page.inner_text("#m-injected"):
                break
            page.wait_for_timeout(500)
        else:
            raise AssertionError("drift injection did not land")
        assert not csp, csp
        assert fonts, "no web font was fetched: the stylesheet or the CSP is broken"
        assert not errors

        browser.close()
