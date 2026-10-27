"""Deploy readiness checks: database, model bundle, key registry, provider config, live instance.

    uv run python scripts/preflight.py
    uv run python scripts/preflight.py --url http://127.0.0.1:8000

Against Heroku, seed and check from your machine. The image deliberately has no
`scripts/` and no `data/`, so nothing here can run with `heroku run`:

    heroku config:get DATABASE_URL -a <app>
    DATABASE_URL=<that url> uv run python scripts/preflight.py --url https://<app>.herokuapp.com

Read-only: it never creates, migrates or writes. Exit code 1 when any check fails.
"""

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def redact(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.hostname or '?'}:{parts.port or '?'}{parts.path}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="also check a running instance over HTTP")
    parser.add_argument("--min-transactions", type=int, default=1000)
    args = parser.parse_args()

    results: list[tuple[bool, str, str]] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        results.append((bool(ok), label, detail))

    from fraud import access, case_facts, serving, store

    # ---- database -------------------------------------------------------
    try:
        with store.connect() as con:
            con.execute("SELECT 1").fetchone()
        check(True, "database reachable", redact(store.dsn()))
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        check(False, "database reachable", f"{type(exc).__name__}: {exc}"[:160])
        report(results)
        return 1

    with store.connect() as con:
        version = con.execute("SHOW server_version").fetchone()[0]
        present = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            ).fetchall()
        }
    check(True, "postgres version", str(version))
    missing = sorted(set(store.TABLES) - present)
    check(not missing, "schema tables", "all present" if not missing else f"missing {', '.join(missing)}")

    counts = store.counts()
    check(
        counts["transactions"] >= args.min_transactions,
        "transactions seeded",
        f"{counts['transactions']:,} rows (min {args.min_transactions:,})",
    )
    check(True, "demo rows", f"{counts['cases_total']} cases, {counts['requests']} requests")

    # ---- model bundle ---------------------------------------------------
    try:
        bundle = serving.Bundle()
        check(True, "model loads", f"{bundle.model.num_trees()} trees, {bundle.model.num_feature()} features")
        check(True, "threshold", f"{bundle.threshold:.6f} from {bundle.card.get('created_at')}")
    except Exception as exc:  # noqa: BLE001
        check(False, "model loads", f"{type(exc).__name__}: {exc}"[:160])

    profile = ROOT / "artifacts" / "final" / "reference.json"
    try:
        import json

        doc = json.loads(profile.read_text(encoding="utf-8"))
        check(True, "drift reference", f"{len(doc.get('features', {}))} features, n={doc.get('n')}")
    except Exception as exc:  # noqa: BLE001
        check(False, "drift reference", f"{type(exc).__name__}: {exc}"[:160])

    # ---- invitation keys ------------------------------------------------
    registry = access.load_registry()
    check(bool(registry), "key registry", f"{len(registry)} keys hashed in configs/key_registry.json")
    if registry:
        expiries = sorted(r.get("expires_at", "") for r in registry.values())
        check(True, "key expiry window", f"{expiries[0][:10]} to {expiries[-1][:10]}")

    # ---- provider config -------------------------------------------------
    # Read the environment before llm_config(), which loads secrets/llm.env into it.
    from_env = any(
        os.getenv(name) for name in ("OPENAI_API_KEY", "OPENCODE_KEY", "OPENCODE_API_KEY", "OPENROUTER_API_KEY")
    )
    config = case_facts.llm_config()
    if config and from_env:
        check(True, "provider key", "found in the environment (survives into a container)")
    elif config:
        check(True, "provider key", "found in a local file only; the image has no .env, set it as a config var")
    else:
        check(False, "provider key", "missing, every summary falls back to the template")
    if config:
        check(True, "provider base", config["base"])
        check(True, "provider model", config["model"])
    check(case_facts.daily_cap() > 0, "daily cap", str(case_facts.daily_cap()))

    # ---- live instance ---------------------------------------------------
    if args.url:
        import httpx

        base = args.url.rstrip("/")
        try:
            with httpx.Client(base_url=base, timeout=30) as client:
                health = client.get("/health")
                check(health.status_code == 200, "GET /health", str(health.status_code))
                if health.status_code == 200:
                    body = health.json()
                    live = body.get("counts", {}).get("transactions")
                    check(True, "live counts", f"{live:,} transactions" if live is not None else "unknown")
                    check(
                        live == counts["transactions"],
                        "live db matches",
                        f"instance {live} vs local {counts['transactions']}",
                    )
                docs = client.get("/docs")
                check(docs.status_code == 404, "docs hidden", str(docs.status_code))
                schema = client.get("/openapi.json")
                check(schema.status_code == 404, "openapi hidden", str(schema.status_code))
                page = client.get("/")
                check(page.status_code == 200, "GET /", str(page.status_code))
        except Exception as exc:  # noqa: BLE001
            check(False, "instance reachable", f"{type(exc).__name__}: {exc}"[:160])

    failed = report(results)
    return 1 if failed else 0


def report(results: list[tuple[bool, str, str]]) -> int:
    width = max(len(label) for _, label, _ in results)
    for ok, label, detail in results:
        print(f"{'ok  ' if ok else 'FAIL'}  {label.ljust(width)}  {detail}")
    failed = sum(1 for ok, _, _ in results if not ok)
    print()
    print(f"{len(results) - failed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
