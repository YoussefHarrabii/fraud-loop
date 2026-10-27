"""Smoke test a running instance: health, score, case, and whether the summary came from the model.

    uv run python scripts/smoke.py http://127.0.0.1:8000
    uv run python scripts/smoke.py https://<host> --key fr_... --expect-llm

`/score` never generates a summary: the report is gated behind an invitation key. So
`--expect-llm` needs `--key`, posts to `/cases/{id}/report`, and fails unless the case ends up
with `summary_source: llm`. That is the only end-to-end proof that the key reached the provider.
"""

import argparse
import os
import sys
import time

import httpx

# Summaries contain typographic quotes; a cp1252 console would otherwise crash on print.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def wait_for_summary(client: httpx.Client, case_id: int, timeout: float = 45.0) -> dict:
    deadline = time.time() + timeout
    case = {}
    while time.time() < deadline:
        case = client.get(f"/cases/{case_id}").json()
        if case.get("summary_source") != "template":
            return case
        time.sleep(1.5)
    return case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", nargs="?", default="http://127.0.0.1:8000")
    parser.add_argument("--expect-llm", action="store_true")
    parser.add_argument("--key", default=os.getenv("SMOKE_ACCESS_KEY"), help="invitation key for the gated report")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    if args.expect_llm and not args.key:
        parser.error("--expect-llm needs --key: the model report is gated, /score only builds a template case")

    with httpx.Client(base_url=args.base.rstrip("/"), timeout=30) as client:
        health = client.get("/health")
        health.raise_for_status()
        h = health.json()
        print(f"health ok | model {h['model_version'][:19]} | threshold {h['threshold']:.4f}")
        print(f"counts {h['counts']}")
        llm = h.get("llm", {})
        print(f"llm used {llm.get('used')} of {llm.get('cap')} for {llm.get('day')}")

        if args.key:
            status = client.get("/access", headers={"X-Access-Key": args.key}).json()
            print(
                f"key {status['remaining']} of {status['max_uses']} left | valid {status['valid']}"
                f" | expires {status['expires_at']}"
            )
            if not status["valid"]:
                print(f"key is not usable: {status['reason']}", file=sys.stderr)
                return 1

        samples = client.get("/samples", params={"limit": args.samples}).json()
        if not samples:
            print("no samples returned; the transactions table is empty", file=sys.stderr)
            return 1

        flagged = None
        for sample in samples:
            payload = {k: sample[k] for k in ("cc_num", "ts", "amt", "category", "merchant", "state", "city_pop")}
            out = client.post("/score", json=payload).json()
            print(f"score {out['score']:.4f} {out['decision']:<4} amt {sample['amt']:.2f} {sample['category']}")
            if out["decision"] == "flag":
                flagged = out
                break

        if flagged is None:
            print(f"no flag in {len(samples)} samples; try --samples 20")
            return 0 if not args.expect_llm else 1

        if args.key:
            queued = client.post(
                f"/cases/{flagged['case_id']}/report", headers={"X-Access-Key": args.key}
            )
            print(f"report {queued.status_code} {queued.json()}")
            if queued.status_code != 202:
                print("the report was not queued", file=sys.stderr)
                return 1

        case = wait_for_summary(client, flagged["case_id"])
        print(f"case {case['id']} | source {case['summary_source']} | status {case['status']}")
        print(f"summary: {case['summary'][:200]}")

        if args.key:
            after = client.get("/access", headers={"X-Access-Key": args.key}).json()
            print(f"key after: {after['remaining']} of {after['max_uses']} left")

        if args.expect_llm and case["summary_source"] != "llm":
            print("expected a model summary; the key or the cap blocked it", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
