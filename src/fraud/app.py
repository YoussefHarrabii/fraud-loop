"""FastAPI scoring service, analyst case queue, and demo UI."""

import os
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fraud import access, case_facts, limits, monitoring, retention, serving, store

BASE = Path(__file__).resolve().parents[2]
STATIC = BASE / "src" / "fraud" / "static"
STATIC.mkdir(parents=True, exist_ok=True)

DOCS = os.getenv("FRAUD_DOCS") == "1"

app = FastAPI(
    title="Fraud / Loop",
    version="0.3",
    # Off unless FRAUD_DOCS=1. A public demo has no reason to publish its schema.
    docs_url="/docs" if DOCS else None,
    redoc_url="/redoc" if DOCS else None,
    openapi_url="/openapi.json" if DOCS else None,
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
store.init_db()
retention.maybe_purge(force=True)

CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; font-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
    response.headers.setdefault("Content-Security-Policy", CSP)
    if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


def guard(scope: str):
    """Rate-limit dependency. 429 with Retry-After when the window is full."""

    def dependency(request: Request) -> None:
        if not limits.allow(scope, limits.client_of(request)):
            raise HTTPException(status_code=429, detail="rate limited", headers={"Retry-After": "60"})

    return dependency

_bundle = None


def get_bundle() -> serving.Bundle:
    global _bundle
    if _bundle is None:
        _bundle = serving.Bundle()
    return _bundle


class Txn(BaseModel):
    cc_num: str
    ts: datetime
    amt: float = Field(gt=0)
    category: str
    merchant: str
    state: str
    city_pop: float
    trans_num: str | None = None


class Decision(BaseModel):
    decision: str
    analyst: str = "analyst"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    b = get_bundle()
    return {
        "status": "ok",
        "threshold": b.threshold,
        "model_version": b.card["created_at"],
        "counts": store.counts(),
        "llm": {**store.llm_usage_today(), "cap": case_facts.daily_cap()},
    }


@app.post("/score", dependencies=[Depends(guard("score"))])
def score(tx: Txn) -> dict:
    retention.maybe_purge()
    b = get_bundle()
    history = store.get_history(tx.cc_num, tx.ts)
    row = serving.build_one(history, tx.model_dump())
    out = b.score(row)
    rid = store.log_request(tx, out, row, b.card["created_at"])
    case_id = None
    if out["decision"] == "flag":
        facts = case_facts.build_facts(tx.model_dump(), history, out, row)
        summary = case_facts.template_summary(facts)
        case_id = store.create_case(
            rid, tx.cc_num, facts, summary, "template", b.card["created_at"],
            recommendation=case_facts.recommend(out, row),
        )
    return {**out, "id": rid, "case_id": case_id, "cc_num": tx.cc_num, "trans_num": tx.trans_num}


@app.get("/cases")
def cases(status: str = "open", limit: int = Query(50, ge=1, le=200)) -> list[dict]:
    return store.list_cases(status or None, limit)


@app.get("/cases/{case_id}")
def case(case_id: int) -> dict:
    out = store.get_case(case_id)
    if out is None:
        raise HTTPException(status_code=404, detail="case not found")
    return out


@app.post("/cases/{case_id}/decision", dependencies=[Depends(guard("decision"))])
def decide(case_id: int, body: Decision) -> dict:
    if body.decision not in {"confirmed_fraud", "false_alarm"}:
        raise HTTPException(status_code=400, detail="decision must be confirmed_fraud or false_alarm")
    out = store.decide_case(case_id, body.decision, body.analyst)
    if out is None:
        raise HTTPException(status_code=404, detail="case not found")
    return out


@app.get("/access", dependencies=[Depends(guard("access"))])
def access_status(x_access_key: str | None = Header(default=None)) -> dict:
    """Validate a key without spending a use. The key travels in a header, never a URL."""
    return access.status(x_access_key)


@app.post("/cases/{case_id}/report", status_code=202, dependencies=[Depends(guard("report"))])
def generate_report(
    case_id: int,
    background: BackgroundTasks,
    x_access_key: str | None = Header(default=None),
) -> dict:
    case = store.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    record, reason, remaining = access.consume(x_access_key)
    if reason != "ok":
        raise HTTPException(status_code=access.REASON_STATUS[reason], detail=reason)
    background.add_task(_write_report, case_id, case["facts"], record)
    return {"status": "queued", "remaining": remaining, "expires_at": record.get("expires_at")}


def _write_report(case_id: int, facts: list[dict], record: dict) -> None:
    """A use is spent when queued and refunded when the provider does not deliver."""
    if case_facts.refresh_case_summary(case_id, facts) != "llm":
        access.refund(record)


@app.get("/requests")
def requests(limit: int = Query(20, ge=1, le=500)) -> list[dict]:
    return store.recent_requests(limit)


@app.get("/monitoring")
def monitoring_report(limit: int = Query(500, ge=1, le=5000)) -> dict:
    return monitoring.report(store.request_features(limit), store.case_stats(), monitoring.load_profile())


@app.get("/samples")
def samples(limit: int = Query(8, ge=1, le=50)) -> list[dict]:
    return store.sample_transactions(limit)


@app.post("/demo/inject-drift", dependencies=[Depends(guard("drift"))])
def inject_drift(n: int = 60) -> dict:
    retention.maybe_purge()
    n = max(1, min(n, 300))
    b = get_bundle()
    picked = store.sample_transactions(n)
    flagged = 0
    for s in picked:
        ts = pd.Timestamp(s["ts"]).replace(hour=2, minute=0, second=0, microsecond=0)
        tx = Txn(
            cc_num=str(s["cc_num"]), ts=ts, amt=float(s["amt"]) * 3.0,
            category="shopping_net", merchant=str(s["merchant"]),
            state=str(s["state"]), city_pop=float(s["city_pop"]),
            trans_num=f"injected-{uuid.uuid4().hex[:12]}",
        )
        history = store.get_history(tx.cc_num, tx.ts)
        row = serving.build_one(history, tx.model_dump())
        out = b.score(row)
        store.log_request(tx, out, row, b.card["created_at"], source="injected")
        flagged += int(out["decision"] == "flag")
    return {"injected": len(picked), "flagged": flagged}


