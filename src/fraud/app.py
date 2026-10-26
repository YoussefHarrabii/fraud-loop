"""FastAPI scoring service, analyst case queue, and demo UI."""

import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fraud import case_facts, monitoring, serving, store

BASE = Path(__file__).resolve().parents[2]
STATIC = BASE / "src" / "fraud" / "static"
STATIC.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Fraud Detection", version="0.3")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
store.init_db()

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
    }


@app.post("/score")
def score(tx: Txn, background: BackgroundTasks) -> dict:
    b = get_bundle()
    history = store.get_history(tx.cc_num, tx.ts)
    row = serving.build_one(history, tx.model_dump())
    out = b.score(row)
    rid = store.log_request(tx, out, row, b.card["created_at"])
    case_id = None
    if out["decision"] == "flag":
        facts = case_facts.build_facts(tx.model_dump(), history, out, row)
        summary = case_facts.template_summary(facts)
        case_id = store.create_case(rid, tx.cc_num, facts, summary, "template", b.card["created_at"])
        background.add_task(case_facts.refresh_case_summary, case_id, facts)
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


@app.post("/cases/{case_id}/decision")
def decide(case_id: int, body: Decision) -> dict:
    if body.decision not in {"confirmed_fraud", "false_alarm"}:
        raise HTTPException(status_code=400, detail="decision must be confirmed_fraud or false_alarm")
    out = store.decide_case(case_id, body.decision, body.analyst)
    if out is None:
        raise HTTPException(status_code=404, detail="case not found")
    return out


@app.get("/requests")
def requests(limit: int = Query(20, ge=1, le=500)) -> list[dict]:
    return store.recent_requests(limit)


@app.get("/monitoring")
def monitoring_report(limit: int = Query(500, ge=1, le=5000)) -> dict:
    return monitoring.report(store.request_features(limit), store.case_stats(), monitoring.load_profile())


@app.get("/samples")
def samples(limit: int = Query(8, ge=1, le=50)) -> list[dict]:
    return store.sample_transactions(limit)


@app.post("/demo/inject-drift")
def inject_drift(n: int = 60) -> dict:
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
