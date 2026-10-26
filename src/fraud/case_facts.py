"""Case facts, LLM summary, and the numeric/citation check. No raw text goes to the model."""

import json
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?%?")
DEFAULT_DAILY_CAP = 40


def _load_env() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "secrets" / "llm.env")


def daily_cap() -> int:
    """LLM calls allowed per UTC day across the whole process."""
    try:
        return max(0, int(os.getenv("LLM_DAILY_CAP", DEFAULT_DAILY_CAP)))
    except ValueError:
        return DEFAULT_DAILY_CAP


def build_facts(tx: dict, history, out: dict, row: dict) -> list[dict]:
    amt = float(tx["amt"])
    n = len(history)
    median = float(history.amt.median()) if n else None
    facts = []
    if not n:
        facts.append({"id": "f1", "text": f"Amount is ${amt:,.2f}, with no prior transactions on this card."})
    elif median and median > 0:
        facts.append({"id": "f1", "text": f"Amount is ${amt:,.2f}, {amt / median:.1f}x this card's median of ${median:,.2f}."})
    else:
        facts.append({"id": "f1", "text": f"Amount is ${amt:,.2f}; the card median is $0.00 across {n} prior transactions."})
    facts.append({"id": "f2", "text": f"Local time is {row['hour']:02d}:00, {'night' if row['is_night'] else 'day'}."})
    facts.append({"id": "f3", "text": f"The card has {n} prior transactions and is {row['card_age_days']:.0f} days old."})
    facts.append({"id": "f4", "text": f"Velocity is {row['tx_count_1h']} in the last hour, {row['tx_count_24h']} in 24 hours, {row['tx_count_7d']} in 7 days."})
    facts.append({"id": "f5", "text": f"The merchant is {'new' if row['new_merchant'] else 'known'} for this card; the category is {'new' if row['new_category'] else 'known'}."})
    if row["secs_since_same_merchant"] >= 0:
        facts.append({"id": "f6", "text": f"The last transaction at this merchant was {row['secs_since_same_merchant'] / 3600:.1f} hours ago."})
    else:
        facts.append({"id": "f6", "text": "There is no prior transaction at this merchant."})
    facts.append({"id": "f7", "text": f"Model score is {out['score']:.3f}; the review threshold is {out['threshold']:.3f}."})
    drivers = ", ".join(d["feature"] for d in out["top_drivers"][:3])
    facts.append({"id": "f8", "text": f"The largest model drivers are {drivers}."})
    return facts


def template_summary(facts: list[dict]) -> str:
    by_id = {f["id"]: f["text"] for f in facts}
    return " ".join(by_id[k] for k in ["f1", "f2", "f4", "f5", "f7"] if k in by_id)


def _numbers(text: str) -> list[float]:
    out = []
    for token in NUMBER.findall(text):
        try:
            out.append(float(token.replace("%", "").replace(",", "")))
        except ValueError:
            pass
    return out


def validate_summary(summary: str, facts: list[dict]) -> bool:
    cited = re.findall(r"\[(f\d+)\]", summary)
    known = {f["id"] for f in facts}
    if not cited or not set(cited) <= known:
        return False
    allowed = _numbers(" ".join(f["text"] for f in facts))
    body = re.sub(r"\[f\d+\]", " ", summary)
    for value in _numbers(body):
        if not any(abs(value - a) < 1e-6 for a in allowed):
            return False
    lowered = summary.lower()
    if any(p in lowered for p in ["is fraud", "confirmed fraud", "definitely fraud", "this is fraudulent"]):
        return False
    return True


def _call_llm(facts: list[dict]) -> str:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("no key")
    base = os.getenv("LLM_BASE_URL")
    if not base:
        base = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else "https://api.openai.com/v1"
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    system = (
        "You summarize fraud case facts for an analyst. Use only the facts. "
        "Do not decide or claim fraud. Cite fact ids like [f1] after each sentence. "
        "Use only numbers that appear in the facts. Three sentences maximum."
    )
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 180,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps([{"id": f["id"], "fact": f["text"]} for f in facts])},
        ],
    }
    r = httpx.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def summarize(facts: list[dict]) -> tuple[str, str]:
    _load_env()
    fallback = template_summary(facts)
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        return fallback, "template"
    from fraud import store

    if not store.consume_llm_credit(daily_cap()):
        return fallback, "template_capped"
    try:
        text = _call_llm(facts)
        if validate_summary(text, facts):
            return text, "llm"
        text = _call_llm(facts)
        if validate_summary(text, facts):
            return text, "llm"
    except Exception:
        pass
    return fallback, "template_fallback"


def refresh_case_summary(case_id: int, facts: list[dict]) -> None:
    from fraud import store

    text, source = summarize(facts)
    if source != "template":
        store.update_case_summary(case_id, text, source)
