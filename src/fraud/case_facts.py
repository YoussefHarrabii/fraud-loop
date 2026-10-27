"""Case facts, LLM summary, and the numeric/citation check. No raw text goes to the model."""

import hashlib
import json
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?%?")
DEFAULT_DAILY_CAP = 40

OPENAI_BASE = "https://api.openai.com/v1"
OPENCODE_BASE = "https://opencode.ai/inference/go/openai/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _load_env() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "secrets" / "llm.env")


def _key_from_file() -> str:
    """Read the key from `LLM_KEY_FILE`. Returns "" when unset or unreadable."""
    path = os.getenv("LLM_KEY_FILE", "").strip()
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def llm_config() -> dict | None:
    """Provider settings from the environment. None when no key is set.

    A key file wins over the environment, so `docker run -v key:/run/key
    -e LLM_KEY_FILE=/run/key` never puts the value in container metadata. With a key file
    set but unreadable there is no fallback: a typo must not quietly switch provider.
    Order otherwise: OpenAI, OpenCode, OpenRouter. `LLM_BASE_URL` wins over the defaults.
    """
    _load_env()
    file_key = _key_from_file()
    if os.getenv("LLM_KEY_FILE") and not file_key:
        return None
    openai_key = os.getenv("OPENAI_API_KEY")
    opencode_key = os.getenv("OPENCODE_KEY") or os.getenv("OPENCODE_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    key = file_key or openai_key or opencode_key or openrouter_key
    if not key:
        return None
    if os.getenv("LLM_BASE_URL"):
        base = os.environ["LLM_BASE_URL"]
    elif openai_key:
        base = OPENAI_BASE
    elif opencode_key:
        base = OPENCODE_BASE
    elif file_key:
        base = OPENAI_BASE
    else:
        base = OPENROUTER_BASE
    model = os.getenv("LLM_MODEL") or os.getenv("OPENCODE_MODEL") or "gpt-4o-mini"
    return {"key": key, "base": base.rstrip("/"), "model": model}


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


def _session_id(facts: list[dict]) -> str:
    """Stable per-case id. OpenCode Go requires one for routing and prompt caching."""
    text = "|".join(f["text"] for f in facts)
    return "fraud-" + hashlib.sha256(text.encode()).hexdigest()[:16]


def _headers(config: dict, facts: list[dict]) -> dict:
    headers = {
        "Authorization": f"Bearer {config['key']}",
        "User-Agent": "fraud-review-loop/0.1",
    }
    if "opencode.ai" in config["base"]:
        headers["x-opencode-session"] = _session_id(facts)
    return headers


def _call_llm(facts: list[dict]) -> str:
    config = llm_config()
    if config is None:
        raise RuntimeError("no key")
    system = (
        "You summarize fraud case facts for an analyst. Use only the facts. "
        "Do not decide or claim fraud. Cite fact ids like [f1] after each sentence. "
        "Use only numbers that appear in the facts. Three sentences maximum."
    )
    payload = {
        "model": config["model"],
        "temperature": 0,
        # Reasoning models spend part of this budget before writing anything, and the
        # answer comes back empty at 180. You pay for tokens used, not for the cap.
        "max_tokens": 2000,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps([{"id": f["id"], "fact": f["text"]} for f in facts])},
        ],
    }
    r = httpx.post(
        f"{config['base']}/chat/completions",
        headers=_headers(config, facts),
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def summarize(facts: list[dict]) -> tuple[str, str]:
    fallback = template_summary(facts)
    if llm_config() is None:
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
