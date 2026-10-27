# Fraud review loop

An end-to-end fraud detection loop built around a model that is already good enough. The
interesting part is everything that turns a score into a reviewable case, and a case into a
retraining signal.

Synthetic card data from [Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation).
LightGBM scorer, Platt-calibrated, threshold picked by cost rather than by a round number.
Analyst queue at card level. LLM case summaries that are checked against the facts and fall
back to a deterministic template. Monitoring, a retrain trigger, a promotion gate, one page
of UI, tests, CI and a container.

## Numbers

1,791,759 transactions, 997 cards, 0.534 percent fraud, January 2019 to December 2020.
Chronological split, no shuffling.

| | |
|---|---|
| Operating point | 90 percent recall on validation, threshold 0.071 |
| Holdout recall | 94.28 percent |
| Holdout precision | 57.67 percent |
| Flags per day | 20 |
| False alarms per day | 8.47 |
| Share of fraud dollars caught | 99.36 percent |
| Holdout PR-AUC | 0.886 |

The figures come after a data audit that removed two leaks: the velocity windows were off by
three orders of magnitude, and the generator tied daypart to the date. The dataset is
generated locally and not committed, so these numbers are from that run.

## The loop

```mermaid
flowchart TD
    A[Sparkov transactions] --> B[Features<br/>past-only history, velocity, card age, merchant novelty]
    B --> C[LightGBM]
    C --> D[Platt calibration]
    D --> E[Cost threshold<br/>90 percent recall on validation]
    E --> F{Flag?}
    F -- pass --> G[Logged, no case]
    F -- flag --> H[Case per card<br/>eight facts computed in code]
    H --> I[Template summary]
    H -. valid invitation key .-> J[Model-written summary]
    J --> K{Every number and citation checks out?}
    K -- no --> I
    K -- yes --> L[Case shows the model summary]
    I --> M[Analyst: confirm or false alarm]
    L --> M
    M --> N[Labels]
    N --> O{Retrain gate<br/>recall and cost}
    O -- pass --> P[Candidate goes live]
    O -- fail --> Q[Held]
    P --> C
```

- **Scoring** is per transaction. **Cases** are per card, because an analyst reviews the
  card, not the row.
- **No blocking.** The decision is written after authorisation, as a review queue.
- **Facts before prose.** The summary is built from eight numbered facts (amount against the
  card median, local hour, velocity, merchant novelty, score against threshold). The model
  only rewrites them. Every number in the output must exist in the facts, every sentence must
  cite a fact id, and no verdict is allowed. Two failures and the template is used.
- **The gate.** A candidate replaces the live model only if its recall does not drop and its
  cost is lower on a paired bootstrap.
- **Monitoring** uses PSI and KS against the training reference, plus the live flag rate and
  realised precision from closed cases. A drift injector pushes 60 transactions so the panels
  move on demand.

## What is real and what is not

Real: the pipeline, the model, the calibration, the threshold, the numbers above, the store,
the case workflow, the validator, the gate, the tests.

Simulated: the transactions. Sparkov is a generator, not production data. There is no real
chargeback feed, so "confirmed fraud" is an analyst click on a synthetic label.

Not built: a chargeback or payment integration, user accounts beyond the invitation keys,
multi-worker deployment.

## Layout

```
src/fraud/       app, serving, features, store, access, limits, retention,
                 case_facts, monitoring, policy, static UI
scripts/         data build, reference profile, retrain trigger, app DB, key mint,
                 demo reset, bundle publish
configs/         model configs, access-key hashes, data manifest
tests/           unit, API, pipeline, LLM, retention, security, concurrency, Playwright
```

## Run

```
uv sync --frozen --all-groups
uv run uvicorn fraud.app:app --host 127.0.0.1 --port 8000
```

The app needs `data/app.db` and `artifacts/final`. Both are gitignored and both are built by
the scripts in `scripts/`.

LLM summaries are optional and off without a key:

```
OPENAI_API_KEY=...        # or OPENROUTER_API_KEY, or any compatible key
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=...          # only for a non-default endpoint
LLM_DAILY_CAP=40          # calls per UTC day, process-wide
```

Set `LLM_KEY_FILE` instead to read the key from a mounted file, which keeps it out of the
process environment. No key means the template summary, which is the same text minus the
rewriting.

## Access keys

The desk is open. The model-written summary is not.

```
uv run python scripts/mint_keys.py --count 100 --uses 5 --days 90
```

That writes `configs/access_keys.json` (SHA-256 hashes, committed) and `secrets/keys.csv`
(plaintext, ignored, the only copy). One key per application.

- `POST /cases/{id}/report` with `X-Access-Key` spends one use and queues the summary.
- `GET /access` reports remaining uses without spending one.
- 401 unknown, 403 expired, 429 exhausted.
- A use is refunded when the provider does not deliver, so a failed call costs nothing.
- The key travels in a header, never a URL, and the browser keeps it in `sessionStorage` for
  the tab.

## Demo data

Cases and requests are what visitors leave behind, so they expire after
`FRAUD_RETENTION_HOURS` (default 24). Key usage never expires: that would hand a spent
invitation its uses back.

```
uv run python scripts/reset_demo.py            # cases, requests, key and llm usage
uv run python scripts/reset_demo.py --keep-keys
```

## Hardening

- No secret in the image, the repo, the URL space or any response.
- Runs as uid 10001; only the database directory is writable.
- Per-client rate limits on the write endpoints and on `/access`, 429 with `Retry-After`.
- `/docs`, `/redoc` and `/openapi.json` are off unless `FRAUD_DOCS=1`.
- CSP, `nosniff`, `DENY` framing, `no-referrer`; HSTS over https only.
- `pip-audit` and a secret scan run in CI next to the tests.

## Tests

```
uv run ruff check src tests scripts
uv run pytest --cov
RUN_UI=1 uv run pytest tests/test_ui_e2e.py
```

127 tests, 97 percent statement coverage on the core modules, gate at 80. CI runs the suite on
Python 3.11 and 3.12, audits dependencies, builds the dependency image, runs the browser test
and scans for secrets.

## Container

```
docker build -t fraud-loop .
docker run --rm -p 8000:8000 --env-file secrets/llm.env fraud-loop
```

The demo database and the model bundle are 119 MB together, too large for git, so they ship as
a release asset. `Dockerfile.deploy` downloads it at build time and verifies its SHA-256, so a
mismatch fails the build instead of shipping the wrong data. The image is 1.38 GB and runs as
a non-root user.
