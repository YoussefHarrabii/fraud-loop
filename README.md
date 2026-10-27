# Fraud / Loop

An end-to-end fraud detection loop built around a model that is already good enough. The
interesting part is what turns a score into a reviewable case, and a case into a retraining
signal.

Synthetic card data from [Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation).
LightGBM, Platt-calibrated, threshold picked by cost rather than by a round number. Card-level
analyst queue. LLM case summaries checked against the facts, with a deterministic fallback.
Monitoring, a retrain trigger and a promotion gate behind one page of UI.

## Numbers

1,791,759 transactions, 997 cards, 0.534 percent fraud, January 2019 to December 2020,
chronological split.

| | |
|---|---|
| Operating point | 90 percent recall on validation, threshold 0.071 |
| Holdout recall | 94.28 percent |
| Holdout precision | 57.67 percent |
| Flags per day | 20 |
| False alarms per day | 8.47 |
| Share of fraud dollars caught | 99.36 percent |
| Holdout PR-AUC | 0.886 |

The figures follow a data audit that removed two leaks: velocity windows off by three orders of
magnitude, and the generator tying daypart to the date. The dataset is generated locally, so
these are from that run.

## The loop

```mermaid
flowchart TD
    A[Sparkov transactions] --> B[Features<br/>past-only history, velocity, card age, merchant novelty]
    B --> C[LightGBM]
    C --> D[Platt calibration]
    D --> E[Cost threshold<br/>90 percent recall on validation]
    E --> F{Flag?}
    F -- pass --> G[Logged, no case]
    F -- flag --> H[Case per card<br/>numbered facts computed in code]
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

Scoring is per transaction, cases are per card. The summary is built from a numbered list of
facts and the model only rewrites them: every number and citation is checked, and two failures
fall back to the template. A triage hint next to the summary (check merchant, contact cardholder,
likely benign) is computed in code, never by the model. A candidate replaces the live model only
if its recall does not drop and its cost is lower on a paired bootstrap.

## What is real

Real: the pipeline, the model, the calibration, the threshold, the numbers above, the store,
the case workflow, the validator, the gate, the tests.

Simulated: the transactions. Sparkov is a generator, not production data, and there is no
chargeback feed, so "confirmed fraud" is an analyst click on a synthetic label.

Not built: a chargeback integration, user accounts beyond the invitation keys, multi-worker
deployment.

## Run

Postgres holds the transaction history and the app tables. There is no local database file.

```
docker compose up -d db
uv sync --frozen --all-groups
uv run python scripts/seed_pg.py --reset          # 1.79M rows, once
uv run uvicorn fraud.app:app --port 8000
```

`DATABASE_URL` defaults to `postgresql://postgres:postgres@127.0.0.1:5432/fraud`, which is
what compose publishes. `artifacts/final` (7.6 MB) is in the repo, so nothing has to be
downloaded or generated before a first run.

The LLM report is optional:

```
OPENCODE_KEY=oc_...       # or OPENAI_API_KEY, or any compatible key with LLM_BASE_URL
OPENCODE_MODEL=deepseek-v4.1-flash
LLM_DAILY_CAP=40
```

Without a key, or when a call fails, the case keeps the template summary.

## Access keys

The desk is open. The model-written report is not: one key per recruiter, five reports each,
ninety days, so a forwarded link does not multiply the spend.

`POST /cases/{id}/report` with `X-Access-Key` spends a use; `GET /access` reports the rest
without spending one. A use is refunded when the provider does not deliver. Keys are stored as
SHA-256 hashes, so the plaintext exists only in the mint output.

```
uv run python scripts/mint_keys.py --count 100 --uses 5 --days 90
```

## Tests

```
docker compose up -d db      # the suite runs against Postgres, database `fraud_test`
uv run ruff check src tests scripts
uv run pytest --cov
RUN_UI=1 uv run pytest tests/test_ui_e2e.py
```

128 tests, 97 percent statement coverage on the core modules, gate at 80. CI runs 3.11 and 3.12
against a Postgres service container, audits dependencies, builds the dependency image, runs the
browser test and scans for secrets. The suite creates `fraud_test` itself and never touches
`fraud`.

## Container

```
docker compose up --build
```

The image is 1.04 GB, runs as a non-root user, needs no writable filesystem, takes the model
from the repo and the database from `DATABASE_URL`. No secret is in a layer.

## Deploy

Heroku, container stack, Basic dyno plus Postgres Essential:

```
heroku stack:set container -a <app>
heroku config:set OPENCODE_KEY=... OPENCODE_MODEL=deepseek-v4.1-flash LLM_DAILY_CAP=40 -a <app>
heroku container:push web -a <app>
heroku container:release web -a <app>

# seed from your machine: the image has no scripts/ and no data/, by design
DATABASE_URL=$(heroku config:get DATABASE_URL -a <app>) uv run python scripts/seed_pg.py --reset
DATABASE_URL=$(heroku config:get DATABASE_URL -a <app>) uv run python scripts/preflight.py \
    --url https://<app>.herokuapp.com
```

`heroku.yml` builds the same `Dockerfile`. There is no bundle to publish and no release asset to
keep alive: the model is in the repo, the transactions go into Postgres once.

## Simulate a deploy locally

The same image, the same lifecycle, without Heroku:

```
docker compose up -d db
docker build -t fraud-loop .
docker run -d --rm --name dyno --read-only --tmpfs /tmp:rw,size=64m \
    --network frauddetection_default -p 8014:9999 \
    -e PORT=9999 -e DATABASE_URL=postgresql://postgres:postgres@db:5432/fraud fraud-loop
uv run python scripts/preflight.py --url http://127.0.0.1:8014
docker restart dyno      # a dyno restart; cases and key counters must survive
```

`--read-only` is the point: it proves the app keeps nothing on the local filesystem, so an
ephemeral dyno cannot lose state. A one-off command is `docker run --rm ... fraud-loop
.venv/bin/python -c "..."`, which is what `heroku run` would be.
