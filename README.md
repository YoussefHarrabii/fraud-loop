# Fraud review loop

An end-to-end fraud detection loop built around a model that is already good enough:
the work here is everything that turns a score into a reviewable case, and a case into
a retraining signal.

Synthetic card data from [Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation).
LightGBM scorer, Platt-calibrated, threshold picked by cost rather than by a round number.
Analyst queue at card level. LLM case summaries that are checked against the facts and
fall back to a deterministic template. Monitoring, a retrain trigger, and a promotion
gate. One page of UI. Tests, CI and a container.

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

The PR-AUC was 0.977 before the data audit. Two bugs inflated it: velocity windows
divided microseconds as if they were nanoseconds, so "last hour" meant 42 days, and the
Sparkov generator tied daypart and category to the date. Both are fixed and the data is
regenerated. 0.886 is the honest number.

## The loop

```
Sparkov CSV
   |
   v
features (past-only history, window counts, card age, merchant novelty)
   |
   v
LightGBM ---> Platt calibration ---> cost threshold ---> score
   |                                                       |
   |                                                       v
   |                                             flag / pass per transaction
   |                                                       |
   |                                                       v
   |                                             case per card, with facts
   |                                                       |
   |                                              template or LLM summary
   |                                                       |
   |                                                       v
   |                                             analyst: confirm / false alarm
   |                                                       |
   +------------------ retrain trigger <-------------------+
                            |
                            v
                   candidate ---> promotion gate ---> final
```

- **Scoring** is per transaction. **Cases** are per card, because an analyst reviews the
  card, not the row.
- **No blocking.** The decision is written after authorisation, as a review queue.
- **Facts before prose.** The summary is built from eight numbered facts (amount versus
  card median, local hour, velocity, merchant novelty, score against threshold). The model
  only rewrites them. Every number in the output must exist in the facts, every sentence
  must cite a fact id, and no verdict is allowed.
- **The gate.** A candidate model replaces the live one only if its recall does not drop
  and its cost is lower on a paired bootstrap.
- **Monitoring** uses PSI and KS against the training reference, plus the live flag rate
  and realised precision from closed cases. A drift injector pushes 60 synthetic
  transactions so the panels move on demand.

## What is real and what is not

Real: the pipeline, the model, the calibration, the threshold, the numbers above, the
store, the case workflow, the validator, the gate, the tests.

Simulated: the transactions. Sparkov is a generator, not production data. There is no
real chargeback feed, so "confirmed fraud" is an analyst click on a synthetic label.

Not built: authentication and per-user keys, a payment or chargeback integration,
multi-worker deployment.

## Layout

```
src/fraud/       app, serving, features, store, case_facts, monitoring, policy, static UI
scripts/         data build, reference profile, retrain trigger, app DB, data publish
configs/         final and candidate model config, data manifest
tests/           unit, API, pipeline, LLM, concurrency, security, Playwright
```

## Run

```
uv sync --frozen --all-groups
uv run uvicorn fraud.app:app --host 127.0.0.1 --port 8000
```

The app needs `data/app.db` and `artifacts/final`, both gitignored and both built by the
scripts in `scripts/`. Without them `GET /health` fails at startup.

LLM summaries are optional. Put a key in `secrets/llm.env` (ignored) or export it:

```
OPENAI_API_KEY=...
LLM_MODEL=gpt-4o-mini
LLM_DAILY_CAP=40
```

No key means the template summary, which is the same text minus the rewriting. A failed
or invalid call also falls back to the template.

## Tests

```
uv run ruff check src tests scripts
uv run pytest --cov
RUN_UI=1 uv run pytest tests/test_ui_e2e.py
```

73 tests, 98 percent statement coverage on the core modules, gate at 80. CI runs the suite
on Python 3.11 and 3.12, builds the dependency image, runs the browser test, and scans for
secrets.

## Deploy

The image needs the model bundle and the 250 MB demo database, which are too large for git.
They ship as a release asset; the build downloads and checksum-verifies it.

```
# once, from a machine that has data/app.db and artifacts/final
$env:GITHUB_TOKEN = "..."        # contents:write
uv run python scripts/publish_data.py --repo <owner>/<repo>
```

That writes `configs/data.json` with the asset URL and its SHA-256. Commit it. Then deploy
`Dockerfile.deploy`, which fetches the bundle, verifies the hash, and unpacks it into the
image. Render reads `render.yaml`; any Docker host works the same way.

Secret handling: the key is set in the host dashboard (`sync: false` in `render.yaml`),
never in git and never in a layer. `LLM_DAILY_CAP` bounds the spend if the URL is public,
on top of a hard monthly cap on the provider key itself.
