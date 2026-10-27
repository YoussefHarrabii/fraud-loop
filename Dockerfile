FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.3 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM base AS app

COPY src ./src
COPY data/app.db ./data/app.db
COPY artifacts/final ./artifacts/final

ENV PYTHONPATH=/app/src \
    FRAUD_DB=/app/data/app.db \
    FRAUD_ARTIFACTS=/app/artifacts/final

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=4s --start-period=15s \
    CMD .venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["sh", "-c", ".venv/bin/python -m uvicorn fraud.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
