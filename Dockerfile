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
# The manifest is mounted for the install and never copied into a layer.
RUN --mount=type=bind,source=pyproject.toml,target=/build/pyproject.toml \
    --mount=type=bind,source=uv.lock,target=/build/uv.lock \
    sh -c 'cp /build/pyproject.toml /build/uv.lock /app/ && uv sync --frozen --no-dev && rm -f /app/pyproject.toml /app/uv.lock'

FROM base AS app

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts \
    && chown 10001:10001 /app /app/artifacts

COPY --chown=10001:10001 src ./src
COPY --chown=10001:10001 configs/key_registry.json ./configs/key_registry.json
COPY --chown=10001:10001 artifacts/final ./artifacts/final

USER appuser

# The transaction history lives in Postgres. Heroku injects DATABASE_URL; compose sets it.
ENV PYTHONPATH=/app/src \
    FRAUD_ARTIFACTS=/app/artifacts/final

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=4s --start-period=15s \
    CMD .venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["sh", "-c", ".venv/bin/python -m uvicorn fraud.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
