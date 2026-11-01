FROM python:3.12.12-slim-bookworm AS build
RUN pip install --no-cache-dir uv==0.12.7
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
COPY backend/app ./app
RUN uv sync --frozen --no-dev --no-editable --compile-bytecode

FROM python:3.12.12-slim-bookworm
RUN groupadd -g 10001 app && useradd -u 10001 -g app -m app && mkdir -p /app/data && chown app:app /app/data
WORKDIR /app/backend
ENV PATH="/app/backend/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY --from=build /app/backend/.venv ./.venv
COPY backend/app ./app
COPY backend/alembic ./alembic
COPY backend/alembic.ini ./
COPY reports/m3-retrieval-v7-final.json reports/pre-m6-v7-live-summary.json /app/reports/
USER app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--no-proxy-headers"]
