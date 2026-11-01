COMPOSE = docker compose --env-file .env --env-file .env.local -f deploy/docker-compose.yml

.PHONY: bootstrap infra up down logs migrate api worker web check test test-parsing test-ingestion test-retrieval eval-retrieval test-chat smoke-chat e2e smoke
bootstrap:
	python3 scripts/bootstrap.py
	cd backend && uv sync --frozen
	cd frontend && npm ci
infra:
	$(COMPOSE) up -d --wait postgres redis
up:
	$(COMPOSE) up -d --build --wait
down:
	$(COMPOSE) down
logs:
	$(COMPOSE) logs --tail=100 -f api worker web
migrate:
	cd backend && uv run alembic upgrade head
	cd backend && uv run python -m app.cli checkpoints
api:
	cd backend && uv run uvicorn app.main:app --reload --port 18000 --no-access-log --no-proxy-headers
worker:
	cd backend && uv run python -m app.workers.run
web:
	cd frontend && npm run dev
check:
	cd backend && uv run ruff check . && uv run ruff format --check . && uv run mypy app
	cd frontend && npm run lint && npm run build
test:
	cd backend && uv run pytest
test-parsing:
	cd backend && uv run pytest tests/test_mineru_*.py tests/test_parsing_*.py
test-ingestion:
	cd backend && uv run pytest tests/test_chunking.py tests/test_embedding_batch.py tests/test_ingestion.py
test-retrieval:
	cd backend && uv run pytest tests/test_retrieval.py
test-chat:
	cd backend && uv run pytest tests/test_chat.py
smoke-chat:
	cd backend && uv run python ../scripts/pre_m6_live.py --confirm-paid --output ../reports/m3-local/chat-$$(date +%Y%m%d-%H%M%S).json
eval-retrieval:
	cd backend && uv run python ../scripts/m3_eval.py --confirm-paid --output ../reports/m3-local/retrieval-$$(date +%Y%m%d-%H%M%S).json
e2e:
	cd backend && uv run python ../scripts/e2e.py
smoke:
	cd backend && uv run python -m app.services.providers.smoke --output ../reports/providers-local.json
