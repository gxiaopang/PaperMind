import os
import subprocess
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from httpx import ASGITransport, AsyncClient
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

from app.core.config import Settings


@pytest.fixture(scope="session")
def infrastructure():
    image = "paradedb/paradedb:0.26.3-pg16"
    with (
        PostgresContainer(
            image, username="papermind_test", password="test-only-password", dbname="papermind_test"
        ) as pg,
        RedisContainer("redis:7.4.6-alpine") as redis,
    ):
        database_url = pg.get_connection_url().replace(
            "postgresql+psycopg2://", "postgresql+asyncpg://"
        )
        values = {
            "DATABASE_URL": database_url,
            "REDIS_URL": f"redis://{redis.get_container_host_ip()}:{redis.get_exposed_port(6379)}/0",
            "APP_ENV": "test",
            "TRACING_ENABLED": "false",
            "MAIL_BACKEND": "memory",
            "MAIL_ENCRYPTION_KEY": Fernet.generate_key().decode(),
            "PUBLIC_BASE_URL": "http://localhost:8080",
        }
        env = {**os.environ, **values}
        subprocess.run(
            [str(Path(__file__).resolve().parents[1] / ".venv/bin/alembic"), "upgrade", "head"],
            env=env,
            check=True,
        )
        yield values


@pytest.fixture
async def app(infrastructure, tmp_path):
    from app.main import create_app

    settings = Settings(
        **{key.lower(): value for key, value in infrastructure.items()},
        storage_root=tmp_path,
        auth_ip_limit=1000,
        auth_email_limit=100,
    )
    instance = create_app(settings)
    async with instance.router.lifespan_context(instance):
        from sqlalchemy import text

        async with instance.state.engine.begin() as conn:
            await conn.execute(
                text(
                    "TRUNCATE email_codes, mail_outbox, account_tokens, sessions, collections, users CASCADE"
                )
            )
        await instance.state.redis.flushdb()
        yield instance


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost:8080",
        headers={"Origin": "http://localhost:8080"},
    ) as client:
        yield client


async def post(client, path, body=None, method="POST"):
    csrf = await client.get("/api/v1/auth/csrf")
    return await client.request(
        method, "/api/v1" + path, json=body, headers={"X-CSRF-Token": csrf.json()["csrf_token"]}
    )


async def register(client, email="a@example.org", password="Strong password 123!"):
    sent = await post(client, "/auth/registration-code", {"email": email})
    assert sent.status_code == 202, sent.text
    mailbox = await delivered(client._transport.app)
    code = next(item["code"] for item in mailbox if item["email"] == email.casefold())
    return await post(
        client, "/auth/register", {"email": email, "password": password, "code": code}
    )


async def login(client, email="a@example.org", password="Strong password 123!"):
    return await post(client, "/auth/login", {"email": email, "password": password})


async def delivered(app):
    from app.services.accounts.mail import send_outbox

    context = {
        "settings": app.state.settings,
        "session_factory": app.state.session_factory,
        "mailbox": [],
    }
    await send_outbox(context)
    return context["mailbox"]
