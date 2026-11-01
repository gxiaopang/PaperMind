"""Explicit, paid M2 acceptance against disposable PostgreSQL/Redis and private files."""

import argparse
import asyncio
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import UUID

import httpx
from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import Worker
from cryptography.fernet import Fernet
from sqlalchemy import select
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import Settings
from app.db.models import Chunk, PaperVersion
from app.main import create_app
from app.services.accounts.mail import send_outbox
from app.services.ingestion.lifecycle import active_chunks, snapshot
from app.workers.ingestion import cleanup, dispatch_outbox, ingest


async def accept(settings: Settings, pdf: Path) -> dict:
    app = create_app(settings)
    started = time.monotonic()
    async with app.router.lifespan_context(app):
        context = {
            "settings": settings,
            "engine": app.state.engine,
            "session_factory": app.state.session_factory,
            "redis": app.state.redis,
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=settings.origin,
            headers={"Origin": settings.origin},
        ) as client:

            async def write(path: str, body=None, method="POST", content=None):
                csrf = (await client.get("/api/v1/auth/csrf")).json()["csrf_token"]
                response = await client.request(
                    method,
                    "/api/v1" + path,
                    json=body,
                    content=content,
                    headers={"X-CSRF-Token": csrf},
                )
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"{path}: HTTP {response.status_code}: {response.text[:300]}"
                    )
                return response.json() if response.status_code != 204 else None

            password = secrets.token_urlsafe(24)
            await write("/auth/registration-code", {"email": "m2-live@example.org"})
            mailbox = []
            await send_outbox({**context, "mailbox": mailbox})
            await write(
                "/auth/register",
                {
                    "email": "m2-live@example.org",
                    "password": password,
                    "code": mailbox[0]["code"],
                },
            )
            await write(
                "/auth/login", {"email": "m2-live@example.org", "password": password}
            )
            owner = UUID((await client.get("/api/v1/auth/me")).json()["id"])
            collection = await write("/collections", {"name": "M2 acceptance"})
            paper = await write(
                f"/collections/{collection['id']}/papers?cloud_consent=true",
                content=await asyncio.to_thread(pdf.read_bytes),
            )
            duplicate = await write(
                f"/collections/{collection['id']}/papers?cloud_consent=true",
                content=await asyncio.to_thread(pdf.read_bytes),
            )
            assert duplicate["id"] == paper["id"]
            await dispatch_outbox(context)
            pool = await create_pool(
                RedisSettings.from_dsn(settings.redis_url.get_secret_value())
            )
            worker = Worker(
                [ingest, cleanup],
                redis_pool=pool,
                queue_name="papermind:ingest",
                ctx=context,
                burst=True,
                max_jobs=1,
                handle_signals=False,
                keep_result=0,
                job_timeout=3600,
            )
            try:
                await worker.async_run()
            finally:
                await worker.close()
            ready = (await client.get(f"/api/v1/papers/{paper['id']}")).json()
            assert ready["status"] == "ready", ready.get("error")
            async with app.state.session_factory() as db:
                chunks = await active_chunks(db, owner)
                citation = await snapshot(db, owner, chunks[0].id)
                citation_id = citation.id
                original_text = citation.source_text
                chunks_count = len(chunks)
                assert all(c.embedding is not None for c in chunks)
                await db.commit()
            usage = (await client.get("/api/v1/usage")).json()
            rebuilt = await write(f"/papers/{paper['id']}/rechunk")
            await ingest(context, str(owner), rebuilt["pending_version_id"])
            next_ready = (await client.get(f"/api/v1/papers/{paper['id']}")).json()
            assert (
                next_ready["status"] == "ready"
                and next_ready["active_version_id"] != ready["active_version_id"]
            )
            after_usage = (await client.get("/api/v1/usage")).json()
            assert after_usage["parse_pages"] == usage["parse_pages"]
            assert after_usage["model_tokens"] == usage["model_tokens"]
            old = (await client.get(f"/api/v1/citations/{citation_id}")).json()
            assert old["source_text"] == original_text
            ranged = await client.get(
                f"/api/v1/papers/{paper['id']}/pdf", headers={"Range": "bytes=0-4"}
            )
            assert ranged.status_code == 206 and ranged.content == b"%PDF-"
            async with app.state.session_factory() as db:
                versions = list(
                    await db.scalars(
                        select(PaperVersion).where(PaperVersion.user_id == owner)
                    )
                )
                assert (
                    len(versions) == 2 and len({v.parse_job_id for v in versions}) == 1
                )
            await write(f"/papers/{paper['id']}", method="DELETE")
            await cleanup(context, str(owner), paper["id"])
            gone = (await client.get(f"/api/v1/citations/{citation_id}")).json()
            assert gone["status"] == "来源已删除" and not gone["source_text"]
            assert (
                await client.get(f"/api/v1/papers/{paper['id']}/pdf")
            ).status_code == 404
            async with app.state.session_factory() as db:
                assert not list(await db.scalars(select(Chunk)))
            return {
                "status": "passed",
                "pdf_pages": paper["page_count"],
                "chunks": chunks_count,
                "embedding_model": settings.embedding_model,
                "dimension": settings.embedding_dim,
                "parse_pages": usage["parse_pages"],
                "embedding_tokens": usage["model_tokens"],
                "embedding_tokens_estimated": usage["model_tokens_estimated"],
                "stored_bytes": int(usage["storage_bytes"]),
                "duplicate_reused": True,
                "rechunk_reused_parse_and_embedding_cache": True,
                "snapshot_survived_rechunk": True,
                "range_passed": True,
                "deletion_revoked_sources_and_files": True,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--confirm-paid", action="store_true", required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/m2-local/live.json"
    )
    args = parser.parse_args()
    with (
        PostgresContainer(
            "paradedb/paradedb:0.26.3-pg16",
            username="m2_live",
            password="test-only",
            dbname="m2_live",
        ) as pg,
        RedisContainer("redis:7.4.6-alpine") as redis,
        tempfile.TemporaryDirectory(prefix="papermind-m2-") as directory,
    ):
        database = pg.get_connection_url().replace(
            "postgresql+psycopg2://", "postgresql+asyncpg://"
        )
        redis_url = (
            f"redis://{redis.get_container_host_ip()}:{redis.get_exposed_port(6379)}/0"
        )
        values = {
            "DATABASE_URL": database,
            "REDIS_URL": redis_url,
            "APP_ENV": "test",
            "TRACING_ENABLED": "false",
            "MAIL_BACKEND": "memory",
            "MAIL_ENCRYPTION_KEY": Fernet.generate_key().decode(),
            "PUBLIC_BASE_URL": "http://localhost:8080",
            "STORAGE_ROOT": directory,
        }
        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=ROOT / "backend",
            env={**os.environ, **values},
            check=True,
        )
        settings = Settings(**{key.lower(): value for key, value in values.items()})
        report = asyncio.run(accept(settings, args.pdf.resolve()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
