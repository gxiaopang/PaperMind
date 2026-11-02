"""Run browser tests against isolated real services, never the user's database/mailbox."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from cryptography.fernet import Fernet
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer


async def seed(database_url: str, pdf_path: str, workbench_pdf_path: str) -> None:
    from pypdf import PdfWriter
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.core.security import hasher
    from app.db.models import User, utcnow

    engine = create_async_engine(database_url)
    try:
        async with async_sessionmaker(engine)() as db:
            db.add(
                User(
                    email="m2-browser@example.org",
                    password_hash=hasher.hash("Browser test password 123!"),
                    email_verified_at=utcnow(),
                )
            )
            db.add(
                User(
                    email="m5-admin@example.org",
                    password_hash=hasher.hash("Browser test password 123!"),
                    role="admin",
                    email_verified_at=utcnow(),
                )
            )
            await db.commit()
    finally:
        await engine.dispose()

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(pdf_path)

    writer2 = PdfWriter()
    writer2.add_blank_page(width=612, height=792)
    writer2.add_blank_page(width=612, height=792)
    writer2.write(workbench_pdf_path)


class FakeEmbedder:
    last_usage = 24

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return [0.1] * 1024

    async def aclose(self) -> None:
        return None


class FakeReranker:
    last_usage = 16

    async def acompress_documents(self, documents: list[Any], query: str) -> list[Any]:
        ordered = sorted(
            documents[:8],
            key=lambda d: (
                0 if any(loc.get("bbox") for loc in d.metadata.get("locations", [])) else 1,
                d.metadata.get("ordinal", 0),
            ),
        )
        result = []
        for idx, doc in enumerate(ordered):
            copied = doc.model_copy(deep=True)
            copied.metadata["rerank_score"] = round(0.98 - idx * 0.05, 4)
            result.append(copied)
        return result


class FakeChatModels:
    async def invoke(
        self,
        run: Any,
        node: str,
        payload: dict[str, Any],
        config: Any,
        *,
        schema: Any = None,
        on_token: Any = None,
    ) -> Any:
        if node == "route":
            return schema.model_validate({"route": "paper_qa"})
        if node in {"rewrite_query", "transform_query"}:
            return schema.model_validate({"query": "scaled dot-product attention BLEU"})
        if node == "plan_paper_tools":
            return schema.model_validate(
                {
                    "calls": [
                        {
                            "name": "search_paper_chunks",
                            "args": {"queries": ["scaled dot-product attention BLEU"]},
                        }
                    ]
                }
            )
        if node == "grade_documents":
            return schema.model_validate({"sufficient": True})
        if node == "verify_grounding":
            return schema.model_validate({"grounded": True})
        if node == "summarize":
            return schema.model_validate({"summary": "研究缩放点积注意力与实验指标对比。", "research_state": {}})
        if node == "direct_answer":
            return "你好，我是 PaperMind 论文助手。"
        # generate node
        if "重生成" in run.question and not payload.get("previous_validation"):
            draft = "这是一版未带引用的草稿文本，将被引用校验器拦截并触发第二次生成。"
            if on_token:
                await on_token(draft)
            return draft
        answer = "缩放点积注意力通过点积除以根号维度计算权重[1]。实验表格显示模型在翻译任务上取得 28.4 BLEU 的表现[2]。"
        if on_token:
            await on_token(answer[:18])
            await asyncio.sleep(0.05)
            await on_token(answer[18:])
        return answer


async def fake_parser(pdf_path: Path, output: Path, settings: Any, **kwargs: Any) -> None:
    from app.services.parsing.mineru_result import (
        Location,
        ParsedBlock,
        ParsedDocument,
        Section,
    )

    await kwargs["checkpoint"]({"stage": "polling", "batch_id": "e2e-batch"})
    doc = ParsedDocument(
        page_count=2,
        blocks=[
            ParsedBlock(
                id="b0",
                kind="text",
                text="Scaled dot-product attention computes the dot products of the query with all keys, divides each by sqrt(d_k), and applies a softmax function.",
                section_id="s1",
                section_path=["1 Introduction", "1.1 Scaled Dot-Product Attention"],
                locations=[Location(page_idx=0, bbox=(0.12, 0.18, 0.88, 0.42))],
            ),
            ParsedBlock(
                id="b1",
                kind="table",
                text="<table><tr><th>Model</th><th>BLEU</th></tr><tr><td>Transformer (big)</td><td>28.4</td></tr></table>",
                section_id="s2",
                section_path=["2 Experiments"],
                locations=[Location(page_idx=1, bbox=None)],
            ),
        ],
        sections=[
            Section(
                id="s1",
                parent_id=None,
                title="1 Introduction",
                level=1,
                path=["1 Introduction"],
            ),
            Section(
                id="s2",
                parent_id=None,
                title="2 Experiments",
                level=1,
                path=["2 Experiments"],
            ),
        ],
        markdown_path="full.md",
        content_list_path="content_list.json",
        layout_path=None,
        assets=["images/fig1.png"],
        warnings=["block-1: bbox 不可用，保留页级定位"],
    )

    def write_files() -> None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "document.json").write_text(doc.model_dump_json())
        (output / "metadata.json").write_text(
            json.dumps(
                {
                    "title": "Attention Is All You Need (M5 Benchmark)",
                    "authors": ["Ashish Vaswani", "Noam Shazeer"],
                    "year": 2017,
                    "venue": "NeurIPS",
                    "abstract": "",
                }
            )
        )
        (output / "job.json").write_text(json.dumps({"stage": "complete", "batch_id": "e2e-batch"}))
        (output / "raw/images").mkdir(parents=True, exist_ok=True)
        (output / "raw/full.md").write_text(
            "# Attention Is All You Need (M5 Benchmark)\n\n"
            "缩放点积注意力公式如下：\n\n"
            "$$\\mathrm{Attention}(Q, K, V) = \\mathrm{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\n\n"
            "| Model | EN-DE BLEU |\n"
            "| --- | --- |\n"
            "| Transformer (big) | 28.4 |\n\n"
            "![架构图](images/fig1.png)\n\n"
            "<script>window.__xss_executed = true;</script>\n\n"
            "[危险链接](javascript:alert(1))\n"
        )
        # 1x1 transparent PNG
        (output / "raw/images/fig1.png").write_bytes(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
            )
        )

    await asyncio.to_thread(write_files)
    await kwargs["checkpoint"]({"stage": "complete", "batch_id": "e2e-batch"})


async def run_e2e_worker(mailbox_file: Path) -> None:
    from arq import create_pool
    from arq.connections import RedisSettings
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.core.config import get_settings
    from app.db.models import Paper, TaskOutbox
    from app.graph.builder import build_graph
    from app.graph.nodes import GraphServices
    from app.services.accounts.mail import send_outbox
    from app.services.retrieval import create_retriever
    from app.workers.chat import execute as execute_chat
    from app.workers.ingestion import cleanup, ingest

    settings = get_settings()
    engine = create_async_engine(
        settings.database_url.get_secret_value(), pool_size=4, max_overflow=4
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url.get_secret_value()))
    embedder = FakeEmbedder()
    reranker = FakeReranker()
    models = FakeChatModels()

    def retriever_factory(
        config: Any, sessions: Any, eng: Any, scope: Any, *, run_id: Any = None
    ) -> Any:
        r = create_retriever(config, sessions, eng, scope, run_id=run_id)
        r.embedder_factory = lambda: embedder
        r.reranker_factory = lambda: reranker
        return r

    pool = AsyncConnectionPool[AsyncConnection[dict[str, Any]]](
        settings.database_url.get_secret_value().replace("postgresql+asyncpg://", "postgresql://"),
        min_size=1,
        max_size=2,
        open=False,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "options": "-c search_path=checkpoints,public",
        },
    )
    await pool.open(wait=True)
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    services = GraphServices(settings, session_factory, engine, redis, models, retriever_factory)
    graphs = {mode: build_graph(services, mode, saver) for mode in ("fast", "agentic")}

    ctx: dict[str, Any] = {
        "settings": settings,
        "engine": engine,
        "session_factory": session_factory,
        "redis": redis,
        "embedder": embedder,
        "parse_pdf": fake_parser,
        "services": services,
        "graphs": graphs,
        "mailbox": [],
    }

    try:
        while True:
            sent = await send_outbox(ctx)
            if sent:
                await asyncio.to_thread(
                    mailbox_file.write_text,
                    json.dumps(ctx["mailbox"], ensure_ascii=False),
                )

            async with session_factory() as db:
                tasks = list(
                    await db.scalars(
                        select(TaskOutbox)
                        .where(TaskOutbox.completed_at.is_(None))
                        .order_by(TaskOutbox.available_at)
                        .limit(20)
                    )
                )
            for t in tasks:
                if t.kind == "ingest":
                    async with session_factory() as db:
                        paper = await db.scalar(
                            select(Paper).where(Paper.pending_version_id == t.resource_id)
                        )
                        page_count = paper.page_count if paper else 1
                    # Keep 1-page PDF in pending state for M2 papers.spec.ts assertion
                    if page_count >= 2:
                        await ingest(ctx, str(t.user_id), str(t.resource_id))
                elif t.kind == "cleanup":
                    await cleanup(ctx, str(t.user_id), str(t.resource_id))
                elif t.kind == "chat":
                    await execute_chat(ctx, str(t.user_id), str(t.resource_id))
            await asyncio.sleep(0.15)
    finally:
        await pool.close()
        await redis.aclose()
        await engine.dispose()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        asyncio.run(run_e2e_worker(Path(os.environ["PAPERMIND_E2E_MAILBOX"])))
        return

    root = Path(__file__).resolve().parents[1]
    with (
        PostgresContainer(
            "paradedb/paradedb:0.26.3-pg16",
            username="e2e",
            password="test-only",
            dbname="e2e",
        ) as pg,
        RedisContainer("redis:7.4.6-alpine") as redis,
        tempfile.TemporaryDirectory(prefix="papermind-browser-") as storage,
    ):
        env = {
            **os.environ,
            "DATABASE_URL": pg.get_connection_url().replace(
                "postgresql+psycopg2://", "postgresql+asyncpg://"
            ),
            "REDIS_URL": f"redis://{redis.get_container_host_ip()}:{redis.get_exposed_port(6379)}/0",
            "APP_ENV": "test",
            "TRACING_ENABLED": "false",
            "MAIL_BACKEND": "memory",
            "MAIL_ENCRYPTION_KEY": Fernet.generate_key().decode(),
            "PUBLIC_BASE_URL": "http://127.0.0.1:8081",
            "AUTH_IP_LIMIT": "1000",
            "CHAT_INITIAL_TOKEN_RESERVE": "128",
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
        }
        env["STORAGE_ROOT"] = storage
        env["PAPERMIND_E2E_PDF"] = str(Path(storage) / "browser.pdf")
        env["PAPERMIND_E2E_WORKBENCH_PDF"] = str(Path(storage) / "workbench.pdf")
        env["PAPERMIND_E2E_MAILBOX"] = str(Path(storage) / "mailbox.json")
        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=root / "backend",
            env=env,
            check=True,
        )
        asyncio.run(
            seed(
                env["DATABASE_URL"],
                env["PAPERMIND_E2E_PDF"],
                env["PAPERMIND_E2E_WORKBENCH_PDF"],
            )
        )
        log = Path("/tmp/papermind-e2e-api.log").open("w")
        worker_log = Path("/tmp/papermind-e2e-worker.log").open("w")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--port",
                "18001",
                "--no-access-log",
                "--no-proxy-headers",
            ],
            cwd=root / "backend",
            env=env,
            stdout=log,
            stderr=log,
        )
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            cwd=root / "backend",
            env={**env, "PYTHONPATH": str(root / "backend")},
            stdout=worker_log,
            stderr=worker_log,
        )
        try:
            with httpx.Client(trust_env=False) as client:
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError("E2E API exited; inspect /tmp/papermind-e2e-api.log")
                    try:
                        if client.get("http://127.0.0.1:18001/readyz").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    time.sleep(0.2)
                else:
                    raise RuntimeError("E2E API failed to become ready")
            result = subprocess.run(
                ["npm", "run", "test:e2e"],
                cwd=root / "frontend",
                env={**env, "PLAYWRIGHT_BROWSERS_PATH": "/tmp/papermind-playwright"},
            )
            if result.returncode:
                raise SystemExit(result.returncode)
        finally:
            for proc in (worker, process):
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            worker_log.close()
            log.close()


if __name__ == "__main__":
    main()
