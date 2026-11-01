import asyncio
import fcntl
import json
import math
import os
import shutil
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

from arq import cron
from arq.connections import RedisSettings
from fastapi import HTTPException
from openai import APIConnectionError, APIStatusError, APITimeoutError
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert

from app.core.config import Settings, get_settings
from app.db.models import (
    Chunk,
    ChunkParent,
    EmbeddingCache,
    Paper,
    PaperVersion,
    ParseJob,
    QuotaReservation,
    Section,
    SystemConfig,
    TaskOutbox,
    UsageRecord,
    User,
    utcnow,
)
from app.services.chunking.academic import references
from app.services.chunking.chunker import CHUNKER_VERSION, chunk_document
from app.services.chunking.paper_map import bind_section_ids, build_paper_map, section_std
from app.services.ingestion.chunk_summaries import enrich_table_summaries
from app.services.ingestion.metadata import AccountedMetadataExtractor
from app.services.ingestion.quota import lock_user, release, reserve, user_settings
from app.services.parsing.errors import ParsingError
from app.services.parsing.metadata import MetadataExtractionSkipped, PaperMetadata
from app.services.parsing.mineru_result import ParsedDocument
from app.services.parsing.pipeline import parse_pdf, private_directory, private_json
from app.services.providers.embedding_batch import AccountedEmbeddings
from app.services.storage.private import directory_bytes, paper_directory, resolve_private
from app.workers.mail import shutdown
from app.workers.mail import startup as mail_startup


class TaskObsolete(Exception):
    pass


async def live(db: Any, user_id: UUID, version_id: UUID) -> tuple[Paper, PaperVersion, ParseJob]:
    await lock_user(db, user_id)
    version = await db.scalar(
        select(PaperVersion).where(PaperVersion.id == version_id, PaperVersion.user_id == user_id)
    )
    paper = (
        await db.scalar(select(Paper).where(Paper.id == version.paper_id, Paper.user_id == user_id))
        if version
        else None
    )
    if (
        not version
        or not paper
        or paper.deleted_at
        or paper.pending_version_id != version_id
        or version.status in {"ready", "deleted", "failed"}
    ):
        raise TaskObsolete()
    job = await db.get(ParseJob, version.parse_job_id)
    if not job or job.user_id != user_id or job.paper_id != paper.id:
        raise TaskObsolete()
    return paper, version, job


async def publish(
    ctx: dict[str, Any], user_id: UUID, paper_id: UUID, version_id: UUID, status: str
) -> None:
    # Durable SQL is authoritative. Redis notifications are a disposable hint.
    try:
        await ctx["redis"].set(
            f"pm:{user_id}:paper:{paper_id}:progress",
            json.dumps({"version_id": str(version_id), "status": status}),
            ex=86400,
        )
    except Exception:
        pass


async def dispatch_outbox(ctx: dict[str, Any]) -> None:
    async with ctx["session_factory"]() as db:
        rows = list(
            await db.scalars(
                select(TaskOutbox)
                .where(
                    TaskOutbox.kind.in_(["ingest", "cleanup"]),
                    TaskOutbox.completed_at.is_(None),
                    TaskOutbox.available_at <= utcnow(),
                    (TaskOutbox.delivered_at.is_(None))
                    | (TaskOutbox.delivered_at < utcnow() - timedelta(seconds=60)),
                )
                .order_by(TaskOutbox.available_at)
                .limit(100)
                .with_for_update(skip_locked=True)
            )
        )
        for row in rows:
            # If Redis loses its queue or a worker dies, unfinished SQL rows are
            # replayed. Deterministic IDs coalesce deliveries still in ARQ.
            await ctx["redis"].enqueue_job(
                "ingest" if row.kind == "ingest" else "cleanup",
                str(row.user_id),
                str(row.resource_id),
                _job_id=f"pm:{row.user_id}:{row.kind}:{row.resource_id}",
                _queue_name="papermind:ingest",
            )
            row.attempts += 1
            row.delivered_at = utcnow()
        await db.commit()


async def checkpoint(
    ctx: dict[str, Any], user_id: UUID, version_id: UUID, manifest: dict[str, Any]
) -> None:
    async with ctx["session_factory"]() as db:
        paper, version, job = await live(db, user_id, version_id)
        job.batch_id = manifest.get("batch_id")
        stage = manifest.get("stage")
        status = "uploading" if stage in {"new", "submitting", "uploading"} else "parsing"
        paper.status, version.status, job.status = status, status, status
        # Once upload is accepted, a cloud parse can consume pages even if the
        # response later gets lost. Keep its conservative estimate explicit.
        if stage in {"polling", "normalizing", "complete"} and not job.charged:
            db.add(
                UsageRecord(
                    user_id=user_id,
                    version_id=version_id,
                    node="parsing",
                    provider="mineru",
                    model=job.config["mineru_model_version"],
                    metric="parse_pages",
                    amount=paper.page_count,
                    estimated=True,
                )
            )
            job.charged = True
            await db.execute(
                update(QuotaReservation)
                .where(
                    QuotaReservation.user_id == user_id,
                    QuotaReservation.version_id == version_id,
                    QuotaReservation.metric == "parse_pages",
                )
                .values(status="settled")
            )
        if stage == "complete":
            job.status, job.error, job.finished_at = "complete", None, utcnow()
        await db.commit()
    await publish(ctx, user_id, paper.id, version_id, status)


async def account_storage(ctx: dict[str, Any], user_id: UUID, version_id: UUID) -> None:
    settings = ctx["settings"]
    async with ctx["session_factory"]() as db:
        paper, version, job = await live(db, user_id, version_id)
        limits = await user_settings(db, settings, user_id)
        directory = paper_directory(settings, user_id, paper.id)
        size = await asyncio.to_thread(directory_bytes, directory)
        others = await db.scalar(
            select(func.coalesce(func.sum(Paper.size_bytes), 0)).where(
                Paper.user_id == user_id, Paper.id != paper.id, Paper.deleted_at.is_(None)
            )
        )
        exceeded = size + int(others or 0) > limits.user_max_storage_mb * 1024**2
        if exceeded and version.operation != "rechunk":
            output = resolve_private(settings, job.result_path)

            # Keep recovery state and already-paid metadata responses, so a
            # retry polls the same remote batch without reissuing model calls.
            # Old versions and their raw parse directories are untouched.
            def prune() -> None:
                for item in output.iterdir():
                    if item.name == "job.json":
                        manifest = json.loads(item.read_text())
                        manifest["stage"] = "polling"
                        private_json(item, manifest)
                    elif item.name in {"metadata-extraction.json", "chunk-summaries"}:
                        continue
                    elif item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

            await asyncio.to_thread(prune)
            size = await asyncio.to_thread(directory_bytes, directory)
        paper.size_bytes = size
        await db.commit()
    if exceeded:
        raise HTTPException(429, "解析结果超过存储额度，请清理其他论文后重试")


async def build(ctx: dict[str, Any], user_id: UUID, version_id: UUID) -> None:
    settings: Settings = ctx["settings"]
    async with ctx["session_factory"]() as db:
        paper, version, job = await live(db, user_id, version_id)
        if (version.embedding_provider, version.embedding_model, version.embedding_dim) != (
            settings.embedding_base_url.rstrip("/"),
            settings.embedding_model,
            settings.embedding_dim,
        ):
            raise ParsingError("Embedding 配置已改变，请重新分块", code="embedding_config_changed")
        if version.chunker_version != CHUNKER_VERSION or settings.embedding_model != "BAAI/bge-m3":
            raise ParsingError(
                "分块算法或分词模型已改变，请重新解析或重新分块以创建新版本",
                code="chunker_config_changed",
            )
        if version.attempts >= settings.ingestion_max_attempts:
            raise ParsingError("入库自动重试次数已耗尽，请手动重试", code="attempts_exhausted")
        version.attempts += 1
        job.attempts += 1
        await db.commit()
    output = resolve_private(settings, job.result_path)
    if version.operation != "rechunk" and job.status != "complete":
        parse_settings = settings.model_copy(update=job.config)
        parser = ctx.get("parse_pdf", parse_pdf)

        async def validate_metadata(db: Any) -> None:
            current, _, _ = await live(db, user_id, version_id)
            if current.metadata_edited:
                raise MetadataExtractionSkipped("论文元数据已手动编辑，已跳过快速模型提取")

        await parser(
            resolve_private(settings, paper.file_path),
            output,
            parse_settings,
            use_llm=True,
            extractor=AccountedMetadataExtractor(
                settings,
                ctx["session_factory"],
                ctx["engine"],
                user_id,
                version_id,
                output,
                validate_metadata,
            ),
            checkpoint=lambda state: checkpoint(ctx, user_id, version_id, state),
        )
        # A complete on-disk manifest returns early; mirror it into SQL on resume.
        if (output / "job.json").is_file():
            await checkpoint(
                ctx, user_id, version_id, json.loads((output / "job.json").read_text())
            )
    await account_storage(ctx, user_id, version_id)
    document = ParsedDocument.model_validate_json((output / "document.json").read_text())
    metadata = PaperMetadata.model_validate_json((output / "metadata.json").read_text())
    async with ctx["session_factory"]() as db:
        paper, version, job = await live(db, user_id, version_id)
        paper.status, version.status = "chunking", "chunking"
        if not paper.metadata_edited:
            for key in ("title", "authors", "year", "venue", "doi", "arxiv_id", "abstract"):
                value = getattr(metadata, key)
                if value:
                    setattr(paper, key, value)
        title, abstract, authors = paper.title, paper.abstract, list(paper.authors or [])
        await db.commit()
    drafts = await asyncio.to_thread(chunk_document, document, title, abstract, authors)

    async def validate_summary(db: Any) -> None:
        await live(db, user_id, version_id)

    await enrich_table_summaries(ctx, user_id, version_id, output, drafts, validate_summary)
    await account_storage(ctx, user_id, version_id)
    if not any(draft.indexable for draft in drafts):
        raise ParsingError("解析结果中没有可索引的文本", code="empty_chunks")
    inputs = {draft.content_hash: draft for draft in drafts if draft.indexable}
    vectors: dict[str, list[float]] = {}
    async with ctx["session_factory"]() as db:
        await live(db, user_id, version_id)
        cached = await db.scalars(
            select(EmbeddingCache).where(
                EmbeddingCache.user_id == user_id,
                EmbeddingCache.provider == version.embedding_provider,
                EmbeddingCache.model == version.embedding_model,
                EmbeddingCache.dimension == version.embedding_dim,
                EmbeddingCache.content_hash.in_(inputs),
            )
        )
        for item in cached:
            vectors[item.content_hash] = list(item.embedding)
        await db.commit()
    missing = [draft for key, draft in inputs.items() if key not in vectors]
    embedder = ctx.get("embedder") or AccountedEmbeddings(settings)
    for start in range(0, len(missing), settings.embedding_batch_size):
        batch = missing[start : start + settings.embedding_batch_size]
        amount = sum(draft.token_count for draft in batch)
        usage_id = uuid4()
        async with ctx["session_factory"]() as db:
            paper, version, _ = await live(db, user_id, version_id)
            paper.status, version.status = "embedding", "embedding"
            reservation = await reserve(db, settings, user_id, version_id, "model_tokens", amount)
            # The intention is durable before external I/O. Unknown outcomes are
            # charged conservatively, so retries cannot evade the daily budget.
            reservation.status = "settled"
            db.add(
                UsageRecord(
                    id=usage_id,
                    user_id=user_id,
                    version_id=version_id,
                    node="embedding",
                    provider=version.embedding_provider,
                    model=version.embedding_model,
                    metric="model_tokens",
                    amount=amount,
                    estimated=True,
                    outcome="uncertain",
                )
            )
            await db.commit()
        await publish(ctx, user_id, paper.id, version_id, "embedding")
        try:
            values = await embedder.aembed_documents([draft.embedding_input for draft in batch])
        except APIStatusError as error:
            if error.status_code < 500:
                async with ctx["session_factory"]() as db:
                    await db.execute(
                        update(UsageRecord)
                        .where(UsageRecord.id == usage_id)
                        .values(amount=0, outcome="rejected")
                    )
                    await db.commit()
            raise
        if len(values) != len(batch) or any(
            len(v) != settings.embedding_dim or any(not math.isfinite(float(n)) for n in v)
            for v in values
        ):
            raise ParsingError("Embedding 返回数量、维度或数值无效", code="invalid_embedding")
        async with ctx["session_factory"]() as db:
            await live(db, user_id, version_id)
            for draft, vector in zip(batch, values, strict=True):
                await db.execute(
                    insert(EmbeddingCache)
                    .values(
                        id=uuid4(),
                        user_id=user_id,
                        provider=version.embedding_provider,
                        model=version.embedding_model,
                        dimension=version.embedding_dim,
                        content_hash=draft.content_hash,
                        embedding=vector,
                    )
                    .on_conflict_do_nothing()
                )
                vectors[draft.content_hash] = vector
            await db.execute(
                update(UsageRecord)
                .where(UsageRecord.id == usage_id)
                .values(
                    outcome="confirmed",
                    amount=getattr(embedder, "last_usage", None)
                    if getattr(embedder, "last_usage", None) is not None
                    else amount,
                    estimated=getattr(embedder, "last_usage", None) is None,
                )
            )
            await db.commit()
    async with ctx["session_factory"]() as db:
        paper, version, job = await live(db, user_id, version_id)
        section_ids = {s.id: uuid4() for s in document.sections}
        # A cancelled or duplicate delivery publishes the same version once.
        await db.execute(
            delete(Chunk).where(Chunk.user_id == user_id, Chunk.version_id == version_id)
        )
        await db.execute(
            delete(ChunkParent).where(
                ChunkParent.user_id == user_id, ChunkParent.version_id == version_id
            )
        )
        await db.execute(
            delete(Section).where(Section.user_id == user_id, Section.version_id == version_id)
        )
        for section in document.sections:
            db.add(
                Section(
                    id=section_ids[section.id],
                    user_id=user_id,
                    version_id=version_id,
                    parent_id=section_ids.get(section.parent_id) if section.parent_id else None,
                    title=section.title,
                    level=section.level,
                    path=section.path,
                    section_std=section_std(section.path),
                )
            )
        await db.flush()
        parents = {
            draft.parent.ordinal: draft.parent for draft in drafts if draft.parent is not None
        }
        parent_ids = {ordinal: uuid4() for ordinal in parents}
        for ordinal, parent in parents.items():
            db.add(
                ChunkParent(
                    id=parent_ids[ordinal],
                    user_id=user_id,
                    paper_id=paper.id,
                    version_id=version_id,
                    section_id=section_ids.get(parent.section_id) if parent.section_id else None,
                    ordinal=ordinal,
                    kind=parent.kind,
                    text=parent.text,
                    locations=parent.locations,
                    child_count=sum(d.parent is parent for d in drafts),
                )
            )
        await db.flush()
        for ordinal, draft in enumerate(drafts):
            pages = [loc["page_idx"] for loc in draft.locations] or [0]
            db.add(
                Chunk(
                    user_id=user_id,
                    collection_id=paper.collection_id,
                    paper_id=paper.id,
                    version_id=version_id,
                    section_id=section_ids.get(draft.section_id) if draft.section_id else None,
                    parent_id=parent_ids[draft.parent.ordinal]
                    if draft.parent is not None
                    else None,
                    ordinal=ordinal,
                    kind=draft.kind,
                    text=draft.text,
                    embedding_input=draft.embedding_input,
                    section_path=draft.section_path,
                    locations=draft.locations,
                    source_spans=draft.source_spans or [],
                    chunk_metadata=draft.chunk_metadata,
                    page_start=min(pages),
                    page_end=max(pages),
                    token_count=draft.token_count,
                    content_hash=draft.content_hash,
                    indexable=draft.indexable,
                    unindexed_reason=draft.unindexed_reason,
                    embedding=vectors.get(draft.content_hash),
                )
            )
        await db.flush()
        paper_map = build_paper_map(document, title, abstract)

        bind_section_ids(paper_map, section_ids)
        version.paper_map = paper_map
        version.reference_entries = references(document)
        version.status, version.error, version.finished_at = "ready", None, utcnow()
        paper.active_version_id, paper.pending_version_id, paper.status, paper.error = (
            version_id,
            None,
            "ready",
            None,
        )
        await release(db, user_id, version_id)
        await db.execute(
            update(TaskOutbox)
            .where(
                TaskOutbox.user_id == user_id,
                TaskOutbox.kind == "ingest",
                TaskOutbox.resource_id == version_id,
            )
            .values(completed_at=utcnow())
        )
        await db.commit()
    await publish(ctx, user_id, paper.id, version_id, "ready")


def failure(error: Exception) -> tuple[str, str, bool]:
    if isinstance(error, ParsingError):
        return error.code, str(error), error.retryable
    if isinstance(error, HTTPException):
        return "quota_or_account", str(error.detail), False
    if isinstance(error, APIStatusError):
        retryable = error.status_code == 429 or error.status_code >= 500
        return (
            "embedding_provider",
            "向量服务暂不可用" if retryable else "向量服务认证、参数或额度错误",
            retryable,
        )
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return "embedding_network", "向量服务网络故障", True
    return "ingestion_error", "入库失败，请重试或联系管理员", False


async def mark_failure(
    ctx: dict[str, Any], user_id: UUID, version_id: UUID, error: Exception
) -> None:
    code, message, retryable = failure(error)
    async with ctx["session_factory"]() as db:
        # This must also settle a task whose owner was disabled during execution.
        await db.scalar(select(User).where(User.id == user_id).with_for_update())
        version = await db.get(PaperVersion, version_id)
        if not version or version.user_id != user_id:
            return
        paper = await db.get(Paper, version.paper_id)
        if not paper or paper.deleted_at or paper.pending_version_id != version_id:
            return
        job = await db.get(ParseJob, version.parse_job_id)
        outbox = await db.scalar(
            select(TaskOutbox).where(
                TaskOutbox.user_id == user_id,
                TaskOutbox.kind == "ingest",
                TaskOutbox.resource_id == version_id,
            )
        )
        assert job and outbox
        if retryable and version.attempts < ctx["settings"].ingestion_max_attempts:
            version.status, paper.status = "pending", "pending"
            outbox.delivered_at = None
            outbox.available_at = utcnow() + timedelta(seconds=10 * 2**version.attempts)
        else:
            version.status, version.finished_at = "failed", utcnow()
            paper.pending_version_id = None
            paper.status = "ready" if paper.active_version_id else "failed"
            outbox.completed_at = utcnow()
            await release(db, user_id, version_id)
        if code == "submission_uncertain" and not job.charged:
            db.add(
                UsageRecord(
                    user_id=user_id,
                    version_id=version_id,
                    node="parsing",
                    provider="mineru",
                    model=job.config["mineru_model_version"],
                    metric="parse_pages",
                    amount=paper.page_count,
                    estimated=True,
                    outcome="uncertain",
                )
            )
            job.charged = True
            await db.execute(
                update(QuotaReservation)
                .where(
                    QuotaReservation.user_id == user_id,
                    QuotaReservation.version_id == version_id,
                    QuotaReservation.metric == "parse_pages",
                )
                .values(status="settled")
            )
        version.error, paper.error = message, message
        if job.status != "complete":
            job.error, job.status = code, "failed"
        await db.commit()


def task_lock(settings: Settings, user_id: UUID, paper_id: UUID) -> int | None:
    directory = settings.storage_root / ".locks"
    private_directory(directory)
    descriptor = os.open(directory / f"{user_id}-{paper_id}", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    return descriptor


async def ingest(ctx: dict[str, Any], owner: str, resource: str) -> None:
    user_id, version_id = UUID(owner), UUID(resource)
    async with ctx["session_factory"]() as db:
        version = await db.get(PaperVersion, version_id)
        if (
            not version
            or version.user_id != user_id
            or version.status in {"ready", "deleted", "failed"}
        ):
            return
        paper_id = version.paper_id
    descriptor = task_lock(ctx["settings"], user_id, paper_id)
    if descriptor is None:
        return
    try:
        # Slots use a dedicated transaction; cancellation or connection loss releases them.
        async with ctx["engine"].connect() as connection:
            slot = None
            for candidate in range(ctx["settings"].provider_concurrency):
                if await connection.scalar(
                    text("SELECT pg_try_advisory_xact_lock(620022, :slot)"), {"slot": candidate}
                ):
                    slot = candidate
                    break
            if slot is None:
                return
            owned_embedder = None if ctx.get("embedder") else AccountedEmbeddings(ctx["settings"])
            try:
                await build(
                    {**ctx, "embedder": owned_embedder or ctx.get("embedder")}, user_id, version_id
                )
            finally:
                if owned_embedder:
                    await owned_embedder.aclose()
    except TaskObsolete:
        return
    except Exception as error:
        await mark_failure(ctx, user_id, version_id, error)
    finally:
        os.close(descriptor)


async def cleanup(ctx: dict[str, Any], owner: str, resource: str) -> None:
    user_id, paper_id = UUID(owner), UUID(resource)
    descriptor = task_lock(ctx["settings"], user_id, paper_id)
    if descriptor is None:
        return
    try:
        async with ctx["session_factory"]() as db:
            paper = await db.get(Paper, paper_id)
            if not paper or paper.user_id != user_id or not paper.deleted_at:
                return
            directory = paper_directory(ctx["settings"], user_id, paper_id)

            def remove_files() -> None:
                if directory.exists():
                    shutil.rmtree(directory)

            await asyncio.to_thread(remove_files)
            await db.execute(
                update(TaskOutbox)
                .where(
                    TaskOutbox.user_id == user_id,
                    TaskOutbox.kind == "cleanup",
                    TaskOutbox.resource_id == paper_id,
                )
                .values(completed_at=utcnow())
            )
            await db.commit()
    finally:
        os.close(descriptor)


async def startup(ctx: dict[str, Any]) -> None:
    await mail_startup(ctx)
    async with ctx["session_factory"]() as db:
        identity = await db.get(SystemConfig, "embedding")
        settings = ctx["settings"]
        if identity is None or identity.value != {
            "model": settings.embedding_model,
            "dimension": settings.embedding_dim,
        }:
            await ctx["engine"].dispose()
            raise RuntimeError("Embedding 模型或维度与数据库不一致")


class WorkerSettings:
    functions = [ingest, cleanup, dispatch_outbox]
    cron_jobs = [cron(dispatch_outbox, second=set(range(0, 60, 10)), run_at_startup=True)]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url.get_secret_value())
    queue_name = "papermind:ingest"
    max_jobs = get_settings().provider_concurrency + 1
    job_timeout = 3600
    max_tries = 100
    keep_result = 0
    health_check_interval = 15
