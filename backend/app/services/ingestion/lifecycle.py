import asyncio
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import (
    Chunk,
    ChunkParent,
    Citation,
    EmbeddingCache,
    Paper,
    PaperVersion,
    ParseJob,
    Section,
    TaskOutbox,
    utcnow,
)
from app.services.chunking.chunker import CHUNKER_VERSION
from app.services.ingestion.quota import release, reserve, user_settings
from app.services.parsing.pdf import PDFInfo
from app.services.parsing.pipeline import private_directory
from app.services.storage.private import paper_directory


async def owned_paper(db: AsyncSession, user_id: UUID, paper_id: UUID) -> Paper:
    item = await db.scalar(
        select(Paper).where(
            Paper.id == paper_id, Paper.user_id == user_id, Paper.deleted_at.is_(None)
        )
    )
    if item is None:
        raise HTTPException(404, "论文不存在")
    return item


async def new_version(
    db: AsyncSession, settings: Settings, paper: Paper, operation: str
) -> PaperVersion:
    settings = await user_settings(db, settings, paper.user_id)
    if paper.pending_version_id:
        raise HTTPException(409, "论文已有入库任务，请等待完成")
    count = await db.scalar(
        select(func.count())
        .select_from(Paper)
        .where(
            Paper.user_id == paper.user_id,
            Paper.deleted_at.is_(None),
            Paper.pending_version_id.is_not(None),
        )
    )
    if (count or 0) >= settings.user_upload_concurrency:
        raise HTTPException(429, "同时入库的论文过多，请稍后重试")
    if operation == "rechunk":
        active = (
            await db.get(PaperVersion, paper.active_version_id) if paper.active_version_id else None
        )
        if not active:
            raise HTTPException(409, "论文尚无可复用的解析结果")
        job = await db.get(ParseJob, active.parse_job_id)
        assert job is not None
    else:
        job_id = uuid4()
        directory = paper_directory(settings, paper.user_id, paper.id) / "parses" / str(job_id)
        job = ParseJob(
            id=job_id,
            user_id=paper.user_id,
            paper_id=paper.id,
            result_path=str(directory.relative_to(settings.storage_root)),
            config={
                key: getattr(settings, key)
                for key in (
                    "mineru_model_version",
                    "mineru_enable_formula",
                    "mineru_enable_table",
                    "mineru_language",
                )
            },
        )
        db.add(job)
        await db.flush()
    version = PaperVersion(
        id=uuid4(),
        user_id=paper.user_id,
        paper_id=paper.id,
        parse_job_id=job.id,
        operation=operation,
        chunker_version=CHUNKER_VERSION,
        embedding_provider=settings.embedding_base_url.rstrip("/"),
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )
    db.add(version)
    await db.flush()
    if operation != "rechunk":
        await reserve(db, settings, paper.user_id, version.id, "parse_pages", paper.page_count)
    paper.pending_version_id, paper.status, paper.error = version.id, "pending", None
    db.add(TaskOutbox(user_id=paper.user_id, kind="ingest", resource_id=version.id))
    return version


async def accept_upload(
    db: AsyncSession,
    settings: Settings,
    user_id: UUID,
    collection_id: UUID,
    temporary: Path,
    pdf: PDFInfo,
) -> Paper:
    settings = await user_settings(db, settings, user_id)
    existing = await db.scalar(
        select(Paper).where(
            Paper.user_id == user_id,
            Paper.collection_id == collection_id,
            Paper.sha256 == pdf.sha256,
            Paper.deleted_at.is_(None),
        )
    )
    if existing:
        return existing
    count, size = (
        await db.execute(
            select(func.count(), func.coalesce(func.sum(Paper.size_bytes), 0)).where(
                Paper.user_id == user_id, Paper.deleted_at.is_(None)
            )
        )
    ).one()
    if (
        count >= settings.user_max_papers
        or size + pdf.size > settings.user_max_storage_mb * 1024**2
    ):
        raise HTTPException(429, "论文数量或存储额度不足，请清理论文后重试")
    paper = Paper(
        id=uuid4(),
        user_id=user_id,
        collection_id=collection_id,
        title=pdf.title or "未命名论文",
        sha256=pdf.sha256,
        file_path="",
        size_bytes=pdf.size,
        page_count=len(pdf.pages),
    )
    db.add(paper)
    await db.flush()
    await new_version(db, settings, paper, "upload")
    directory = paper_directory(settings, user_id, paper.id)
    private_directory(settings.storage_root / str(user_id))
    private_directory(directory)
    destination = directory / "original.pdf"
    await asyncio.to_thread(temporary.replace, destination)
    paper.file_path = str(destination.relative_to(settings.storage_root))
    return paper


async def tombstone(db: AsyncSession, paper: Paper) -> None:
    paper.deleted_at, paper.status, paper.error = utcnow(), "deleted", None
    paper.active_version_id, paper.pending_version_id = None, None
    paper.title, paper.authors, paper.abstract = "来源已删除", [], None
    paper.venue, paper.doi, paper.arxiv_id, paper.year = None, None, None, None
    await db.execute(
        update(Citation)
        .where(Citation.user_id == paper.user_id, Citation.paper_id == paper.id)
        .values(source_text="", locations=[], deleted_at=paper.deleted_at)
    )
    versions = list(
        await db.scalars(
            select(PaperVersion.id).where(
                PaperVersion.user_id == paper.user_id, PaperVersion.paper_id == paper.id
            )
        )
    )
    await db.execute(
        update(PaperVersion)
        .where(PaperVersion.id.in_(versions))
        .values(status="deleted", paper_map={})
    )
    await db.execute(
        update(ParseJob)
        .where(ParseJob.user_id == paper.user_id, ParseJob.paper_id == paper.id)
        .values(status="deleted")
    )
    for version_id in versions:
        await release(db, paper.user_id, version_id)
    await db.execute(
        update(TaskOutbox)
        .where(
            TaskOutbox.user_id == paper.user_id,
            TaskOutbox.kind == "ingest",
            TaskOutbox.resource_id.in_(versions),
        )
        .values(completed_at=utcnow())
    )
    # Remove content and cache immediately. The worker's final transaction takes
    # this same user lock, so it cannot publish chunks after this transaction.

    await db.execute(
        delete(Chunk).where(Chunk.user_id == paper.user_id, Chunk.paper_id == paper.id)
    )
    await db.execute(
        delete(ChunkParent).where(
            ChunkParent.user_id == paper.user_id, ChunkParent.paper_id == paper.id
        )
    )
    await db.execute(
        delete(Section).where(Section.user_id == paper.user_id, Section.version_id.in_(versions))
    )
    await db.execute(delete(EmbeddingCache).where(EmbeddingCache.user_id == paper.user_id))
    db.add(TaskOutbox(user_id=paper.user_id, kind="cleanup", resource_id=paper.id))


async def active_chunks(db: AsyncSession, user_id: UUID) -> list[Chunk]:
    return list(
        await db.scalars(
            select(Chunk)
            .join(Paper, Paper.active_version_id == Chunk.version_id)
            .where(
                Paper.user_id == user_id,
                Chunk.user_id == user_id,
                Paper.deleted_at.is_(None),
                Chunk.indexable.is_(True),
            )
        )
    )


async def snapshot(db: AsyncSession, user_id: UUID, chunk_id: UUID) -> Citation:
    from app.db.models import User

    await db.scalar(select(User).where(User.id == user_id).with_for_update())
    # M4 attaches these snapshots to messages in its final persistence transaction.
    chunk = await db.scalar(
        select(Chunk)
        .join(Paper, Paper.active_version_id == Chunk.version_id)
        .where(
            Chunk.id == chunk_id,
            Chunk.user_id == user_id,
            Paper.user_id == user_id,
            Paper.deleted_at.is_(None),
        )
    )
    if chunk is None:
        raise HTTPException(404, "来源不存在")
    citation = Citation(
        user_id=user_id,
        paper_id=chunk.paper_id,
        version_id=chunk.version_id,
        chunk_id=chunk.id,
        source_text=chunk.text,
        locations=chunk.locations,
    )
    db.add(citation)
    await db.flush()
    return citation
