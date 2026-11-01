"""Offline SQL coverage of parent reconstruction; not pg_search/concurrency tests."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import (
    Base,
    Chunk,
    ChunkParent,
    Collection,
    Paper,
    PaperVersion,
    ParseJob,
    Section,
    User,
    utcnow,
)
from app.services.retrieval.store import RetrievalStore
from app.services.retrieval.types import RetrievalScope
from tests.test_ingestion_metadata import AsyncSessionFacade


class LocalJSONCompiler(SQLiteTypeCompiler):
    def visit_JSONB(self, type_, **kwargs):
        return "JSON"


@pytest.fixture
def parent_store():
    engine = create_engine("sqlite://")
    engine.dialect.type_compiler_instance = LocalJSONCompiler(engine.dialect)
    Base.metadata.create_all(
        engine,
        tables=[
            m.__table__
            for m in (User, Collection, Paper, ParseJob, PaperVersion, Section, ChunkParent, Chunk)
        ],
    )
    settings = Settings(_env_file=None, tracing_enabled=False)
    owner, foreign = uuid4(), uuid4()
    papers, all_chunks, parents, versions = [], [], [], []
    with Session(engine) as db:
        for i, user_id in enumerate((owner, foreign)):
            db.add(
                User(
                    id=user_id,
                    email=f"{i}@example.org",
                    password_hash="unused",
                    email_verified_at=utcnow(),
                )
            )
            collection = Collection(id=uuid4(), user_id=user_id, name="Research")
            db.add(collection)
            paper = Paper(
                id=uuid4(),
                user_id=user_id,
                collection_id=collection.id,
                title="Paper",
                sha256=str(i) * 64,
                file_path="unused",
                size_bytes=1,
                page_count=5,
                status="ready",
            )
            db.add(paper)
            job = ParseJob(
                id=uuid4(), user_id=user_id, paper_id=paper.id, result_path="unused", config={}
            )
            db.add(job)
            version = PaperVersion(
                id=uuid4(),
                user_id=user_id,
                paper_id=paper.id,
                parse_job_id=job.id,
                operation="upload",
                embedding_provider=settings.embedding_base_url.rstrip("/"),
                embedding_model=settings.embedding_model,
                embedding_dim=1024,
                status="ready",
            )
            db.add(version)
            paper.active_version_id = version.id
            chunks = []
            for ordinal, count in enumerate((3, 1, 1)):
                parent = ChunkParent(
                    id=uuid4(),
                    user_id=user_id,
                    paper_id=paper.id,
                    version_id=version.id,
                    ordinal=ordinal,
                    kind="text",
                    text="Original parent",
                    locations=[],
                    child_count=count,
                )
                db.add(parent)
                parents.append(parent.id)
                for _ in range(count):
                    index = len(chunks)
                    chunk = Chunk(
                        id=uuid4(),
                        user_id=user_id,
                        collection_id=collection.id,
                        paper_id=paper.id,
                        version_id=version.id,
                        parent_id=parent.id,
                        ordinal=index,
                        kind="formula" if index == 1 else "text",
                        text=f"Original source {index}.",
                        embedding_input=f"Source {index}",
                        section_path=["Results"] if index == 4 else ["Methods"],
                        locations=[{"page_idx": index, "bbox": None}],
                        source_spans=[{"block_id": str(index), "start": 0, "end": 18}],
                        page_start=index,
                        page_end=index,
                        token_count=10,
                        content_hash=str(index) * 64,
                        indexable=True,
                        embedding=[0.1] * 1024,
                    )
                    db.add(chunk)
                    chunks.append(chunk.id)
            all_chunks.append(chunks)
            papers.append(paper.id)
            versions.append(version.id)
        db.commit()

    @asynccontextmanager
    async def sessions():
        with Session(engine) as db:
            yield AsyncSessionFacade(db)

    context = SimpleNamespace(
        engine=engine,
        settings=settings,
        owner=owner,
        foreign=foreign,
        papers=papers,
        chunks=all_chunks,
        parents=parents,
        versions=versions,
        store=RetrievalStore(sessions, settings),
        scope=RetrievalScope(owner),
    )
    yield context
    engine.dispose()


def hit(context, index=1, foreign=False):
    with Session(context.engine) as db:
        chunk = db.get(Chunk, context.chunks[int(foreign)][index])
        return context.store.document(chunk, "Paper")


async def test_store_expands_single_hit_window_without_crossing_section(parent_store):
    c = parent_store
    docs = await c.store.expand_context(c.scope, [hit(c)])
    assert [d.metadata["ordinal"] for d in docs] == [0, 1, 2]
    assert {d.metadata["user_id"] for d in docs} == {str(c.owner)}
    assert all(d.metadata["paper_id"] == str(c.papers[0]) for d in docs)
    assert all(d.page_content == f"Original source {d.metadata['ordinal']}." for d in docs)
    assert [d.metadata["locations"][0]["page_idx"] for d in docs] == [0, 1, 2]
    assert [d.metadata["retrieval_role"] for d in docs] == ["neighbor", "hit", "neighbor"]
    assert all(d.metadata["parent_complete"] for d in docs[:3])


async def test_single_text_hit_gets_one_neighbor_each_side(parent_store):
    c = parent_store
    docs = await c.store.expand_context(c.scope, [hit(c, 0)])
    assert [d.metadata["ordinal"] for d in docs] == [0, 1]


async def test_foreign_or_out_of_scope_hits_cannot_expand(parent_store):
    c = parent_store
    assert await c.store.expand_context(c.scope, [hit(c, foreign=True)]) == []
    assert await c.store.expand_context(RetrievalScope(c.owner, paper_ids=()), [hit(c)]) == []
    assert (
        await c.store.expand_context(RetrievalScope(c.owner, paper_ids=(c.papers[1],)), [hit(c)])
        == []
    )


@pytest.mark.parametrize("revocation", ["deleted", "version", "disabled", "incompatible"])
async def test_revoked_sources_are_removed_before_context_expansion(parent_store, revocation):
    c = parent_store
    original = hit(c)
    with Session(c.engine) as db:
        if revocation == "disabled":
            db.get(User, c.owner).status = "disabled"
        elif revocation == "version":
            db.get(Paper, c.papers[0]).active_version_id = None
        elif revocation == "incompatible":
            db.get(PaperVersion, c.versions[0]).embedding_model = "different"
        else:
            db.get(Paper, c.papers[0]).deleted_at = utcnow()
        db.commit()
    assert await c.store.expand_context(c.scope, [original]) == []


async def test_window_switch_source_limit_and_nonindexable_sibling(parent_store):
    c = parent_store
    c.settings.retrieval_neighbor_window = 0
    c.settings.retrieval_parent_max_chunks = 1
    docs = await c.store.expand_context(c.scope, [hit(c)])
    assert len(docs) == 1 and not docs[0].metadata["parent_complete"]
    c.settings.retrieval_parent_max_chunks = 32
    with Session(c.engine) as db:
        db.get(Chunk, c.chunks[0][2]).indexable = False
        db.commit()
    docs = await c.store.expand_context(c.scope, [hit(c, 0), hit(c, 1)])
    assert [d.metadata["ordinal"] for d in docs] == [0, 1, 2]
    assert all(d.metadata["parent_complete"] for d in docs)


async def test_delete_after_anchor_refresh_returns_empty_instead_of_reading_stale_parent(
    parent_store, monkeypatch
):
    c = parent_store
    refresh = c.store.refresh
    calls = 0

    async def deleting(scope, docs):
        nonlocal calls
        result = await refresh(scope, docs)
        calls += 1
        if calls == 1:
            with Session(c.engine) as db:
                for chunk in db.scalars(select(Chunk).where(Chunk.user_id == c.owner)):
                    db.delete(chunk)
                for parent in db.scalars(select(ChunkParent).where(ChunkParent.user_id == c.owner)):
                    db.delete(parent)
                db.commit()
        return result

    monkeypatch.setattr(c.store, "refresh", deleting)
    assert await c.store.expand_context(c.scope, [hit(c)]) == []
