import asyncio
import hashlib
from uuid import UUID, uuid4

import httpx
import pytest
import respx
from fastapi import HTTPException
from langchain_core.documents import Document
from sqlalchemy import select, update

from app.db.models import (
    Chunk,
    ChunkParent,
    Collection,
    Paper,
    PaperVersion,
    ParseJob,
    Section,
    UsageRecord,
    User,
    utcnow,
)
from app.services.providers.reranker import RerankCompressor
from app.services.retrieval import RetrievalScope, RetrievalUnavailable, create_retriever
from app.services.retrieval.hybrid import budgeted, fuse
from tests.conftest import login, post, register


def doc(key, text="evidence"):
    return Document(page_content=text, metadata={"chunk_id": key, "embedding_input": text})


def test_rrf_deduplicates_and_fuses_ranks():
    a, b, c = doc("a"), doc("b"), doc("c")
    result = fuse({"vector": [a, b, a], "bm25": [b, c]}, 60)
    assert [d.metadata["chunk_id"] for d in result] == ["b", "a", "c"]
    assert result[0].metadata["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert "rrf_score" not in b.metadata


def test_context_budget_counts_utf8_and_skips_oversized():
    assert [d.metadata["chunk_id"] for d in budgeted([doc("a", "中" * 4), doc("b", "yes")], 5)] == [
        "b"
    ]


@pytest.mark.parametrize(
    "rows",
    [
        [{"index": 0, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.7}],
        [{"index": 4, "relevance_score": 0.8}, {"index": 1, "relevance_score": 0.7}],
        [{"index": True, "relevance_score": 0.8}, {"index": 1, "relevance_score": 0.7}],
        [{"index": 0, "relevance_score": float("inf")}, {"index": 1, "relevance_score": 0.7}],
        [],
    ],
)
async def test_rerank_rejects_malformed(settings, rows):
    settings.rerank_top_n = 2
    # JSON has no infinity; exercise validator through a decoded mocked response.
    with respx.mock:
        route = respx.post(settings.rerank_base_url.rstrip("/") + "/rerank")
        response = httpx.Response(200, json={"results": []})
        response.json = lambda: {"results": rows}
        route.mock(return_value=response)
        with pytest.raises(ValueError):
            await RerankCompressor(settings=settings).acompress_documents(
                [doc("a"), doc("b")], "query"
            )


@pytest.fixture
def settings():
    from app.core.config import Settings

    return Settings(app_env="test", tracing_enabled=False)


async def test_rerank_orders_scores_and_captures_usage(settings):
    settings.rerank_top_n = 2
    compressor = RerankCompressor(settings=settings)
    with respx.mock:
        respx.post(settings.rerank_base_url.rstrip("/") + "/rerank").respond(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.2},
                    {"index": 1, "relevance_score": 0.9},
                ],
                "usage": {"total_tokens": 12},
            },
        )
        result = await compressor.acompress_documents([doc("a"), doc("b")], "query")
    assert [d.metadata["chunk_id"] for d in result] == ["b", "a"]
    assert compressor.last_usage == 12


class Embedder:
    last_usage = 3
    error = None
    gate = None
    entered = None
    calls = 0

    async def aembed_documents(self, texts):
        self.calls += 1
        if self.entered:
            self.entered.set()
        if self.gate:
            await self.gate.wait()
        if self.error:
            raise self.error
        return [[1.0] + [0.0] * 1023]

    async def aclose(self):
        pass


class Reranker:
    last_usage = 7
    error = None
    seen = None

    async def acompress_documents(self, documents, query):
        self.seen = documents
        if self.error:
            raise self.error
        return list(reversed(documents))[:8]


@pytest.fixture
async def retrieval_setup(app, client):
    await register(client)
    await login(client)
    owner = UUID((await client.get("/api/v1/auth/me")).json()["id"])
    other = uuid4()
    collections, papers, chunks, versions = [], [], [], []
    async with app.state.session_factory() as db:
        db.add(
            User(
                id=other, email="b@example.org", password_hash="unused", email_verified_at=utcnow()
            )
        )
        await db.flush()
        for i, user_id in enumerate([owner, owner, other]):
            collection = Collection(id=uuid4(), user_id=user_id, name=f"Scope {i}")
            db.add(collection)
            await db.flush()
            paper = Paper(
                id=uuid4(),
                user_id=user_id,
                collection_id=collection.id,
                title=f"Paper {i}",
                sha256=str(i) * 64,
                file_path="unused",
                page_count=1,
                size_bytes=1,
                status="ready",
            )
            db.add(paper)
            await db.flush()
            job = ParseJob(
                id=uuid4(), user_id=user_id, paper_id=paper.id, result_path="unused", config={}
            )
            db.add(job)
            await db.flush()
            for old in [True, False]:
                version = PaperVersion(
                    id=uuid4(),
                    user_id=user_id,
                    paper_id=paper.id,
                    parse_job_id=job.id,
                    operation="upload",
                    embedding_provider=app.state.settings.embedding_base_url.rstrip("/"),
                    embedding_model=app.state.settings.embedding_model,
                    embedding_dim=1024,
                    status="ready",
                )
                db.add(version)
                await db.flush()
                text = "OLD secret" if old else f"attention mechanism transformer 方法 隔离证据 {i}"
                chunk = Chunk(
                    id=uuid4(),
                    user_id=user_id,
                    collection_id=collection.id,
                    paper_id=paper.id,
                    version_id=version.id,
                    ordinal=0,
                    kind="text",
                    text=text,
                    embedding_input=text,
                    section_path=["Methods"],
                    locations=[{"page_idx": 0, "bbox": None}],
                    page_start=0,
                    page_end=0,
                    token_count=len(text.encode()),
                    content_hash=hashlib.sha256(text.encode()).hexdigest(),
                    indexable=True,
                    embedding=[1.0] + [0.0] * 1023,
                )
                db.add(chunk)
                if not old:
                    paper.active_version_id = version.id
                    chunks.append(chunk)
                    versions.append(version)
            collections.append(collection)
            papers.append(paper)
        await db.commit()
    embedder, reranker = Embedder(), Reranker()

    def build(scope=None):
        retriever = create_retriever(
            app.state.settings,
            app.state.session_factory,
            app.state.engine,
            scope or RetrievalScope(owner),
        )
        retriever.embedder_factory = lambda: embedder
        retriever.reranker_factory = lambda: reranker
        return retriever

    return owner, collections, papers, chunks, versions, embedder, reranker, build


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["vector", "bm25", "hybrid", "hybrid_rerank"])
async def test_real_channels_filter_owner_active_version_and_scopes(app, retrieval_setup, variant):
    owner, collections, papers, chunks, versions, embed, rerank, build = retrieval_setup
    result = await build().search("attention 方法", variant=variant)
    assert {d.metadata["chunk_id"] for d in result.documents} == {str(c.id) for c in chunks[:2]}
    assert not result.degraded
    for scope in [
        RetrievalScope(owner, collection_ids=(collections[0].id,)),
        RetrievalScope(owner, paper_ids=(papers[0].id,)),
    ]:
        scoped = await build(scope).search("attention 方法", variant=variant)
        assert [d.metadata["chunk_id"] for d in scoped.documents] == [str(chunks[0].id)]
    assert not (
        await build(
            RetrievalScope(owner, collection_ids=(collections[0].id,), paper_ids=(papers[1].id,))
        ).search("attention", variant=variant)
    ).documents
    assert not (
        await build(RetrievalScope(owner, paper_ids=())).search("attention", variant=variant)
    ).documents
    if variant == "hybrid_rerank":
        assert all(d.metadata["user_id"] == str(owner) for d in rerank.seen)


@pytest.mark.integration
async def test_configured_weights_reach_production_fusion_with_authorized_sources(
    app, retrieval_setup, monkeypatch
):
    owner, *_, build = retrieval_setup
    retriever = build()
    authorized = await retriever.store.vector(RetrievalScope(owner), [1.0] + [0.0] * 1023, 40)
    assert len(authorized) == 2

    async def vector(*args):
        return authorized

    async def bm25(*args):
        return list(reversed(authorized))

    monkeypatch.setattr(retriever.store, "vector", vector)
    monkeypatch.setattr(retriever.store, "bm25", bm25)
    app.state.settings.rerank_top_n = 1
    for vector_weight, expected in [(0.9, authorized[0]), (0.1, authorized[1])]:
        app.state.settings.rrf_vector_weight = vector_weight
        app.state.settings.rrf_bm25_weight = 1 - vector_weight
        result = await retriever.search("attention", variant="hybrid")
        assert len(result.documents) == 1
        assert result.documents[0].metadata["chunk_id"] == expected.metadata["chunk_id"]
        assert result.documents[0].metadata["user_id"] == str(owner)
        assert not result.degraded


async def test_scope_rejected_before_external_call(retrieval_setup):
    owner, collections, papers, chunks, versions, embed, rerank, build = retrieval_setup
    for scope in [
        RetrievalScope(owner, paper_ids=(papers[2].id,)),
        RetrievalScope(owner, collection_ids=(collections[2].id,)),
        RetrievalScope(owner, paper_ids=(uuid4(),)),
    ]:
        with pytest.raises(HTTPException) as error:
            await build(scope).search("attention")
        assert error.value.status_code == 404
    assert embed.calls == 0


@pytest.mark.integration
async def test_partial_channel_failure_and_total_failure(retrieval_setup, monkeypatch):
    *_, embed, rerank, build = retrieval_setup
    embed.error = RuntimeError("provider secret must not escape")
    result = await build().search("attention")
    assert result.documents and result.degraded == ["vector_unavailable"]
    retriever = build()

    async def broken(*args):
        raise RuntimeError("db failed")

    monkeypatch.setattr(retriever.store, "bm25", broken)
    with pytest.raises(RetrievalUnavailable):
        await retriever.search("attention")


@pytest.mark.integration
async def test_bm25_failure_uses_vector_and_rerank_failure_uses_rrf(retrieval_setup, monkeypatch):
    *_, embed, rerank, build = retrieval_setup
    rerank.error = TimeoutError()
    retriever = build()

    async def broken(*args):
        raise RuntimeError("bm25 failed")

    monkeypatch.setattr(retriever.store, "bm25", broken)
    result = await retriever.search("attention")
    assert result.documents and result.degraded == ["bm25_unavailable", "rerank_unavailable"]
    assert result.documents[0].metadata["rrf_score"] >= result.documents[1].metadata["rrf_score"]


@pytest.mark.integration
async def test_deletion_during_rerank_revokes_results(app, retrieval_setup):
    owner, collections, papers, chunks, versions, embed, rerank, build = retrieval_setup

    class DeletingReranker(Reranker):
        async def acompress_documents(self, documents, query):
            async with app.state.session_factory() as db:
                await db.execute(
                    update(Paper).where(Paper.user_id == owner).values(deleted_at=utcnow())
                )
                await db.commit()
            return documents

    retriever = build()
    retriever.reranker_factory = DeletingReranker
    assert not (await retriever.search("attention")).documents


@pytest.mark.integration
async def test_version_switch_and_unindexed_excluded(app, retrieval_setup):
    owner, collections, papers, chunks, versions, embed, rerank, build = retrieval_setup
    async with app.state.session_factory() as db:
        await db.execute(update(Chunk).where(Chunk.id == chunks[0].id).values(indexable=False))
        await db.execute(
            update(PaperVersion)
            .where(PaperVersion.id == versions[1].id)
            .values(embedding_model="different")
        )
        await db.commit()
    result = await build().search("attention")
    assert not result.documents and not result.degraded


@pytest.mark.integration
async def test_neighbors_obey_scope_and_budget(app, retrieval_setup):
    owner, collections, papers, chunks, versions, embed, rerank, build = retrieval_setup
    retriever = build(RetrievalScope(owner, paper_ids=(papers[0].id,)))
    foreign = retriever.store.document(chunks[2], "foreign")
    assert await retriever.store.neighbors(retriever.scope, [foreign], budget=1000) == []
    result = await retriever.search("attention")
    assert await retriever.store.neighbors(retriever.scope, result.documents, budget=0) == []


@pytest.mark.integration
async def test_usage_is_operation_scoped_and_charged(app, retrieval_setup):
    owner, *_, build = retrieval_setup
    result = await build().search("attention")
    async with app.state.session_factory() as db:
        rows = list(
            await db.scalars(
                select(UsageRecord).where(UsageRecord.operation_id == result.operation_id)
            )
        )
        assert {r.node for r in rows} == {"query_embedding", "rerank"}
        assert all(r.user_id == owner and r.version_id is None and not r.estimated for r in rows)
        assert sum(r.amount for r in rows) == 10


@pytest.mark.integration
async def test_budget_checked_before_call_and_unknown_spend_retained(app, retrieval_setup):
    owner, *_, embed, rerank, build = retrieval_setup
    app.state.settings.user_daily_model_tokens = 2
    with pytest.raises(HTTPException) as error:
        await build().search("attention")
    assert error.value.status_code == 429 and embed.calls == 0
    app.state.settings.user_daily_model_tokens = 100000
    embed.error = TimeoutError()
    await build().search("attention")
    async with app.state.session_factory() as db:
        row = await db.scalar(select(UsageRecord).where(UsageRecord.node == "query_embedding"))
        assert row.estimated and row.outcome == "uncertain" and row.amount == len("attention") + 8


@pytest.mark.integration
async def test_provider_rejection_does_not_charge(app, retrieval_setup):
    *_, embed, rerank, build = retrieval_setup
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.org"))
    embed.error = httpx.HTTPStatusError("rate", request=response.request, response=response)
    await build().search("attention")
    async with app.state.session_factory() as db:
        row = await db.scalar(select(UsageRecord).where(UsageRecord.node == "query_embedding"))
        assert row.amount == 0 and row.outcome == "rejected"


@pytest.mark.integration
async def test_concurrency_slots_release_after_cancel(app, retrieval_setup):
    *_, embed, rerank, build = retrieval_setup
    app.state.settings.user_chat_concurrency = 1
    embed.gate, embed.entered = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(build().search("attention"))
    await embed.entered.wait()
    with pytest.raises(HTTPException) as error:
        await build().search("attention")
    assert error.value.status_code == 429
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    embed.gate = None
    assert (await build().search("attention")).documents


@pytest.mark.integration
async def test_search_api_auth_scope_and_extra_fields(client, app, retrieval_setup, monkeypatch):
    owner, collections, papers, chunks, versions, embed, rerank, build = retrieval_setup
    monkeypatch.setattr(
        "app.api.v1.retrieval.create_retriever",
        lambda settings, sessions, engine, scope: build(scope),
    )
    response = await post(
        client, "/retrieval/search", {"query": "attention", "paper_ids": [str(papers[0].id)]}
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "ok" and len(result["documents"]) == 1
    assert "embedding_input" not in result["documents"][0]
    for body, status in [
        ({"query": "attention", "user_id": str(owner)}, 422),
        ({"query": "attention", "paper_ids": [str(papers[2].id)]}, 404),
        ({"query": "   "}, 422),
        ({"query": "中" * 1024}, 422),
    ]:
        assert (await post(client, "/retrieval/search", body)).status_code == status
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost:8080",
        headers={"Origin": "http://localhost:8080"},
    ) as anonymous:
        assert (
            await post(anonymous, "/retrieval/search", {"query": "attention"})
        ).status_code == 401
    async with app.state.session_factory() as db:
        await db.execute(update(User).where(User.id == owner).values(email_verified_at=None))
        await db.commit()
    assert (await post(client, "/retrieval/search", {"query": "attention"})).status_code == 403


@pytest.mark.integration
async def test_filter_before_top_k_with_many_foreign_candidates(app, retrieval_setup):
    owner, collections, papers, chunks, versions, embed, rerank, build = retrieval_setup
    foreign = chunks[2]
    async with app.state.session_factory() as db:
        for ordinal in range(1, 51):
            db.add(
                Chunk(
                    id=uuid4(),
                    user_id=foreign.user_id,
                    collection_id=foreign.collection_id,
                    paper_id=foreign.paper_id,
                    version_id=foreign.version_id,
                    ordinal=ordinal,
                    kind="text",
                    text="attention " * 10,
                    embedding_input="attention " * 10,
                    section_path=[],
                    locations=[{"page_idx": 0}],
                    page_start=0,
                    page_end=0,
                    token_count=100,
                    content_hash=str(ordinal).zfill(64),
                    indexable=True,
                    embedding=[1.0] + [0.0] * 1023,
                )
            )
        await db.commit()
    app.state.settings.retrieval_top_k = 1
    for variant in ["vector", "bm25"]:
        result = await build(RetrievalScope(owner, paper_ids=(papers[0].id,))).search(
            "attention", variant=variant
        )
        assert [d.metadata["chunk_id"] for d in result.documents] == [str(chunks[0].id)]


@pytest.mark.integration
async def test_concurrent_budget_cannot_overspend(app, retrieval_setup):
    *_, embed, rerank, build = retrieval_setup
    app.state.settings.user_daily_model_tokens = 600
    embed.gate, embed.entered = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(build().search("x" * 400, variant="vector"))
    await embed.entered.wait()
    with pytest.raises(HTTPException) as error:
        await build().search("x" * 400, variant="vector")
    assert error.value.status_code == 429 and embed.calls == 1
    embed.gate.set()
    await task


@pytest.mark.integration
async def test_disabled_account_and_global_budget_stop_paid_calls(app, retrieval_setup):
    owner, *_, embed, rerank, build = retrieval_setup
    app.state.settings.global_daily_model_tokens = 2
    with pytest.raises(HTTPException) as error:
        await build().search("attention")
    assert error.value.status_code == 429 and embed.calls == 0
    async with app.state.session_factory() as db:
        await db.execute(update(User).where(User.id == owner).values(status="disabled"))
        await db.commit()
    with pytest.raises(HTTPException) as error:
        await build().search("attention", variant="bm25")
    assert error.value.status_code == 403


@pytest.mark.integration
async def test_api_distinguishes_degraded_empty_and_unavailable(
    client, retrieval_setup, monkeypatch
):
    owner, collections, papers, chunks, versions, embed, rerank, build = retrieval_setup
    monkeypatch.setattr(
        "app.api.v1.retrieval.create_retriever",
        lambda settings, sessions, engine, scope: build(scope),
    )
    rerank.error = TimeoutError()
    result = await post(client, "/retrieval/search", {"query": "attention"})
    assert result.json()["status"] == "degraded" and result.json()["degraded"] == [
        "rerank_unavailable"
    ]
    empty = await post(client, "/retrieval/search", {"query": "attention", "paper_ids": []})
    assert empty.json()["status"] == "ok" and empty.json()["documents"] == []

    async def broken(*args):
        raise RuntimeError("private provider error")

    from app.services.retrieval.store import RetrievalStore

    monkeypatch.setattr(RetrievalStore, "bm25", broken)
    embed.error = TimeoutError()
    result = await post(client, "/retrieval/search", {"query": "attention"})
    assert result.status_code == 503 and "private" not in result.text


def test_frozen_dataset_and_metric_contract():
    from app.eval.retrieval import ROOT, load_dataset, metrics

    dataset, corpus = load_dataset(ROOT / "eval/datasets/m3-gold-v7.json")
    assert len(dataset["cases"]) == 34 and len(corpus) == 2
    result = metrics(
        [
            {"relevant_ranks": [2, 4], "relevant_count": 4},
            {"relevant_ranks": [], "relevant_count": 1},
        ]
    )
    assert result == {
        "hit_at_8": 0.5,
        "mrr_at_8": 0.25,
        "precision_at_8": 0.125,
        "recall_at_8": 0.25,
    }


async def test_rerank_siliconflow_meta_usage(settings):
    settings.rerank_top_n = 1
    compressor = RerankCompressor(settings=settings)
    with respx.mock:
        respx.post(settings.rerank_base_url.rstrip("/") + "/rerank").respond(
            200,
            json={
                "results": [{"index": 0, "relevance_score": 0.9}],
                "meta": {"tokens": {"input_tokens": 150, "output_tokens": 10}},
            },
        )
        await compressor.acompress_documents([doc("a")], "query")
    assert compressor.last_usage == 160


@pytest.mark.integration
async def test_langchain_retriever_contract_includes_health(retrieval_setup):
    *_, build = retrieval_setup
    documents = await build().ainvoke("attention")
    assert documents and all(d.metadata["retrieval_degraded"] == [] for d in documents)
    assert all(UUID(d.metadata["operation_id"]) for d in documents)


async def test_navigation_api_uses_postgres_exact_labels_and_scoped_l1(
    app, client, retrieval_setup
):
    owner, _, papers, chunks, versions, embedder, _, _ = retrieval_setup
    section_id, parent_id = uuid4(), uuid4()
    full_table = "| Model | Score |\n| --- | --- |\n| Ours | 92.3 |"
    async with app.state.session_factory() as db:
        db.add(
            Section(
                id=section_id,
                user_id=owner,
                version_id=versions[0].id,
                title="3 Method",
                level=1,
                path=["3 Method"],
                section_std="method",
            )
        )
        await db.flush()
        db.add(
            ChunkParent(
                id=parent_id,
                user_id=owner,
                paper_id=papers[0].id,
                version_id=versions[0].id,
                section_id=section_id,
                ordinal=0,
                kind="section",
                text=full_table,
                locations=[],
                child_count=1,
            )
        )
        await db.flush()
        chunk = await db.get(Chunk, chunks[0].id)
        chunk.section_id, chunk.parent_id, chunk.text = section_id, parent_id, full_table
        chunk.kind, chunk.indexable, chunk.embedding = "table", False, None
        chunk.chunk_metadata = {"defined_labels": ["table:2"], "markdown_table": full_table}
        version = await db.get(PaperVersion, versions[0].id)
        version.paper_map = {
            "title": "Paper",
            "abstract": "Full abstract",
            "toc": [],
            "elements": [],
        }
        await db.commit()
    scope = {"paper_ids": [str(papers[0].id)]}
    response = await post(client, "/retrieval/paper-maps", scope)
    assert (
        response.status_code == 200
        and response.json()["paper_maps"][0]["abstract"] == "Full abstract"
    )
    response = await post(client, "/retrieval/read-section", {**scope, "section_std": "method"})
    assert response.status_code == 200 and response.json()["status"] == "ok"
    assert response.json()["parents"][0]["text"] == full_table
    response = await post(client, "/retrieval/inspect-element", {**scope, "label": "Table 2"})
    assert response.status_code == 200 and response.json()["documents"][0]["text"] == full_table
    assert "user_id" not in response.json()["documents"][0]
    response = await post(client, "/retrieval/inspect-element", {**scope, "label": "Table 20"})
    assert response.json()["status"] == "not_found" and not response.json()["documents"]
    response = await post(client, "/retrieval/paper-maps", {"paper_ids": [str(papers[2].id)]})
    assert response.status_code == 404
    assert embedder.calls == 0
