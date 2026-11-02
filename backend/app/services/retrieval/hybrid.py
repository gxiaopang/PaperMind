import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any, Literal
from uuid import UUID, uuid4

import structlog
from fastapi import HTTPException
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from app.core.config import Settings
from app.core.observability import observation
from app.services.chunking.paper_map import section_std
from app.services.providers.embedding_batch import AccountedEmbeddings
from app.services.providers.reranker import RerankCompressor
from app.services.retrieval.accounting import RetrievalAccounting
from app.services.retrieval.planning import expand_acronyms, fusion_policy, route_query
from app.services.retrieval.store import RetrievalStore
from app.services.retrieval.types import RetrievalResult, RetrievalScope, RetrievalUnavailable

log = structlog.get_logger()
Variant = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]
ChannelResults = list[list[Document] | BaseException]


def fuse(
    channels: dict[str, list[Document]], k: int, weights: dict[str, float] | None = None
) -> list[Document]:
    docs: dict[str, Document] = {}
    for channel, rows in channels.items():
        weight = weights.get(channel, 1.0) if weights is not None else 1.0
        if not math.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError("RRF weights must be finite and in (0, 1]")
        seen = set()
        for rank, source in enumerate(rows, 1):
            key = source.metadata["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            if key not in docs:
                docs[key] = source.model_copy(deep=True)
                docs[key].metadata.update(rrf_score=0.0, ranks={})
            docs[key].metadata["rrf_score"] += weight / (k + rank)
            docs[key].metadata["ranks"][channel] = rank
    return sorted(docs.values(), key=lambda d: (-d.metadata["rrf_score"], d.metadata["chunk_id"]))


def budgeted(documents: list[Document], budget: int, *, input_text: bool = False) -> list[Document]:
    selected = []
    for doc in documents:
        content = (
            doc.metadata.get("embedding_input", doc.page_content)
            if input_text
            else doc.page_content
        )
        size = len(content.encode())
        if size <= budget:
            selected.append(doc)
            budget -= size
    return selected


class HybridRetriever(BaseRetriever):
    """Bound by server code to an immutable tenant scope; async production path.

    search() returns explicit health metadata; ainvoke() implements LangChain's
    retriever contract and includes that metadata on every returned Document.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    settings: Settings = Field(exclude=True)
    scope: RetrievalScope
    store: RetrievalStore = Field(exclude=True)
    accounting: RetrievalAccounting = Field(exclude=True)
    embedder_factory: Callable[[], Any] = Field(exclude=True)
    reranker_factory: Callable[[], Any] = Field(exclude=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise NotImplementedError("Use ainvoke")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        result = await self.search(query)
        for doc in result.documents:
            doc.metadata["retrieval_degraded"] = result.degraded
            doc.metadata["operation_id"] = str(result.operation_id)
        return result.documents

    async def search(
        self,
        query: str,
        *,
        variant: Variant = "hybrid_rerank",
        probes: list[str] | None = None,
        section_hint: str | None = None,
        modality_hint: str | None = None,
    ) -> RetrievalResult:
        with observation("retrieval.search", variant=variant) as span:
            result = await self._scoped_search(
                query,
                variant=variant,
                probes=probes,
                section_hint=section_hint,
                modality_hint=modality_hint,
            )
            span.update(
                operation_id=str(result.operation_id),
                document_count=len(result.documents),
                channel_counts=result.channel_counts,
                degraded=result.degraded,
                intent=result.plan.get("intent"),
                context_bytes=sum(len(d.page_content.encode()) for d in result.documents),
            )
            return result

    async def _scoped_search(
        self,
        query: str,
        *,
        variant: Variant,
        probes: list[str] | None,
        section_hint: str | None,
        modality_hint: str | None,
    ) -> RetrievalResult:
        query = query.strip()
        if not query or len(query.encode()) > self.settings.retrieval_query_bytes:
            raise HTTPException(422, "检索问题为空或超过长度限制")
        if probes is not None and (
            not probes
            or len(probes) > self.settings.retrieval_max_subqueries
            or any(
                not p.strip() or len(p.encode()) > self.settings.retrieval_query_bytes
                for p in probes
            )
        ):
            raise HTTPException(422, "多探针查询为空或超过限制")
        await self.store.validate_scope(self.scope)
        operation_id = uuid4()
        if self.scope.collection_ids == () or self.scope.paper_ids == ():
            return RetrievalResult([], operation_id=operation_id)
        async with self.accounting.slot(self.scope.user_id):
            return await self._search(
                query, variant, operation_id, probes, section_hint, modality_hint
            )

    async def _search(
        self,
        query: str,
        variant: Variant,
        operation_id: UUID,
        probes: list[str] | None = None,
        section_hint: str | None = None,
        modality_hint: str | None = None,
    ) -> RetrievalResult:
        degraded: list[str] = []
        channels: dict[str, list[Document]] = {}
        plan = route_query(query, max_queries=self.settings.retrieval_max_subqueries)
        if probes is not None:
            plan = replace(plan, queries=tuple(dict.fromkeys(p.strip() for p in probes)))
        if plan.intent == "overview" and probes is None and len(plan.queries) == 1:
            with observation("retrieval.overview") as span:
                hits = await self.store.overview(self.scope)
                span.update(candidate_count=len(hits))
            with observation("retrieval.expand_candidates") as span:
                units = await self.store.expand_candidates(self.scope, hits, overview=True)
                span.update(document_count=len(units))
            for unit in units:
                unit.metadata["retrieval_intent"] = plan.intent
            with observation("retrieval.finalize_context") as span:
                documents = await self.store.finalize_context(self.scope, units)
                span.update(document_count=len(documents))
            return RetrievalResult(
                documents,
                channel_counts={"overview": len(hits)},
                operation_id=operation_id,
                plan={"intent": plan.intent, "queries": [query]},
            )

        dictionary = await self.store.acronym_dictionary(self.scope)
        queries = list(
            dict.fromkeys(
                expand_acronyms(q, dictionary, max_bytes=self.settings.retrieval_query_bytes)
                for q in plan.queries
                if len(q.encode()) <= self.settings.retrieval_query_bytes
            )
        )
        scopes = [self.scope]
        if plan.intent == "comparison" and self.scope.paper_ids and len(self.scope.paper_ids) > 1:
            scopes = [
                RetrievalScope(self.scope.user_id, self.scope.collection_ids, (paper,))
                for paper in self.scope.paper_ids
            ]

        async def channel(name: str, calls: list[Awaitable[list[Document]]]) -> ChannelResults:
            with observation(
                f"retrieval.{name}", query_count=len(queries), scope_count=len(scopes)
            ) as span:
                results = await asyncio.gather(*calls, return_exceptions=True)
                failures = sum(isinstance(rows, BaseException) for rows in results)
                span.update(
                    candidate_count=sum(len(rows) for rows in results if isinstance(rows, list)),
                    failed_count=failures,
                    status="degraded" if failures else "ok",
                )
                return results

        async def vector() -> ChannelResults:
            embedder = self.embedder_factory()
            try:
                with observation("retrieval.query_embedding", query_count=len(queries)):
                    values = await self.accounting.call(
                        self.scope.user_id,
                        operation_id,
                        "query_embedding",
                        sum(len(q.encode()) + 8 for q in queries),
                        lambda: embedder.aembed_documents(queries),
                        lambda: embedder.last_usage,
                    )
                    if len(values) != len(queries) or any(
                        len(value) != self.settings.embedding_dim
                        or any(not math.isfinite(float(n)) for n in value)
                        or not any(value)
                        for value in values
                    ):
                        raise ValueError("Invalid query vector")
                return await channel(
                    "vector",
                    [
                        self.store.vector(scope, value, self.settings.retrieval_top_k, q)
                        for scope in scopes
                        for q, value in zip(queries, values, strict=True)
                    ],
                )
            finally:
                await embedder.aclose()

        tasks: dict[str, Awaitable[ChannelResults]] = {}
        if variant != "bm25":
            tasks["vector"] = vector()
        if variant != "vector":
            tasks["bm25"] = channel(
                "bm25",
                [
                    self.store.bm25(scope, q, self.settings.retrieval_top_k)
                    for scope in scopes
                    for q in queries
                ],
            )
        if plan.metric_focus and variant in {"hybrid", "hybrid_rerank"}:
            tasks["metadata"] = channel(
                "metadata",
                [
                    self.store.metadata(scope, q, self.settings.retrieval_top_k)
                    for scope in scopes
                    for q in queries
                ],
            )
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        def failed(name: str, error: BaseException) -> None:
            # Quota/auth failures and cancellations never become degradation.
            if isinstance(error, HTTPException) or not isinstance(error, Exception):
                raise error
            marker = f"{name}_unavailable"
            if marker not in degraded:
                degraded.append(marker)
                log.warning(
                    "retrieval_channel_failed",
                    channel=name,
                    error_type=type(error).__name__,
                    operation_id=str(operation_id),
                )

        for name, result in zip(tasks, results, strict=True):
            if isinstance(result, BaseException):
                failed(name, result)
            else:
                for index, rows in enumerate(result):
                    if isinstance(rows, BaseException):
                        failed(name, rows)
                    else:
                        channels[name if len(result) == 1 else f"{name}:{index}"] = rows
        if not channels:
            raise RetrievalUnavailable("所有检索通道暂时不可用")
        policy = fusion_policy(
            plan,
            self.settings.rrf_vector_weight,
            self.settings.rrf_bm25_weight,
            self.settings.rrf_k,
        )
        weights = {
            "vector": policy["vector"],
            "bm25": policy["bm25"],
            "metadata": self.settings.rrf_metadata_weight,
        }
        with observation("retrieval.rrf") as span:
            documents = fuse(
                channels,
                policy["k"],
                {
                    name: weights[name.split(":")[0]] / (len(queries) * len(scopes))
                    for name in channels
                },
            )
            span.update(candidate_count=len(documents))
        for doc in documents:
            beta = 1.0
            if section_hint and section_std(doc.metadata.get("section_path", [])) == section_hint:
                beta *= 1.3
            if modality_hint and doc.metadata.get("kind") == modality_hint:
                beta *= 1.3
            doc.metadata["soft_weight_beta"] = beta
            doc.metadata["rrf_score"] *= beta
        documents.sort(key=lambda d: (-d.metadata["rrf_score"], d.metadata["chunk_id"]))
        if plan.intent == "comparison":
            # Round-robin papers before the global candidate cap, so one paper
            # cannot consume the entire comparison's recall budget.
            by_paper: dict[str, list[Document]] = {}
            for doc in documents:
                by_paper.setdefault(doc.metadata["paper_id"], []).append(doc)
            documents = [
                rows[i]
                for i in range(max((len(rows) for rows in by_paper.values()), default=0))
                for rows in by_paper.values()
                if i < len(rows)
            ]
        with observation("retrieval.expand_candidates") as span:
            documents = await self.store.expand_candidates(
                self.scope, documents[: self.settings.retrieval_top_k]
            )
            documents = budgeted(documents, self.settings.rerank_candidate_bytes, input_text=True)
            span.update(document_count=len(documents))
        for doc in documents:
            doc.metadata["retrieval_intent"] = plan.intent
        if documents and variant == "hybrid_rerank":
            reranker = self.reranker_factory()
            rerank_query = queries[0]
            # Query can be paired with every candidate by the provider tokenizer.
            amount = sum(
                len(d.metadata["embedding_input"].encode()) + len(rerank_query.encode()) + 8
                for d in documents
            )
            try:
                with observation("retrieval.rerank", candidate_count=len(documents)) as span:
                    documents = list(
                        await self.accounting.call(
                            self.scope.user_id,
                            operation_id,
                            "rerank",
                            amount,
                            lambda: reranker.acompress_documents(documents, rerank_query),
                            lambda: reranker.last_usage,
                        )
                    )
                    documents = [
                        doc
                        for doc in documents
                        if doc.metadata.get("rerank_score", 1.0)
                        >= self.settings.rerank_score_threshold
                    ]
                    span.update(document_count=len(documents))
            except HTTPException:
                raise
            except Exception as error:
                degraded.append("rerank_unavailable")
                log.warning(
                    "retrieval_rerank_failed",
                    error_type=type(error).__name__,
                    operation_id=str(operation_id),
                )
        with observation("retrieval.finalize_context") as span:
            documents = await self.store.finalize_context(
                self.scope, documents[: self.settings.rerank_top_n]
            )
            span.update(
                document_count=len(documents),
                context_bytes=sum(len(d.page_content.encode()) for d in documents),
            )
        counts: dict[str, int] = {}
        for name, rows in channels.items():
            base = name.split(":")[0]
            counts[base] = counts.get(base, 0) + len(rows)
        return RetrievalResult(
            documents,
            degraded,
            counts,
            operation_id,
            {
                "intent": plan.intent,
                "queries": queries,
                "alpha": policy["alpha"],
                "rrf_k": policy["k"],
                "section_hint": section_hint,
                "modality_hint": modality_hint,
            },
        )


def create_retriever(
    settings: Settings,
    sessions: Any,
    engine: Any,
    scope: RetrievalScope,
    *,
    run_id: UUID | None = None,
) -> HybridRetriever:
    return HybridRetriever(
        settings=settings,
        scope=scope,
        store=RetrievalStore(sessions, settings),
        accounting=RetrievalAccounting(sessions, engine, settings, run_id=run_id),
        embedder_factory=lambda: AccountedEmbeddings(settings),
        reranker_factory=lambda: RerankCompressor(settings=settings),
    )
