import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langfuse import Langfuse
from langgraph.graph import END, START, StateGraph
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.core.config import Settings
from app.core.observability import (
    CONTENT_DISABLED,
    MAX_CONTENT_CHARS,
    Observation,
    export_mask,
    observation,
    publish_result,
    run_trace,
    run_trace_id,
    trace_context,
    tracing_client,
)
from app.services.chat.telemetry import result_metrics
from app.services.retrieval.accounting import RetrievalAccounting
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.store import RetrievalStore
from app.services.retrieval.types import RetrievalScope
from app.workers.chat import graph_config


def make_run(**overrides):
    now = datetime.now(UTC)
    values = dict(
        id=uuid4(),
        user_id=uuid4(),
        conversation_id=uuid4(),
        mode="fast",
        config={"prompt_version": "test-v1"},
        status="completed",
        result_status="answered",
        retrieval_count=2,
        generation_count=3,
        recoveries=2,
        created_at=now,
        finished_at=now + timedelta(seconds=5),
    )
    return SimpleNamespace(**{**values, **overrides})


@pytest.fixture
def sdk_client():
    clients = []

    def factory(*, content_enabled=False):
        exporter = InMemorySpanExporter()
        client = Langfuse(
            public_key=f"pk-test-{uuid4().hex}",
            secret_key="test-only-secret",
            base_url="http://localhost:1",
            tracer_provider=TracerProvider(),
            span_exporter=exporter,
            mask_otel_spans=export_mask(content_enabled=content_enabled),
            environment="test",
        )
        # Scores normally use a separate HTTP ingestion queue. Keep this test local.
        client.create_score = Mock()
        clients.append(client)
        return client, exporter

    yield factory
    for client in clients:
        client.shutdown()


def attrs(exporter, name):
    return next(s.attributes for s in exporter.get_finished_spans() if s.name == name)


def exported_json(exporter):
    return json.dumps(
        [
            {
                "name": s.name,
                "attributes": dict(s.attributes),
                "status": s.status.description,
                "events": [
                    {"name": e.name, "attributes": dict(e.attributes or {})} for e in s.events
                ],
            }
            for s in exporter.get_finished_spans()
        ],
        ensure_ascii=False,
    )


def metadata(attributes, key):
    value = attributes.get(
        f"langfuse.observation.metadata.{key}", attributes.get(f"langfuse.trace.metadata.{key}")
    )
    return json.loads(value)


def test_export_masks_content_and_preserves_diagnostics(sdk_client):
    client, exporter = sdk_client()
    run = make_run()
    with run_trace(client, run):
        with client.start_as_current_observation(
            name="sensitive",
            input={"question": "private question"},
            output="private answer",
            metadata={
                "run_id": str(run.id),
                "document_count": 2,
                "question": "private question",
                "documents": [{"text": "private paper"}],
                "memory": "private memory",
            },
        ):
            pass
    client.flush()
    exported = attrs(exporter, "sensitive")
    assert exported["langfuse.observation.input"] == CONTENT_DISABLED
    assert exported["langfuse.observation.output"] == CONTENT_DISABLED
    assert metadata(exported, "run_id") == str(run.id)
    assert metadata(exported, "document_count") == 2
    assert not any(k.endswith((".question", ".documents", ".memory")) for k in exported)
    assert "private" not in exported_json(exporter)


def test_enabled_content_redacts_secrets_and_limits_size(sdk_client):
    client, exporter = sdk_client(content_enabled=True)
    with client.start_as_current_observation(
        name="opt-in",
        input={
            "question": "hello",
            "api_key": "private-key",
            "nested": [{"password": "private-password"}],
            "message": "me@example.org Bearer private-token",
        },
        output="a" * (MAX_CONTENT_CHARS * 2),
    ):
        pass
    client.flush()
    exported = attrs(exporter, "opt-in")
    decoded = json.loads(exported["langfuse.observation.input"])
    assert decoded["question"] == "hello"
    assert decoded["api_key"] == decoded["nested"][0]["password"] == "[redacted]"
    assert "example.org" not in decoded["message"]
    assert "private" not in decoded["message"]
    assert len(exported["langfuse.observation.output"]) <= MAX_CONTENT_CHARS


async def test_graph_callbacks_share_stable_trace_and_keep_node_parent(sdk_client):
    client, exporter = sdk_client()
    run = make_run()
    settings = SimpleNamespace(
        langfuse_public_key=SimpleNamespace(get_secret_value=lambda: client._resources.public_key)
    )

    async def node(state):
        with observation("inside-node"):
            await asyncio.sleep(0)
        return {"answer": "private answer", "question": state["question"]}

    graph = StateGraph(dict)
    graph.add_node("retrieve", node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", END)
    compiled = graph.compile()
    for attempt in (1, 2):
        with run_trace(client, run, attempt=attempt):
            config = graph_config({"tracing": client, "settings": settings}, run)
            await asyncio.wait_for(
                compiled.ainvoke(
                    {"question": "private question", "memory": "private memory"}, config
                ),
                timeout=15,
            )
    client.flush()
    spans = exporter.get_finished_spans()
    tid = run_trace_id(run.user_id, run.id)
    assert {f"{s.context.trace_id:032x}" for s in spans} == {tid}
    roots = [s for s in spans if s.name == "papermind.chat.attempt"]
    assert len({s.context.span_id for s in roots}) == 2
    assert {metadata(s.attributes, "attempt") for s in roots} == {1, 2}
    by_id = {s.context.span_id: s for s in spans}
    for span in spans:
        if span.name == "inside-node":
            assert by_id[span.parent.span_id].name == "retrieve"
    assert "private" not in exported_json(exporter)


async def test_concurrent_runs_and_summary_do_not_share_context(sdk_client):
    client, exporter = sdk_client()
    runs = [make_run(), make_run()]

    async def work(run):
        with run_trace(client, run):
            with observation("parallel"):
                await asyncio.sleep(0)
                assert trace_context(run)["trace_id"] == run_trace_id(run.user_id, run.id)
        with run_trace(client, run, kind="summarize"):
            assert trace_context(run, "summarize")["trace_id"] != run_trace_id(run.user_id, run.id)

    await asyncio.gather(*(work(run) for run in runs))
    with observation("outside-run"):
        pass
    client.flush()
    spans = exporter.get_finished_spans()
    for span in spans:
        run_id = metadata(span.attributes, "run_id")
        run = next(r for r in runs if str(r.id) == run_id)
        kind = "summarize" if span.name == "papermind.summarize.attempt" else "chat"
        assert f"{span.context.trace_id:032x}" == run_trace_id(run.user_id, run.id, kind)
    assert not any(s.name == "outside-run" for s in spans)
    assert not trace.get_current_span().get_span_context().is_valid


def test_errors_and_cancellation_keep_types_without_private_messages(sdk_client):
    client, exporter = sdk_client()
    run = make_run()
    for error in (RuntimeError("private provider response"), asyncio.CancelledError("private")):
        with pytest.raises(type(error)), run_trace(client, run):
            with observation("failing-step"):
                raise error
    client.flush()
    spans = exporter.get_finished_spans()
    assert {metadata(s.attributes, "error_type") for s in spans} == {
        "RuntimeError",
        "CancelledError",
    }
    assert "private" not in exported_json(exporter)
    assert all(not s.events for s in spans)


def test_tracing_failures_do_not_swallow_business_errors():
    client = Mock()
    client.start_as_current_observation.side_effect = RuntimeError("SDK down")
    with run_trace(client, make_run()), observation("disabled"):
        pass
    with pytest.raises(ValueError, match="business failure"), run_trace(client, make_run()):
        raise ValueError("business failure")
    with run_trace(None, make_run()), observation("disabled") as span:
        assert span.span is None


def test_tracing_initialization_failure_is_optional(monkeypatch):
    constructor = Mock(side_effect=RuntimeError("SDK initialization failed"))
    monkeypatch.setattr("app.core.observability.Langfuse", constructor)
    settings = Settings(_env_file=None, tracing_enabled=False)
    assert tracing_client(settings) is None
    constructor.assert_not_called()
    settings.tracing_enabled = True
    assert tracing_client(settings) is None
    constructor.assert_called_once()


@pytest.mark.parametrize(
    "status,result_status,verification,citation,grounding",
    [
        (
            "completed",
            "answered",
            {"citations_valid": True, "grounding": "not_run"},
            "passed",
            "not_run",
        ),
        (
            "completed",
            "answered",
            {"citations_valid": True, "grounding": "passed"},
            "passed",
            "passed",
        ),
        (
            "completed",
            "safe_fallback",
            {"sources_revoked": True, "final_citations_valid": True},
            "not_applicable",
            "not_run",
        ),
        (
            "completed",
            "insufficient",
            {"grounding": "insufficient"},
            "not_applicable",
            "not_applicable",
        ),
        (
            "completed",
            "chitchat",
            {"grounding": "not_applicable"},
            "not_applicable",
            "not_applicable",
        ),
        ("failed", None, {}, "not_run", "not_run"),
    ],
)
def test_final_metrics_distinguish_skipped_checks(
    status, result_status, verification, citation, grounding
):
    run = make_run(status=status, result_status=result_status)
    metrics = result_metrics(run, verification, {"tokens": 42, "estimated": True}, 1)
    assert metrics["citation_validation"] == citation
    assert metrics["grounding"] == grounding
    assert metrics["retrieval_retries"] == 1
    assert metrics["generation_retries"] == 2
    assert metrics["recoveries"] == 1
    assert metrics["duration_seconds"] == 5
    assert metrics["tokens_estimated"] is True
    assert metrics["result_status"] == (result_status or status)


def test_final_scores_have_stable_ids_across_redelivery():
    client, run = Mock(), make_run()
    metrics = result_metrics(run, {}, {"tokens": 0, "estimated": False}, 0)
    for _ in range(2):
        publish_result(client, run, metrics, Observation())
    calls = [c.kwargs for c in client.create_score.call_args_list]
    assert calls[:6] == calls[6:]
    assert len({c["score_id"] for c in calls}) == 6
    assert next(c for c in calls if c["name"] == "grounding")["value"] == "not_run"


class Store(RetrievalStore):
    async def validate_scope(self, scope):
        pass

    async def acronym_dictionary(self, scope):
        return {}

    async def vector(self, *args):
        await asyncio.sleep(0)
        return [
            Document(
                page_content="private evidence",
                metadata={
                    "chunk_id": "chunk",
                    "paper_id": "paper",
                    "embedding_input": "private evidence",
                },
            )
        ]

    async def bm25(self, *args):
        return await self.vector()

    async def metadata(self, *args):
        return await self.vector()

    async def expand_candidates(self, scope, documents, **kwargs):
        return documents

    async def finalize_context(self, scope, documents):
        return documents


class Accounting(RetrievalAccounting):
    @asynccontextmanager
    async def slot(self, user_id):
        yield

    async def call(self, user_id, operation_id, node, amount, callback, usage, **kwargs):
        return await callback()


@pytest.mark.parametrize("rerank_fails", [False, True])
async def test_retrieval_stages_parallel_parenting_and_degradation(sdk_client, rerank_fails):
    client, exporter = sdk_client()
    run = make_run()
    settings = Settings(_env_file=None, tracing_enabled=False)

    class Embedder:
        last_usage = 3

        async def aembed_documents(self, queries):
            return [[1.0] + [0.0] * (settings.embedding_dim - 1) for _ in queries]

        async def aclose(self):
            pass

    class Reranker:
        last_usage = 3

        async def acompress_documents(self, documents, query):
            if rerank_fails:
                raise TimeoutError("private provider response")
            return documents

    retriever = HybridRetriever(
        settings=settings,
        scope=RetrievalScope(run.user_id),
        store=Store(None, settings),
        accounting=Accounting(None, None, settings),
        embedder_factory=Embedder,
        reranker_factory=Reranker,
    )
    with run_trace(client, run):
        result = await retriever.search("attention mechanism")
    client.flush()
    spans = exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    stages = {
        "query_embedding",
        "vector",
        "bm25",
        "rrf",
        "expand_candidates",
        "rerank",
        "finalize_context",
    }
    assert {f"retrieval.{stage}" for stage in stages} <= by_name.keys()
    search_id = by_name["retrieval.search"].context.span_id
    assert all(by_name[f"retrieval.{stage}"].parent.span_id == search_id for stage in stages)
    assert result.degraded == (["rerank_unavailable"] if rerank_fails else [])
    assert metadata(by_name["retrieval.search"].attributes, "degraded") == result.degraded
    assert metadata(by_name["retrieval.finalize_context"].attributes, "document_count") == 1
    if rerank_fails:
        assert metadata(by_name["retrieval.rerank"].attributes, "error_type") == "TimeoutError"
    assert "private" not in exported_json(exporter)
