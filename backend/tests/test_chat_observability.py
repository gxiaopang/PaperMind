import asyncio

import pytest
from pydantic import SecretStr

from app.core.observability import run_trace_id
from app.services.ingestion.lifecycle import tombstone
from app.services.retrieval.types import RetrievalScope
from app.workers.chat import execute
from tests.test_chat import chat_setup, run_result, submit_run  # noqa: F401
from tests.test_observability import exported_json, metadata, sdk_client  # noqa: F401
from tests.test_retrieval import retrieval_setup  # noqa: F401


def attach(ctx, factory):
    tracing, exporter = factory()
    ctx["tracing"] = tracing
    ctx["settings"].langfuse_public_key = SecretStr(tracing._resources.public_key)
    return tracing, exporter


@pytest.mark.parametrize("scenario", ["fast", "fallback", "failure", "degraded", "revoked"])
async def test_committed_result_scores(client, chat_setup, sdk_client, scenario):  # noqa: F811
    owner, cid, ctx, models, papers, _ = chat_setup
    tracing, exporter = attach(ctx, sdk_client)
    if scenario == "fallback":
        models.answers = ["没有引用的事实。"]
    elif scenario == "failure":
        models.error_node = "route"
    elif scenario == "degraded":
        ctx["services"].retriever_factory(
            ctx["settings"], ctx["session_factory"], ctx["engine"], RetrievalScope(owner)
        ).reranker_factory().error = TimeoutError("private provider body")
    elif scenario == "revoked":
        original = models.invoke

        async def invoke(run, node, *args, **kwargs):
            output = await original(run, node, *args, **kwargs)
            if node == "verify_grounding":
                async with ctx["session_factory"]() as db:
                    for paper in papers[:2]:
                        await tombstone(db, await db.get(type(paper), paper.id))
                    await db.commit()
            return output

        models.invoke = invoke
    rid = await submit_run(
        client,
        cid,
        mode="fast" if scenario == "fast" else "agentic",
        scope={} if scenario == "revoked" else None,
    )
    result = await run_result(client, ctx, owner, rid)
    scores = {c.kwargs["name"]: c.kwargs for c in tracing.create_score.call_args_list}
    assert scores["result_status"]["value"] == (result["result_status"] or result["status"])
    assert scores["retrieval_retries"]["value"] == max(0, result["retrieval_count"] - 1)
    if scenario == "fast":
        assert scores["grounding"]["value"] == "not_run"
        assert scores["citation_validation"]["value"] == "passed"
    elif scenario in {"fallback", "revoked"}:
        assert scores["result_status"]["value"] == "safe_fallback"
        assert scores["citation_validation"]["value"] == "not_applicable"
    elif scenario == "failure":
        assert scores["result_status"]["value"] == "failed"
        assert scores["grounding"]["value"] == "not_run"
    else:
        assert scores["degraded"]["value"] == 1
    await execute(ctx, str(owner), rid)
    assert {c.kwargs["score_id"] for c in tracing.create_score.call_args_list} == {
        c["score_id"] for c in scores.values()
    }
    tracing.flush()
    assert "private provider body" not in exported_json(exporter)
    final = [s for s in exporter.get_finished_spans() if s.name == "papermind.chat.attempt"][-1]
    assert metadata(final.attributes, "result_status") == (
        result["result_status"] or result["status"]
    )
    assert metadata(final.attributes, "tokens") == result["usage"]["tokens"]
    assert metadata(final.attributes, "degraded") == (scenario == "degraded")
    assert metadata(final.attributes, "sources_revoked") == (scenario == "revoked")


async def test_resumed_worker_reuses_trace_and_emits_final_scores(client, chat_setup, sdk_client):  # noqa: F811
    owner, cid, ctx, models, _, _ = chat_setup
    tracing, exporter = attach(ctx, sdk_client)
    rid = await submit_run(client, cid)
    models.crash = True
    with pytest.raises(asyncio.CancelledError):
        await execute(ctx, str(owner), rid)
    assert not tracing.create_score.called
    result = await run_result(client, ctx, owner, rid)
    assert result["status"] == "completed"
    tracing.flush()
    roots = [s for s in exporter.get_finished_spans() if s.name == "papermind.chat.attempt"]
    assert {f"{s.context.trace_id:032x}" for s in roots} == {run_trace_id(owner, rid)}
    assert {metadata(s.attributes, "attempt") for s in roots} == {1, 2}
    assert metadata(roots[-1].attributes, "resumed") is True
    assert metadata(roots[-1].attributes, "generation_retries") == 1
    assert metadata(roots[-1].attributes, "recoveries") == 1
